#!/usr/bin/env python3
"""Turn a handful of clean wake-word recordings into thousands of augmented
training positives for openWakeWord.

This mirrors what `openwakeword.data.augment_clips` does, but is written against
numpy/scipy alone so it runs on macOS without torch, speechbrain, audiomentations
or the Linux-only Piper toolchain. Output is 16 kHz mono 16-bit PCM WAV, sized to
`--total-length` samples, ready to drop into the training pipeline's
`positive_train/` and `positive_test/` directories.

Per output clip the chain is:

  1. random speed/pitch perturbation      (varies delivery)
  2. random spectral tilt                 (varies mic/room colouration)
  3. reverb via room impulse response     (real corpus, or synthesised)
  4. placement in the fixed-size window    ending near the window end, matching
                                           `create_fixed_size_clip` semantics
  5. background noise mixed at random SNR (real corpus, or synthesised)
  6. random gain, with clipping guarded

Real room impulse responses and background audio give the best results. Point
--rir-dir and --bg-dir at them when you have them; without them the script
synthesises both so you can still produce a usable set.

Examples
--------
Preview what one clip becomes, without writing a full set:
    .venv/bin/python scripts/augment_positives.py --in voice_data/positives_raw \\
        --out voice_data/preview --rounds 3 --limit 2

Full run, 40 variants per recording, with real corpora:
    .venv/bin/python scripts/augment_positives.py --in voice_data/positives_raw \\
        --out voice_data/positives_augmented --rounds 40 \\
        --rir-dir ./mit_rirs --bg-dir ./audioset_16k
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly, fftconvolve, lfilter

SR = 16000


# --------------------------------------------------------------------------- #
# io
# --------------------------------------------------------------------------- #

def read_wav(path: Path) -> np.ndarray:
    """Read a WAV as mono float32 in [-1, 1], resampling to 16 kHz if needed."""
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SR:
        g = np.gcd(int(sr), SR)
        data = resample_poly(data, SR // g, sr // g).astype(np.float32)
    return data


def write_wav(path: Path, x: np.ndarray) -> None:
    wavfile.write(path, SR, (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if x.size else 0.0


# --------------------------------------------------------------------------- #
# augmentation stages
# --------------------------------------------------------------------------- #

def trim_silence(x: np.ndarray, pad_ms: int = 60) -> np.ndarray:
    """Strip leading/trailing silence, keeping a short pad.

    Placement aligns the *end of the array* with the end of the window, so any
    trailing silence on the source would push the phrase earlier and teach the
    model an alignment that does not match how it fires when streaming.
    """
    frame = 160
    n = (len(x) // frame) * frame
    if n == 0:
        return x
    fr = np.sqrt((x[:n].reshape(-1, frame).astype(np.float64) ** 2).mean(axis=1))
    thresh = max(float(np.percentile(fr, 20)) * 4.0, float(fr.max()) * 0.08, 1e-4)
    speech = np.where(fr > thresh)[0]
    if speech.size == 0:
        return x
    pad = pad_ms // 10
    a = max(0, speech[0] - pad)
    b = min(len(fr), speech[-1] + 1 + pad)
    return x[a * frame: b * frame]


def speed_perturb(x: np.ndarray, rng, lo=0.90, hi=1.12) -> np.ndarray:
    """Resample to change speed and pitch together, as a real speaker would vary."""
    factor = rng.uniform(lo, hi)
    up = 1000
    down = max(1, int(round(up * factor)))
    return resample_poly(x, up, down).astype(np.float32)


def spectral_tilt(x: np.ndarray, rng) -> np.ndarray:
    """Mild single-pole tilt, standing in for microphone and room colouration."""
    a = rng.uniform(-0.4, 0.4)
    if abs(a) < 0.02:
        return x
    # a > 0 brightens (high-pass-ish), a < 0 darkens (low-pass-ish)
    return lfilter([1.0, -a], [1.0], x).astype(np.float32)


def synth_rir(rng) -> np.ndarray:
    """Synthesise a plausible room impulse response: direct path plus early
    reflections over an exponentially decaying noise tail."""
    rt60 = rng.uniform(0.12, 0.65)
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    tail = rng.normal(0, 1, n) * np.exp(-6.9 * t / rt60)

    ir = np.zeros(n, dtype=np.float32)
    ir[0] = 1.0  # direct path
    for _ in range(rng.integers(3, 9)):  # early reflections
        d = rng.integers(int(0.002 * SR), max(int(0.002 * SR) + 1, int(0.05 * SR)))
        if d < n:
            ir[d] += rng.uniform(-0.6, 0.6)
    ir += (tail * rng.uniform(0.1, 0.45)).astype(np.float32)
    return ir / (np.abs(ir).max() + 1e-9)


def apply_reverb(x: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve with an RIR, preserving the original level and onset alignment."""
    before = rms(x)
    wet = fftconvolve(x, rir)[: len(x) + len(rir)]
    # keep the direct-path onset where it was so end-placement stays meaningful
    peak = int(np.argmax(np.abs(rir)))
    wet = wet[peak: peak + len(x) + SR // 4]
    after = rms(wet)
    return (wet * (before / after)).astype(np.float32) if after > 0 else x


def place_in_window(x: np.ndarray, n_samples: int, rng, end_jitter=0.200) -> np.ndarray:
    """Pad into a fixed-size window so the utterance ends near the window end.

    Mirrors `openwakeword.data.create_fixed_size_clip`: streaming detection fires
    just after the phrase completes, so positives must sit at the window end.
    """
    out = np.zeros(n_samples, dtype=np.float32)
    if len(x) >= n_samples:
        return (x[:n_samples] if rng.random() >= 0.5 else x[-n_samples:]).astype(np.float32)
    jitter = int(rng.uniform(0, end_jitter) * SR)
    start = max(0, n_samples - (len(x) + jitter))
    out[start: start + len(x)] = x[: n_samples - start]
    return out


def background_for(n_samples: int, rng, bg_clips) -> np.ndarray:
    """A window of background audio: sampled from the corpus, else synthesised."""
    if bg_clips:
        bg = read_wav(bg_clips[rng.integers(len(bg_clips))])
        if len(bg) < n_samples:  # loop short clips up to length
            bg = np.tile(bg, int(np.ceil(n_samples / max(1, len(bg)))))
        off = rng.integers(0, max(1, len(bg) - n_samples))
        return bg[off: off + n_samples]
    # synthesised fallback: pink-ish noise via a one-pole filter on white noise
    white = rng.normal(0, 1, n_samples).astype(np.float32)
    return lfilter([1.0], [1.0, -rng.uniform(0.85, 0.98)], white).astype(np.float32)


def mix_at_snr(sig: np.ndarray, bg: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix background under signal at a target SNR in dB."""
    s, n = rms(sig), rms(bg)
    if n <= 0 or s <= 0:
        return sig
    return (sig + bg * (s / n) / (10 ** (snr_db / 20.0))).astype(np.float32)


# --------------------------------------------------------------------------- #

def augment_one(clip: np.ndarray, rng, args, rirs, bg_clips) -> np.ndarray:
    x = speed_perturb(clip, rng, args.speed_min, args.speed_max)
    x = spectral_tilt(x, rng)

    if rng.random() < args.reverb_prob:
        rir = read_wav(rirs[rng.integers(len(rirs))]) if rirs else synth_rir(rng)
        x = apply_reverb(x, rir)

    x = place_in_window(x, args.total_length, rng)

    if rng.random() < args.noise_prob:
        bg = background_for(args.total_length, rng, bg_clips)
        x = mix_at_snr(x, bg, rng.uniform(args.snr_min, args.snr_max))

    x = x * rng.uniform(args.gain_min, args.gain_max)

    peak = float(np.abs(x).max())
    if peak > 0.99:  # guard the clipping the gain stage can introduce
        x = x * (0.99 / peak)
    return x


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--in", dest="in_dir", default="voice_data/positives_raw")
    p.add_argument("--out", dest="out_dir", default="voice_data/positives_augmented")
    p.add_argument("--rounds", type=int, default=40, help="variants per input recording")
    p.add_argument("--limit", type=int, default=0, help="use only the first N recordings (0 = all)")
    p.add_argument("--total-length", type=int, default=32000,
                   help="output window in samples (32000 = 2s, the pipeline default)")
    p.add_argument("--rir-dir", default=None, help="directory of room impulse response WAVs")
    p.add_argument("--bg-dir", default=None, help="directory of background/noise WAVs")
    p.add_argument("--reverb-prob", type=float, default=0.8)
    p.add_argument("--noise-prob", type=float, default=0.9)
    p.add_argument("--snr-min", type=float, default=5.0)
    p.add_argument("--snr-max", type=float, default=20.0)
    p.add_argument("--speed-min", type=float, default=0.90)
    p.add_argument("--speed-max", type=float, default=1.12)
    p.add_argument("--gain-min", type=float, default=0.35)
    p.add_argument("--gain-max", type=float, default=1.0)
    p.add_argument("--no-trim", action="store_true",
                   help="skip silence-trimming of inputs (they must already be tight)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    in_dir = Path(args.in_dir)
    clips = sorted(q for q in in_dir.glob("*.wav") if not q.name.startswith("."))
    if not clips:
        raise SystemExit(f"no .wav files in {in_dir} - record some first with scripts/record_wakeword.py")
    if args.limit:
        clips = clips[: args.limit]

    rirs = sorted(Path(args.rir_dir).glob("**/*.wav")) if args.rir_dir else []
    bg_clips = sorted(Path(args.bg_dir).glob("**/*.wav")) if args.bg_dir else []

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{len(clips)} recordings x {args.rounds} rounds = {len(clips) * args.rounds} clips")
    print(f"window: {args.total_length} samples ({args.total_length / SR:.2f}s)")
    print(f"RIRs: {len(rirs) or 'synthesised'}   background: {len(bg_clips) or 'synthesised'}")

    n = 0
    for src in clips:
        clean = read_wav(src)
        if not args.no_trim:
            clean = trim_silence(clean)
        for r in range(args.rounds):
            write_wav(out_dir / f"{src.stem}_aug{r:03d}.wav",
                      augment_one(clean, rng, args, rirs, bg_clips))
            n += 1
        print(f"  {src.name} -> {args.rounds}", end="\r", flush=True)

    (out_dir / "augment_config.json").write_text(json.dumps(vars(args), indent=2))
    print(f"\nwrote {n} clips to {out_dir}/")
    print("\nNext: copy into the training pipeline, holding some back for validation:")
    print(f"  positive_train/  <- most of {out_dir}/")
    print("  positive_test/   <- a held-out slice (from recordings not used in train)")


if __name__ == "__main__":
    main()
