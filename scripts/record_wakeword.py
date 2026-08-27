#!/usr/bin/env python3
"""Record wake-word samples in your own voice, in the format openWakeWord expects
(16 kHz, mono, 16-bit PCM WAV).

Capture goes through ffmpeg's avfoundation input, so no PortAudio/PyAudio/system
packages are required on macOS. Only numpy and scipy are needed, both of which
live in the project venv.

Two capture modes:

  takes    One clip per take. You press ENTER, it records, and you keep or redo
           it. Slower, but you hear every clip and control quality per sample.

  session  One long continuous recording in which you repeat the phrase with a
           pause between each. The file is then split on silence into one clip
           per utterance. Much faster for collecting 50-100+ samples.

Every saved clip is auto-trimmed to the utterance (with a short pad), checked for
clipping and low level, and written as 16 kHz mono 16-bit PCM.

Examples
--------
List microphones:
    .venv/bin/python scripts/record_wakeword.py --list-devices

60 individual takes of "hey seeree":
    .venv/bin/python scripts/record_wakeword.py --phrase "hey seeree" -n 60

One continuous 3-minute session, auto-split into utterances:
    .venv/bin/python scripts/record_wakeword.py --phrase "hey seeree" \\
        --mode session --session-secs 180
"""

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile

SR = 16000
FRAME = 160  # 10 ms at 16 kHz
WARMUP = 0.6  # seconds avfoundation needs to open the mic before audio flows


# --------------------------------------------------------------------------- #
# audio helpers
# --------------------------------------------------------------------------- #

def frame_rms(x: np.ndarray) -> np.ndarray:
    """Short-time RMS over 10 ms frames of a float array in [-1, 1]."""
    n = (len(x) // FRAME) * FRAME
    if n == 0:
        return np.zeros(0)
    frames = x[:n].reshape(-1, FRAME).astype(np.float64)
    return np.sqrt((frames ** 2).mean(axis=1))


def speech_mask(rms: np.ndarray) -> tuple:
    """Boolean mask of frames that look like speech, plus the threshold used.

    The noise floor is estimated from the quietest 20% of frames so the
    threshold adapts to the room rather than assuming a fixed level.
    """
    if rms.size == 0:
        return np.zeros(0, dtype=bool), 0.0
    noise_floor = float(np.percentile(rms, 20))
    thresh = max(noise_floor * 4.0, float(rms.max()) * 0.08, 1e-4)
    return rms > thresh, thresh


def trim(x: np.ndarray, pad_ms: int = 100):
    """Trim leading/trailing silence, leaving `pad_ms` of padding on each side.

    Returns None when the clip contains no detectable speech.
    """
    rms = frame_rms(x)
    mask, _ = speech_mask(rms)
    if not mask.any():
        return None
    pad = pad_ms // 10
    first = max(0, int(np.argmax(mask)) - pad)
    last = min(len(mask), len(mask) - int(np.argmax(mask[::-1])) + pad)
    return x[first * FRAME: last * FRAME]


def split_on_silence(x: np.ndarray, min_silence_ms: int = 350, pad_ms: int = 100):
    """Split a long recording into utterances separated by silence."""
    rms = frame_rms(x)
    mask, _ = speech_mask(rms)
    if not mask.any():
        return []

    gap = min_silence_ms // 10
    pad = pad_ms // 10
    segments, start, silence = [], None, 0

    for i, is_speech in enumerate(mask):
        if is_speech:
            if start is None:
                start = i
            silence = 0
        elif start is not None:
            silence += 1
            if silence >= gap:
                segments.append((start, i - silence))
                start = None
    if start is not None:
        segments.append((start, len(mask)))

    clips = []
    for a, b in segments:
        a, b = max(0, a - pad), min(len(mask), b + pad)
        clips.append(x[a * FRAME: b * FRAME])
    return clips


def describe(x: np.ndarray) -> tuple:
    """Return (peak, message, ok) describing the level of a clip."""
    peak = float(np.abs(x).max()) if x.size else 0.0
    dur = len(x) / SR
    if peak >= 0.99:
        return peak, f"CLIPPING (peak {peak:.2f}) - move back or lower gain", False
    if peak < 0.05:
        return peak, f"very quiet (peak {peak:.2f}) - move closer or raise gain", False
    return peak, f"ok  {dur:.2f}s  peak {peak:.2f}", True


def save(x: np.ndarray, path: Path) -> None:
    wavfile.write(path, SR, (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16))


def read_wav(path: Path) -> np.ndarray:
    sr, data = wavfile.read(path)
    if sr != SR:
        raise SystemExit(f"expected {SR} Hz from ffmpeg, got {sr} Hz in {path}")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32) / 32768.0


# --------------------------------------------------------------------------- #
# ffmpeg capture
# --------------------------------------------------------------------------- #

def list_devices() -> None:
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True, text=True,
    ).stderr
    audio = out.split("AVFoundation audio devices:")
    if len(audio) < 2:
        print(out)
        return
    print("Audio input devices (use the number with --device):")
    for line in audio[1].splitlines():
        m = re.search(r"\[(\d+)\]\s+(.*)", line)
        if m:
            print(f"  {m.group(1)}: {m.group(2).strip()}")


def capture(seconds: float, device: str, path: Path, on_ready=None) -> np.ndarray:
    """Record `seconds` of audio from `device` straight to 16 kHz mono 16-bit.

    avfoundation takes a moment to open the device, during which nothing is
    captured. So ffmpeg is spawned first and given `WARMUP` seconds to come up
    before `on_ready` cues the speaker, and that warm-up is added to the
    requested length so the usable window after the cue is the full `seconds`.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "avfoundation", "-i", f":{device}",
        "-t", str(seconds + WARMUP), "-ar", str(SR), "-ac", "1", "-sample_fmt", "s16",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(WARMUP)
    if on_ready is not None:
        on_ready()
    _, stderr = proc.communicate()

    if proc.returncode != 0 or not path.exists():
        raise SystemExit(
            "ffmpeg capture failed. If this is the first run, macOS may need "
            "microphone permission for your terminal "
            "(System Settings > Privacy & Security > Microphone).\n\n"
            + stderr
        )
    return read_wav(path)


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #

def run_takes(args, out_dir: Path, slug: str, tmp: Path) -> int:
    saved = existing_count(out_dir, slug)
    target = saved + args.n
    print(f"\nRecording {args.n} takes of \"{args.phrase}\" ({saved} already on disk).")
    print("ENTER records a take. Type 's' to skip, 'q' to stop early.\n")

    while saved < target:
        cmd = input(f"[{saved + 1}/{target}] ENTER to record > ").strip().lower()
        if cmd == "q":
            break
        if cmd == "s":
            continue

        print("   opening mic...", end="", flush=True)
        raw = capture(
            args.take_secs, args.device, tmp,
            on_ready=lambda: print("\r   SPEAK NOW      ", end="", flush=True),
        )
        print("\r                     ", end="\r")

        clip = trim(raw, args.pad_ms)
        if clip is None:
            print("   no speech detected - retrying this take")
            continue

        peak, msg, ok = describe(clip)
        print(f"   {msg}")
        if not ok and not args.keep_bad:
            print("   discarded - retrying this take")
            continue

        saved += 1
        save(clip, out_dir / f"{slug}_{saved:04d}.wav")

    return saved


def run_session(args, out_dir: Path, slug: str, tmp: Path) -> int:
    saved = existing_count(out_dir, slug)
    print(f"\nContinuous session: {args.session_secs:.0f}s of \"{args.phrase}\".")
    print("Say the phrase, pause ~1 second, say it again. Vary distance, volume,")
    print("speed and tone as you go - that variety is what makes the model robust.\n")
    input("ENTER to start > ")

    print("   opening mic...", end="", flush=True)
    raw = capture(
        args.session_secs, args.device, tmp,
        on_ready=lambda: print("\r   RECORDING - go!   ", flush=True),
    )
    print("   done, splitting...")

    clips = split_on_silence(raw, args.min_silence_ms, args.pad_ms)
    kept = skipped = 0
    for clip in clips:
        dur = len(clip) / SR
        if not (args.min_dur <= dur <= args.max_dur):
            skipped += 1
            continue
        _, _, ok = describe(clip)
        if not ok and not args.keep_bad:
            skipped += 1
            continue
        saved += 1
        kept += 1
        save(clip, out_dir / f"{slug}_{saved:04d}.wav")

    print(f"   found {len(clips)} utterances, kept {kept}, skipped {skipped} "
          f"(outside {args.min_dur}-{args.max_dur}s or bad level)")
    return saved


def existing_count(out_dir: Path, slug: str) -> int:
    """Highest index already on disk, so repeat runs append instead of overwrite."""
    indices = [
        int(m.group(1))
        for p in out_dir.glob(f"{slug}_*.wav")
        if (m := re.search(rf"{re.escape(slug)}_(\d+)\.wav$", p.name))
    ]
    return max(indices, default=0)


# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--phrase", default="hey seeree", help="wake word / phrase to record")
    p.add_argument("--speaker", default=None,
                   help="speaker name; recordings go to voice_data/<speaker>/positives_raw/. "
                        "Keeping speakers in separate directories is what lets the "
                        "train/test split stay speaker-aware and lets you add a speaker later.")
    p.add_argument("--out", default=None,
                   help="explicit output directory (overrides --speaker)")
    p.add_argument("--device", default="0", help="avfoundation audio device number")
    p.add_argument("--list-devices", action="store_true", help="list microphones and exit")
    p.add_argument("--mode", choices=["takes", "session"], default="takes")
    p.add_argument("-n", type=int, default=60, help="number of takes (takes mode)")
    p.add_argument("--take-secs", type=float, default=2.5, help="seconds per take")
    p.add_argument("--session-secs", type=float, default=180.0, help="session length")
    p.add_argument("--min-silence-ms", type=int, default=350, help="split gap (session mode)")
    p.add_argument("--pad-ms", type=int, default=100, help="padding kept around each utterance")
    p.add_argument("--min-dur", type=float, default=0.35, help="shortest kept utterance (s)")
    p.add_argument("--max-dur", type=float, default=2.5, help="longest kept utterance (s)")
    p.add_argument("--keep-bad", action="store_true", help="keep clipped/quiet clips too")
    args = p.parse_args()

    if args.list_devices:
        list_devices()
        return

    slug = re.sub(r"[^a-z0-9]+", "_", args.phrase.lower()).strip("_")

    if args.out:
        out_dir = Path(args.out)
    elif args.speaker:
        speaker = re.sub(r"[^a-z0-9]+", "_", args.speaker.lower()).strip("_")
        out_dir = Path("voice_data") / speaker / "positives_raw"
    else:
        raise SystemExit("pass --speaker NAME (recommended) or --out DIR")
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / ".capture.wav"

    try:
        if args.mode == "takes":
            total = run_takes(args, out_dir, slug, tmp)
        else:
            total = run_session(args, out_dir, slug, tmp)
    except KeyboardInterrupt:
        print("\ninterrupted")
        total = existing_count(out_dir, slug)
    finally:
        tmp.unlink(missing_ok=True)

    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps({
        "phrase": args.phrase,
        "speaker": args.speaker,
        "slug": slug,
        "sample_rate": SR,
        "clips": sorted(p.name for p in out_dir.glob(f"{slug}_*.wav")),
    }, indent=2))

    print(f"\n{total} clips in {out_dir}/  (manifest: {manifest})")
    if total:
        durs = [len(read_wav(p)) / SR for p in sorted(out_dir.glob(f"{slug}_*.wav"))]
        print(f"duration: mean {np.mean(durs):.2f}s  min {min(durs):.2f}s  max {max(durs):.2f}s")


if __name__ == "__main__":
    main()
