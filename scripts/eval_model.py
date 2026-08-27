#!/usr/bin/env python3
"""Score a wake-word model against real recordings and against audio that must
not trigger it.

Reports the score distribution over positives (how reliably it catches the
phrase) and over negatives (how close it comes to firing when it should not),
at several thresholds.

Examples
--------
    .venv/bin/python scripts/eval_model.py --model models/hey_seeree.onnx \\
        --positives ~/Documents/Repos/openwakeword-training/my_real_samples/jay \\
        --negatives tests/data
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from openwakeword.model import Model

FRAME = 1280  # 80 ms at 16 kHz, the model's native step


def score_clip(model: Model, key: str, path: Path, pad_post: float = 1.0) -> float:
    """Peak streaming score over a clip, mimicking how detection runs live.

    Trailing silence is appended because the model scores a 16-frame (~1.28 s)
    context window and fires just *after* the phrase completes. A bare clip that
    ends on the last syllable never lets the window slide over the whole phrase,
    so isolated short clips score near zero no matter how good the model is.
    Live from a microphone audio always continues past the phrase, so padding
    here reproduces real conditions rather than flattering the model.
    """
    model.reset()
    sr, dat = wavfile.read(path)
    if dat.ndim > 1:
        dat = dat.mean(axis=1).astype(np.int16)
    if sr != 16000:
        return float("nan")
    if pad_post > 0:
        dat = np.concatenate([dat, np.zeros(int(pad_post * sr), dtype=np.int16)])
    best = 0.0
    for i in range(0, len(dat) - FRAME, FRAME):
        best = max(best, model.predict(dat[i:i + FRAME])[key])
    return best


def report(name: str, scores: np.ndarray, thresholds, positive: bool) -> None:
    if scores.size == 0:
        print(f"{name}: no clips")
        return
    print(f"\n{name}  ({len(scores)} clips)")
    print(f"  mean {scores.mean():.3f}   median {np.median(scores):.3f}   "
          f"min {scores.min():.3f}   max {scores.max():.3f}")
    for t in thresholds:
        n = int((scores >= t).sum())
        if positive:
            print(f"  threshold {t:.2f}: detected {n}/{len(scores)} "
                  f"({100 * n / len(scores):.0f}%)   missed {len(scores) - n}")
        else:
            print(f"  threshold {t:.2f}: FIRED on {n}/{len(scores)} "
                  f"({100 * n / len(scores):.0f}%)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--positives", default=None, help="dir of clips that SHOULD trigger")
    p.add_argument("--negatives", default=None, help="dir of clips that should NOT trigger")
    p.add_argument("--thresholds", default="0.5,0.3,0.1")
    p.add_argument("--pad-post", type=float, default=1.0,
                   help="seconds of trailing silence appended to each clip; see score_clip")
    p.add_argument("--show-worst", type=int, default=5)
    args = p.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",")]
    model = Model(wakeword_models=[args.model], inference_framework="onnx")
    key = list(model.models.keys())[0]
    print(f"model: {args.model}  (output '{key}')")

    if args.positives:
        clips = sorted(Path(args.positives).glob("*.wav"))
        scored = [(score_clip(model, key, c, args.pad_post), c) for c in clips]
        scored = [(s, c) for s, c in scored if not np.isnan(s)]
        arr = np.array([s for s, _ in scored])
        report("POSITIVES (should trigger)", arr, thresholds, positive=True)
        if args.show_worst and scored:
            print(f"  worst {args.show_worst}:")
            for s, c in sorted(scored)[: args.show_worst]:
                print(f"    {s:.3f}  {c.name}")

    if args.negatives:
        clips = sorted(Path(args.negatives).glob("*.wav"))
        scored = [(score_clip(model, key, c, args.pad_post), c) for c in clips]
        scored = [(s, c) for s, c in scored if not np.isnan(s)]
        arr = np.array([s for s, _ in scored])
        report("NEGATIVES (should NOT trigger)", arr, thresholds, positive=False)
        if scored:
            print("  highest scoring:")
            for s, c in sorted(scored, reverse=True)[: args.show_worst]:
                print(f"    {s:.3f}  {c.name}")


if __name__ == "__main__":
    main()
