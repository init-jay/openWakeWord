#!/usr/bin/env python3
"""Live wake-word detection from the microphone.

Streams audio from ffmpeg's avfoundation input straight into the model, so no
PortAudio/PyAudio is needed (the example script in examples/ requires pyaudio).
Shows a live score meter and announces activations.

Examples
--------
    .venv/bin/python scripts/detect_live.py --model models/hey_seeree.onnx
    .venv/bin/python scripts/detect_live.py --model models/hey_seeree.onnx \\
        --threshold 0.3 --device 1
"""

import argparse
import subprocess
import sys
import time

import numpy as np

from openwakeword.model import Model

SR = 16000
FRAME = 1280           # 80 ms, the model's native step
BYTES = FRAME * 2      # int16


def meter(score: float, width: int = 32) -> str:
    filled = int(score * width)
    return "#" * filled + "-" * (width - filled)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default="0", help="avfoundation audio device number")
    p.add_argument("--refractory", type=float, default=1.5,
                   help="seconds to suppress repeat activations after one fires")
    p.add_argument("--vad", type=float, default=0.0,
                   help="optional Silero VAD threshold (0 disables); suppresses "
                        "activations when no speech is present")
    args = p.parse_args()

    kwargs = {"vad_threshold": args.vad} if args.vad > 0 else {}
    model = Model(wakeword_models=[args.model], inference_framework="onnx", **kwargs)
    key = list(model.models.keys())[0]

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "avfoundation", "-i", f":{args.device}",
        "-ar", str(SR), "-ac", "1", "-f", "s16le", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    print(f"listening for '{key}'  (threshold {args.threshold})   Ctrl-C to stop\n")
    last_fire = 0.0
    peak_since = 0.0
    n_fired = 0

    try:
        while True:
            buf = proc.stdout.read(BYTES)
            if not buf or len(buf) < BYTES:
                err = proc.stderr.read().decode(errors="replace")
                raise SystemExit(f"audio stream ended.\n{err}")

            audio = np.frombuffer(buf, dtype=np.int16)
            score = model.predict(audio)[key]
            peak_since = max(peak_since, score)

            now = time.time()
            if score >= args.threshold and (now - last_fire) > args.refractory:
                n_fired += 1
                last_fire = now
                print(f"\r  *** DETECTED ***  score {score:.3f}   "
                      f"(activation #{n_fired})".ljust(70))
                peak_since = 0.0
            else:
                print(f"\r  {meter(score)}  {score:.3f}   peak {peak_since:.3f}",
                      end="", flush=True)
    except KeyboardInterrupt:
        print(f"\n\nstopped. {n_fired} activation(s).")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
