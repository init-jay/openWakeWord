# Wake word latency plan

**Goal: reduce the time a user waits between finishing the wake phrase and the
detection firing**, subject to a CPU budget that keeps an always-on detector viable
on the target device.

Current felt latency: **191 ms median** from end of speech, at a 40 ms prediction step
(56/56 real clips at threshold 0.5). It was 220 ms at the original 80 ms step.

> **Done: option 2, the finer frame step.** `step_samples=640` cuts 29.5 ms and
> improves detection from 55/56 to 56/56, for 2x the CPU (1.83% -> 3.62% of a core).
> Default behaviour is unchanged and bit-identical; this is opt-in.

The uncomfortable headline, measured rather than assumed:

> **Every CPU-side lever available, combined, is worth about 60 ms of the original
> 220 ms** — roughly half of which is now banked. The remaining ~170 ms is the model
> needing to see the phrase before its score rises. Only a training change moves it,
> and that is the option with by far the highest ceiling.

Compute optimization is **not** a latency lever. It reaches felt latency through one
narrow channel — funding a finer frame step — and is otherwise a duty-cycle concern.
All of it lives in `compute-cost.md`; the 1.43x banked there changes felt
responsiveness by ~0.3%.

---

## Method

- Machine: MacBookPro18,2 (M1 Pro, 10 cores), macOS 15.5
- Model: `openwakeword/resources/models/hey_seeree.onnx`, ONNX runtime
- Corpus: 56 real `hey_seeree` recordings
- Each clip is streamed frame by frame with 3 s of lead-in silence (to flush the
  feature-buffer priming) and 2 s trailing. Latency is the sample offset at which the
  score first crosses threshold, minus the clip's end-of-speech offset.
- "End of speech" is the last sample above 2% of peak amplitude

Latency is measured in **audio time**, not wall clock, which is the right frame for a
real-time system: audio arrives at 1x no matter how fast the model runs.

---

## Where the latency goes

Time from end of speech to the score first crossing 0.5:

| | ms |
|---|---:|
| median | **229** |
| mean | 182 |
| p10 - p90 | -107 - 307 |
| min - max | -370 - 420 |
| end of speech to *peak* score | 384 |

Fired on 54/56 clips at threshold 0.5. The negative fast tail is an artifact of the
amplitude-based end-of-speech marker: trailing breath or room noise pushes the marker
later than the spoken word.

```
  frame step     ████████                                  0-80 ms, avg ~40
  window fill    ████████████████████████████████████████  ~190 ms
  model compute  ▏                                         1.5 ms (0.65%)
                 └─────────┴──────────┴─────────┴──────────┴─────┘
                 0         50         100       150        200   229 ms
```

- **frame step** — inference only runs once 1,280 samples have accumulated, so a
  phrase ending mid-frame waits up to 80 ms.
- **window fill** — the wakeword model reads a sliding window covering **1.96 s** of
  audio. After the phrase ends the window keeps advancing and the score keeps
  climbing; the median clip peaks 384 ms after end of speech and crosses 0.5 well
  before that.
- **model compute** — **1.5 ms**, and only the *final* frame's compute counts. Frame
  N is processed while frame N+1's audio is still arriving, so every earlier frame's
  compute is hidden behind the wait for sound.

**The first two bars are audio time: the system is waiting for sound to physically
arrive.** At 1.5 ms of work per 80 ms of audio the detector is idle ~98% of the time.
Making the models faster does not shorten the window fill, because nothing is being
computed during it. A 7.5x faster embedding model would leave the detector idle 99.7%
of the time and fire at the same moment.

---

## Why the window fill exists: how the sliding windows nest

Three windows sit inside each other, each sliding at a different rate. All figures
were read back from the loaded model, not from documentation.

**1. Audio to melspectrogram.** Each step reads 480 samples of history plus the
1,280 new ones, so the spectrogram frames line up across step boundaries:

```
                  ├── 480 ───┤├───── 1,280 new samples ──────┤
                  ├────── 1,760 samples read each step ──────┤
                                       │
                                       ▼  melspectrogram model
                  ░░░░░░░░░░░░░░░░████████   8 new frames, 10 ms hop = 80 ms
                  └ already buffered
```

**2. Melspectrogram to embedding.** The embedding model reads a 76-frame window that
advances only 8 frames per step — so 89% of its input is re-read every step:

```
                  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  melspec buffer
                  ├──────── 76 frames = 760 ms ────────┤  step N-1
                      ├──────── 76 frames = 760 ms ────────┤  step N
                  ├──┤  stride 8 frames = 80 ms; 68 of 76 (89%) re-read
```

**3. Embedding to wakeword.** One embedding frame is produced per step, and the
wakeword model reads the last 16 — 94% shared with the previous step:

```
                  ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●  feature buffer
                  ├───────── 16 frames ──────────┤  step N-1
                    ├───────── 16 frames ──────────┤  step N
                  ├┤  stride 1 frame = 80 ms; 15 of 16 (94%) shared
```

**4. What one prediction actually sees.** Composing them: 16 embedding frames spaced
8 melspec frames apart, each 76 frames wide, is `76 + 15x8 = 196` melspec frames:

```
                  ├──── 196 melspec frames = 1.96 s of audio ─────┤
                  ├┤ one prediction advances the whole field 80 ms
```

**The commonly quoted "1.28 s context window" is the stride span** (16 x 80 ms). It
omits the 760 ms width of the embedding window itself. The true receptive field is
**1.96 s**, and it is the direct cause of the window fill: the phrase must sit far
enough inside that field for the score to rise.

This is also the number a training-side change would attack — see option 3.

---

## Remaining options

Ordered by ceiling, not by ease.

### 1. Shorten the model's receptive field — highest ceiling, unmeasured

The ~170 ms of window fill is set by the 1.96 s receptive field, and the receptive
field is set by how many embedding frames the wakeword head consumes:

| head input frames | receptive field | vs today |
|---:|---:|---:|
| 16 (today) | 1.96 s | — |
| 12 | 1.64 s | -320 ms |
| 10 | 1.48 s | -480 ms |
| 8 | 1.32 s | -640 ms |

This is a **config-level change in the existing training pipeline**, not model
surgery. `openwakeword.train.Model` takes `input_shape=(16, 96)` as a first-class
argument, and at `train.py:823` it is derived from the shape of the generated
feature array — so it follows from how training examples are cut, via
`seconds_per_example`. The embedding model is untouched; only the small wakeword head
is retrained.

Unknowns that must be measured, not assumed:

- **Latency does not necessarily fall 1:1 with the receptive field.** A shorter field
  may simply need the phrase positioned differently within it. Measure, do not
  extrapolate from the table above.
- **Shorter context costs accuracy.** "hey seeree" is ~0.8 s spoken, so a 1.32 s field
  still contains it, but with less margin for slow speakers and less surrounding
  context to reject near-misses. Expect a worse false-accept rate.
- A related and possibly cheaper variant: keep 16 frames but **change label
  alignment** so the model is trained to fire earlier in the phrase rather than after
  it completes.

This is the only option that attacks the 74% of latency everything else cannot reach.
It should be tried before any further compute work.

### 2. Finer frame step — DONE, 29.5 ms, costs CPU linearly

**Implemented.** `AudioFeatures` takes a `step_samples` argument, plumbed through
`Model(**kwargs)`:

```python
Model(wakeword_models=[path], inference_framework="onnx", step_samples=640)
```

It must divide 1280 evenly (1280 / 640 / 320 / 160 = 80 / 40 / 20 / 10 ms), because
the wakeword model always consumes 16 embeddings spaced 80 ms apart. A smaller step
does not change that grid — it interleaves `1280 // step_samples` phases of it, and
`get_features` strides by that factor to pick one out. No model change, no retraining.

Measured over all 56 clips, with a realistic noise floor rather than digital silence:

| step | detected | median | p90 | CPU per s of audio |
|---:|---:|---:|---:|---:|
| 1280 (80 ms) | 55/56 | 220.2 ms | 296.1 ms | 18.3 ms (1.83%) |
| **640 (40 ms)** | **56/56** | **190.7 ms** | 295.2 ms | 36.2 ms (3.62%) |

**29.5 ms faster, and it detects every clip** — slightly better than the 24 ms the
phase-interleaving simulation predicted. Cost is exactly 2x CPU, as expected.

Correctness: at the default step the output is **bit-identical** to before the change
(692 scores across all three `predict` paths, `max abs diff 0.0`). At 640 the
melspectrograms and wakeword inputs are **bit-identical to the 80 ms stream** given
any realistic noise floor.

One caveat, worth knowing because it affects the eval scripts: with *exact digital
silence* the two step sizes diverge on 8 of 633 melspectrogram frames, always at the
boundary where silence meets speech. This is the sensitivity `_streaming_melspectrogram`
already documents ("padding with 0 or very small values seems to demonstrate the
differences well") — the amount of lookback inside one melspectrogram call differs
between step sizes, and near-zero input makes that visible. It does not occur with mic
audio. `eval_model.py` and `compare_models.py` pad with `np.zeros`, so they sit exactly
on this case; padding with a few LSB of noise instead would remove the artifact.

Remaining headroom: 320 and 160 sample steps are supported and would cut the frame
quantization further, at 4x and 8x CPU. Extrapolating, the full remaining gain is only
another ~10-15 ms, so 640 is likely the sweet spot unless CPU is free.

#### How it was measured before implementing

Simulated by running two interleaved phases 640 samples apart and merging the score
streams — no model change needed, just 2x the embedding work.

| threshold | 80 ms step (now) | | 40 ms step (2 phases) | |
|---|---:|---:|---:|---:|
| | median | p90 | median | p90 |
| 0.5 | **229 ms** | 307 | 205 ms | 297 |
| 0.4 | 229 ms | 301 | 205 ms | 284 |
| 0.3 | 220 ms | 296 | 203 ms | 290 |
| 0.2 | 220 ms | 296 | 200 ms | 282 |
| 0.1 | 200 ms | 290 | **170 ms** | 278 |

Detection rate is 54/56 at 0.5 and 55/56 at 0.3 and below, in both step sizes.

Halving the step buys **~24 ms** for **2x the embedding compute** (duty cycle
1.82% -> 3.6%). Its ceiling is the melspec hop: a 10 ms step would remove frame
quantization almost entirely at 8x the compute, worth perhaps 35-40 ms total —
**extrapolated from one measured halving, not measured**.

This is the only option that compute work funds, and the reason `compute-cost.md`
exists.

### 3. Lower the threshold — measured, ~29 ms, weakest lever

Dropping 0.5 to 0.1 buys only **~29 ms**, because the score rises from below 0.1 to
above 0.5 within a single 80 ms frame in the median case. There is very little rising
edge to catch.

It also costs false accepts, and **that cost is not measured here** — the repo's
negative set is 3 clips, far too small. Any threshold change needs a proper negative
corpus first. Worst side effects, smallest gain: treat as a last resort.

### Not a latency option

Everything in `compute-cost.md` — threads, int8 quantization, the streaming
conversion of the embedding model. Those change duty cycle. They appear here only as
the budget that makes option 2 affordable.

---

## Reproducing

Latency is measured by streaming each real clip (3 s lead-in silence to flush the
feature-buffer priming, 2 s trailing) frame by frame, recording the sample offset at
which the score first crossed threshold, and subtracting the clip's end-of-speech
offset. A finer frame step is simulated by re-running each clip with the lead-in
padded by an extra 640 samples and merging the two score streams in time order.

Environment setup and the compute-side measurements are in `compute-cost.md`.
