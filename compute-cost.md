# Per-frame compute and duty cycle

Everything about how much CPU the always-on detector burns: where the time goes, what
has been optimized, and what is left to try.

Split out of `plan.md` deliberately. **Compute is 0.65% of felt latency** — 1.5 ms of
a 229 ms wait — so none of this makes the wake word feel faster. It matters for two
things only:

1. **Duty cycle** — how much of a core the detector burns, which decides whether it
   is viable on a Pi-class target at all.
2. **Funding the one latency lever it touches** — a finer frame step costs CPU
   linearly, so headroom here is what makes it affordable. See `plan.md`.

Every option below should be justified on those two grounds, never on responsiveness.

---

## Method

- Machine: MacBookPro18,2 (M1 Pro, 10 cores), macOS 15.5
- Model: `openwakeword/resources/models/hey_seeree.onnx`, `inference_framework="onnx"`,
  onnxruntime 1.29.0
- Workload: 30-40 s of audio through `Model.predict` in 1280-sample (80 ms) frames,
  timed at steady state (raw buffer full) after a warm-up pass
- 1.0 ms/frame = 80x realtime on one core

Correctness gate: 692 scores captured across the three `predict` paths (1280-sample,
sub-1280, and multi-chunk >1280), compared before/after each change.

---

## Where the time goes

Baseline was **2.146 ms/frame** (37x realtime). Component costs, measured
individually:

| Component | ms/frame | Note |
|---|---:|---|
| embedding model (ONNX) | 1.312 | Google `speech_embedding`, 76x32 window |
| `list(deque)` copy + `np.array` | 0.333 | pure waste, now removed |
| melspectrogram model (ONNX) | 0.102 | |
| wakeword model (ONNX) | 0.015 | |
| `vstack` on feature/melspec buffers | 0.008 | bounded buffers, not a problem |

The `list(deque)` cost is measured at a partly-filled buffer and grows to ~0.6 ms
once the 10 s buffer fills, because it is O(buffer length).

### The pipeline

Three models run back to back on every 80 ms frame. One of them is the whole cost:

```mermaid
flowchart LR
    IN["80 ms of audio<br/>1,280 samples"]
    RB["raw audio FIFO<br/>contiguous int16<br/>~0 ms"]
    MEL["melspectrogram<br/>model<br/>0.102 ms"]
    MB["melspec buffer<br/>76 x 32 window"]
    EMB["embedding model<br/>19 fused conv layers<br/>1.312 ms — 87%"]
    FB["feature buffer<br/>16 x 96"]
    WW["wakeword model<br/>0.015 ms"]
    OUT["score 0-1"]

    IN --> RB
    RB -->|last 1,760 samples| MEL
    MEL --> MB
    MB -->|slides 8 frames| EMB
    EMB --> FB
    FB -->|last 16 frames| WW
    WW --> OUT

    classDef hot fill:#c0392b,stroke:#7b241c,color:#ffffff
    classDef warm fill:#5d6d7e,stroke:#34495e,color:#ffffff
    classDef cool fill:#d5dbdb,stroke:#85929e,color:#1c2833
    class EMB hot
    class MEL,WW warm
    class IN,RB,MB,FB,OUT cool
```

The red box is the entire optimization problem. `melspectrogram` and `wakeword`
together are 0.117 ms — even making both instant would buy less than 8%.

### Inside the embedding model

Per-node ONNX Runtime profiling over 60 runs, 1396 us of kernel time per run:

| Op | us/run | nodes | share |
|---|---:|---:|---:|
| FusedConv | 1268.6 | 19 | 90.9% |
| Max | 61.2 | 19 | 4.4% |
| MaxPool | 49.1 | 5 | 3.5% |
| Conv | 12.9 | 1 | 0.9% |
| Reshape | 4.5 | 2 | 0.3% |

There is no wasted op to delete. It is genuinely compute-bound on convolution, which
is why every remaining option is "run the same convolutions less, or cheaper, or on
more cores".

---

## Done

### 1. Replace the raw-audio deque with a contiguous numpy FIFO

`_streaming_melspectrogram` read its input as
`list(self.raw_data_buffer)[-n_samples-480:]`. That converted **all 160,000**
buffered samples into a Python list on every 80 ms frame, only to slice off the
last ~1,760. It cost more per frame than the melspectrogram model itself, and the
cost grew with uptime.

Replaced with `_RawAudioBuffer` in `openwakeword/utils.py`: a fixed-capacity int16
FIFO backed by one contiguous array, with 2x slack so compaction amortizes to
roughly one memcpy per 125 frames. Reads are now a view (`tail(n)`), not a copy.

| | before | after |
|---|---:|---:|
| per frame | 2.146 ms | 1.506 ms |
| realtime | 37x | 53x |

Output bit-identical (`max abs diff: 0.0` over all 692 captured scores). All 17
tests pass. The buffer is private to `utils.py` — nothing else in the repo touches
`raw_data_buffer`.

### 2. Cache the feature-buffer priming used by `reset()`

`AudioFeatures.reset()` re-embedded 4 s of fresh random audio on every call:
**56 ms**. The priming is now computed once in `__init__` and copied on reset:
**0.0 ms**.

This does not affect steady-state throughput, but `reset()` sits on the live path
when re-arming after a detection, and its own docstring warned it "may not be
efficient when called too frequently". Side benefit: repeated resets are now
reproducible rather than re-priming with different noise each time.

### What it bought

| | CPU per second of audio | one core |
|---|---:|---:|
| before | 26.8 ms | 2.68% |
| after | 18.2 ms | 1.82% |

**8.6 ms less CPU per second of audio — a third of the always-on budget freed.**

In latency terms this is worth ~0.3%. Its value is that halving the frame step
doubles the embedding work, so this headroom is roughly what a 40 ms step costs: the
optimization makes that step affordable at 3.6%, where before it would have cost
5.4%.

### Current configuration

**Library defaults stay conservative** — `step_samples=1280` (80 ms) and `ncpu=1` —
so that a single-core or power-constrained target is never silently charged for
tuning it did not ask for. The Pi tuning is applied at the call site instead:

```bash
python examples/detect_from_microphone.py --step_samples 640 --ncpu 2
```

That is **4.39% of a core with 31x realtime headroom** on this machine. A Pi core is
several times slower, so expect proportionally more — still comfortably real-time at
a 40 ms step, with two cores left free.

Note that `--chunk_size` now follows `--step_samples`. Reading larger chunks from the
microphone than the prediction step throws the latency benefit away: `predict` will
process both phases at once, but the result still only arrives once per read.

---

## Measured and rejected — do not retry

| Idea | Result | Verdict |
|---|---|---|
| Batch the embedding model | 1.354 / 1.322 / 1.329 / 1.343 ms per window at batch 1 / 8 / 32 / 128 | Flat. No throughput gain whatsoever. |
| CoreML execution provider | 4.074 ms vs 1.31 ms | 3x **slower**; ORT splits the graph into 20 partitions, only 44 of 65 nodes supported |
| tflite / LiteRT instead of ONNX | embedding 1.274 ms (1 thread), melspec 0.193 ms vs ONNX 0.102 ms | Embedding is a wash, melspec is worse. Not worth switching. |
| `ORT_ENABLE_ALL` graph optimization | 1.342 vs 1.378 ms | Within noise; already the default |
| Disable intra-op spin-wait | 1.356 ms | No effect |

---

## Remaining options

### A. Thread count — DONE, default is now `ncpu=2`

`ncpu` sets the intra-op thread count for the melspectrogram and embedding models:

```python
Model(wakeword_models=[path], inference_framework="onnx", ncpu=2)  # now the default
```

**The important part is not the thread count — it is that spinning had to be turned
off.** ORT's intra-op pool busy-waits for the next inference by default. That is right
for back-to-back batch work and badly wrong for an always-on detector idle ~95% of the
time between steps. Measured on the embedding model, inferring once per 40 ms:

| | ms/infer | CPU actually consumed |
|---|---:|---:|
| `ncpu=1` | 1.342 | 9.8% of a core |
| `ncpu=2`, spinning on (ORT default) | 0.851 | **36.9%** |
| `ncpu=2`, spinning off | 1.117 | 14.4% |
| `ncpu=4`, spinning on (ORT default) | 0.625 | **99.7% — a whole core burned idling** |
| `ncpu=4`, spinning off | 1.089 | 17.1% |

`AudioFeatures` now sets `session.intra_op.allow_spinning=0` whenever `ncpu > 1`, and
sets `inter_op_num_threads=1` because the graph runs sequentially and inter-op threads
would only be allocated and never used.

End-to-end through `Model.predict` at a 640-sample step, with spinning off:

| | wall per step | CPU per step | duty cycle | realtime headroom |
|---|---:|---:|---:|---:|
| `ncpu=1` | 1.454 ms | 1.453 ms | 3.63% | 28x |
| **`ncpu=2`** | **1.278 ms** | 1.757 ms | 4.39% | 31x |
| `ncpu=4` | 1.195 ms | 2.123 ms | 5.31% | 33x |

Read that carefully: threading lowers the *wall-clock* cost of a step by 12% but
raises the *total CPU* it consumes by 21%, because the parallel work does not come for
free. It buys scheduling headroom, not efficiency.

`ncpu=4` is not worth it — 6% more wall-clock headroom than `ncpu=2` for 21% more CPU
again. These models are small and stop scaling past two threads. On a 4-core Pi,
`ncpu=2` leaves two cores for everything else.

Numerics: threading changes scores by at most **1.3e-6** (parallel reduction order);
peak scores are identical to six decimals. `ncpu=1` remains bit-identical to the
pre-optimization baseline.

### B. int8 quantization — unproven, needs an accuracy gate

Dynamic quantization gave only **1.309 -> 1.063 ms (1.23x)**, because
`quantize_dynamic` does little for Conv. Static quantization with a calibration set
should do better on ARM's QLinearConv kernels, but it changes scores.

If pursued:

1. Build a calibration set from real 16 kHz audio (a few hundred 76x32 embedding
   inputs sampled across the positive and negative clips).
2. Quantize to QDQ int8, embedding model only. Leave melspec (0.10 ms) and the
   wakeword model (0.015 ms) alone — there is nothing to win there.
3. Gate on accuracy before accepting, against the 56 real positive clips:
   - no clip changes side of the 0.5, 0.3 or 0.1 threshold
   - mean absolute score delta < 0.01
   - the two current weak clips (`hey_seeree_0011.wav` at 0.002,
     `hey_seeree_0008.wav` at 0.333) do not degrade further
4. Require >= 1.5x on the embedding model, else drop the idea — 1.23x is not worth
   shipping a second model artifact and a numerics risk.

### C. Streaming conversion of the embedding model — biggest upside

Exploit the 89% window overlap by giving each conv layer a ring buffer holding the
last `kernel_time - 1` frames of its own input, and computing only the new frames per
step. Total state is ~37 KB.

**This is mathematically exact and reuses the existing weights — no retraining.**
Three structural facts, read back from `embedding_model.onnx`, make that so; all
three had to hold, and all three do:

1. **The time axis is never padded.** Those `pads=[0,1,0,1]` are
   `[H_begin, W_begin, H_end, W_end]` in NCHW where H is time — the padding is on
   *frequency*. Every time convolution is `valid`, so there are no boundary artifacts
   between the windowed and streaming forms.
2. **The receptive field is exactly the window.** Walking back from one output frame
   through all 20 convs and 5 pools lands on exactly 76 input frames — the declared
   input height. No global op entangles the window.
3. **Temporal downsampling matches the hop.** Three pools have time-stride 2, so
   total downsampling is 8, and the pipeline advances 8 melspec frames per step.
   Feeding 8 new frames yields exactly 1 new output, with no drift.

Work per step drops from 83.91 to 11.23 MFLOPs — a **7.48x ceiling**
(1.312 -> 0.176 ms). Expect 2-4x in practice: a conv over 8 time frames is far less
efficient per element than over 74, and 20 small ops carry ORT dispatch overhead of
roughly 20-100 us against a 176 us target.

**Gate it on a half-day spike before building anything.** Construct a graph with the
same convolutions at streaming time dimensions (8/4/2/1), time it, and compare
against 1.312 ms. **If it lands above ~0.9 ms, stop** — `ncpu=2` is then a better
return for a fraction of the effort. If it passes, the build is ~2-4 days: rebuild
with explicit cache tensors as model inputs/outputs, gate on ~1e-5 agreement with the
windowed model over 60 s of continuous audio (a short test misses slow cache drift),
wire into `_streaming_features` only — `embed_clips` and the training pipeline keep
the windowed model — and handle cache priming in `reset()`.

**On its own this buys zero felt latency.** The detector is idle during the window
fill, not computing. Its entire latency value is indirect: it makes a 10 ms frame
step affordable (3.7% duty cycle, roughly what a 40 ms step costs today), worth an
extrapolated 35-40 ms.

Prior art: Google's [`kws_streaming`](https://github.com/google-research/google-research/tree/master/kws_streaming)
(paper: [Streaming keyword spotting on mobile devices](https://arxiv.org/abs/2005.06720))
automates exactly this conversion with a `Stream` layer wrapper that holds the ring
buffer. Read it before writing any of this. Caveat: it wraps *Keras* models and
targets TFLite, and our embedding model is a frozen ONNX artifact — so it is a
reference for the pattern, not a drop-in.

### D. Not investigated

- Pruning or distilling the embedding model. Large effort, changes the model, would
  invalidate every trained wakeword head that depends on these embeddings.
- Python-level overhead in `Model.predict` is ~0.08 ms/frame. Real, but small enough
  that it is not worth touching until the conv cost comes down.

---

## Reproducing

```bash
uv venv .venv-perf --python 3.12
VIRTUAL_ENV=.venv-perf uv pip install numpy onnxruntime scipy tqdm requests scikit-learn pytest mock

# correctness gate: capture scores, change code, capture again, compare
# (seed numpy before constructing Model so reset() priming is deterministic)

# tests (the repo's pytest addopts pull in coverage/flake8/mypy plugins)
PYTHONPATH=. .venv-perf/bin/python -m pytest tests/ -q -p no:cacheprovider -c /dev/null
```

Duty cycle is the steady-state per-frame cost times 12.5 frames per second.

Note that running the test suite downloads the pretrained models into
`openwakeword/resources/models/`.
