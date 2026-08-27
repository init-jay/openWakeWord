#!/usr/bin/env python3
"""Assemble the positive training set for a general wake-word model that is
extra-strong on specific speakers.

The model stays general because the bulk of its positives are synthetic TTS
clips covering many voices. It gets extra-strong on your household because your
own augmented recordings are blended in and oversampled on top of that.

Given per-speaker augmented directories (from augment_positives.py) and a
directory of synthetic TTS positives, this writes the pipeline's
`positive_train/` and `positive_test/` directories.

Two things it is careful about:

  Speaker-disjoint, recording-disjoint split
      Augmented variants of one recording are near-duplicates. If some land in
      train and others in test, validation recall is inflated and meaningless.
      The split happens at the level of the *source recording*, so every variant
      of a recording lands wholly on one side.

  Honest oversampling
      --speaker-ratio sets the share of training positives that come from your
      recorded voices. Pushing it too high trades away the generality that makes
      the model work for guests; the script warns past a sane bound.

Adding a speaker later is just another --speaker-dir plus a re-run.

Examples
--------
You only, for now:
    .venv/bin/python scripts/assemble_training_set.py \\
        --synthetic-dir ./my_custom_model/positive_train_synthetic \\
        --speaker-dir voice_data/jay/positives_augmented \\
        --out-dir ./my_custom_model

Later, with your wife's recordings too:
    .venv/bin/python scripts/assemble_training_set.py \\
        --synthetic-dir ./my_custom_model/positive_train_synthetic \\
        --speaker-dir voice_data/jay/positives_augmented \\
        --speaker-dir voice_data/wife/positives_augmented \\
        --out-dir ./my_custom_model
"""

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

# augment_positives.py names outputs "<source-recording-stem>_augNNN.wav"
AUG_SUFFIX = re.compile(r"_aug\d+$")


def source_recording(path: Path) -> str:
    """The source recording a clip came from, so variants stay grouped."""
    return AUG_SUFFIX.sub("", path.stem)


def group_by_recording(clips):
    groups = defaultdict(list)
    for c in clips:
        groups[source_recording(c)].append(c)
    return groups


def split_recordings(groups, test_frac, rng):
    """Hold out whole recordings, never individual augmented variants."""
    names = sorted(groups)
    rng.shuffle(names)
    n_test = max(1, int(round(len(names) * test_frac))) if len(names) > 1 else 0
    return names[n_test:], names[:n_test]


def link_or_copy(src: Path, dst: Path, use_copy: bool) -> None:
    """Hardlink by default: these sets get large and duplicate bytes add up."""
    if dst.exists():
        dst.unlink()
    if use_copy:
        shutil.copy2(src, dst)
        return
    try:
        dst.hardlink_to(src)
    except (OSError, AttributeError):
        shutil.copy2(src, dst)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--synthetic-dir", default=None,
                   help="directory of synthetic TTS positives (the generality bulk)")
    p.add_argument("--speaker-dir", action="append", default=[], metavar="DIR",
                   help="augmented recordings for one speaker; repeat per speaker")
    p.add_argument("--out-dir", default="./my_custom_model",
                   help="pipeline output dir; positive_train/ and positive_test/ go here")
    p.add_argument("--speaker-ratio", type=float, default=0.25,
                   help="target share of training positives drawn from recorded voices")
    p.add_argument("--test-frac", type=float, default=0.15,
                   help="fraction of source recordings held out for validation")
    p.add_argument("--copy", action="store_true", help="copy files instead of hardlinking")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    train_dir, test_dir = out_dir / "positive_train", out_dir / "positive_test"
    for d in (train_dir, test_dir):
        d.mkdir(parents=True, exist_ok=True)

    if not args.speaker_dir:
        raise SystemExit("pass at least one --speaker-dir")

    # ---- per-speaker clips, split by source recording -------------------- #
    train_speaker, test_speaker, summary = [], [], {}
    for sdir in args.speaker_dir:
        sp_path = Path(sdir)
        clips = sorted(sp_path.glob("*.wav"))
        if not clips:
            raise SystemExit(f"no .wav files in {sp_path} - run augment_positives.py first")
        name = sp_path.parent.name if sp_path.name.startswith("positives") else sp_path.name

        groups = group_by_recording(clips)
        tr_names, te_names = split_recordings(groups, args.test_frac, rng)
        # Carry the speaker with each clip: speakers number their recordings
        # independently, so bare filenames collide across speakers and would
        # silently overwrite each other in the shared output directory.
        tr = [(c, name) for n in tr_names for c in groups[n]]
        te = [(c, name) for n in te_names for c in groups[n]]
        train_speaker += tr
        test_speaker += te
        summary[name] = {
            "source_recordings": len(groups),
            "augmented_clips": len(clips),
            "train_recordings": len(tr_names), "train_clips": len(tr),
            "test_recordings": len(te_names), "test_clips": len(te),
        }
        print(f"{name}: {len(groups)} recordings -> {len(clips)} clips "
              f"({len(tr_names)} rec/{len(tr)} clips train, "
              f"{len(te_names)} rec/{len(te)} clips test)")

    # ---- synthetic clips, sized to hit the requested ratio ---------------- #
    synth = sorted(Path(args.synthetic_dir).glob("*.wav")) if args.synthetic_dir else []
    if synth:
        # n_speaker / (n_speaker + n_synth_used) = ratio
        want = int(round(len(train_speaker) * (1 - args.speaker_ratio) / args.speaker_ratio))
        if want > len(synth):
            actual = len(train_speaker) / (len(train_speaker) + len(synth))
            print(f"\nWARNING: only {len(synth)} synthetic clips available, wanted {want}.")
            print(f"  Recorded voices will be {actual:.0%} of training positives, not "
                  f"{args.speaker_ratio:.0%}.")
            print("  Generate more synthetic positives to keep the model general.")
            use_synth = synth
        else:
            idx = rng.choice(len(synth), size=want, replace=False)
            use_synth = [synth[i] for i in sorted(idx)]
    else:
        use_synth = []
        print("\nWARNING: no --synthetic-dir given. Training on recorded voices alone "
              "produces a model that fits you two and generalises poorly to anyone else.")

    n_total = len(train_speaker) + len(use_synth)
    real_share = len(train_speaker) / n_total if n_total else 0
    if real_share > 0.5:
        print(f"\nWARNING: recorded voices are {real_share:.0%} of training positives. "
              "Above ~50% the model starts overfitting your mics and rooms.")

    # ---- materialise ------------------------------------------------------ #
    for clip, spk in train_speaker:
        link_or_copy(clip, train_dir / f"spk_{spk}_{clip.name}", args.copy)
    for clip in use_synth:
        link_or_copy(clip, train_dir / f"syn_{clip.name}", args.copy)
    for clip, spk in test_speaker:
        link_or_copy(clip, test_dir / f"spk_{spk}_{clip.name}", args.copy)

    # Every intended clip must exist on disk. A shortfall means destination names
    # collided and clips were silently overwritten, quietly shrinking the set.
    for d, expected, label in ((train_dir, n_total, "positive_train"),
                               (test_dir, len(test_speaker), "positive_test")):
        found = len(list(d.glob("*.wav")))
        if found != expected:
            raise SystemExit(
                f"{label}: wrote {expected} clips but found {found} on disk - "
                "destination filenames collided. Check for duplicate names across "
                "--speaker-dir inputs."
            )

    manifest = {
        "speaker_ratio_requested": args.speaker_ratio,
        "speaker_ratio_actual": round(real_share, 4),
        "train_clips_total": n_total,
        "train_clips_speaker": len(train_speaker),
        "train_clips_synthetic": len(use_synth),
        "test_clips_speaker": len(test_speaker),
        "speakers": summary,
    }
    (out_dir / "training_set_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\npositive_train/: {n_total} clips "
          f"({len(train_speaker)} recorded = {real_share:.0%}, {len(use_synth)} synthetic)")
    print(f"positive_test/:  {len(test_speaker)} clips (held-out recordings only)")
    print(f"manifest: {out_dir / 'training_set_manifest.json'}")
    print("\nNote: positive_test/ holds only your voices, so validation recall measures "
          "how well the model catches YOU, which is what you are optimising for.")


if __name__ == "__main__":
    main()
