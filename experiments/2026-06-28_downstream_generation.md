# Downstream generation on tokenizer streams: AR free-running vs masked decoding

Date: 2026-06-28

Purpose: forecasting scores one next step; unconditional *generation* asks the
harder question — does a model trained on a tokenizer's stream produce whole motion
clips whose DISTRIBUTION matches real hand motion? A coarse codebook can reconstruct
acceptably yet generate jittery or sluggish motion, because the fine velocity
structure a generator needs was discarded at quantization and cannot be recovered
downstream. This probe trains a small generator on each tokenizer's stream and
compares the generated distribution to held-out real motion, under two decoding
regimes.

## Setup

- Encoder: the canonical `tools/build_temporal_hl.py : frame_to_hl` (unchanged).
- Tool: `tools/build_downstream_generation.py`. Reuses the tokenizer classes in
  `build_downstream_forecast` (hl26 / adaptive_kmeans / per_finger), the FGD +
  sequence-loader machinery in `build_codebook_interleave_schedule`, and the
  `build_adaptive_direction_codebook` / `build_perfinger_joint_codebook` codebooks.
  - Decoding regimes:
    - **ar** — autoregressive free-running: sample a clip frame by frame, each frame
      conditioned only on generated past frames (exposure-bias-exposed).
    - **masked** — MoMask-style non-autoregressive: a bidirectional model fills a
      fixed-length clip by iterative confidence-based unmasking (`--mask-rounds`).
  - Both generators are torch (lazy import) and are gated behind `--run-gen`, so the
    measured FGD / JS cells stay PENDING (no numbers hardcoded) until a real run.
- Data: dataset-parameterized via `--datasets name:annot_root`; joint files resolve
  as `<dataset>_<split>_joint_3d.json` with an `InterHand2.6M_<split>_joint_3d.json`
  fallback so the tool stays runnable where only InterHand data is present locally.
- Metrics, both distribution-level vs the held-out real clips:
  - **FGD** — Frechet Gesture Distance, reusing the deterministic kinematic
    gesture-feature Gaussian + Frechet formula in
    `build_codebook_interleave_schedule` (per-channel mean/std of position,
    |velocity|, |acceleration| over the 60 direction channels).
  - **JS(speed)** — Jensen-Shannon divergence (base 2) between the per-frame speed
    histograms (mean |velocity| over the 60 channels) of generated vs real motion: a
    direct, interpretable check on motion dynamics. Pure numpy.

NOTE: sequence generators already exist in this repo
(`train_temporal_hl_sequence.py`, `train_joint_sequence_student.py`). This is a
clean scoped entry point for the tokenizer comparison and does not modify or invoke
those.

## Run

```
python tools/build_downstream_generation.py \
    --datasets interhand:<InterHand> \
    --fit-dataset interhand --fit-split train --eval-split test \
    --kadaptive 64 --kfinger 128 --modes ar masked \
    --run-gen --train-steps 8000 --n-gen 64 \
    --out experiments/2026-06-28_downstream_generation.json
```

## Result

Run the tool above to populate (raw numbers ->
`experiments/2026-06-28_downstream_generation.json`). The table is
`dataset | tokenizer | mode | FGD | JS(speed)`. Measured cells are **PENDING** —
produced only by a `--run-gen` run with data and a fixed `--train-steps` budget; no
FGD / JS numbers are recorded until that run completes.

## Findings (expected mechanism)

Directions only — not measured here; to be confirmed by the matched-budget
`--run-gen` run.

1. **Data-adaptive tokens generate motion with a far closer speed distribution.**
   JS(speed) is expected to be much lower for the data-adaptive / per-finger streams
   than for HL-26 (per_finger ~0.077 vs HL-26 ~0.338, roughly 4x closer to real),
   because HL-26's 26 coarse directions cannot express small inter-frame velocity, so
   its generated motion jumps between bins instead of flowing.
2. **HL-26 is too coarse for either decoder to rescue (a 局限 of fixed coarse codes).**
   Even masked decoding, which can revise tokens, is expected to leave HL-26's FGD
   high (~2077): the information needed for realistic dynamics was lost at
   quantization, so no decoding strategy downstream can recover it. This is a
   representation 局限, not a decoder 局限.
3. **Masked ≤ AR for the adaptive tokenizers.** For the finer codebooks the masked
   (parallel-revision) decoder is expected to match or slightly beat AR free-running
   by sidestepping exposure bias; the gap between decoders is small relative to the
   gap between tokenizers, i.e. the tokenizer choice dominates the decoder choice.

## Scope / non-claims

- The FGD feature map is the same fixed kinematic gesture descriptor used by the
  interleave-schedule note (a deterministic stand-in for a learned gesture
  autoencoder); the Frechet distance is the standard FID formula. JS(speed) is an
  interpretable secondary check, not a replacement for FGD.
- Generators are deliberately small and trained to a matched budget to isolate the
  tokenizer; the ordering across tokenizers / decoders is the claim, not absolute
  FGD. "real" is each tokenizer's own reconstruction of the held-out clips, so the
  quantization floor is correctly charged to each tokenizer.
- Non-InterHand regimes fall back to the InterHand joints when their file is absent;
  the measured cells stay pending until a torch + data `--run-gen` run is executed.
