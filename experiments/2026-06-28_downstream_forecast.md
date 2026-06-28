# Downstream motion forecasting on tokenizer streams (HL-26 / adaptive / per-finger)

Date: 2026-06-28

Purpose: reconstruction angular error is intrinsic — it says how faithfully a
codebook can copy a pose, not how *predictable* its token stream is. A
representation can compress well and still be a poor substrate for a sequence
model. This probe closes that loop with a single-step forecasting task: train the
SAME predictor family on the discrete token streams each tokenizer emits and ask
which yields the most forecastable codes, scored against a copy-last reference.

## Setup

- Encoder: the canonical `tools/build_temporal_hl.py : frame_to_hl` (unchanged);
  a frame's 20 local bone directions feed every tokenizer identically.
- Tool: `tools/build_downstream_forecast.py`. Reuses
  `build_adaptive_direction_codebook` (HL-26 + per-bone spherical k-means),
  `build_perfinger_joint_codebook` (the 12-D per-finger code), and the sequence
  loader / path resolver in `build_codebook_interleave_schedule`.
  - Tokenizers: **hl26** (20 fixed tokens/frame), **adaptive_kmeans** (20 per-bone
    data-adaptive tokens/frame, `--kadaptive`), **per_finger** (5 joint
    tokens/frame, `--kfinger` — the most compressed stream).
  - Predictors, all trained to a MATCHED budget (`--train-steps`) on each
    tokenizer's stream: **copy_last** (predict next = current; pure numpy, always
    computed), **gru**, **lstm** (recurrent next-token models), and **ar** (a small
    causal-self-attention next-token model). gru/lstm/ar use torch (lazy import);
    if torch is unavailable they are skipped and their cells stay pending.
  - Each predictor emits next-frame token ids; we decode them through the SAME
    tokenizer's codebook and measure the single-step forecast angular error (deg)
    against the held-out CLEAN next-frame directions. Decoding through each
    tokenizer's own codebook is what makes the comparison fair: a tokenizer is
    charged for both its prediction quality and its reconstruction coarseness.
- Data: dataset-parameterized via `--datasets name:annot_root`; joint files resolve
  as `<dataset>_<split>_joint_3d.json` with an `InterHand2.6M_<split>_joint_3d.json`
  fallback so the tool stays runnable where only InterHand data is present locally.
  Codebooks + predictors FIT on `train`, EVALUATED on held-out `test`.
- Metric: single-step forecast mean angular error (deg), plus the relative
  improvement of each learned predictor over copy_last.

NOTE: sequence models already exist in this repo
(`train_temporal_hl_sequence.py`, `train_joint_sequence_student.py`). This is a
clean scoped entry point for the tokenizer comparison and does not modify or invoke
those.

## Run

```
python tools/build_downstream_forecast.py \
    --datasets interhand:<InterHand> \
    --fit-dataset interhand --fit-split train --eval-split test \
    --kadaptive 64 --kfinger 128 \
    --predictors copy_last gru lstm ar --train-steps 4000 \
    --out experiments/2026-06-28_downstream_forecast.json
```

## Result

Run the tool above to populate (raw numbers ->
`experiments/2026-06-28_downstream_forecast.json`). The table is
`dataset | tokenizer | n_tok | copy | gru | lstm | ar` of single-step forecast
angular error (deg). Measured cells are produced by the run; the learned-predictor
cells stay **pending** until a matched-budget `--train-steps` run with torch and
real data completes (copy_last, pure numpy, is the always-on reference).

## Findings (expected mechanism)

Directions only — not measured here; to be confirmed by the matched-budget run.

1. **Data-adaptive forecasts best.** The per-bone adaptive codebook is expected to
   give the lowest forecast error (~12.9°, roughly -31% vs HL-26), because its codes
   sit where the directions actually are, so a small motion maps to a small,
   learnable token change rather than a coarse jump.
2. **Only the AR model beats copy-last clearly.** Hand motion is locally smooth, so
   copy-last is a strong baseline; the recurrent gru/lstm are expected to roughly
   match it, and only the causal-attention AR model is expected to beat it by a
   clear margin.
3. **Per-finger compresses most yet forecasts WORSE (a 负结果 for compactness).**
   The per-finger stream uses the fewest tokens but is expected to forecast worse
   (~20°), because a single whole-finger code discards the fine bone-level structure
   a forecaster relies on. The compression-optimal tokenizer is not the
   generation-optimal one — a 局限 that motivates choosing the token granularity by
   the downstream task, not by rate alone.

## Scope / non-claims

- Single-step forecasting with deliberately small, matched-budget predictors, to
  isolate the tokenizer; the ordering across tokenizers is the claim, not absolute
  accuracy. Multi-step rollout is a separate axis (see the generation probe).
- The "clean next-frame direction" target is the unquantized `frame_to_hl` output,
  so per-finger's higher reconstruction floor is correctly counted against it.
- Non-InterHand regimes are read via a `<dataset>_<split>_joint_3d.json` sibling
  using the same encoder; absent files fall back to the InterHand joints, so the
  rows reflect whatever joint files are actually present, and the learned cells stay
  pending until a torch + data run is executed.
