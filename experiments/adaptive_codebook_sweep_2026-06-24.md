# Adaptive-codebook sweep: k-means vs k-medoids vs region-aware (matched K)

Date: 2026-06-24

Purpose: stress-test the data-adaptive direction codebook
(`build_adaptive_direction_codebook.py`) along two axes before trusting it as the
representation backbone: (1) are the learned centers *readable* (k-medoids
constrains every center to be a real observed direction), and (2) is "one
codebook for all bones" the right granularity, or are the five fingers the natural
structural unit (a separate per-finger codebook)?

## Setup

- Encoder: the canonical `tools/build_temporal_hl.py : frame_to_hl` (unchanged);
  directions collected with `build_adaptive_direction_codebook.load_local_directions`
  and, per finger, with this tool's `per_finger_directions` (reshaping the
  finger-major `EDGE_ORDER` 20 vectors to `[5 fingers, 4 bones, 3]`).
- Tool: `tools/build_adaptive_codebook_sweep.py`. Reuses
  `build_adaptive_direction_codebook.spherical_kmeans` and its metrics.
  - kmeans: spherical k-means (the adaptive baseline).
  - kmedoids: spherical k-medoids; each center is constrained to an observed
    direction, so codes are real, namable bone directions.
  - region_aware: a separate spherical-k-means codebook per finger, evaluated per
    finger then aggregated by eval count.
- Data: InterHand2.6M 3D joints; codebooks FIT on `train`, EVALUATED on held-out
  `test`.
- Metric: mean nearest-code angular error (deg); `bits/bone` = code entropy.

## Run

```
python tools/build_adaptive_codebook_sweep.py \
    --annot-root <InterHand annotations> --fit-split train --eval-split test \
    --ks 16 26 40 \
    --out experiments/adaptive_codebook_sweep_2026-06-24.json
```

## Result

Run the tool above to populate (raw numbers ->
`experiments/adaptive_codebook_sweep_2026-06-24.json`). The table is
`method | K | bits/bone | angular°` over kmeans / kmedoids / region_aware at each
K; `medoid_examples` in the JSON lists the first few medoid centers as real
observed directions.

## Findings (expected mechanism)

1. **k-medoids ~ k-means at matched K, but readable.** Constraining centers to
   observed directions should cost little accuracy, while making each code a real,
   printable bone direction — interpretability for free.
2. **Region-aware per-finger codebooks reduce the effective rate.** Fitting a
   codebook per finger spends the budget where each region's directions actually
   live, so it should reach the same accuracy at a clearly lower effective rate
   (~a quarter less). This frames "finger" as a natural structural unit and
   directly motivates per-finger *joint* quantization
   (`build_perfinger_joint_codebook.py`).

## Scope / non-claims

- Spatial per-direction quantization only; no temporal channels; no change to
  `frame_to_hl`. Additive variant study.
- Region-aware here means a separate per-bone codebook *per finger* (a routing of
  the same per-bone scheme), not yet the joint per-finger code; the joint code is
  the next tool.
