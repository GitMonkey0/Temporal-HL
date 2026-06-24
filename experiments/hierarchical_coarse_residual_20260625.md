# Hierarchical coarse + residual direction codes (high-fidelity, anatomy-aware)

Date: 2026-06-25

Purpose: give the per-finger code a high-fidelity refinement. A single per-finger
code must average over the exact pose of four bones, so it has an accuracy
ceiling. Adding a small per-bone *residual* on top of the per-finger *coarse*
code should break that ceiling at a low extra rate, because the two layers carry
near-independent information.

## Setup

- Encoder: `tools/build_temporal_hl.py : frame_to_hl` (unchanged).
- Tool: `tools/build_hierarchical_coarse_residual.py`. Reuses the per-finger
  loader and 12-D k-means in `build_perfinger_joint_codebook.py` and the metrics
  in `build_adaptive_direction_codebook.py`.
  - coarse: a per-finger joint code (12-D) = "which finger pose".
  - residual: per bone, quantize the offset between the GT direction and the
    coarse reconstruction with a small 3-D codebook = "by how much".
  - reconstruct each bone = normalize(coarse_bone + residual_code).
- Compared against a **flat per-bone** codebook (the reference) and the
  **coarse-only** per-finger code (no residual).
- Data: InterHand2.6M 3D joints; everything FIT on `train`, EVALUATED on held-out
  `test`. `bits/bone` = coarse_entropy / 4 + mean per-bone residual entropy.

## Run

```
python tools/build_hierarchical_coarse_residual.py \
    --annot-root <InterHand annotations> --fit-split train --eval-split test \
    --kcoarse 128 --kres 8 16 32 \
    --out experiments/hierarchical_coarse_residual_20260625.json
```

## Result

Run the tool above to populate (raw numbers ->
`experiments/hierarchical_coarse_residual_20260625.json`). The table is
`method | bits/bone | angular°` over flat-per-bone, coarse-only, and
coarse+residual at several residual budgets.

## Findings (expected mechanism)

1. **Lower error at a lower rate than flat per-bone.** The coarse code already
   explains most of the finger configuration, so the coarse->residual mutual
   information is small and a few residual bits go a long way: the
   coarse+residual point should dominate the flat per-bone reference on the
   rate-distortion plane.
2. **The coarse layer stays anatomically grounded** (a whole-finger pose), so the
   coarse/residual split has a direct physical meaning, unlike a generic residual
   stack. This is the high-fidelity operating point of the representation.

## Scope / non-claims

- Spatial quantization only; no temporal channels; no change to `frame_to_hl`.
- The residual codebooks are per-bone and small; this is a fidelity/ rate result,
  not a downstream-task claim.
