# Per-finger joint quantization (anatomy-aware) vs independent per-bone

Date: 2026-06-24

Purpose: go beyond *fitting the data* to *respecting hand structure*. A
data-adaptive per-bone codebook still treats the four bones of a finger as
independent, but they are strongly coupled (a finger curls as a unit). Quantizing
a whole finger jointly should reach the same accuracy at a much lower rate, and
give each token a readable physical meaning (a finger pose).

## Setup

- Encoder: the canonical `tools/build_temporal_hl.py : frame_to_hl` (unchanged).
  `EDGE_ORDER` is finger-major, so a frame's 20 local bone directions reshape to
  `[5 fingers, 4 bones, 3]`.
- Tool: `tools/build_perfinger_joint_codebook.py`. Reuses the spherical k-means
  and metrics in `build_adaptive_direction_codebook.py`.
  - independent per-bone: a separate 3-D codebook per bone (`spherical k-means`).
  - per-finger joint: concatenate a finger's 4 bones into a 12-D vector and learn
    one codebook over finger configurations (`euclid_kmeans` on the product of
    four unit spheres). One token = one finger pose; reconstruct each bone from
    the code's 3-D sub-vector.
- Data: InterHand2.6M 3D joints; codebooks FIT on `train`, EVALUATED on held-out
  `test`.
- Metric: mean per-bone angular error (deg); `bits/bone` = code entropy / 4, so
  joint and independent codes are compared at the same per-bone rate.

## Run

```
python tools/build_perfinger_joint_codebook.py \
    --annot-root <InterHand annotations> --fit-split train --eval-split test \
    --kbone 16 26 --kfinger 64 128 256 \
    --out experiments/perfinger_joint_20260624.json
```

## Result

Run the tool above to populate (raw numbers -> `experiments/perfinger_joint_20260624.json`).
The table is `method | bits/bone | angular°`, comparing per-bone-independent
against per-finger-joint at several budgets.

## Findings (expected mechanism)

1. **Same accuracy, roughly one third the rate.** Because intra-finger bones are
   highly correlated, a joint code stops paying separate bits for information one
   bone already implies about its neighbors — so the per-finger curve should sit
   at clearly lower `bits/bone` for the same angular error than independent
   per-bone codes.
2. **One token = a readable finger pose** (extended / slightly bent / curled),
   which a per-bone code cannot express in isolation. This interpretable,
   anatomy-grounded structure is the basis for later editing and denoising
   behavior.

## Scope / non-claims

- Spatial quantization only; no temporal channels; no change to `frame_to_hl`.
- The joint codebook is data-dependent (it encodes finger correlations), so it
  benefits from diverse fitting data; this is a per-finger compression result,
  not a claim about every downstream task.
