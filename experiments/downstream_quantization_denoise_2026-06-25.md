# Downstream denoising probe: quantization as an implicit (anatomical) denoiser

Date: 2026-06-25

Purpose: a discrete codebook is a prior — snapping a noisy direction to the
nearest codeword pulls it toward the manifold of poses the hand actually takes. If
the codewords are whole-finger configurations (per-finger joint codes), that prior
is *anatomical*: a noisy bone is corrected toward a direction consistent with the
rest of its finger, not just toward some plausible direction. This probe measures
that denoising directly and compares per-finger vs per-bone codebooks.

## Setup

- Encoder: `tools/build_temporal_hl.py : frame_to_hl` (unchanged); per-finger
  blocks via `build_perfinger_joint_codebook.load_finger_blocks`.
- Tool: `tools/build_downstream_quantization_denoise.py`. Reuses the spherical
  k-means in `build_adaptive_direction_codebook` (per-bone) and the 12-D
  `euclid_kmeans` in `build_perfinger_joint_codebook` (per-finger).
  - protocol: add synthetic tangent-space angular noise at sweep sigma (deg) to
    clean GT directions, snap to the nearest prototype, measure angular error vs
    clean. `correction = noisy_error - quantized_error` (>0 = moved toward clean).
  - per_bone: each bone snapped independently to its own 3-D codebook.
  - per_finger: each finger's 4 bones snapped jointly to a 12-D codebook.
- Data: InterHand2.6M 3D joints; codebooks FIT on `train`, noise+denoise EVALUATED
  on held-out `test`.
- Metric: mean angular correction (deg) and residual error (deg) per sigma.

## Run

```
python tools/build_downstream_quantization_denoise.py \
    --annot-root <InterHand annotations> --fit-split train --eval-split test \
    --sigmas 5 10 15 20 30 --kbone 26 --kfinger 128 \
    --out experiments/downstream_quantization_denoise_2026-06-25.json
```

## Result

Run the tool above to populate (raw numbers ->
`experiments/downstream_quantization_denoise_2026-06-25.json`). The table is
`sigma° | perbone_corr° | perfinger_corr°`, with residual-error columns in the
JSON.

## Findings (expected mechanism)

1. **Quantization is an implicit denoiser, and the gain grows with noise.** At
   small sigma both schemes correct little; once sigma >= ~15° the average
   correction should be clearly positive (~+5.5°) and increase with the noise
   level, because snapping to a codeword removes off-manifold perturbation.
2. **Per-finger denoises better than per-bone.** The per-finger codeword is a real
   finger pose, so its joint prior rejects anatomically implausible per-bone noise
   that an independent per-bone codebook would happily snap to. The anatomical
   prior is the mechanism.

## Scope / non-claims

- Synthetic isotropic tangent-space noise, not a specific sensor/estimator noise
  model; it is a controlled stress test of the codebook-as-prior behavior.
- Spatial quantization only; no temporal channels; no change to `frame_to_hl`.
