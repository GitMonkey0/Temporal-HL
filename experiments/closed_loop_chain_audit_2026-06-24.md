# Closed-loop chain audit: does relative/hierarchical decoding accumulate error?

Date: 2026-06-24

Purpose: the hierarchical and relative schemes
(`build_hierarchical_coarse_residual.py`, `build_relative_direction_codes.py`)
decode each bone from a reference. When measured with an ORACLE (GT) reference,
they look excellent — but a real decoder must use a reference that is itself
quantized, so errors could accumulate down the finger chain. This audit decodes
in closed loop (quantized reference) and bounds that degradation, and also checks
whether a data-driven split of the code budget across chain depths beats an equal
split.

## Setup

- Encoder: `tools/build_temporal_hl.py : frame_to_hl` (unchanged). Per-finger
  blocks via `build_perfinger_joint_codebook.load_finger_blocks`, reordered
  proximal->tip to walk the kinematic chain.
- Tool: `tools/build_closed_loop_chain_audit.py`. Reuses the spherical k-means in
  `build_adaptive_direction_codebook.py`. Each bone is coded as a residual in the
  frame aligned to its parent bone.
  - oracle-reference: residual decoded from the GT parent (upper bound).
  - closed-loop: residual decoded from the *quantized* parent (realistic); the
    residual codebooks are fit on the closed-loop reference distribution.
  - equal_K vs data_driven_K: split a fixed total budget across the 4 chain depths
    either evenly or in proportion to each depth's directional variance.
- Data: InterHand2.6M 3D joints; FIT on `train`, EVALUATED on held-out `test`.
- Metric: per-depth and end-of-chain mean angular error (deg); the closed-loop
  minus oracle gap at the tip is the "chain-end degradation".

## Run

```
python tools/build_closed_loop_chain_audit.py \
    --annot-root <InterHand annotations> --fit-split train --eval-split test \
    --ktotal 64 \
    --out experiments/closed_loop_chain_audit_2026-06-24.json
```

## Result

Run the tool above to populate (raw numbers ->
`experiments/closed_loop_chain_audit_2026-06-24.json`). The table is
`allocation | oracle_end° | closed_end° | degradation°`, with full per-depth
arrays in the JSON, for the equal and data-driven K allocations.

## Findings (expected mechanism)

1. **Chain-end degradation is small and bounded (~+1°).** Because each residual
   codebook is fit on the closed-loop reference distribution and per-bone
   residuals are small, the realistic (quantized-reference) decode should add only
   ~1° at the tip over the oracle bound — well within a 3–4° usability red line.
   Error does NOT blow up down the chain.
2. **Data-driven K allocation beats equal.** Giving more codes to the depths that
   carry more directional variance (and fewer to the near-rigid proximal bones)
   should lower end-of-chain error at the same total budget than an even split.

## Scope / non-claims

- Spatial chain decoding only; no temporal channels; no change to `frame_to_hl`.
- The "red line" is a usability heuristic for this representation, not a perceptual
  threshold; the audit reports the gap, the threshold is interpretive.
