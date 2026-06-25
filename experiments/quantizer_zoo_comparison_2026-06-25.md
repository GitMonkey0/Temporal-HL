# Quantizer-zoo comparison: FSQ / PQ / VQ / RVQ / BSQ x raw-3D / spherical

Date: 2026-06-25

Purpose: ground the choice of plain k-means (a 1-layer VQ) for HL direction
quantization by placing it next to the standard quantizer zoo at MATCHED bit
budgets, and in BOTH the raw 3-D direction space and the spherical-coordinate
`(theta, phi)` space. This rules out "we just didn't try the right quantizer" and
isolates the cost of the spherical-coordinate pole singularities.

## Setup

- Encoder: `tools/build_temporal_hl.py : frame_to_hl` (unchanged); directions via
  `build_adaptive_direction_codebook.load_local_directions`.
- Tool: `tools/build_quantizer_zoo_comparison.py`. Reuses
  `build_adaptive_direction_codebook` for the directions; all quantizers
  (PQ/FSQ/RVQ/BSQ and a numpy k-means VQ) are implemented compactly in numpy here.
  - VQ: single-layer k-means.
  - RVQ: multi-stage residual k-means.
  - PQ: per-coordinate product codebooks.
  - FSQ: per-coordinate uniform scalar levels.
  - BSQ: sign bits on fixed projections, re-normalized to the sphere.
  - Each is run in `raw3d` (the unit 3-vector) and `sphere` (`theta` from +Y,
    `phi`), always measuring angular error back in 3-D.
- Data: InterHand2.6M 3D joints; FIT on `train`, EVALUATED on held-out `test`,
  all at a matched `--bits` budget per direction.
- Metric: reconstruction angular error (deg) at matched bits.

## Run

```
python tools/build_quantizer_zoo_comparison.py \
    --annot-root <InterHand annotations> --fit-split train --eval-split test \
    --bits 6 \
    --out experiments/quantizer_zoo_comparison_2026-06-25.json
```

## Result

Run the tool above to populate (raw numbers ->
`experiments/quantizer_zoo_comparison_2026-06-25.json`). The table is
`method | domain | bits | angular°` over the five quantizers in both domains.

## Findings (expected mechanism)

1. **Single-layer k-means VQ is competitive with a well-tuned learned VQ**, so the
   geometric baseline is not leaving fidelity on the table at a fixed budget
   (the learned-VQ upper bound is quantified separately in
   `learned_vq_upper_bound`).
2. **RVQ is the high-fidelity end**: stacking residual stages should push
   reconstruction below ~2°, the operating point for high-fidelity decoding.
3. **Raw-3D beats spherical-coord at matched bits.** The `(theta, phi)`
   parameterization has pole singularities, so scalar/grid quantizers that look
   natural there waste resolution near the poles and do worse than the same method
   in the ambient 3-D space — a concrete reason to keep HL direction work in 3-D.
4. **BSQ is dominated** at these budgets.

## Scope / non-claims

- Per-direction quantization only; no temporal channels; no change to
  `frame_to_hl`. The implementations are compact references for a fair matched-bit
  comparison, not optimized production codecs.
- `bits` is the idealized code rate (log2 codebook / level count), not an
  entropy-coded stream length.
