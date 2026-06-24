# Geometric-basis diagnostic: does a denser fixed grid beat HL-26?

Date: 2026-06-20

Purpose: rule out the simplest explanation for HL-26's quantization error before
introducing a data-adaptive codebook — namely that HL-26 just has too few
directions and any finer fixed grid (or a different coordinate system) would
already fix it. This is a negative control; it motivates the adaptive codebook.

## Setup

- Encoder: the canonical `tools/build_temporal_hl.py : frame_to_hl` (unchanged);
  directions are collected with `build_adaptive_direction_codebook.load_local_directions`.
- Tool: `tools/build_geometry_basis_diagnostic.py`. Every codebook here is
  **fixed / data-agnostic** (constructed analytically, only *evaluated* on data),
  so there is no fitting and no train/test split — all numbers are on held-out
  directions.
- Codebooks: HL-26; uniform `(theta, phi)` grid with the pole on `+Z` (naive)
  and on `+Y`; equal-area grid (pole `+Y`); Fibonacci near-uniform.
- Metric: mean nearest-code angle (deg); `bits` = code-assignment entropy;
  `dead%` = fraction of unused codes.

## Run

```
python tools/build_geometry_basis_diagnostic.py \
    --annot-root <InterHand annotations> --eval-split test \
    --out experiments/geometry_basis_20260620.json
```

## Result

Run the tool above to populate (raw numbers land in
`experiments/geometry_basis_20260620.json`). The table is `name | K | bits |
angular° | dead%`.

## Findings (mechanism the control is testing)

1. **A naive uniform spherical grid should be *worse* than HL-26, not better.**
   Hand bones concentrate on `+Y`; placing the grid pole on the sparse `+Z` axis
   crowds cells where there is no data, so a large fraction of codes go unused
   (`dead%` high) while the populated region stays coarse.
2. **Pole correction + equal-area recovers most of the gap but not the win.**
   Moving the pole to `+Y` and using equal-area cells should approach Fibonacci
   sampling — clearly better than naive, but still a *uniform* layout with a
   fixed ceiling.
3. **Lesson.** The bottleneck is not resolution or coordinates; it is that a
   fixed geometry does not match the data distribution. That is what the
   data-adaptive codebook (`build_adaptive_direction_codebook.py`) addresses.

## Scope / non-claims

- Spatial per-direction quantization only; no temporal channels, no change to
  `frame_to_hl`. Additive diagnostic.
- All codebooks are geometric (no learning); this brackets the best a fixed
  layout can do.
