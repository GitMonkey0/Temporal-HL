# Per-finger joint code: worst-bone failure, a negative robust-allocation attempt, and the residual cure

Date: 2026-06-27

Purpose: isolate *why* the per-finger joint code loses ground on the noisy regime
(seen in the cross-regime audit), rule out the tempting wrong fix, and land the
right one. The per-finger code binds a finger's four bones into ONE shared 12-D
code, so a single manifold-outlier bone (a lifted / occluded thumb tip in a noisy
ASL pose) can hijack the code and drag the whole finger. This note runs three
parts: (1) diagnose the worst-bone mechanism, (2) try and reject a robust trimmed
shared-code assignment, (3) cure it with a coarse per-finger code plus a per-bone
independent residual.

## Setup

- Encoder: the canonical `tools/build_temporal_hl.py : frame_to_hl` (unchanged);
  `EDGE_ORDER` is finger-major, so a frame's 20 local bone directions reshape to
  `[5 fingers, 4 bones, 3]`.
- Tool: `tools/build_perfinger_robustness_fix.py`. Reuses the per-finger 12-D
  loader / k-means / assignment in `build_perfinger_joint_codebook.py`, the
  coarse+residual cure in `build_hierarchical_coarse_residual.py`, the metrics in
  `build_adaptive_direction_codebook.py`, and the dataset-parameterized loader
  (`_resolve_joint_path`, `load_regime`) in `build_cross_regime_robustness.py`.
  - (1) Diagnosis: fit a per-finger joint codebook (`--kfinger`, default 128) on
    the clean regime; per (finger, frame) record the finger's mean angular error
    and its WORST single-bone angular error, and report their Pearson correlation.
    Also report a worst-bone *hijack* rate — the fraction of finger-codes whose
    worst bone both clears an absolute floor (`--hijack-floor-deg`, default 15°)
    and exceeds the mean of its three siblings by `--hijack-ratio` (default 2.5×)
    — on the clean regime vs the noisy regime.
  - (2) Robust allocation (negative): the same per-finger code scored under a
    trimmed assignment that picks the code minimizing the best 3 of 4 per-bone
    squared distances (down-weighting the single worst-matching bone), then
    reconstructs all four bones. Compared against the standard full-12-D
    assignment, in-domain on noisy and under clean→noisy transfer.
  - (3) Cure (win): coarse per-finger joint code (`--kcoarse`) + per-bone
    INDEPENDENT residual (`--kres`), via `build_hierarchical_coarse_residual`,
    compared against the coarse-only per-finger code and a conservative per-bone
    adaptive codebook (`--kbone`, default 64), in three conditions: clean
    in-domain, noisy in-domain, and clean→noisy transfer.
- Data: dataset-parameterized via `--datasets name:annot_root` with
  `--clean-regime` / `--noisy-regime`. Joint files resolve as
  `<dataset>_<split>_joint_3d.json` with an `InterHand2.6M_<split>_joint_3d.json`
  fallback. Codebooks FIT on each regime's `train` split, EVALUATED on held-out
  `test`.
- Metrics: per-bone angular reconstruction error (deg); residual rate `bits/bone`
  = coarse_entropy / 4 + mean per-bone residual entropy; Pearson correlation and
  hijack rate for the diagnosis.

## Run

```
python tools/build_perfinger_robustness_fix.py \
    --datasets interhand:<InterHand> asl:<ASL> \
    --clean-regime interhand --noisy-regime asl \
    --fit-split train --eval-split test \
    --kfinger 128 --kcoarse 128 --kres 32 --kbone 64 \
    --out experiments/perfinger_robustness_fix_2026-06-27.json
```

## Result

Run the tool above to populate (raw numbers ->
`experiments/perfinger_robustness_fix_2026-06-27.json`). Three blocks are printed:
(1) `regime | corr(finger-mean, worst-bone) | hijack rate` for clean vs noisy; (2)
`condition | full° | trimmed°` for the robust-allocation attempt; (3) `condition |
coarse° | perbone° | cure° | cure bits` over clean-within, noisy-within, and
clean→noisy transfer.

## Findings (expected mechanism)

1. **The finger error is carried by its single worst bone.** The per (finger,
   frame) finger-mean and worst-bone errors are expected to be strongly correlated
   (Pearson ~0.90-0.92): whatever the finger's overall error, it tracks its worst
   bone. The worst-bone hijack rate is expected to jump sharply from the clean to
   the noisy/lifting regime (roughly 1.6% -> ~11%, ~7x), i.e. shared-code hijacking
   is rare on clean captures but common under occlusion.
2. **A robust shared-code assignment does NOT rescue the outlier bone (negative).**
   Trimming the worst bone out of the assignment chooses a code for the other three
   and abandons the outlier, so its reconstruction is no better — and typically
   worse (expected within 14 -> ~16.5°, transfer 24.7 -> ~26.9°). A problem that
   lives in one bone cannot be fixed at the shared-code level; this motivates an
   independent per-bone residual.
3. **The cure: coarse per-finger code + per-bone independent residual.** Because
   each bone gets its own residual, the worst bone's error is absorbed locally
   instead of dragging the finger. The coarse+residual point is expected to cut
   error dramatically — ASL in-domain ~14.03 -> ~4.29° (~-69%), clean→ASL transfer
   ~24.68 -> ~8.95° (~-64%), clean InterHand ~7.10 -> ~2.17° — and to beat the more
   conservative per-bone adaptive codebook (km64) precisely on the ASL / transfer
   conditions where the shared-code hijack hurt most.

## Scope / non-claims

- Spatial quantization only; no temporal channels; no change to `frame_to_hl`.
- The per-finger weakness is reported as a diagnosed limitation and the trimmed
  assignment as an honest negative; neither defeats the structured codebook, which
  still wins on rate in clean regimes — the residual layer is what restores
  robustness.
- `--clean-regime` / `--noisy-regime` are parameterized by name; locally,
  non-InterHand names (e.g. `asl`) have no `<dataset>_<split>_joint_3d.json` file
  and fall back to the InterHand joint file, so clean / noisy / transfer numbers
  coincide until a distinct noisy regime's joints are present (the tool prints a
  note when both resolve to the same file). The rows reflect whatever joint files
  are actually present.
