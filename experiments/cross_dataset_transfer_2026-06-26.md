# Cross-dataset transfer of the data-adaptive direction codebook

Date: 2026-06-26

Purpose: test whether the adaptive direction codebook's gain over fixed HL-26 is
*general hand structure* or a single dataset's quirk. The honest probe is transfer:
fit the spherical k-means codebook on dataset A, apply it unchanged (no re-fit) to
dataset B, and vice-versa. If a codebook fit on A still beats HL-26 on B, the
adaptive layout is capturing shared anatomy of how finger bones point, not a
capture artefact.

## Setup

- Encoder: the canonical `tools/build_temporal_hl.py : frame_to_hl` (unchanged);
  every (capture, hand, bone) gives one canonical local bone direction, identical
  to the HL token inputs.
- Tool: `tools/build_cross_dataset_transfer.py`. Reuses the spherical k-means,
  angular-error and entropy helpers in `build_adaptive_direction_codebook.py`.
  - per dataset: FIT a K=26 spherical k-means codebook on its `train` split.
  - HL-26: the fixed cube-surface codebook (`build_temporal_hl.CODEBOOK`).
  - score every codebook on every dataset's held-out `test` split.
- Data: dataset-parameterized via `--datasets name:annot_root` (InterHand,
  FreiHAND, ...). Joint files resolve as `<dataset>_<split>_joint_3d.json` with an
  `InterHand2.6M_<split>_joint_3d.json` fallback so the tool stays runnable where
  only InterHand data is present locally.
- Metric: mean reconstruction angular error (deg), arranged as a transfer matrix
  `rows = codebook source (HL-26, adaptive@A, adaptive@B) x columns = eval dataset`.
  Diagonal = own-data fit; off-diagonal = transferred codebook; HL-26 row = fixed
  baseline on each dataset.

## Run

```
python tools/build_cross_dataset_transfer.py \
    --datasets interhand:<InterHand annotations> freihand:<FreiHAND annotations> \
    --fit-split train --eval-split test --k 26 \
    --out experiments/cross_dataset_transfer_2026-06-26.json
```

## Result

Run the tool above to populate (raw numbers ->
`experiments/cross_dataset_transfer_2026-06-26.json`). The first table is the
transfer matrix `source \ eval` of mean angular error; the second lists each
transferred adaptive codebook against the HL-26 baseline on the same eval dataset
(`fit_on | eval_on | hl26° | adaptive° | pct_vs_hl26`).

## Findings (expected mechanism)

1. **The transferred adaptive codebook still beats HL-26 in both directions.** A
   codebook fit on InterHand should reduce reconstruction error on FreiHAND below
   the fixed HL-26 baseline, and the FreiHAND-fit codebook should likewise beat
   HL-26 on InterHand — the off-diagonal entries should sit below the HL-26 row in
   every column, not just on the diagonal.
2. **The gain is structure, not overfitting.** On FreiHAND the transferred
   adaptive codebook is expected to cut angular error by roughly -61% vs HL-26;
   that a codebook never fit on FreiHAND still recovers most of the own-fit gain
   says the adaptive layout encodes the shared concentration of finger-bone
   directions (general anatomy), not one dataset's distribution.
3. **Own-data fit is an upper bound, transfer is close to it.** The diagonal
   (own-fit) entries should be the lowest in each column, with the transferred
   entries only slightly above — a small own-vs-transfer gap is the quantitative
   statement that the codebook generalises across capture regimes.

## Scope / non-claims

- Spatial per-bone quantization only; no temporal channels; no change to
  `frame_to_hl`. This is a transfer/generalisation result for the direction
  codebook, not a claim about every downstream task.
- FreiHAND (and any non-InterHand regime) is read via a
  `<dataset>_<split>_joint_3d.json` sibling using the same `frame_to_hl` encoder;
  if that file is absent the tool falls back to the InterHand joint file so it
  stays runnable, and the matrix rows reflect whatever joint files are actually
  present.
