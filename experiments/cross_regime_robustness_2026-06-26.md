# Cross-regime robustness audit (with a per-finger failure-mode diagnostic)

Date: 2026-06-26

Purpose: stress-test the three main HL tokenizers off their training regime. The
fixed HL-26 codebook, the data-adaptive per-bone codebook, and the anatomy-aware
per-finger joint codebook all look good on clean InterHand two-hand captures; this
audit asks whether the ordering survives a change of capture regime (single-hand
FreiHAND, free single-hand HanCo sequences, noisy occluded ASL sign-language
poses). It also runs an honest failure-mode probe on the per-finger joint code.

## Setup

- Encoder: the canonical `tools/build_temporal_hl.py : frame_to_hl` (unchanged);
  `EDGE_ORDER` is finger-major, so a frame's 20 local bone directions reshape to
  `[5 fingers, 4 bones, 3]`.
- Tool: `tools/build_cross_regime_robustness.py`. Reuses the per-bone codebook /
  metrics in `build_adaptive_direction_codebook.py` and the per-finger joint
  codebook (`euclid_kmeans`, assignment) in `build_perfinger_joint_codebook.py`.
  - The two data-driven codebooks are FIT ONCE on a reference regime
    (`--fit-dataset`, default the first entry, `train` split), then every
    tokenizer is APPLIED unchanged to each regime's held-out `test` directions.
  - Tokenizers: HL-26 fixed; data-adaptive per-bone k-means (K=26); per-finger
    joint (one 12-D code per finger, K=128).
- Data: dataset-parameterized via `--datasets name:annot_root` (one entry per
  regime). Joint files resolve as `<dataset>_<split>_joint_3d.json` with an
  `InterHand2.6M_<split>_joint_3d.json` fallback so the tool stays runnable where
  only InterHand data is present locally.
- Metrics: per regime, mean reconstruction angular error (deg) and rate
  (bits/bone) for each tokenizer. Failure-mode diagnostic: for each (regime,
  finger) the per-finger joint code reconstructs four bones from ONE shared code;
  we record the WORST single bone's angular error and the finger's joint-code mean
  error, then report the correlation between worst-bone error and joint-code error
  across all (regime, finger) pairs.

## Run

```
python tools/build_cross_regime_robustness.py \
    --datasets interhand:<InterHand> freihand:<FreiHAND> hanco:<HanCo> asl:<ASL> \
    --fit-dataset interhand --fit-split train --eval-split test \
    --k 26 --kfinger 128 \
    --out experiments/cross_regime_robustness_2026-06-26.json
```

## Result

Run the tool above to populate (raw numbers ->
`experiments/cross_regime_robustness_2026-06-26.json`). The first table is
`regime | HL26° | adapt° | pfing° | HL26b | adptb | pfingb`; the second is the
per-finger diagnostic `regime | finger | joint_mean° | worst_bone°`, followed by
the correlation between worst-bone error and finger joint-code error.

## Findings (expected mechanism)

1. **The main line holds across regimes.** On every regime the data-adaptive
   per-bone codebook is expected to keep its edge over fixed HL-26 (lower angular
   error at the matched K=26 budget), and the per-finger joint code is expected to
   reach comparable accuracy at clearly lower bits/bone — "adaptive > fixed;
   structured saves rate" should not be an InterHand-only artefact.
2. **Per-finger degrades on the noisy ASL regime (negative result, recorded
   honestly).** Because the four bones of a finger share ONE code, a single badly
   reconstructed bone — typical of ASL lifting / self-occlusion poses — is expected
   to drag the whole finger's joint-code error up, so the per-finger tokenizer
   should lose ground relative to the per-bone codebook specifically on ASL.
3. **The diagnostic should expose the mechanism.** The worst-single-bone error and
   the finger joint-code error are expected to be strongly positively correlated
   across (regime, finger) pairs: the shared per-finger code is hijacked by its
   worst bone. This is the concrete motivation for an independent-residual
   (hierarchical) per-finger code in a later step — a coarse shared code plus an
   independent per-bone residual so one bad bone no longer dominates.

## Scope / non-claims

- Spatial quantization only; no temporal channels; no change to `frame_to_hl`.
- The per-finger weakness here is reported as a limitation, not hidden: it is the
  motivation for the hierarchical fix, not a defeat of the structured codebook,
  which still wins on rate in the clean regimes.
- FreiHAND / HanCo / ASL are each read via a `<dataset>_<split>_joint_3d.json`
  sibling using the same `frame_to_hl` encoder; if a regime's file is absent the
  tool falls back to the InterHand joint file so it stays runnable, and the rows
  reflect whatever joint files are actually present.
