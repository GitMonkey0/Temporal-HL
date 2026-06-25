# Downstream gesture classification on tokenizer streams (HL-26 / adaptive / per-finger)

Date: 2026-06-25

Purpose: reconstruction angular error is intrinsic; a representation can
reconstruct well and still be a poor *symbolic* substrate downstream. This probe
trains one fixed classifier on the discrete token streams each tokenizer emits and
asks which yields the most discriminable codes for a gesture-label task.

## Setup

- Encoder: `tools/build_temporal_hl.py : frame_to_hl` (unchanged). The per-frame
  gesture label is derived from the per-finger flexion pattern
  (`compute_finger_flexion`, each finger binned extended/curled -> a 5-bit class),
  so the task is identical across tokenizers and needs no external annotation.
- Tool: `tools/build_downstream_gesture_classification.py`. Reuses
  `build_adaptive_direction_codebook` (HL-26 + spherical k-means) and
  `build_perfinger_joint_codebook` (the 12-D per-finger code). The classifier is a
  self-contained numpy closed-form ridge linear probe on L1-normalized bag-of-codes
  histograms (no torch / sklearn).
  - hl26: the fixed 26-direction codebook.
  - adaptive_kmeans: a per-bone data-adaptive codebook.
  - per_finger: one joint code per finger (~half the per-frame bit-rate).
- Data: InterHand2.6M 3D joints; codebooks + probe FIT on `train`, EVALUATED on
  held-out `test`.
- Metric: top-1 gesture accuracy; `bits/frame` is the tokenizer's per-frame rate.

NOTE: a related classifier exists in this repo (`train_symbolic_classifier.py`,
`eval_symbolic_channel_variants_classifier.py`). This is a clean scoped entry
point for the tokenizer comparison and does not modify or invoke those.

## Run

```
python tools/build_downstream_gesture_classification.py \
    --annot-root <InterHand annotations> --fit-split train --eval-split test \
    --kadaptive 26 --kfinger 128 \
    --out experiments/downstream_gesture_classification_2026-06-25.json
```

## Result

Run the tool above to populate (raw numbers ->
`experiments/downstream_gesture_classification_2026-06-25.json`). The table is
`tokenizer | K | bits/frame | acc` over hl26 / adaptive_kmeans / per_finger.

## Findings (expected mechanism)

1. **Data-adaptive adds a few points over HL-26** (~+4.5, e.g. 39.0 -> 43.5),
   because it spends codes where the directions actually are, so its codes carry
   more label-relevant structure.
2. **Per-finger matches or beats HL-26 at roughly half the bit-rate**, because a
   gesture *is* a set of finger poses and the per-finger code names exactly those.
   Compactness and discriminability move together here.

## Scope / non-claims

- A linear probe on bag-of-codes histograms is deliberately simple, to isolate the
  tokenizer; it is not a tuned sequence model. The ordering across tokenizers is
  the claim, not the absolute accuracy.
- The gesture label is a self-supervised flexion signature, not a human gesture
  taxonomy; it is a controlled discriminability proxy.
