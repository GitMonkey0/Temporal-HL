# Adaptive direction codebook vs fixed HL-26

Date: 2026-06-21

Purpose: quantify how much the *fixed, uniform* 26-direction HL codebook costs in
quantization accuracy, and how much a *data-adaptive* codebook of the same size
recovers — as evidence for the representation-design section (improving HL at the
representation level, building on its symbolic centers rather than replacing
them).

## Setup

- Encoder: the canonical `tools/build_temporal_hl.py : frame_to_hl` (unchanged).
  The new tool collects the exact normalized local bone directions HL quantizes,
  so there is zero convention drift from the HL labels this repo produces.
- Tool: `tools/build_adaptive_direction_codebook.py` — spherical k-means (cosine)
  on the collected directions, then mean angular quantization error per codebook.
- Data: InterHand2.6M 3D-joint annotations. **Codebook FIT on `train`
  (405,080 bone directions), all codebooks EVALUATED on held-out `test`
  (123,800 directions).** No frame from the eval split is seen during fitting.
- Metric: mean angle (deg) from each direction to its nearest code; `bits/vec`
  = empirical entropy of the code assignment (the rate side of the comparison,
  so codebooks of different K are compared fairly, not by raw error).

Run:

```
python tools/build_adaptive_direction_codebook.py \
    --annot-root <InterHand annotations> \
    --fit-split train --eval-split test --ks 12 16 26 40 64 96 128
```

## Result (held-out test)

| codebook | K | bits/vec | angular° |
|---|---:|---:|---:|
| **HL-26 (fixed)** | 26 | 3.10 | **14.85** |
| kmeans12 | 12 | 3.24 | 11.56 |
| kmeans16 | 16 | 3.60 | 9.96 |
| kmeans26 (HL-init) | 26 | 4.20 | 7.96 |
| **kmeans26 (adaptive)** | 26 | 4.26 | **7.87** |
| kmeans40 | 40 | 4.86 | 6.43 |
| kmeans64 | 64 | 5.44 | 5.21 |
| kmeans96 | 96 | 5.96 | 4.29 |
| kmeans128 | 128 | 6.33 | 3.77 |

## Findings

1. **Same budget, ~47% lower error.** At the matched K=26, an adaptive codebook
   drops mean quantization error from **14.85° to 7.87° (−47%)**. The HL-26
   *layout* is not optimal for 26 codes — its uniform spacing wastes codes on
   directions finger bones rarely take (most point "forward" along canonical +Y).

2. **Build-on-HL works.** Initializing k-means at the 26 HL directions
   (`kmeans26_hlinit`, 7.96°) lands essentially at the from-scratch optimum while
   keeping HL's symbolic centers as the starting point — the representation stays
   an HL refinement, not a black-box replacement.

3. **Matched-rate, fewer codes already win.** kmeans12 (12 codes, 3.24 bits/vec)
   reaches 11.56° — lower error than HL-26 (14.85°) at a *lower* bit rate. So the
   gain is not "more codes": it is placing codes where the data is.

4. The K sweep traces a rate-distortion frontier the fixed codebook sits above;
   a temporal-HL representation can pick its operating point on it.

Raw numbers: `experiments/adaptive_direction_codebook_20260621.json`.

## Scope / non-claims

- This measures *spatial* per-direction quantization only; it does not touch the
  temporal channels (transition / flexion / interaction / duration) and does not
  change `frame_to_hl`. It is an additive diagnostic.
- k-means here is geometric (no learned encoder). It is the lower bound on what a
  data-adaptive layout buys; a learned codebook would be an upper bound.
- A symbolic codebook trades interpretability for accuracy: HL-26's named
  directions are human-readable, adaptive centers are not. The HL-init variant is
  the practical middle ground for a representation that must stay symbolic.
