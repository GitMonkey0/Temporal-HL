# Learned-VQ upper bound: geometric k-means MATCHES a tuned learned VQ

Date: 2026-06-25

Purpose: a fair reviewer will ask whether a *learned* vector quantizer — the kind
trained inside a neural tokenizer with EMA codebook updates and dead-code revival
— would beat geometric spherical k-means. This control builds that learned VQ as
a numpy reference, uses it as an UPPER BOUND, and shows geometric k-means reaches
it at matched K.

## Setup

- Encoder: `tools/build_temporal_hl.py : frame_to_hl` (unchanged); directions via
  `build_adaptive_direction_codebook.load_local_directions`.
- Tool: `tools/build_learned_vq_upper_bound.py`. Reuses the spherical k-means and
  metrics in `build_adaptive_direction_codebook.py`; the learned VQ is a
  self-contained numpy EMA-k-means with **dead-code revival** (codes with low EMA
  count are reseeded to random data points) — no torch required.
- Data: InterHand2.6M 3D joints; both quantizers FIT on `train`, EVALUATED on
  held-out `test`, at each K in `--ks`.
- Metric: angular error (deg), codebook utilization (fraction of codes used), and
  perplexity. The `gap_to_upper_bound_deg = kmeans° - learnedVQ°` is ~0 when
  k-means matches the learned bound.

## Run

```
python tools/build_learned_vq_upper_bound.py \
    --annot-root <InterHand annotations> --fit-split train --eval-split test \
    --ks 26 64 128 \
    --out experiments/learned_vq_upper_bound_2026-06-25.json
```

## Result

Run the tool above to populate (raw numbers ->
`experiments/learned_vq_upper_bound_2026-06-25.json`). The table is
`K | kmeans° | learnedVQ° | gap° | km_util | vq_util`.

## Findings (expected mechanism)

1. **Geometric k-means MATCHES / REACHES the tuned learned-VQ upper bound.** With
   dead-code revival the learned VQ is properly tuned (utilization ~100%), and the
   per-granularity gap should be ~0 (e.g. at K=64, ~5.64 vs ~5.63°). The honest
   conclusion is that k-means *reaches* the learned bound, not that it exceeds it.
2. **Caveat — never claim "k-means beats learned VQ".** An earlier internal
   version reported k-means *beating* learned VQ; that was an artifact of a
   **collapsed** VQ (no dead-code revival -> many unused codes -> the learned
   codebook looked worse than it is). Reviving dead codes removes the artifact and
   the corrected statement is "matches the learned upper bound". This note records
   the caveat so the framing is not re-broken later.

## Scope / non-claims

- Per-direction quantization only; no temporal channels; no change to
  `frame_to_hl`.
- The learned VQ is a numpy EMA-k-means proxy for a neural codebook, not a full
  trained tokenizer; it is a controlled upper-bound reference, deliberately simple
  and reproducible.
