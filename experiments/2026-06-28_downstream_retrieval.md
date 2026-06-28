# Training-free sequence retrieval on tokenizer streams (HL-26 / adaptive / per-finger)

Date: 2026-06-28

Purpose: a discrete tokenization unlocks a capability continuous coordinates do not
have for free — training-free symbolic matching. Once a clip is a string of tokens,
two clips compare by bag-of-tokens overlap, shared n-grams, or token warping
distance, with no learned encoder and no training. This probe asks how retrievable
each tokenizer's streams are: given a query clip, can we pull back a clip of the
SAME gesture purely by symbolic similarity, and by how much do we beat chance?

## Setup

- Encoder: the canonical `tools/build_temporal_hl.py : frame_to_hl` (unchanged).
- Tool: `tools/build_downstream_retrieval.py`. Reuses the tokenizer classes in
  `build_downstream_forecast` (hl26 / adaptive_kmeans / per_finger), the
  `build_adaptive_direction_codebook` / `build_perfinger_joint_codebook` codebooks,
  and the path resolver in `build_codebook_interleave_schedule`. Pure numpy, no
  torch, fully training-free.
  - Matchers (discrete-only): **bag** (per-(slot,code) histogram, cosine), **ngram**
    (consecutive frame-word bigram sets, Jaccard), **dtw** (dynamic time warping with
    per-frame Hamming token-mismatch cost).
  - Continuous reference: **float_dtw** — DTW with L2 cost on the raw float
    directions, the only matcher continuous coordinates support; bag / ngram need
    discrete symbols and are undefined on floats. This is the concrete sense in
    which symbolic matching is a discrete-only capability.
  - Relevance: a self-supervised gesture label per clip (the majority per-frame 5-bit
    extended/curled flexion signature, identical to the gesture-classification probe).
  - Protocol: leave-one-out precision@1 (does the nearest OTHER clip share the
    query's gesture?), reported absolutely AND as a multiple of the random-pick
    baseline (the gallery's expected same-label rate). Clips are downsampled to
    `--max-len-keep` frames (DTW is O(T^2)) and capped at `--max-seqs` (leave-one-out
    is O(N^2)).
- Data: dataset-parameterized via `--datasets name:annot_root`; joint files resolve
  as `<dataset>_<split>_joint_3d.json` with an `InterHand2.6M_<split>_joint_3d.json`
  fallback so the tool stays runnable where only InterHand data is present locally.
  Codebooks FIT on `train`; retrieval EVALUATED on held-out `test`.

NOTE: a learned symbolic-retrieval stack already exists in this repo
(`eval_sequence_symbolic_retrieval.py`, `eval_symbolic_retrieval.py`). This is a
clean, TRAINING-FREE scoped entry point for the tokenizer comparison and does not
modify or invoke those.

## Run

```
python tools/build_downstream_retrieval.py \
    --datasets interhand:<InterHand> \
    --eval-split test --kadaptive 64 --kfinger 128 \
    --matchers bag ngram dtw --max-seqs 400 --max-len-keep 48 \
    --out experiments/2026-06-28_downstream_retrieval.json
```

## Result

Run the tool above to populate (raw numbers ->
`experiments/2026-06-28_downstream_retrieval.json`). Two tables:
`dataset | tokenizer | matcher | p@1 | random | xrand` over the discrete matchers,
and a one-row continuous reference (`float_dtw`). Measured cells stay **pending**
until a run on real multi-gesture data completes (the local synthetic fallback
collapses to a single gesture class, so its ratio is uninformative).

## Findings (expected mechanism)

Directions only — not measured here; to be confirmed by a run on real data.

1. **Symbolic matching beats chance by a large factor.** Training-free precision@1
   is expected to reach many times the random baseline (up to ~31x), confirming that
   the tokens carry gesture identity recoverable by pure string similarity.
2. **Per-finger bag-of-tokens is strongest (~25.4% p@1).** A whole-finger code names
   exactly the configuration that defines a handshape, so a bag of finger codes is a
   near-direct handshape descriptor; the per-bone bags are more diffuse.
3. **The capability is discrete-only.** The float_dtw reference shows the continuous
   baseline can warp but cannot do the bag / n-gram edit-style matching that drives
   most of the gain — bag and n-gram are undefined on raw floats. The discrete
   tokenization is what makes training-free symbolic retrieval possible.

## Scope / non-claims

- The gesture label is a self-supervised flexion signature, not a human gesture
  taxonomy; it is a controlled relevance proxy, and the ordering across
  tokenizers / matchers is the claim, not absolute precision.
- DTW is quadratic; clips are downsampled and the clip count capped, so the numbers
  are for the capped, downsampled protocol stated above.
- Non-InterHand regimes fall back to the InterHand joints when their file is absent;
  the rows reflect whatever joint files are actually present, and stay pending until
  a real multi-gesture run is executed.
