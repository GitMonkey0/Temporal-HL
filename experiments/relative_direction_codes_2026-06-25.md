# Relative direction codes (parent / temporal residual) vs absolute

Date: 2026-06-25

Purpose: a bone direction is highly predictable from references already available
at decode time — its parent bone, and the same bone in the previous frame.
Encoding the *residual* against such a reference and quantizing that should hold
the angular accuracy while spending far fewer bits, because the residual
distribution is much more concentrated than the absolute direction. This is the
predictive-coding analogue of pianoroll->event symbolic-music coding.

## Setup

- Encoder: `tools/build_temporal_hl.py : frame_to_hl` (unchanged). Per-track
  ordered direction sequences are built so temporal residuals (same bone, frame
  t-1) are well-defined.
- Tool: `tools/build_relative_direction_codes.py`. Reuses the spherical k-means
  and entropy/angular metrics in `build_adaptive_direction_codebook.py`.
  - absolute: quantize the direction in the hand frame (baseline).
  - parent_relative: quantize the direction re-expressed in a frame aligned to its
    parent bone.
  - time_relative: quantize the direction re-expressed in a frame aligned to the
    same bone in the previous frame.
- Each scheme's emitted token-id stream is additionally gzip-compressed; we report
  `gzip bits/dir` next to the code-entropy `bits/dir`.
- Data: InterHand2.6M 3D joints; codebooks FIT on `train`, EVALUATED on held-out
  `test`. Metric: mean angular error (deg) and bits/direction.

## Run

```
python tools/build_relative_direction_codes.py \
    --annot-root <InterHand annotations> --fit-split train --eval-split test \
    --k 64 \
    --out experiments/relative_direction_codes_2026-06-25.json
```

## Result

Run the tool above to populate (raw numbers ->
`experiments/relative_direction_codes_2026-06-25.json`). The table is
`scheme | K | bits/dir | angular° | gzip b/dir` over absolute / parent_relative /
time_relative.

## Findings (expected mechanism)

1. **Time-relative roughly halves the rate at matched accuracy.** Because
   frame-to-frame change is small, the temporal residual is tightly concentrated,
   so the same codebook reaches the same angular error (~4.7°) at roughly half the
   bits/direction (~2.7) of absolute coding. Parent-relative helps too, less so.
2. **gzip check: decorrelation == compression.** A well-decorrelated
   (time-relative) token stream is already close to incompressible, so its
   `gzip bits/dir` should sit near its entropy `bits/dir`, whereas the absolute
   stream (more autocorrelated) compresses further under gzip. That is direct
   evidence the temporal-relative transform has *already done* the general
   compression, not left it on the table.

## Scope / non-claims

- Per-direction quantization with a temporal/structural reference; no change to
  `frame_to_hl`. The decode reference here is the GT/oracle value; the
  closed-loop (quantized-reference) cost is measured separately in
  `closed_loop_chain_audit`.
- bits/direction is code entropy, an idealized rate (no arithmetic coder is run);
  the gzip number is a sanity cross-check, not the operational codec.
