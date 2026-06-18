# Encoding consistency note

Date: 2026-06-18

Purpose: pin down the canonical per-frame HL-26 direction encoding so the
symbolic representation stays stable, and record a cross-implementation check.

## Canonical encoder

`tools/build_temporal_hl.py : frame_to_hl` is canonical. Per frame, per hand:

1. canonical hand frame — origin at the wrist; `+Y` toward the middle-finger
   base; palm plane spanned by index/pinky bases (`z = orthogonalize(pinky_base
   - index_base, y)`, `x = y × z`);
2. 20 regional vectors `child - parent` expressed in that frame;
3. each direction quantized to the nearest of the 26 cuboid-surface symbols
   (`{-1,0,1}^3 \ {origin}`, normalized; nested x,y,z iteration fixes the id
   order).

## Regression guard

`tools/verify_hl_encoding.py` independently re-derives the codebook, the
canonical basis, and the quantization (reusing only `EDGE_ORDER` /
`parent_index`) and asserts token-for-token agreement with `frame_to_hl`.

Verified 2026-06-18:

- `python -m tools.verify_hl_encoding` (3000 seeded synthetic frames):
  120000/120000 tokens identical (100.00%).
- `python -m tools.verify_hl_encoding --joints <InterHand val joint_3d.json>`
  (4000 frames): 74240/74240 tokens identical (100.00%).

## Cross-codebase check

The same canonical encoding was reproduced by a fully separate geometric
tokenizer pipeline (independent codebase: regional vectors → canonical frame →
26-direction argmax, with reconstruction-error evaluation). On real InterHand
(val+test), once aligned to this convention it matches `frame_to_hl` exactly:

- right hand: 6400/6400 tokens identical (100.00%)
- left hand:  2220/2220 tokens identical (100.00%)

Two convention points that must match for exact agreement (both are part of the
canonical definition here):

1. **Canonical basis** — use this repo's basis (`z` from the index→pinky palm
   span orthogonalized against `y`). A different-but-valid palm-normal
   construction (`z` from `cross(palm_normal, y)`) agreed only ~97%, diverging
   on regional vectors that sit near a cuboid-bin boundary.
2. **Left hand is not mirrored** — the left hand uses its own basis. Folding the
   left hand into the right-hand frame (mirroring) drops left-hand agreement to
   ~61% and is not used here.
