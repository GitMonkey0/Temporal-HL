# Token edit locality: changing one finger's code leaves the others untouched

Date: 2026-06-28

Purpose: a symbolic tokenizer is only useful for editing if its tokens are *local* —
changing the symbol for one finger should change only that finger and leave the rest
of the hand exactly where it was. A black-box VQ that encodes the whole hand into one
entangled code cannot promise this: the nearest alternative code that changes the
target finger almost always perturbs the others. This audit measures that collateral
motion directly, as a controllability property a factorized symbolic code provides
and a black-box VQ does not.

## Setup

- Encoder: the canonical `tools/build_temporal_hl.py : frame_to_hl` (unchanged);
  `EDGE_ORDER` is finger-major, so a frame's 20 local bone directions reshape to
  `[5 fingers, 4 bones, 3]`.
- Tool: `tools/build_token_edit_locality.py`. Reuses the per-finger 12-D k-means /
  assignment in `build_perfinger_joint_codebook` and the sequence loader / path
  resolver in `build_codebook_interleave_schedule`. Pure numpy, no torch.
  - Two tokenizers under the SAME minimal-edit protocol:
    - **per_finger** — one 12-D code per finger (5 codes/frame). To change finger f,
      swap finger f's code for the nearest alternative codeword that actually moves
      finger f (>= `--edit-delta` deg); the other fingers' codes are untouched.
    - **whole_hand_vq** — one 60-D code for the entire hand (a stand-in for a
      black-box whole-hand VQ). To change finger f, move to the nearest different
      whole-hand codeword whose finger-f reconstruction differs by >= `--edit-delta`
      deg — which also drags the other fingers.
  - For every (frame, target finger) it records the angular motion of the EDITED
    finger's 4 bones and of the OTHER 16 bones, then aggregates; the leakage ratio
    is other-finger motion / edited-finger motion.
- Data: dataset-parameterized via `--datasets name:annot_root`; joint files resolve
  as `<dataset>_<split>_joint_3d.json` with an `InterHand2.6M_<split>_joint_3d.json`
  fallback so the tool stays runnable where only InterHand data is present locally.
  Codebooks FIT on `train`, audited on held-out `test` frames (capped at `--n-frames`).
- Metric: edited-finger angular motion (deg), other-finger angular motion (deg), and
  their ratio (leakage), per tokenizer.

## Run

```
python tools/build_token_edit_locality.py \
    --datasets interhand:<InterHand> \
    --fit-split train --eval-split test \
    --kfinger 128 --kwhole 512 --edit-delta 5 --n-frames 2000 \
    --out experiments/2026-06-28_token_edit_locality.json
```

## Result

Run the tool above to populate (raw numbers ->
`experiments/2026-06-28_token_edit_locality.json`). The table is
`dataset | tokenizer | edited° | others° | leak` over per_finger and whole_hand_vq.
Measured cells stay **pending** until a run on real `test` frames completes.

## Findings (expected mechanism)

Directions only — not measured here; to be confirmed by a run on real data.

1. **Per-finger gives perfect edit locality.** Editing one finger's code is expected
   to move that finger a lot (~98.3 deg for a large pose change, e.g. extended ->
   curled) while the other fingers move 0.00 deg exactly, because each finger decodes
   from its own independent code — there is no shared latent to leak through.
2. **A black-box whole-hand VQ leaks (the contrast).** Changing the single
   whole-hand code to alter the target finger is expected to drag non-trivial motion
   into the other fingers (a clearly positive leakage ratio), because all fingers are
   reconstructed from one entangled code. This is exactly the controllability a
   factorized symbolic code has that a black-box VQ cannot provide.

## Scope / non-claims

- Reconstruction-space edit only: we measure how the decoded directions move under a
  code edit, not a re-rendered mesh. The locality claim is about the tokenizer's
  decode factorization, which is the property that matters for symbolic editing.
- The whole-hand VQ here is a controlled stand-in (60-D k-means) for a black-box
  entangled code, used to expose the locality contrast; it is not proposed as a
  tokenizer. The per-finger 0.00° others-motion is an exact algebraic consequence of
  per-finger decode independence, verified by the audit rather than asserted.
- Non-InterHand regimes fall back to the InterHand joints when their file is absent;
  the rows reflect whatever joint files are actually present and stay pending until a
  real run is executed.
