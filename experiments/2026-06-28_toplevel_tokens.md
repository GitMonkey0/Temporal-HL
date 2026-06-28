# Top-level HL tokens: whole-hand shape, relational, and motion-state (with a negative)

Date: 2026-06-28

Purpose: the per-bone / per-finger HL tokens describe the hand bottom-up. Many
downstream uses want a single readable symbol per frame — "what shape", "how the
fingers relate", "is it moving". This tool adds three orthogonal TOP-LEVEL tokens on
top of the existing HL stream and measures what each contributes, including an
explicit 负结果.

## Setup

- Encoder: the canonical `tools/build_temporal_hl.py : frame_to_hl` (unchanged); the
  per-frame flexion scores from `compute_finger_flexion` drive the gesture label and
  the relational symbol.
- Tool: `tools/build_toplevel_tokens.py`. Reuses
  `build_downstream_gesture_classification` (its ridge probe + HL-26 bag features),
  `build_perfinger_joint_codebook` (k-means), `build_adaptive_direction_codebook`,
  and the path resolver in `build_codebook_interleave_schedule`. Pure numpy, no torch.
  - Three top-level tokens:
    1. **whole-hand shape** — one k-means code over the 60-D (20 bones x 3) frame
       vector (`--kwhole`): a single holistic shape symbol per frame. Tested by
       whether it BOOSTS the fixed gesture classifier on top of the per-bone HL bag.
    2. **relational** — per-finger extended / bent / curl (3 levels from the flexion
       score via `--ext-thr` / `--curl-thr`) -> a 5-symbol relation code. Tested by
       its normalized mutual information (NMI) with the gesture label and by printing
       the most frequent codes as readable handshapes.
    3. **motion-state** — one k-means code over a causal-window recent-speed feature
       (`--kmotion`, `--motion-window`): static / slow / fast. Tested, like (1), by
       whether it helps the gesture classifier.
- Data: dataset-parameterized via `--datasets name:annot_root`; joint files resolve
  as `<dataset>_<split>_joint_3d.json` with an `InterHand2.6M_<split>_joint_3d.json`
  fallback so the tool stays runnable where only InterHand data is present locally.
  Codebooks + probe FIT on `train`, EVALUATED on held-out `test`.
- Metrics: top-1 gesture accuracy for the shape / motion-state tokens (vs the HL-26
  bag baseline); NMI(relational code, gesture label) + readable top codes for the
  relational token.

NOTE: a gesture classifier already exists in this repo
(`build_downstream_gesture_classification.py`, `train_symbolic_classifier.py`). This
tool REUSES the former's probe + loader utilities and adds the three top-level
tokens; it does not modify them.

## Run

```
python tools/build_toplevel_tokens.py \
    --datasets interhand:<InterHand> \
    --fit-split train --eval-split test \
    --kwhole 64 --kmotion 8 --motion-window 5 \
    --out experiments/2026-06-28_toplevel_tokens.json
```

## Result

Run the tool above to populate (raw numbers ->
`experiments/2026-06-28_toplevel_tokens.json`). Three blocks are printed: (1)
`base` vs `base + whole-hand shape` gesture accuracy; (2) NMI(relational, gesture)
and the top relational EBC codes with names; (3) `base + motion-state` gesture
accuracy. Measured cells stay **pending** until a run on real `train`/`test` data
completes (the local synthetic fallback collapses to a single gesture class).

## Findings (expected mechanism)

Directions only — not measured here; to be confirmed by a run on real data.

1. **Whole-hand shape boosts gesture classification.** Adding the single holistic
   shape token on top of the per-bone bag is expected to lift accuracy by a few
   points (~+5.8 -> ~49.3%), a cheap global feature the per-bone bag lacks.
2. **The relational token yields readable handshapes.** NMI with the gesture label
   is expected around ~0.301 and the frequent relation codes are interpretable: all
   five fingers extended = "5" (open hand), index+middle extended = "V", all curled =
   fist "S". This is the human-readable, ASL-style layer on top of HL.
3. **Motion-state is a 负结果 (recorded as a 局限, not hidden).** The motion-state
   token is expected NOT to help a static-pose gesture classifier (delta ~ 0),
   because pose identity and motion state are orthogonal — knowing the hand is moving
   fast says little about which handshape it is in. Motion-state is useful for a
   different task (dynamics / segmentation), not for static gesture pose; we report
   the null result rather than overclaiming a universal gain.

## Scope / non-claims

- The three tokens are deliberately simple (single k-means symbol / threshold code)
  and added on a fixed linear probe, to isolate each token's marginal contribution;
  the direction of each effect is the claim, not absolute accuracy.
- The gesture label is the self-supervised flexion signature shared with the
  gesture-classification probe; NMI is the sqrt-normalized form and is 0 when either
  labeling is constant (guarded against float noise).
- Non-InterHand regimes fall back to the InterHand joints when their file is absent;
  the rows reflect whatever joint files are actually present and stay pending until a
  real multi-gesture run is executed.
