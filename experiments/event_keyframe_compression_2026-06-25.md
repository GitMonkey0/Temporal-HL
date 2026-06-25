# Event-level keyframe temporal compression (RDP + slerp) vs uniform subsampling

Date: 2026-06-25

Purpose: compress the *number of frames* (orthogonal to the per-frame bit cost the
other tools target). Hand motion is piecewise-smooth, so most frames are
predictable from their neighbors by interpolation; only the curvature-bearing
"event" frames carry new information. Keeping those keyframes and reconstructing
the rest is the symbolic-music pianoroll->event move applied to the temporal axis
of HL.

## Setup

- Encoder: `tools/build_temporal_hl.py : frame_to_hl` (unchanged); each (capture,
  hand, bone) gives a unit-direction path `[T,3]`.
- Tool: `tools/build_event_keyframe_compression.py`.
  - RDP keyframe selection: a Ramer-Douglas-Peucker recursion on each path that
    keeps a frame when the slerp of its bracketing keyframes deviates from it by
    more than a tolerance (a *curvature* criterion); the tolerance is binary-
    searched per track to hit a target keep-ratio.
  - slerp reconstruction: dropped frames are filled by spherical-linear
    interpolation between kept keyframes.
  - baseline: uniform frame subsampling at the same keep-ratio, slerp-decoded.
- Runs PER DATASET (HanCo and InterHand handled separately, via `--datasets
  name:root`) because their frame rates and motion statistics differ.
- Metric: mean reconstruction angular error (deg) vs keep-ratio; `floor°` is the
  keep-all (lossless) error.

## Run

```
python tools/build_event_keyframe_compression.py \
    --datasets hanco:<HanCo annotations> interhand:<InterHand annotations> \
    --split test --keep-ratios 0.45 0.57 \
    --out experiments/event_keyframe_compression_2026-06-25.json
```

## Result

Run the tool above to populate (raw numbers ->
`experiments/event_keyframe_compression_2026-06-25.json`). The table is
`dataset | keep | rdp+slerp° | uniform° | floor°`, one block per dataset.

## Findings (expected mechanism)

1. **Dropping ~half the frames barely loses accuracy.** Curvature-aware keyframes
   plus slerp should reconstruct the path almost as well as keeping everything
   (e.g. HanCo keep 45% -> ~4.93° vs a ~4.29° floor; InterHand keep 57% ->
   ~5.91°), because the dropped frames were interpolation-predictable.
2. **Beats uniform subsampling at every rate.** Spending the kept-frame budget on
   curvature (where the path bends) rather than evenly should win at every
   keep-ratio.
3. **Criterion must match the decoder.** A slerp decoder pairs with a *curvature*
   criterion (this tool); a hold/zero-order decoder would instead pair with a
   *drift* criterion. Mismatching them wastes keyframes — a design note for the
   event-level layer.

## Scope / non-claims

- Temporal frame-count compression only; per-frame quantization is untouched, and
  the two axes are intended to compose.
- HanCo support is via a `<dataset>_<split>_joint_3d.json` sibling using the same
  `frame_to_hl` encoder; if that file is absent the tool falls back to the
  InterHand joint file so it stays runnable, and the per-dataset rows reflect
  whatever joint files are actually present.
