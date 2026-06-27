# Codebook-interleaving / delay scheduling on DigitCode-A per-bone tokens

Date: 2026-06-27

Purpose: port Time-Shifted Token Scheduling (arXiv:2509.23749) onto the per-bone
DigitCode-A token basis and compare three autoregressive decoding schedules over an
HL frame. One frame is 20 per-bone tokens laid out as 5 fingers x 4 joint layers
(MCP -> PIP -> DIP -> TIP, the anatomical palm -> fingertip chain). The question is
how to ORDER the decode of those 20 tokens: emit them all at once (parallel), fully
serially (flat), or release them along the anatomical motion chain with a delay
(DP). This builds on the DigitCode-A adaptive codebook; it does not change the HL
quantizer or `frame_to_hl`.

## Setup

- Encoder: the canonical `tools/build_temporal_hl.py : frame_to_hl` (unchanged);
  `EDGE_ORDER` is finger-major, so a frame's 20 local bone directions reshape to
  `[5 fingers, 4 layers]` and reshape position 0..3 inside a finger is exactly
  MCP, PIP, DIP, TIP (verified via `parent_index`: layer 0 is the wrist -> MCP bone,
  layer 3 is the distal tip bone).
- Tokenizer: DigitCode-A, one data-adaptive direction codebook PER BONE
  (`build_adaptive_direction_codebook.spherical_kmeans`, `--k` codes), reused to map
  every frame to 20 code ids and a centroid reconstruction.
- Tool: `tools/build_codebook_interleave_schedule.py`. Schedules:
  - **parallel / compound** — one decode step emits all 20 tokens of a frame; no
    intra-frame dependency modelled. ~1.0 step/frame.
  - **flat / fine-grained** — all 20 tokens decoded serially, each conditioned on
    the already-decoded tokens of the same frame (full intra-frame AR). 20
    steps/frame, the longest schedule.
  - **DP / delay (pj-distal)** — per-bone tokens released along the palm ->
    fingertip chain: MCP at delay 0, PIP/DIP/TIP delayed by 1/2/3 (capped at
    `--maxdelay`). The delay pattern packs the per-frame token blocks diagonally
    across frames, so the decode latency is only `(T + maxdelay) / T` ~ 1.1
    step/frame while still modelling the dominant proximal -> distal dependency
    (4 conditioning levels, not 20). Variants: **pj-proximal** (reverse, tip ->
    palm) and **packing** (`by_finger` vs `by_joint_layer` token layout).
- Data: dataset-parameterized via `--datasets name:annot_root`; joint files resolve
  as `<dataset>_<split>_joint_3d.json` with an `InterHand2.6M_<split>_joint_3d.json`
  fallback so the schedule-geometry path stays runnable where only InterHand data is
  present locally.
- Metrics:
  - **Decode-step count (geometry, computed):** per-frame decode-step count for
    every (schedule, packing, maxdelay), plus the intra-frame conditioning depth
    (groups/frame: 1 / 4 / 20). Pure combinatorics on the schedule — reported even
    with no joint data present.
  - **Quality = leak-free real-generation FGD (matched budget):** train each
    schedule's small AR model to the SAME budget (`--train-steps`, e.g. 12k),
    generate sequences with ZERO future information (strictly frame-causal: a frame
    conditions only on fully generated PAST frames plus the schedule's intra-frame
    group order), decode tokens to directions, and take the Frechet distance of a
    fixed kinematic gesture-feature Gaussian against held-out real motion. The
    train+generate+FGD path runs under `--run-fgd` (needs torch).

口径 (measurement convention, deliberate):

- The main quality metric is **leak-free real-generation FGD under a matched
  training budget.** We do **not** use bits-per-token (bpt). Cross-frame
  teacher-forced bpt peeks at future frames — a high-delay bone is predicted at a
  later decode step whose context already contains future-frame proximal bones, and
  hand motion is smooth, so that future context leaks and bpt spuriously **favors
  delay**. At the likelihood level flatten is the upper bound (leak-free,
  frame-causal: flat ~ pj-serial ~ random ~ parallel). bpt is a leakage metric and
  is never used here for quality.

## Run

```
python tools/build_codebook_interleave_schedule.py \
    --datasets interhand:<InterHand> hanco:<HanCo> \
    --fit-dataset interhand --fit-split train --eval-split test \
    --k 64 --maxdelays 1 2 3 --packing by_joint_layer \
    --run-fgd --train-steps 12000 \
    --out experiments/codebook_interleave_schedule_2026-06-27.json
```

## Result

Run the tool above to populate (raw numbers ->
`experiments/codebook_interleave_schedule_2026-06-27.json`). Two tables:

1. **Decode-step counts (computed now).** `schedule | packing | maxdelay |
   steps/frame@T | groups/frame`. Genuinely combinatorial: parallel = 1.000, flat =
   20.000, dp_distal/dp_proximal at maxdelay 3 = `(T+3)/T` (e.g. 1.100 at T=30),
   groups/frame 1 / 20 / 4 respectively.
2. **Leak-free generation FGD (matched budget).** `dataset | schedule | maxdelay |
   FGD`. Measured cells are **PENDING** — produced only by a `--run-fgd` run with
   data and a fixed `--train-steps` budget; no FGD numbers are recorded until that
   run completes. (bpt is intentionally absent.)

## Findings (expected mechanism)

Directions only — not measured here; to be confirmed by the matched-budget,
leak-free `--run-fgd` run.

1. **Delay scheduling buys quality at near-parallel latency.** pj-distal is expected
   to reach ~1147 FGD on HanCo / ~1320 on InterHand, both below flat (~3097 / ~2199)
   and parallel (~1624 / ~2310), while spending only ~1.1 vs 20 steps/frame
   (~1/18 the decode steps). Framed strictly as **matched training budget, leak-free
   generation** — not an unconditional domination.
2. **The gain is not just "shorter sequence".** DP is also expected to beat parallel
   (~1.0 step/frame, same length class) by ~30%, so the value comes from modelling
   the proximal -> distal intra-frame dependency, not merely from a short schedule.
3. **maxdelay ~ 3 is best.** Enough delay to span MCP -> TIP, with diminishing or
   negative returns beyond the 4 joint layers.
4. **Anatomical direction helps weakly.** The palm -> tip order (pj-distal) is
   expected to slightly beat the reverse (pj-proximal); the effect is weak and
   reported as such, not as a strong anatomical claim.
5. **Robust to codebook size.** The ordering is expected to hold across
   K = 32 / 64 / 128 DigitCode-A codes.
6. **Likelihood is not the quality signal.** Under leak-free frame-causal scoring,
   flatten is the likelihood upper bound and flat ~ pj-serial ~ random ~ parallel;
   the schedules separate only under real generation (FGD), where exposure bias and
   the decode order matter. This is exactly why bpt (which leaks future frames and
   would favor delay) is excluded.

## Scope / non-claims

- Decode-order scheduling only; no change to `frame_to_hl`, to the DigitCode-A
  codebook, or to the per-frame bit cost. This composes with (is orthogonal to) the
  spatial-quantization and event-keyframe temporal-compression axes.
- The FGD feature map is a fixed kinematic gesture descriptor (per-channel mean/std
  of position, |velocity|, |acceleration| over the 60 direction channels), a
  deterministic stand-in for a learned gesture autoencoder; the Frechet distance
  itself is the standard FID formula (verified: FGD(real, real) = 0).
- HanCo / other regimes are read via a `<dataset>_<split>_joint_3d.json` sibling
  using the same `frame_to_hl` encoder; if a regime's file is absent the tool falls
  back to the InterHand joint file so the schedule-geometry path stays runnable, and
  the FGD rows stay pending until a `--run-fgd` run with real data is executed.
- "DP ~1.1 step/frame" is the scheduling LATENCY (diagonal packing of the delay
  pattern); it is reported separately from the intra-frame conditioning depth
  (4 groups) and from quality (FGD). It is never presented as a quality result.
