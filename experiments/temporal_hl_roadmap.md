# Temporal HL Experiment Roadmap

This file is an experiment spec, not a paper draft. It defines what must be
true for a temporal Hand Labanotation upgrade to be strong enough for a top
conference review cycle.

## Local evidence already verified

- Workspace currently contains no runnable codebase, only [AGENTS.md](/opt/tiger/hand/AGENTS.md) and `AuthorKit27.zip`.
- Local data is available under `/opt/tiger/InterHand`.
- `InterHand2.6M_val_data.json` loads locally and contains `380125` images and
  `380125` annotations.
- Validation split metadata exposes explicit temporal and multi-view keys:
  `capture`, `seq_name`, `camera`, `frame_idx`.
- Validation split contains `11` action sequences, each with `139` cameras and
  `99` to `742` unique frames.
- Validation split frame deltas are consistently `6`, matching the `5fps`
  release and making fixed-length temporal windows straightforward.
- `InterHand2.6M_val_joint_3d.json` contains per-frame 3D joints with hand type
  labels `right`, `left`, and `interacting`.
- `InterHand2.6M_val_MANO_NeuralAnnot.json` provides MANO parameters for a large
  subset of frames, enabling geometry-aware or reconstruction-style evaluation.
- `tools/build_temporal_hl.py` now exports deterministic state and transition
  labels from InterHand 3D joints.
- `experiments/generated/temporal_hl_val_summary.json` verifies that the current
  deterministic construction uses all `26` state bins on `val`, with non-trivial
  temporal label counts for `stay`, `minor_shift`, and `major_shift`.
- Split structure is not identical: `train` has a dominant frame delta of `18`,
  while `val/test` have a dominant frame delta of `6`.

## Hard requirement: avoid the weak version of the project

Do not define the project as:

- "Use a temporal transformer to predict per-frame HL."
- "Add temporal smoothing to HL logits."
- "Use sequence loss on top of MHLFormer."

Those are model upgrades, not representation upgrades. Reviewers can dismiss
them as sequence modeling on top of an existing label space.

## The actual target

Upgrade HL from a frame-wise symbol inventory into a sequence-native symbolic
representation.

The new representation should encode at least one of:

- change events between adjacent frames
- duration / persistence of stable hand states
- motion phase or sub-action structure
- finger-group or cross-hand coordination over time

If none of these enter the representation itself, the work is too weak.

## Recommended problem formulation

### Primary formulation: state + transition temporal HL

Represent each time step using two coupled symbolic channels:

- `S_t`: the original HL state token set for 20 regional vectors per hand
- `T_t`: a temporal token set describing the transition from `t-1` to `t`

Recommended transition token families:

- `stay`: no meaningful change
- `minor_shift`: same coarse direction, local angular refinement
- `major_shift`: cross-bin motion
- `open/close`: finger flexion-extension trend
- `approach/separate`: cross-hand distance trend for interacting sequences
- `synchronous/asynchronous`: optional coordination token for grouped fingers

This is the cleanest upgrade because it stays connected to HL while making
temporal structure explicit.

### Secondary formulation: segmental temporal HL

Compress consecutive frames into variable-length symbolic segments:

- segment label = canonical hand state
- segment duration = discrete duration bucket
- boundary token = onset / offset / contact / reversal

This is stronger in abstraction, but harder to annotate robustly and harder to
evaluate fairly in the first iteration.

### Optional third formulation: learned motion code on top of HL

Keep HL states interpretable, but learn a discrete codebook over short temporal
motifs of regional-vector trajectories. The model predicts:

- explicit HL state tokens
- learned temporal motif tokens

This can be strong, but only if the learned codes are shown to be reusable
across tasks and not just latent noise.

## Data construction plan

### Stage A: deterministic temporal annotation from InterHand

Use the available 3D world coordinates to derive temporal labels without manual
relabeling.

Per frame:

- compute the original 20 regional vectors per hand
- map them to HL state tokens

Per transition:

- compute angular displacement of each regional vector between `t-1` and `t`
- bucket magnitude into `stay / minor / major`
- compute signed flexion trends per finger chain
- for interacting sequences, compute cross-hand proximity and contact trends

Current local implementation status:

- frame-wise `26`-way state tokens: implemented
- per-vector transition labels: implemented
- hand-level opening / closing trend: implemented
- cross-hand approach / separate trend: implemented
- per-frame persistence buckets for hand state / hand activity / interaction:
  implemented
- segment boundary tokens: not implemented yet

### Stage B: temporal grouping

Derive short windows and segment boundaries from transition density:

- low transition density -> stable hold
- sharp, synchronized change -> motion event
- repeated oscillation -> temporal motif candidate

### Stage C: difficulty tags for evaluation

Precompute evaluation subsets:

- one-hand vs interacting-hand sequences
- occluded vs no-occlusion sequences
- finger-only vs wrist-dominant motion
- short vs long sequences
- low-motion vs high-motion transitions

Without these subsets, claims about robustness or temporal benefit will be weak.

## Model plan

### Baseline family 1: frame-wise HL translation

Necessary weak baselines:

- image -> HL state only
- image window -> center-frame HL state only

Purpose:

- prove that temporal HL gains are not just from extra context at test time

### Baseline family 2: sequence prediction over old HL labels

Necessary stronger baselines:

- CNN/ViT + temporal transformer predicting frame-wise HL
- 3D pose -> temporal transformer -> frame-wise HL
- CTC or autoregressive decoder over frame-wise HL sequences

Purpose:

- isolate "better sequence model" from "better representation"

### Proposed family: temporal HL joint prediction

Recommended model heads:

- state head for `S_t`
- transition head for `T_t`
- optional segment boundary head

Recommended encoders:

- multi-view image encoder if image-based submission remains central
- 3D-pose encoder baseline to test representation quality independent of vision

## Must-have evaluation

### Core metrics

- frame-wise HL accuracy
- transition token accuracy / macro-F1
- event boundary F1 if segments are used
- edit distance at sequence level
- temporal consistency score:
  compare predicted state changes against true state changes, not just state IDs

### Stronger sequence metrics

- event IoU over derived motion phases
- run-length error for stable segments
- transition confusion by motion magnitude
- per-finger temporal accuracy

Current local status:

- persistence-aware sequence retrieval is implemented in
  [eval_sequence_symbolic_retrieval.py](/opt/tiger/hand/tools/eval_sequence_symbolic_retrieval.py)
- current audit shows:
  - persistence helps plain frame-signature retrieval
  - persistence hurts current RLE and bag-of-features classifier pipelines
  - event-level DTW restores a sensible temporal matcher after the weak RLE
    baseline
  - full segment-duration labels are now also exported and tested
  - segment duration is slightly better than elapsed persistence under
    event-DTW, but still does not beat plain temporal event matching
  - phase-level event grouping compresses sequences further while preserving
    event-DTW top-1, but it sharply reduces similarity margin
  - adding persistence or segment duration on top of the current phase taxonomy
    causes a clear accuracy drop
  - a refined phase taxonomy is now materially better than the coarse one
  - refined phase + segment duration recovers event-DTW top-1 while still using
    fewer events than exact-state matching
  - however, pushing refined phase into the current bag-of-features pretrain ->
    finetune classifier does not reproduce the same strength
  - a refined-phase sequence encoder is better matched than the MLP head and
    restores positive transfer under `186/96`
  - but the compressed refined-phase branch still trails the main temporal HL
    baselines by a large margin
  - a frame-level temporal HL sequence encoder also shows positive transfer, but
    still underperforms the current histogram-style symbolic mainline
  - but persistence still does not surpass temporal-without-persistence under
    the current event design
  - therefore persistence is promising as representation content, but not yet a
    drop-in mainline feature

Important protocol constraint:

- temporal windows must be defined in time-aware units, not raw frame count
  alone, because the effective frame stride differs across splits locally
  (`train` vs `val/test`)

### Downstream metrics

At least one downstream task must demonstrate that temporal HL is useful beyond
notation.

Recommended options:

- gesture / sequence classification from symbolic streams
- action retrieval using symbolic sequence distance
- hand motion reconstruction from temporal HL
- robot control smoothing / jitter reduction from temporal symbols

The downstream task should be chosen for clean evaluation, not for demo appeal.

## Required ablations

Minimum ablation grid:

- old HL vs temporal HL
- state-only vs transition-only vs state+transition
- single-view vs multi-view
- image encoder vs 3D pose encoder
- fixed temporal window lengths
- no duration token vs duration token
- no interacting-hand token vs interacting-hand token

If learned discrete motion codes are added:

- no codebook vs codebook
- different codebook sizes
- frozen vs jointly trained codebook

## Evidence needed for a top-tier bar

The work is not ready until the experiments can defend all of the following:

- the temporal label space is not arbitrary
- temporal HL improves sequence understanding beyond frame smoothing
- gains hold on interacting-hand and occluded settings
- the representation is more compact or more stable than dense pose streams
- the representation remains interpretable
- reconstruction or retrieval quality degrades gracefully under discretization

## High-risk failure modes

- transition labels become heuristic and noisy
- temporal gains come only from vision context, not the representation
- interacting-hand sequences dominate and hide one-hand weakness
- sequence splits leak by capture or motion pattern
- codebook tokens collapse into frequency artifacts

## Immediate implementation order

1. Build deterministic temporal annotation pipeline from InterHand 3D joints.
2. Export sequence windows with old HL and new temporal HL side by side.
3. Benchmark symbolic-only baselines first.
4. Add image-based translation only after the label space is stable.
5. Add one downstream task with exact quantitative evaluation.

## Exit criteria for the first serious experimental milestone

The project reaches the first credible milestone only if all items below are
true:

- temporal labels are generated reproducibly for train/val/test
- at least one sequence-level metric exists and is meaningful
- at least one strong sequence baseline is implemented
- temporal HL beats old HL on a sequence-level task
- error analysis identifies where temporal HL helps and where it fails
