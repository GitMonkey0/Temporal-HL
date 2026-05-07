# Reviewer Risk Analysis

## R1. "The gain is decoder-specific, so the representation may not be generally useful."

Risk level: high

Why a reviewer may say this:
- Temporal-HL improves reconstruction under the Transformer decoder.
- Static-HL is slightly better under GRU and TCN.
- A reviewer may conclude the result is an implementation artifact rather than a representation benefit.

Current evidence:
- We explicitly report the negative GRU/TCN results.
- We already have a clear ablation showing `keyframe` and `motion + keyframe` matter under the Transformer decoder.

What is still missing:
- A stronger Transformer-family check to show the gain is not tied to a single lightweight decoder instance.

Action:
- Add a stronger Transformer reconstruction variant with larger width/depth.

Update:
- Done.
- Result: larger Transformer does **not** preserve the Temporal-HL gain.
- New implication: the paper should not claim broad decoder-family robustness.
- New highest-value fix is seed robustness or claim reduction.

## R2. "Keyframes seem to do all the work. Why do you need motion tokens?"

Risk level: high

Why a reviewer may say this:
- `static + keyframe` already improves over static.
- `static + motion` does not.
- The full gain may appear to come mostly from keyframe anchors.

Current evidence:
- The full temporal notation is still best.
- Motion-only underperforms, which means the signal is not trivially useful.

What is still missing:
- A more explicit explanation that motion tokens are complementary and only help when anchored by event structure.
- Preferably a qualitative example where motion tokens distinguish same-pose but different-transition segments.

Action:
- Add one qualitative sequence comparison highlighting identical static states but different motion evolution.

Update:
- Still relevant.
- Since multi-seed robustness is weak, this qualitative point is now supporting material rather than a substitute for robustness.

## R3. "The paper claims temporal notation, but the notation translation task becomes worse on static accuracy."

Risk level: medium-high

Why a reviewer may say this:
- Static-HL gets better static-token accuracy than Temporal-HL under joint training.
- A reviewer may interpret this as a regression.

Current evidence:
- Temporal-HL predicts two additional temporal targets with reasonable quality.
- The main paper claim is not token classification, but motion recoverability.

What is still missing:
- Stronger framing: this is a richer notation, not a drop-in replacement for maximizing static token accuracy.
- Possibly a balanced-capacity or multi-task weighting note in the experiment section.

Action:
- Rewrite claim language to emphasize recoverability rather than raw static classification.

Update:
- This is now mandatory, not optional.

## R4. "The evaluation is too synthetic. Why should reconstruction in normalized joint space matter?"

Risk level: high

Why a reviewer may say this:
- Reconstruction is currently measured in normalized coordinate L1.
- Reviewers may want a more application-facing justification.

Current evidence:
- Temporal-HL is motivated by documentation, editing, and robot control.
- We have a sequence-aware retrieval proxy.

What is still missing:
- A stronger qualitative reconstruction figure and an application-facing explanation.
- If possible, one additional metric tied to trajectory shape or segment ordering.

Action:
- Add qualitative reconstruction comparisons and emphasize recoverability as a documentation criterion.

Update:
- Still valid.
- Should be coupled with a statement that current reconstruction is in normalized joint space.

## R5. "Auxiliary downstream tasks are weak."

Risk level: medium

Why a reviewer may say this:
- The action classification benchmark saturates.
- The retrieval benchmark is a proxy rather than a standard task.

Current evidence:
- We already decided not to oversell classification.

What is still missing:
- Clear distinction between main and auxiliary experiments.

Action:
- Keep action classification as a sanity check only.
- Keep retrieval as a supporting analysis only.

## R6. "The keyframe detector is heuristic."

Risk level: medium

Why a reviewer may say this:
- Local minima over motion energy is simple and hand-designed.

Current evidence:
- The heuristic already yields measurable gain.

What is still missing:
- Either a learned variant, or a clear statement that keyframe detection is not the paper's main novelty.

Action:
- Treat learned keyframe detection as optional future work unless time permits.

Update:
- Keep as future work for now. It is not the main bottleneck compared with robustness.

## R7. "The notation itself may be overfit to this dataset."

Risk level: medium

Why a reviewer may say this:
- Labels are generated from one dataset split with fixed FPS and clip length.

Current evidence:
- The representation is geometry-defined, not class-defined.

What is still missing:
- Stronger text in the method section explaining that the notation is rule-defined and dataset-agnostic.

Action:
- Strengthen the method description and avoid overclaiming generalization beyond the current benchmark.

## R8. "The reorganized split may be confusing or potentially leaky."

Risk level: high

Why a reviewer may say this:
- The reorganized dataset uses new train/test clip splits, but the original metadata still contains `source_split` from InterHand.
- Some clips in the new train split come from original `source_split='test'`.
- Capture IDs, sequence names, and cameras overlap across the reorganized train/test files.

Current evidence:
- The current paper uses the reorganized split consistently.
- The task is representation benchmarking on the reorganized dataset, not reproduction of the original InterHand benchmark.

What is still missing:
- Explicit explanation that `source_split` is inherited provenance metadata rather than the split used in this paper.
- A sentence clarifying that the current benchmark is a newly organized clip-level split.
- Avoid using the easy seq-name classification result as strong evidence, because sequence-name overlap makes that task weak.

Action:
- Clarify the split protocol in the dataset section.
- Demote the seq-name classifier to a sanity check only.

## Priority

Highest-value next actions:
1. Seed-robustness repair for the main Transformer result.
2. Better qualitative comparison for keyframe + motion complementarity.
3. Rewrite claims to center on motion recoverability and decoder dependence.
4. Clarify the reorganized split protocol to pre-empt leakage concerns.
