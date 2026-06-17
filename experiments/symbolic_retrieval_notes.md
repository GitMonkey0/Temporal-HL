# Symbolic Retrieval Notes

This is an experiment log, not paper text.

## Task definition

- Gallery split: `val`
- Query split: `test`
- Only evaluate sequence labels that overlap between `val` and `test`
- Resulting query set size: `20`
- Gallery class count: `11`

Rationale:

- `train` does not share sequence names with `val/test`
- `val` and `test` share the ROM action family cleanly
- This makes `val -> test` a compact cross-subject symbolic retrieval benchmark

## Inputs

- Gallery labels: [temporal_hl_val.json](/opt/tiger/hand/experiments/generated/temporal_hl_val.json)
- Query labels: [temporal_hl_test.json](/opt/tiger/hand/experiments/generated/temporal_hl_test.json)
- Evaluation script: [eval_symbolic_retrieval.py](/opt/tiger/hand/tools/eval_symbolic_retrieval.py)

## Verified command

```bash
python tools/eval_symbolic_retrieval.py \
  --gallery experiments/generated/temporal_hl_val.json \
  --query experiments/generated/temporal_hl_test.json \
  --output experiments/generated/symbolic_retrieval_val_to_test_weighted.json
```

## Current result

- `state_only`: `0.85` top-1
- `temporal_hl` with weights `state=1.0`, `transition=0.5`,
  `hand_motion=0.5`, `interaction=0.2`: `0.90` top-1
- `temporal_hl` with channel-wise `late` fusion needs an explicit `tempo`
  channel to match `0.90`; without it, the same channel weights fall back to
  `0.85`

## Observed pattern

- Naive equal-weight temporal concatenation underperforms state-only
- Calibrated temporal weighting improves over state-only
- Temporal cues help recover:
  - `ROM09_Interaction_Fingers_Touching`
  - one instance of `ROM05_RT_Wrist_ROM`
- Remaining hard cases are still confusable with `ROM03_RT_No_Occlusion`

## Small ablation

Measured on the same `val -> test` protocol:

- `state_only`: `0.85`
- `state + transition`: `0.90`
- `state + hand_motion`: `0.90`
- `state + interaction`: `0.90`
- `state + transition + hand_motion`: `0.85`
- `state + transition + interaction`: `0.90`
- `state + all`: `0.90`

Fusion ablation:

- `feature` fusion, no tempo: `0.90`
- `late` fusion, no tempo: `0.85`
- `late` fusion, with tempo weight `0.2`: `0.90`

Late-fusion tempo sweep:

- `tempo_weight=0.0`: `0.85`
- `tempo_weight=0.05`: `0.85`
- `tempo_weight=0.1`: `0.85`
- `tempo_weight=0.2`: `0.90`
- `tempo_weight=0.3`: `0.90`
- `tempo_weight=0.5`: `0.90`
- `tempo_weight=1.0`: `0.85`

## Immediate implication

The temporal signal is useful, but not all temporal channels should be fused
naively. The next baseline should treat temporal subspaces as partially
redundant and test:

- separate projection heads
- channel gating
- sequence normalization by effective time stride
- tempo channels should be softly weighted, not treated as dominant features

## Broad family-correction audit update

The earlier family-correction benchmark was too restrictive because the old
correction scripts only triggered when the symbolic top-1 prediction was
already inside the occlusion / wrist / finger-occlusion family.

New audit script:

- [audit_family_disagreement_broad.py](/opt/tiger/hand/tools/audit_family_disagreement_broad.py)

New generated reports:

- [family_disagreement_audit_broad_mainline_widegap.json](/opt/tiger/hand/experiments/generated/family_disagreement_audit_broad_mainline_widegap.json)
- [family_disagreement_audit_broad_mainline_fraction05_widegap.json](/opt/tiger/hand/experiments/generated/family_disagreement_audit_broad_mainline_fraction05_widegap.json)

Important implementation detail:

- the first version of the broad audit still inherited a hidden target-family
  filter through `build_query_events`
- this was fixed by rebuilding query events for all query sequences while
  keeping the retrieval gallery family-restricted

### What changed scientifically

The benchmark is no longer disagreement-degenerate once the trigger is defined
by model-side uncertainty / family proximity rather than by
`mainline top1 in family`.

For the strongest full-data symbolic mainline:

- `uncertain_margin <= 2.0` yields `21` candidate rows with
  `7` positive and `2` negative disagreements
- `top3_or_margin <= 2.0` yields `32` candidate rows with
  `7` positive and `2` negative disagreements
- `family_gap <= 6.0` yields `35` candidate rows with
  `7` positive and `5` negative disagreements

For the `fraction=0.5` mainline:

- `uncertain_margin <= 1.0` yields `17` candidate rows with
  `8` positive and `2` negative disagreements
- `top3_or_margin <= 1.0` yields `32` candidate rows with
  `13` positive and `2` negative disagreements
- `family_gap <= 4.0` yields `34` candidate rows with
  `13` positive and `4` negative disagreements

### Current benchmark recommendation

Use one moderate trigger family rather than the extreme wide-gap setting.

Recommended starting points:

- full-data benchmark:
  `uncertain_margin <= 2.0`
- low-data benchmark:
  `uncertain_margin <= 1.0`

Reason:

- they are no longer degenerate
- retrieval is still materially better than the original symbolic prediction
- the candidate set is not inflated to nearly the whole benchmark

What not to use as the main benchmark:

- very wide `family_gap` triggers such as `>= 8.0`
- very loose `margin <= 4.0`

These do create negatives, but they also pull in almost the entire evaluation
set and dilute the correction problem into a near-global reranking task.

## Correction results on the non-degenerate benchmark

Updated evaluation scripts:

- [eval_symbolic_retrieval_fusion.py](/opt/tiger/hand/tools/eval_symbolic_retrieval_fusion.py)
- [eval_symbolic_learned_gate.py](/opt/tiger/hand/tools/eval_symbolic_learned_gate.py)

Both scripts now support explicit candidate-benchmark parameters:

- `candidate_trigger`
- `candidate_scope`
- `candidate_margin_max`
- `candidate_family_gap_max`

This removes the old hidden assumption that only `top1-in-family` rows are
eligible for correction.

### Full-data benchmark

Benchmark definition:

- `candidate_trigger=uncertain_margin`
- `candidate_scope=all`
- `candidate_margin_max=2.0`

Artifacts:

- [symbolic_retrieval_fusion_mainline_uncertain2.json](/opt/tiger/hand/experiments/generated/symbolic_retrieval_fusion_mainline_uncertain2.json)
- [symbolic_learned_gate_mainline_uncertain2.json](/opt/tiger/hand/experiments/generated/symbolic_learned_gate_mainline_uncertain2.json)

Observed candidate baseline:

- candidate rows: `21`
- original candidate accuracy: `0.6190`

Best zero-harm result:

- candidate accuracy: `0.6190 -> 0.9524`
- improved cases: `7`
- harmed cases: `0`

For retrieval fusion, one representative setting is:

- `mainline_max_margin <= 2.0`
- `retrieval_min_margin >= 0.05`

### Low-data benchmark

Benchmark definition:

- `candidate_trigger=uncertain_margin`
- `candidate_scope=all`
- `candidate_margin_max=1.0`

Artifacts:

- [symbolic_retrieval_fusion_fraction05_uncertain1.json](/opt/tiger/hand/experiments/generated/symbolic_retrieval_fusion_fraction05_uncertain1.json)
- [symbolic_learned_gate_fraction05_uncertain1.json](/opt/tiger/hand/experiments/generated/symbolic_learned_gate_fraction05_uncertain1.json)

Observed candidate baseline:

- candidate rows: `17`
- original candidate accuracy: `0.4118`

Best zero-harm result:

- candidate accuracy: `0.4118 -> 0.8824`
- improved cases: `8`
- harmed cases: `0`

### Current interpretation

- the correction line is now scientifically meaningful because the candidate
  cohort contains both positive and negative disagreements before gating
- under the recommended uncertainty-based benchmark, both fusion gating and the
  learned gate still recover strong zero-harm gains
- the learned gate appears almost threshold-insensitive on the current cohort,
  which suggests the benchmark is now non-degenerate but still relatively easy
  once the candidate set is defined well

Immediate next question:

- can we build a harder candidate benchmark where retrieval fusion and learned
  gating separate more clearly, rather than both collapsing to the same
  zero-harm frontier?

## Why fusion and learned gating still collapse

I ran a harder comparison on the broader candidate benchmarks where negative
disagreements are already present:

- full-data hard benchmark:
  `candidate_trigger=family_gap`, `candidate_scope=all`,
  `candidate_family_gap_max=6.0`
- low-data hard benchmark:
  `candidate_trigger=family_gap`, `candidate_scope=all`,
  `candidate_family_gap_max=4.0`

Artifacts:

- [symbolic_retrieval_fusion_mainline_gap6.json](/opt/tiger/hand/experiments/generated/symbolic_retrieval_fusion_mainline_gap6.json)
- [symbolic_learned_gate_mainline_gap6.json](/opt/tiger/hand/experiments/generated/symbolic_learned_gate_mainline_gap6.json)
- [symbolic_retrieval_fusion_fraction05_gap4.json](/opt/tiger/hand/experiments/generated/symbolic_retrieval_fusion_fraction05_gap4.json)
- [symbolic_learned_gate_fraction05_gap4.json](/opt/tiger/hand/experiments/generated/symbolic_learned_gate_fraction05_gap4.json)

Observed result:

- full-data hard benchmark:
  candidate accuracy `0.7714 -> 0.9714`, `7` improved, `0` harmed
- low-data hard benchmark:
  candidate accuracy `0.5882 -> 0.9706`, `13` improved, `0` harmed

The best learned-gate frontier is still identical to the fusion frontier.
Case-by-case comparison confirms the best applied sets are the same under the
recommended uncertainty benchmarks.

### Feature ablation diagnosis

To test whether the learned gate is using genuinely richer structure or just
rediscovering the same margin rule, I ran feature-preset ablations on the hard
benchmarks.

Artifacts:

- [symbolic_learned_gate_mainline_gap6_changedonly.json](/opt/tiger/hand/experiments/generated/symbolic_learned_gate_mainline_gap6_changedonly.json)
- [symbolic_learned_gate_mainline_gap6_margins.json](/opt/tiger/hand/experiments/generated/symbolic_learned_gate_mainline_gap6_margins.json)
- [symbolic_learned_gate_fraction05_gap4_changedonly.json](/opt/tiger/hand/experiments/generated/symbolic_learned_gate_fraction05_gap4_changedonly.json)
- [symbolic_learned_gate_fraction05_gap4_margins.json](/opt/tiger/hand/experiments/generated/symbolic_learned_gate_fraction05_gap4_margins.json)

Results:

- `changed_only` is not enough and introduces harm:
  - full-data hard benchmark:
    `0.7714 -> 0.8571`, with `4` harmed
  - low-data hard benchmark:
    `0.5882 -> 0.8824`, with `3` harmed
- `margins_only` fully recovers the best zero-harm frontier:
  - full-data hard benchmark:
    `0.7714 -> 0.9714`, `0` harmed
  - low-data hard benchmark:
    `0.5882 -> 0.9706`, `0` harmed

### Current interpretation

- the current non-degenerate correction benchmark is real, but the best
  decisions are still almost completely determined by mainline/retrieval margin
  geometry
- the learned gate is not yet exploiting richer structure than the simple
  fusion rule
- therefore the next useful experiment is not "another gate architecture"
  but adding candidate-side signals that are not reducible to the current
  margins, e.g. symbolic family structure, sequence-shape mismatch, or
  retrieval-consensus features across multiple neighbors

## Added non-margin correction features

I extended the learned-gate pipeline with retrieval-consensus features from the
top-`k` family ranking (`k=3,5,8`), including:

- label vote ratios and vote margins
- top-`k` score mean / std / range
- support ratios for the retrieval top-1 label and the original symbolic label

These are now available through feature presets in
[eval_symbolic_learned_gate.py](/opt/tiger/hand/tools/eval_symbolic_learned_gate.py):

- `topk_only`
- `non_margin`
- `structure_only`

### Hard-benchmark diagnostic results

Artifacts:

- [symbolic_learned_gate_mainline_gap6_topkonly.json](/opt/tiger/hand/experiments/generated/symbolic_learned_gate_mainline_gap6_topkonly.json)
- [symbolic_learned_gate_mainline_gap6_nonmargin.json](/opt/tiger/hand/experiments/generated/symbolic_learned_gate_mainline_gap6_nonmargin.json)
- [symbolic_learned_gate_mainline_gap6_structureonly.json](/opt/tiger/hand/experiments/generated/symbolic_learned_gate_mainline_gap6_structureonly.json)
- [symbolic_learned_gate_fraction05_gap4_topkonly.json](/opt/tiger/hand/experiments/generated/symbolic_learned_gate_fraction05_gap4_topkonly.json)
- [symbolic_learned_gate_fraction05_gap4_nonmargin.json](/opt/tiger/hand/experiments/generated/symbolic_learned_gate_fraction05_gap4_nonmargin.json)
- [symbolic_learned_gate_fraction05_gap4_structureonly.json](/opt/tiger/hand/experiments/generated/symbolic_learned_gate_fraction05_gap4_structureonly.json)

Results:

- `topk_only` has real signal but does not reach the best frontier
  - full-data hard benchmark:
    `0.7714 -> 0.9143`
  - low-data hard benchmark:
    `0.5882 -> 0.9412`
- `non_margin` fully matches the best frontier
  - full-data hard benchmark:
    `0.7714 -> 0.9714`
  - low-data hard benchmark:
    `0.5882 -> 0.9706`
- `structure_only` also fully matches the best frontier
  - full-data hard benchmark:
    `0.7714 -> 0.9714`
  - low-data hard benchmark:
    `0.5882 -> 0.9706`

Interpretation:

- top-`k` consensus is useful, but it is not the deciding factor on the current
  hard benchmark
- the benchmark is still largely explained by symbolic family structure and
  label-pair identity

## Pair-rule baseline

To test whether the current benchmark has collapsed into pair memorization, I
added a strict leave-one-seed-out pair-rule baseline:

- [eval_symbolic_pair_rule.py](/opt/tiger/hand/tools/eval_symbolic_pair_rule.py)

Artifacts:

- [symbolic_pair_rule_mainline_gap6.json](/opt/tiger/hand/experiments/generated/symbolic_pair_rule_mainline_gap6.json)
- [symbolic_pair_rule_fraction05_gap4.json](/opt/tiger/hand/experiments/generated/symbolic_pair_rule_fraction05_gap4.json)

Results:

- full-data hard benchmark:
  `0.7714 -> 0.9429`
- low-data hard benchmark:
  `0.5882 -> 0.9706`

This means:

- the low-data hard benchmark is fully explainable by pair memorization
- the full-data hard benchmark is almost fully explainable by pair memorization
  and misses only one positive case

The missing full-data case is:

- `ROM03_RT_No_Occlusion -> ROM05_RT_Wrist_ROM` on seed `2`

This pair appears only once in the current cohort, so the LOSO pair-rule cannot
learn it from the remaining seeds.

### Updated correction-line conclusion

- the current hard candidate benchmarks are stronger than the old degenerate
  benchmark, but they are still too close to deterministic label-structure
  repair
- if we want a correction result that looks genuinely stronger than a compact
  symbolic rule system, the next benchmark must force decisions that cannot be
  recovered from label-pair identity alone

## Pair-hard unseen subset

To make the diagnosis explicit, I added a subset summarizer:

- [summarize_pair_hard_subset.py](/opt/tiger/hand/tools/summarize_pair_hard_subset.py)

This derives LOSO pair status from the pair-rule benchmark and then evaluates
other methods only on the `unseen` subset, optionally restricted to changed
rows.

Artifacts:

- [pair_hard_mainline_gap6_unseen_changed.json](/opt/tiger/hand/experiments/generated/pair_hard_mainline_gap6_unseen_changed.json)
- [pair_hard_fraction05_gap4_unseen_changed.json](/opt/tiger/hand/experiments/generated/pair_hard_fraction05_gap4_unseen_changed.json)

### Full-data hard benchmark

Subset definition:

- benchmark: `family_gap <= 6.0`
- keep only changed rows
- keep only rows whose `(original_label, retrieval_label)` pair is `unseen`
  under LOSO

Observed subset size:

- only `2` rows remain
- among them, only `1` row is a true positive fix case

Results:

- pair-rule:
  `0.0 -> 0.0`
- structure-only learned gate:
  `0.0 -> 0.5`
- non-margin learned gate:
  `0.0 -> 0.5`

Interpretation:

- after removing pair-memorization-easy cases, the current full-data benchmark
  retains exactly one meaningful positive correction example
- the learned gate can recover it, but the sample size is too small to support
  a strong headline claim

### Low-data hard benchmark

Subset definition:

- benchmark: `family_gap <= 4.0`
- keep only changed rows
- keep only LOSO-`unseen` pairs

Observed subset size:

- only `2` rows remain
- there is no unseen positive fix case
- there is `1` harmful retrieval case and `1` neutral case

Results:

- pair-rule:
  `0.5 -> 0.5`
- structure-only learned gate:
  `0.5 -> 0.5`
- non-margin learned gate:
  `0.5 -> 0.5`

Interpretation:

- the low-data benchmark has no pair-hard positive signal at all
- therefore it is not suitable as a primary correction benchmark for claiming
  extra model intelligence beyond symbolic pair structure

## Practical conclusion for the correction line

The current evidence now supports a sharper boundary:

- the broad uncertainty / family-gap benchmarks are useful to show correction
  value exists
- but they are still too easy to carry a strong “learning to correct” claim
- the pair-hard remainder is presently too small to serve as the central
  experiment

So the next correction experiment should focus on **benchmark construction**
rather than another gating model:

- enlarge the pair-hard candidate pool
- include candidates with conflicting neighbor evidence
- include candidates where multiple retrieval labels from the same symbolic
  family compete closely
- add sequence-shape residual features only after such a benchmark exists

## Candidate-pool constructor and gap-band diagnosis

I added two utilities to turn the benchmark-construction step into repeatable
artifacts:

- [build_correction_benchmark_candidates.py](/opt/tiger/hand/tools/build_correction_benchmark_candidates.py)
- [evaluate_method_on_candidate_subset.py](/opt/tiger/hand/tools/evaluate_method_on_candidate_subset.py)

Generated candidate pools:

- [correction_candidates_mainline_gap6.json](/opt/tiger/hand/experiments/generated/correction_candidates_mainline_gap6.json)
- [correction_candidates_fraction05_gap4.json](/opt/tiger/hand/experiments/generated/correction_candidates_fraction05_gap4.json)

These candidate pools annotate each row with:

- LOSO pair status: `seen_fix`, `seen_harm`, `unseen`
- retrieval top-2 score gap
- retrieval top-1 vote margin
- number of distinct labels in the top-`k` neighbor set

### Important finding: the current conflict metrics are still too weak

Under the current retrieval ranking:

- every changed row already satisfies the loose conflict condition
- every changed row also has `ranking_top1_vote_margin = 0`
- every changed row has `ranking_unique_labels = 8`

So these generic “neighbor conflict” checks do not yet isolate a meaningfully
harder subset. The practically useful scalar is still the retrieval top-2 score
gap.

### Gap-band subsets

The constructor now exports explicit gap-band subsets such as:

- `gap_le_0.01`
- `gap_le_0.02`
- `gap_le_0.04`
- `gap_le_0.06`

Method evaluation artifacts:

- [method_subset_mainline_gap6_gap004.json](/opt/tiger/hand/experiments/generated/method_subset_mainline_gap6_gap004.json)
- [method_subset_mainline_gap6_gap006.json](/opt/tiger/hand/experiments/generated/method_subset_mainline_gap6_gap006.json)
- [method_subset_fraction05_gap4_gap004.json](/opt/tiger/hand/experiments/generated/method_subset_fraction05_gap4_gap004.json)
- [method_subset_fraction05_gap4_gap006.json](/opt/tiger/hand/experiments/generated/method_subset_fraction05_gap4_gap006.json)

### What the gap bands reveal

For the full-data hard benchmark:

- `gap <= 0.04` gives `5` rows and is effectively a negative / abstention band
  - all methods stay at `0.8 -> 0.8`
- `gap <= 0.06` gives `11` rows and becomes a mixed repair band
  - all methods jump together to `0.3636 -> 0.9091`

For the low-data hard benchmark:

- `gap <= 0.04` gives `4` rows and is again an abstention band
  - all methods stay at `0.75 -> 0.75`
- `gap <= 0.06` gives `14` rows and all methods again jump together
  - `0.2143 -> 0.9286`

### Updated diagnosis

The correction line is currently dominated by a very simple regime split:

- very small retrieval score gap:
  everyone abstains
- slightly larger but still small gap:
  everyone applies the same repair

This is stronger evidence that the present benchmark is still too close to a
scalar gap-threshold problem, even after removing the most obvious pair-rule
degeneracy.

### Immediate consequence

The next benchmark-construction target should not be “more rows with small gap”.
It should be rows where:

- multiple plausible repair labels compete within the same family
- pair status is `unseen`
- and the decision cannot be explained by a single global score-gap threshold

## Subset-frontier search

To move from manual inspection to a systematic check, I added:

- [search_correction_subset_frontier.py](/opt/tiger/hand/tools/search_correction_subset_frontier.py)

Artifacts:

- [correction_subset_frontier_mainline_gap6.json](/opt/tiger/hand/experiments/generated/correction_subset_frontier_mainline_gap6.json)
- [correction_subset_frontier_fraction05_gap4.json](/opt/tiger/hand/experiments/generated/correction_subset_frontier_fraction05_gap4.json)

This search scans simple subsets over:

- pair-status mode:
  `all`, `unseen`, `seen_fix`, `seen_harm`, `not_seen_fix`, `not_seen_harm`
- target mode:
  `all`, `positive`, `negative`, `ambiguous`, `nonnegative`
- retrieval top-2 score gap threshold

### What the frontier says

For the full-data hard benchmark:

- the largest clean subset is still dominated by seen-fix pairs
  - e.g. `not_seen_harm + nonnegative + gap<=0.08` gives `8` rows with
    `7` positives, but only `2` unseen rows
- the only fully pair-hard region is tiny
  - `unseen + all + gap<=0.08` gives only `2` rows, with just `1` positive

For the low-data hard benchmark:

- every sizeable positive subset is almost entirely `seen_fix`
  - e.g. `all + nonnegative + gap<=0.08` gives `14` rows with `13` positives,
    but only `1` unseen row
- the unseen region never carries a meaningful positive pool
  - the two unseen rows are not useful positive correction cases

### Updated benchmark verdict

Under the current InterHand ROM correction protocol:

- there is no discovered subset that is simultaneously
  - nontrivial in size
  - rich in positive fixes
  - low in harmful cases
  - and meaningfully pair-hard

This is the strongest current evidence that the correction line should be
treated as a secondary result unless a better candidate-generation protocol is
built.
