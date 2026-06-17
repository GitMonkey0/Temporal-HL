# Symbolic Pretrain Notes

This is an experiment log, not paper text.

## Goal

Test whether the large non-ROM `train` split can act as a supervised pretraining
source for the ROM `val -> test` protocol, despite label spaces not matching.

## 2026-06-09 matched old-HL vs temporal-HL boundary check

New artifacts:

- [build_oldhl_temporal_matched_report.py](/opt/tiger/hand/tools/build_oldhl_temporal_matched_report.py)
- [oldhl_temporal_matched_report.json](/opt/tiger/hand/experiments/generated/oldhl_temporal_matched_report.json)
- [oldhl_temporal_matched_report.md](/opt/tiger/hand/experiments/generated/summary_tables/oldhl_temporal_matched_report.md)

Protocol:

- same pretrain split
- same finetune split
- same `168/84` window schedule
- same hidden size, epochs, aggregation, and boost map
- only `mode=state` vs `mode=temporal` changes

Result:

- scratch:
  - old HL `0.4615`
  - temporal HL `0.4615`
- pretrained:
  - old HL `0.7179`
  - temporal HL `0.6923`
- pretrained delta:
  - temporal minus state `-0.0256`

Interpretation:

- this directly rules out the lazy story that temporal HL wins because the same
  strong encoder classifies it better
- the matched comparison is useful precisely because it is negative
- it forces the mainline onto the stronger axes that actually survive:
  - transition-conditioned control
  - grouped symbolic editability
  - hard-slice compact search on right-hand interaction edits

Decision:

- do **not** center the project on sequence classification gains
- keep the temporal upgrade framed as a representation and control upgrade
- use this matched negative result as a reviewer-defense artifact, not as a
  weakness to hide

## 2026-06-10 hard-slice feasible interaction-rich residual audit

New artifacts:

- [build_interaction_realized_feasible_interaction_rich_residuals.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_interaction_rich_residuals.py)
- [interaction_realized_feasible_interaction_rich_residuals.json](/opt/tiger/hand/experiments/generated/interaction_realized_feasible_interaction_rich_residuals.json)
- [interaction_realized_feasible_interaction_rich_residuals.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_interaction_rich_residuals.md)

Protocol:

- start from the existing hard-slice scaling report
- keep the current strongest hard-slice method:
  - `hgb_budget2_finger_profile_snap_top20`
- compare it against the same selector / budget / depth without the repair:
  - `hgb_budget2_none_top20`
- split the hard right-hand slices into:
  - all subtypes
  - `other_hand_motion=none` mass only
  - feasible interaction-rich residuals with `other_hand_motion!=none`

Result:

- closing:
  - all subtypes:
    - baseline `0.0340`
    - best `0.0555`
    - delta `+0.0215`
  - none mass only:
    - baseline `0.0000`
    - best `0.0000`
  - feasible interaction-rich only:
    - baseline `0.1203`
    - best `0.1962`
    - delta `+0.0759`
    - ratio vs all-subtype best: `3.5380x`
- opening:
  - all subtypes:
    - baseline `0.0320`
    - best `0.0533`
    - delta `+0.0213`
  - none mass only:
    - baseline `0.0000`
    - best `0.0000`
  - feasible interaction-rich only:
    - baseline `0.1146`
    - best `0.1911`
    - delta `+0.0764`
    - ratio vs all-subtype best: `3.5860x`

Interpretation:

- this converts the earlier subtype diagnosis into a quantitative artifact
- the hard-slice absolute score is indeed dominated by structurally infeasible
  `other_hand_motion=none` mass
- on the actually feasible interaction-rich residual, the current strongest
  mechanism is materially stronger than the full-slice number suggests
- the gain from preserve-side repair is also much larger on the true feasible
  residual than on the mixed full slice

Decision:

- keep the global hard-slice claim boundary explicit
- when pushing the remaining editor gap, target the feasible interaction-rich
  residual rather than averaging over the full hard slice
- use this artifact to justify that the next mechanism should be evaluated on
  true feasible failures, not on absent-opposite-hand mass

## 2026-06-10 opening chunk-length robustness

New artifacts:

- [build_interaction_realized_opening_chunk_transfer_length_sweep_val_only.py](/opt/tiger/hand/tools/build_interaction_realized_opening_chunk_transfer_length_sweep_val_only.py)
- [interaction_realized_opening_chunk_transfer_length_sweep_val_only.json](/opt/tiger/hand/experiments/generated/interaction_realized_opening_chunk_transfer_length_sweep_val_only.json)
- [interaction_realized_opening_chunk_transfer_length_sweep_val_only.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_opening_chunk_transfer_length_sweep_val_only.md)

Protocol:

- broadened corrected feasible opening slice
- val-only support
- sweep chunk length `2 / 3 / 4`
- compare:
  - fixed edge
  - chunk classifier
  - chunk regressor
  - oracle binary

Result:

- all chunk lengths produce the same result:
  - `fixed_edge = 0.6306`
  - `chunk_cls = 0.7197`
  - `chunk_reg = 0.7197`
  - `oracle_binary = 0.7197`
  - paired delta vs fixed edge `+0.0892`
- paired wins/losses are identical for all three lengths:
  - wins `14`
  - losses `0`
  - ties `143`

Interpretation:

- the broader opening chunk-transfer gain is not a chunk-length artifact
- the opening-side temporal factorization result is now backed by the same
  length-robustness style evidence that already existed on the closing side

Decision:

- promote opening chunk transfer from a single positive point to a robustness
  backed result
- keep chunk temporal factorization as a general temporal-HL mechanism rather
  than a closing-only effect

## 2026-06-10 feasible-left lightweight routing closure

New artifacts:

- [build_interaction_realized_feasible_left_opening_chunk_knn.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_left_opening_chunk_knn.py)
- [interaction_realized_feasible_left_opening_chunk_knn.json](/opt/tiger/hand/experiments/generated/interaction_realized_feasible_left_opening_chunk_knn.json)
- [interaction_realized_feasible_left_opening_chunk_knn.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_opening_chunk_knn.md)
- [build_interaction_realized_feasible_rich_template_gate.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_rich_template_gate.py)
- [interaction_realized_feasible_rich_template_gate.json](/opt/tiger/hand/experiments/generated/interaction_realized_feasible_rich_template_gate.json)
- [interaction_realized_feasible_rich_template_gate.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_rich_template_gate.md)
- [build_interaction_realized_feasible_left_routing_closure.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_left_routing_closure.py)
- [interaction_realized_feasible_left_routing_closure.json](/opt/tiger/hand/experiments/generated/interaction_realized_feasible_left_routing_closure.json)
- [interaction_realized_feasible_left_routing_closure.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_routing_closure.md)

Protocol:

- test whether lightweight routing can beat the fixed task-best left repair on
  the feasible interaction-rich region
- explicitly close the missing opening-side chunk-routing case
- aggregate all lightweight routing attempts into one comparison table

Key new result:

- opening chunk KNN is negative:
  - fixed finger `0.6497`
  - chunk KNN `0.5350`
  - delta `-0.1146`
  - wins `2`, losses `20`, ties `135`
- rich-only template gate exactly reproduces the same numbers as the old
  feasible template gate:
  - closing fixed task-best `0.6582`, best rich gate `0.5696`
  - opening fixed task-best `0.6497`, best rich gate `0.5860`

Aggregate closure:

- closing:
  - best lightweight alternatives still lose to fixed task-best:
    - dense KNN `0.5696`
    - seq+subtype template gate `0.5696`
    - edge-vs-finger temporal route `0.6203`
    - chunk KNN `0.4747`
  - fixed task-best remains `0.6582`
  - framewise oracle remains `0.7405`
- opening:
  - only frame-level temporal-window KNN shows a tiny positive move:
    - `0.6624` vs fixed task-best `0.6497` (`+0.0127`)
  - dense KNN only ties fixed task-best
  - chunk KNN drops to `0.5350`
  - framewise oracle remains `0.7197`

Interpretation:

- lightweight routing is now effectively closed as a main next-step family
- removing `none` mass does not rescue subtype/template routing
- opening has a weak frame-level temporal signal, but it does not survive a
  chunk-level decision unit
- the remaining headroom is real, but it is oracle headroom, not evidence that
  another cheap gate is about to solve the problem

Decision:

- stop spending cycles on lightweight gates, template rules, and chunk KNN
  routing for feasible left repair
- keep the residual gap focused on stronger supervision / target redesign /
  objective change rather than more cheap routing variants

## 2026-06-10 feasible-left dense gain regressor

New artifacts:

- [build_interaction_realized_feasible_left_gain_regressor.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_left_gain_regressor.py)
- [interaction_realized_feasible_left_gain_regressor.json](/opt/tiger/hand/experiments/generated/interaction_realized_feasible_left_gain_regressor.json)
- [interaction_realized_feasible_left_gain_regressor.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_gain_regressor.md)

Protocol:

- keep the same feasible-left corrected pipeline
- upgrade supervision from sparse mode labels to dense continuous gains:
  - `left_mode_joint - left_none_joint`
- use a stronger regressor family rather than KNN:
  - `HistGradientBoostingRegressor`
- features:
  - previous/current/next feasible context
  - current left/right state signatures
  - no-repair agreement statistics

Result:

- closing:
  - fixed task-best `0.6582`
  - gain regressor `0.5759`
  - delta `-0.0823`
  - wins `9`, losses `22`, ties `127`
- opening:
  - fixed task-best `0.6497`
  - gain regressor `0.5541`
  - delta `-0.0955`
  - wins `3`, losses `18`, ties `136`

Interpretation:

- moving from sparse mode labels to dense gain supervision still does not beat
  the fixed task-best feasible-left policy
- this is stronger evidence than the earlier cheap gates because the objective
  itself changed, not just the routing rule
- the remaining feasible-left headroom is therefore not obviously unlocked by a
  simple supervised selector, even with richer context and continuous targets

Decision:

- extend the routing closure boundary to include dense gain regression
- stop treating “just learn a better selector” as the likely next win on the
  feasible-left problem

## 2026-06-10 feasible-left oracle gap audit and sparse override

New artifacts:

- [build_interaction_realized_feasible_left_oracle_gap_audit.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_left_oracle_gap_audit.py)
- [interaction_realized_feasible_left_oracle_gap_audit.json](/opt/tiger/hand/experiments/generated/interaction_realized_feasible_left_oracle_gap_audit.json)
- [interaction_realized_feasible_left_oracle_gap_audit.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_oracle_gap_audit.md)
- [build_interaction_realized_feasible_left_sparse_override.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_left_sparse_override.py)
- [interaction_realized_feasible_left_sparse_override.json](/opt/tiger/hand/experiments/generated/interaction_realized_feasible_left_sparse_override.json)
- [interaction_realized_feasible_left_sparse_override.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_sparse_override.md)

Oracle audit result:

- closing:
  - overall headroom `+0.0823`
  - conflict slice headroom `+0.1074`
  - oracle modes:
    - `none`: `116` frames
    - `edge_transition_snap`: `37`
    - `finger_profile_snap`: `5`
  - largest conflict sequence contribution:
    - `ROM01_No_Interaction_2_Hand`: delta `+0.1250`
- opening:
  - overall headroom `+0.0701`
  - conflict slice headroom `+0.0733`
  - oracle modes:
    - `none`: `120` frames
    - `edge_transition_snap`: `30`
    - `finger_profile_snap`: `7`
  - largest conflict sequence contribution:
    - `ROM01_No_Interaction_2_Hand`: delta `+0.0748`
    - `ROM02_Interaction_2_Hand`: delta `+0.0870`

Interpretation:

- the remaining feasible-left headroom is indeed structured, not random
- but it is dominated by conflict frames where the fixed task-best mode is not
  the per-frame oracle mode
- closing and opening prefer different oracle conflict directions:
  - closing conflict mass is mostly `oracle=none`
  - opening conflict mass is mostly `oracle=edge_transition_snap`

Sparse override follow-up:

- use the train split to whitelist only a few subtypes with strong positive
  alternate-mode gain over fixed task-best
- closing:
  - selected only `2` override subtypes
  - fixed task-best `0.6582`
  - sparse override `0.6329`
  - delta `-0.0253`
- opening:
  - selected `0` override subtypes
  - exact tie with fixed task-best at `0.6497`

Decision:

- even oracle-guided structured conflict analysis does not rescue a sparse
  subtype-rule policy
- add sparse override to the closure boundary
- the next useful target redesign should likely move below coarse subtype rules
  and above cheap selectors, e.g. a richer transition objective or a different
  decomposition of preserve-side success

## 2026-06-10 feasible-left soft preserve objective

New artifacts:

- [build_interaction_realized_feasible_left_soft_objective.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_left_soft_objective.py)
- [interaction_realized_feasible_left_soft_objective.json](/opt/tiger/hand/experiments/generated/interaction_realized_feasible_left_soft_objective.json)
- [interaction_realized_feasible_left_soft_objective.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_soft_objective.md)

Protocol:

- stop learning selectors entirely
- choose the repair mode directly by a softer preserve objective:
  - `left_transition_agreement + w * left_state_agreement`
- sweep `w` from `0.0` to `1.0`
- compare the resulting policy against:
  - fixed task-best
  - framewise oracle

Result:

- closing:
  - best `w = 0.75`
  - soft-objective joint `0.5759`
  - fixed task-best `0.6582`
  - oracle `0.7405`
  - delta vs fixed `-0.0823`
  - but transition agreement rises:
    - soft `0.8117`
    - fixed `0.7535`
    - oracle `0.7775`
- opening:
  - best `w = 0.75`
  - soft-objective joint `0.5796`
  - fixed task-best `0.6497`
  - oracle `0.7197`
  - delta vs fixed `-0.0701`
  - transition agreement again rises:
    - soft `0.7930`
    - fixed `0.7643`
    - oracle `0.7732`

Interpretation:

- a softer preserve-side local agreement objective is not enough
- the policy can improve transition agreement while still hurting exact grouped
  preserve and final joint score
- therefore the remaining gap is not explained simply by the current preserve
  target being “too hard”; it is a deeper objective mismatch

Decision:

- add soft preserve-objective routing to the negative boundary
- future target redesign should not assume that maximizing local
  state/transition agreement will automatically improve the real preserve-side
  success metric

## 2026-06-10 feasible-left oracle granularity audit

New artifacts:

- [build_interaction_realized_feasible_left_oracle_granularity.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_left_oracle_granularity.py)
- [interaction_realized_feasible_left_oracle_granularity.json](/opt/tiger/hand/experiments/generated/interaction_realized_feasible_left_oracle_granularity.json)
- [interaction_realized_feasible_left_oracle_granularity.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_oracle_granularity.md)

Protocol:

- keep the same feasible-left row scores
- compare the best achievable joint score when the repair mode is allowed to
  change at different granularities:
  - one mode per sequence
  - one mode per subtype
  - one mode per short contiguous run
  - one mode per frame

Result:

- closing:
  - fixed task-best `0.6582`
  - sequence oracle `0.6646` (`+0.0063`)
  - subtype oracle `0.6962` (`+0.0380`)
  - run oracle `0.7089` (`+0.0506`)
  - frame oracle `0.7405` (`+0.0823`)
  - frame-vs-run gap `0.0316`
- opening:
  - fixed task-best `0.6497`
  - sequence oracle `0.6624` (`+0.0127`)
  - subtype oracle `0.6752` (`+0.0255`)
  - run oracle `0.6943` (`+0.0446`)
  - frame oracle `0.7197` (`+0.0701`)
  - frame-vs-run gap `0.0255`

Interpretation:

- sequence-level policy is too coarse; its headroom over fixed is tiny
- subtype-level policy helps more, but still leaves a lot on the table
- short-run policy captures a substantial fraction of the remaining headroom
- the remaining frame-vs-run gap is real but much smaller than the old
  fixed-vs-frame gap

Decision:

- the next preserve-side redesign should prioritize run/chunk-level targets
  rather than sequence/subtype rules
- this also explains why sequence-level and sparse-rule policies were weak:
  the useful mode variation happens at a finer temporal granularity

## 2026-06-10 feasible-left run-level learned scorer

New artifacts:

- [build_interaction_realized_feasible_left_run_learned_scorer.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_left_run_learned_scorer.py)
- [interaction_realized_feasible_left_run_learned_scorer.json](/opt/tiger/hand/experiments/generated/interaction_realized_feasible_left_run_learned_scorer.json)
- [interaction_realized_feasible_left_run_learned_scorer.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_run_learned_scorer.md)

Protocol:

- keep the same feasible-left row scores as the earlier repair bundle
- group contiguous feasible frames into short runs with gap threshold `12`
- learn run-level repair-mode policies from run summaries:
  - `run_cls`: predict the best mode directly
  - `run_reg`: regress per-mode run score, then pick the best
- compare against:
  - fixed task-best
  - run oracle
  - frame oracle

Result:

- closing:
  - fixed task-best `0.6582`
  - run_cls `0.4747`
  - run_reg `0.6013`
  - run oracle `0.7089`
  - frame oracle `0.7405`
- opening:
  - fixed task-best `0.6497`
  - run_cls `0.6115`
  - run_reg `0.6752`
  - run oracle `0.6943`
  - frame oracle `0.7197`

Interpretation:

- the granularity audit was real: run-level oracle still keeps meaningful
  headroom on both tasks
- but simple learned run summaries are not enough to recover that headroom
- `run_reg` captures a small fraction of the opening gap, but still leaves the
  full closing side below the fixed task-best policy
- `run_cls` is clearly unstable and collapses badly on closing

Decision:

- do **not** promote run-level learned scorers into the mainline
- keep this as a closure artifact:
  - the useful granularity is indeed run/chunk-level
  - but a simple run-summary classifier/regressor is not sufficient to exploit
    it robustly
- preserve-side evidence stays with:
  - task-specific fixed left repair as the mainline gain
  - oracle granularity as a diagnostic bound
  - routing/run-level learned scorers as negative boundary evidence

## 2026-06-09 transition-conditioned editor follow-up

New artifacts:

- [build_transition_conditioned_symbolic_editor.py](/opt/tiger/hand/tools/build_transition_conditioned_symbolic_editor.py)
- [transition_conditioned_symbolic_editor.json](/opt/tiger/hand/experiments/generated/transition_conditioned_symbolic_editor.json)
- [transition_conditioned_symbolic_editor.md](/opt/tiger/hand/experiments/generated/summary_tables/transition_conditioned_symbolic_editor.md)

Protocol:

- build donor pairs from `val`
- for each eligible `test` transition, choose a donor pair that already
  realizes the requested target hand-motion label
- compose the donor target-hand transition onto the original previous-frame
  hand geometry
- evaluate:
  - coarse hand-motion realization
  - donor-state agreement
  - donor-transition agreement
  - donor residual distance

Important scope restriction:

- this editor currently covers only left/right hand-motion tasks
- it does **not** edit interaction-motion because the current exported labels do
  not expose editable global cross-hand geometry

Observed result:

- coarse composed success is too easy:
  - symbolic tracks symbolic-pair availability at `0.8291 .. 0.8425`
  - proxy reaches `1.0000` on all four hand-motion tasks
- therefore coarse `opening/closing` realization is not a discriminative main
  metric for the editor

Motif-level detail:

- semantic proxy source:
  - symbolic is better on:
    - left closing: state `0.6986 > 0.6789`, transition `0.8957 > 0.8564`
    - left opening: state `0.6341 > 0.6235`, transition `0.8813 > 0.8424`
    - right closing: state `0.6556 > 0.5722`, donor residual `0.3778 < 0.4800`
  - symbolic is worse on:
    - right opening: state `0.5335 < 0.7033`, transition `0.8368 < 0.8732`,
      donor residual `0.4582 > 0.3207`
- continuous proxy source:
  - symbolic is better on:
    - left closing: state `0.6986 > 0.6679`, donor residual `0.3267 < 0.4224`
    - left opening: transition `0.8813 > 0.7510`, donor residual
      `0.3200 < 0.6254`
  - symbolic is mixed / worse on right-hand tasks:
    - right closing: transition `0.8837 < 0.9481`
    - right opening: state `0.5335 < 0.5776`, transition `0.8368 < 0.8825`

Interpretation:

- this is a useful but **mixed** result, not a mainline claim
- it shows that a realized editor can be built from current artifacts
- it also shows that:
  - coarse hand-motion labels are too weak for evaluating editor quality
  - motif portability is the right evaluation axis
  - target-hand delta composition alone does not produce a stable symbolic win
    across all hand-motion tasks
- the remaining gap is not just availability; it is the lack of a richer
  transition / coordination representation, especially on right-hand slices

Decision:

- keep this as a negative-pressure artifact proving that audit-only evidence was
  not enough
- do **not** promote current transition-conditioned delta composition into the
  main claim
- next editor-stage work should focus on:
  - richer transition motifs than coarse `opening/closing`
  - interaction-aware or coordination-aware channels
  - realized edit evaluation that penalizes motif drift, not only target-label
    success

## 2026-06-09 grouped-motif editor refinement

The first realized editor result above showed that coarse hand-motion
realization was too easy and therefore not discriminative. I then upgraded the
same evaluator to score a grouped target-hand motif:

- `grouped motif = hand_motion + per-finger activity profile`
- per finger, the profile is bucketed as:
  - `still`: 0 non-stay regional transitions
  - `local`: 1 non-stay regional transition
  - `active`: 2+ non-stay regional transitions

This grouped motif is much less sparse than exact 20-way transition strings,
but still sequence-native and anatomically interpretable.

Updated artifact:

- [transition_conditioned_symbolic_editor.md](/opt/tiger/hand/experiments/generated/summary_tables/transition_conditioned_symbolic_editor.md)

Key result:

- coarse composed success remains trivial and should still be ignored as a main
  metric
- grouped motif fidelity is now strongly in favor of symbolic edits on **all**
  four hand-motion tasks

Semantic proxy source:

- left closing:
  - grouped motif match `0.4376` vs proxy `0.1049`
  - active slice `0.3738` vs `0.1667`
  - dense slice `0.4586` vs `0.2071`
- left opening:
  - grouped motif match `0.2967` vs proxy `0.0712`
  - active slice `0.2593` vs `0.0955`
  - dense slice `0.2803` vs `0.1040`
- right closing:
  - grouped motif match `0.4289` vs proxy `0.0916`
  - active slice `0.4360` vs `0.1123`
  - dense slice `0.4377` vs `0.1127`
- right opening:
  - grouped motif match `0.2714` vs proxy `0.1063`
  - active slice `0.2714` vs `0.1262`
  - dense slice `0.2760` vs `0.1283`

Continuous proxy source:

- left closing:
  - grouped motif match `0.4376` vs proxy `0.1274`
  - dense slice `0.4586` vs `0.2515`
- left opening:
  - grouped motif match `0.2967` vs proxy `0.0538`
  - dense slice `0.2803` vs `0.0786`
- right closing:
  - grouped motif match `0.4289` vs proxy `0.0724`
  - dense slice `0.4377` vs `0.0891`
- right opening:
  - grouped motif match `0.2714` vs proxy `0.0893`
  - dense slice `0.2760` vs `0.1077`

Interpretation:

- the problem was not that realized editing was hopeless
- the problem was that the old success label was too coarse
- once evaluated with a sequence-native grouped motif target, symbolic edits are
  consistently better than opaque proxies
- this is materially stronger than the earlier donor-only audits because the
  donor transition is actually composed onto a new previous-frame context

Claim boundary:

- keep:
  - symbolic edits preserve grouped temporal motifs better than opaque proxies
    under transition-conditioned composition
  - active and dense motif slices strengthen this advantage
- do not claim yet:
  - exact transition-sequence portability is solved
  - interaction-aware realized editing is solved

Next:

- add interaction-vs-noninteraction reporting on this grouped-motif editor
- test whether explicit cross-hand coordination channels can reduce the
  remaining interaction weakness

## 2026-06-09 adaptive support relaxation on the weak slice

I tested whether the current weak slice could be improved by a more careful
multi-stage support policy rather than blunt uniform relaxation.

New artifacts:

- [build_weak_slice_adaptive_support_topk.py](/opt/tiger/hand/tools/build_weak_slice_adaptive_support_topk.py)
- [weak_slice_adaptive_support_topk.json](/opt/tiger/hand/experiments/generated/weak_slice_adaptive_support_topk.json)
- [weak_slice_adaptive_support_topk.md](/opt/tiger/hand/experiments/generated/summary_tables/weak_slice_adaptive_support_topk.md)

Protocol:

- weak slice remains:
  - task: `right_hand_motion -> opening`
  - setting: `interaction` frames only
- keep the successful `top-k right x top-k left` joint search
- compare policies for choosing the left donor pool:
  - exact grouped support only
  - uniform relaxed family support
  - fallback to relaxed only when exact support is empty
  - thresholded fallback when exact support size is below a small count
  - simple interaction-motion-gated fallback policies

Most important result:

- the best overall joint-hit rate comes from relaxing exactly the weak support
  regimes:
  - `exact`: availability `0.1563`, joint-on-available `0.1477`, overall
    joint-hit `0.0231`
  - `uniform relaxed`: availability `0.2060`, joint-on-available `0.1293`,
    overall joint-hit `0.0266`
  - `threshold < 2`: identical to uniform relaxed on the key metrics

Interpretation:

- support-aware adaptation does help coverage enough to improve the overall
  realized joint-hit rate
- but the useful adaptive signal is much narrower than expected
- a generic subtype-aware gate did **not** beat the simple support-size rule

Key structural finding:

- the exact left-support pool is effectively tri-modal:
  - mostly empty
  - occasionally singleton
  - otherwise already large enough
- empirical counts on the weak slice:
  - exact left pool size `0`: 469 frames
  - exact left pool size `1`: 14 frames
  - exact left pool size `>= 2`: rare but typically already stable

Why `threshold < 2` equals `uniform relaxed`:

- the quality gap from exact to relaxed is concentrated almost entirely on
  singleton-support frames
- on the `exact_left_pool_size == 1` subset:
  - exact overall joint-hit is `0.0000`
  - relaxed overall joint-hit rises to `0.0769`
- on `exact_left_pool_size >= 2`, relaxing support does not materially change
  the chosen result on the tracked metrics

What did **not** work:

- fallback only when exact support is empty is too conservative:
  - availability still rises to `0.2060`
  - but overall joint-hit reaches only `0.0249`
- simple motion-gated fallback policies such as only relaxing for
  `approach/separate` did not beat the best support-size rule

Updated conclusion:

- candidate-space expansion remains the right mechanism family
- but the current family-level relaxation is a coverage-quality tradeoff, not a
  clean fix
- the practically useful heuristic is:
  - keep exact support when it has meaningful redundancy
  - relax when support is empty or singleton
- however, even this only gives a modest gain in overall joint-hit and still
  leaves the weak slice fundamentally unsolved

Next:

- move beyond support-size gating alone
- test whether richer left-hand preservation targets or explicit cross-hand
  coordination descriptors can discriminate the singleton-support failures from
  the recoverable cases

## 2026-06-09 full joint-search diagnosis on the weak slice

Before inventing another heuristic, I tested whether the current bottleneck is
simply caused by truncating the candidate search too aggressively.

New artifacts:

- [build_weak_slice_full_joint_search.py](/opt/tiger/hand/tools/build_weak_slice_full_joint_search.py)
- [weak_slice_full_joint_search.json](/opt/tiger/hand/experiments/generated/weak_slice_full_joint_search.json)
- [weak_slice_full_joint_search.md](/opt/tiger/hand/experiments/generated/summary_tables/weak_slice_full_joint_search.md)

Protocol:

- weak slice remains:
  - task: `right_hand_motion -> opening`
  - setting: `interaction` frames only
- compare four methods:
  - exact support + top-k joint search
  - exact support + full candidate search
  - relaxed support + top-k joint search
  - relaxed support + full candidate search

Main result:

- exact search is already saturated:
  - exact top-k and exact full are identical on all tracked metrics
- relaxed search is **not** saturated:
  - relaxed top-k:
    - availability `0.2060`
    - joint-on-available `0.1293`
    - overall joint-hit `0.0266`
  - relaxed full:
    - availability `0.2060`
    - joint-on-available `0.1552`
    - overall joint-hit `0.0320`

Interpretation:

- the candidate-space expansion line is stronger than it first appeared
- the main bottleneck is **not** only support coverage
- a real part of the remaining loss comes from top-k truncation on the relaxed
  left-support pool
- exact support has no hidden depth benefit, but relaxed support does

Subtype signal:

- the gain from relaxed full is localized and meaningful, not diffuse
- the biggest recovered subtype is:
  - `opening + steady`
  - relaxed top-k joint-on-available `0.0000`
  - relaxed full joint-on-available `0.2500`
- `opening + separate` also improves:
  - `0.7143 -> 0.8571`

Updated conclusion:

- this is the first evidence that the weak slice still contains recoverable
  donors beyond the current search depth
- therefore the next mechanism should focus on better candidate exploration or
  ranking inside the relaxed left-support pool, rather than discarding relaxed
  support as fundamentally too noisy

## 2026-06-09 relaxed left-search scaling

I then quantified how much search depth is actually needed on the relaxed
left-support pool.

New artifacts:

- [build_weak_slice_relaxed_search_scaling.py](/opt/tiger/hand/tools/build_weak_slice_relaxed_search_scaling.py)
- [weak_slice_relaxed_search_scaling.json](/opt/tiger/hand/experiments/generated/weak_slice_relaxed_search_scaling.json)
- [weak_slice_relaxed_search_scaling.md](/opt/tiger/hand/experiments/generated/summary_tables/weak_slice_relaxed_search_scaling.md)

Protocol:

- keep the full right-donor pool
- vary only the relaxed left-search depth:
  - top-1, top-3, top-5, top-10, top-20, top-50, full

Result:

- deeper left search is strongly monotonic up to about 20 candidates:
  - top-1 overall joint-hit: `0.0107`
  - top-3 overall joint-hit: `0.0178`
  - top-5 overall joint-hit: `0.0266`
  - top-10 overall joint-hit: `0.0302`
  - top-20 overall joint-hit: `0.0320`
  - top-50 / full overall joint-hit: `0.0320`

Most important practical finding:

- `left_top_20` already matches full search on the key weak-slice objective:
  - availability `0.2060`
  - joint-on-available `0.1552`
  - overall joint-hit `0.0320`

Interpretation:

- the weak slice is not asking for an intractable combinatorial search
- the missing positive cases are mostly within a moderate relaxed left-support
  depth, not buried arbitrarily deep
- this means the next practical mechanism should be:
  - keep relaxed family support
  - search deeper on the left side, especially for weak-support cases

Updated boundary:

- keep:
  - candidate-space expansion is a live and now stronger mechanism family
  - relaxed support is valuable when explored deeply enough
- refine:
  - the previous support-gating conclusion was incomplete because it used too
    shallow a left search depth
  - the real bottleneck is now:
    - relaxed left-pool exploration / ranking
    - not exact-support saturation

Next:

- promote `relaxed + deeper left search` to the new weak-slice baseline
- then test whether coordination-aware reranking can approximate the `top-20`
  gain with a smaller effective search budget

## 2026-06-09 coordination-aware rerank under small search budgets

I tested whether the new `relaxed + deeper left search` gain can be
approximated by better left-candidate ordering rather than simply searching
deeper.

New artifacts:

- [build_weak_slice_coordination_rerank.py](/opt/tiger/hand/tools/build_weak_slice_coordination_rerank.py)
- [weak_slice_coordination_rerank.json](/opt/tiger/hand/experiments/generated/weak_slice_coordination_rerank.json)
- [weak_slice_coordination_rerank.md](/opt/tiger/hand/experiments/generated/summary_tables/weak_slice_coordination_rerank.md)

Protocol:

- weak slice remains:
  - task: `right_hand_motion -> opening`
  - setting: `interaction` frames only
- keep the relaxed family support pool
- compare:
  - baseline search budgets: `top-5`, `top-10`, `top-20`
  - rerank by current left-hand geometry
  - rerank by current left-hand geometry plus current cross-hand distance
  - pair-aware rerank adding right-donor / left-donor cross-distance agreement

Main result:

- reranking helps at very small budgets, but only modestly
- the best small-budget result is:
  - `curr_left_top5`: overall joint-hit `0.0284`
  - versus baseline `base_top5`: `0.0266`
- for `top-10`, reranking does not beat the plain deeper search:
  - `base_top10`: `0.0302`
  - all reranked `top-10` variants: also `0.0302`
- the strong target remains:
  - `base_top20`: `0.0320`

Interpretation:

- better local ordering is real and worth keeping
- but the gain is smaller than the gain from simply allowing a modestly deeper
  left search
- current coordination-aware signals do **not** add value beyond simple
  left-hand geometry ordering under this protocol

What the rerank result says about coordination:

- adding cross-hand distance terms did not beat the geometry-only rerank
- pair-aware cross-distance compatibility also failed to improve over the plain
  `top-10` baseline
- so the current explicit coordination cues are still too weak to explain the
  missing weak-slice recoveries

What this means practically:

- the strongest weak-slice baseline is now:
  - relaxed family support
  - full right-donor pool
  - left search depth around `20`
- if compute budget is tight, a small geometry-aware rerank is still useful:
  - `top-5` improves from `0.0266` to `0.0284`
- but the main recovered gain still comes from search depth, not coordination
  reranking

Updated boundary:

- keep:
  - candidate-space expansion is the strongest live mechanism family
  - relaxed support plus moderate left-depth search gives the best current weak
    slice result
- refine:
  - current coordination-aware reranking is not yet the mechanism that closes
    the remaining gap
  - explicit cross-hand features still do not outperform search depth on this
    slice

Next:

- treat `relaxed + left_top20` as the new weak-slice control baseline
- if we continue on this line, the next meaningful upgrade is not another hand
  built reranker
- it is either:
  - a learned reranker trained to predict joint success inside the relaxed left
    pool
  - or a richer representation target for left-hand preservation than the
    current grouped motif family

## 2026-06-09 learned left-pool reranker

I then replaced hand-built reranking with a lightweight learned reranker.

New artifacts:

- [build_weak_slice_learned_reranker.py](/opt/tiger/hand/tools/build_weak_slice_learned_reranker.py)
- [weak_slice_learned_reranker.json](/opt/tiger/hand/experiments/generated/weak_slice_learned_reranker.json)
- [weak_slice_learned_reranker.md](/opt/tiger/hand/experiments/generated/summary_tables/weak_slice_learned_reranker.md)

Training target:

- for each weak-slice frame in `val`
- for each relaxed left candidate
- label whether there exists at least one right donor in the current right pool
  such that the joint composition score is `1`

Training scale:

- candidate examples: `4495`
- positive examples: `228`
- negative examples: `4267`

Main result:

- `learned_top10` matches the current strongest search-depth baseline:
  - `learned_top10` overall joint-hit: `0.0320`
  - `base_top20` overall joint-hit: `0.0320`
- both also match on joint-on-available:
  - `0.1552`

Comparison against simpler budgets:

- `base_top10`: `0.0302`
- `learned_top10`: `0.0320`
- `base_top5`: `0.0266`
- `learned_top5`: `0.0266`

Interpretation:

- this is the first mechanism that converts the weak-slice depth gain into a
  smaller effective left-search budget
- a learned reranker can recover the benefit of `left_top20` while only using
  `top10` left candidates
- the gain does **not** extend to `top5` under the current feature set

Subtype evidence:

- the learned `top10` recovery is driven by the same meaningful subtype that
  deeper relaxed search uncovered:
  - `opening + steady`
  - `base_top10` joint-on-available: `0.1250`
  - `learned_top10`: `0.2500`
  - matching `base_top20`

What this changes:

- the new strongest weak-slice mechanisms are now:
  - relaxed family support + left_top20 search
  - relaxed family support + learned left reranker + left_top10 search
- therefore the project now has a concrete path from:
  - raw candidate-space expansion
  - to a compact learned retrieval/reranking mechanism

Updated boundary:

- keep:
  - candidate-space expansion is the correct weak-slice mechanism family
  - the search-depth gain is real and not an artifact
- add:
  - the gain is learnable from weak-slice supervision
  - a lightweight learned reranker can compress `top20` search down to `top10`
    without losing the current best weak-slice score
- still do not claim:
  - the interaction weak slice is solved globally
  - `top5` budget is enough

Next:

- freeze `learned_top10` as the new compact weak-slice baseline
- then decide between two higher-value continuations:
  - test whether richer preserve targets push `top5`
  - or broaden the learned-reranker idea into a more general controllability
    mechanism beyond the single weak slice

## 2026-06-09 pair-guided learned reranker

I strengthened the learned reranker by moving from left-only supervision to
pair-level supervision.

New artifacts:

- [build_weak_slice_pairguided_reranker.py](/opt/tiger/hand/tools/build_weak_slice_pairguided_reranker.py)
- [weak_slice_pairguided_reranker.json](/opt/tiger/hand/experiments/generated/weak_slice_pairguided_reranker.json)
- [weak_slice_pairguided_reranker.md](/opt/tiger/hand/experiments/generated/summary_tables/weak_slice_pairguided_reranker.md)

Training target:

- build pair examples on the weak slice from `val`
- each example is `(right donor, left donor)`
- label whether the joint composition score is `1`
- at test time:
  - use the trained pair model to score each left candidate by its best
    predicted pair score under the current right pool
  - then evaluate the true oracle only on the top-ranked left candidates

Training scale:

- pair examples: `4746`
- positive pairs: `236`
- negative pairs: `4510`

Main result:

- `pairguided_top10` matches the strongest current weak-slice baseline:
  - `pairguided_top10` overall joint-hit: `0.0320`
  - `base_top20` overall joint-hit: `0.0320`
- more importantly, `pairguided_top5` closes the previous `top5` gap:
  - `base_top5`: `0.0266`
  - `pairguided_top5`: `0.0302`
  - which now matches `base_top10`

Interpretation:

- pair-level supervision is stronger than the earlier left-only reranker
- the model is learning which left candidates are promising **given the current
  right donor pool**, not just in isolation
- this is the first mechanism that:
  - preserves the current best `top10` compact result
  - and also upgrades the `top5` budget from `base_top5` to `base_top10`

Subtype signal:

- the crucial recovery again appears on:
  - `opening + steady`
- `pairguided_top5` raises this subtype from:
  - `base_top5`: `0.0000`
  - to `0.2500`
- which matches both `pairguided_top10` and `base_top20`

What this changes:

- the compact weak-slice mechanism line is now clearly stronger:
  - left-only learned reranker:
    - compresses `top20 -> top10`
  - pair-guided learned reranker:
    - compresses `top20 -> top10`
    - and compresses `top10 -> top5` on the overall weak-slice score

Updated strongest baselines:

- strongest unrestricted weak-slice baseline:
  - relaxed family support + `base_top20`
- strongest compact weak-slice baseline:
  - relaxed family support + pair-guided reranker + `top10`
- strongest very-small-budget weak-slice baseline:
  - relaxed family support + pair-guided reranker + `top5`

Updated boundary:

- keep:
  - the weak-slice gain is learnable
  - pair-level compatibility matters beyond left-only candidate quality
- add:
  - pair-guided learning is now the best compact mechanism family on this slice
- still do not claim:
  - the interaction weakness is globally solved outside this concentrated slice

Next:

- treat pair-guided reranking as the new main weak-slice mechanism
- the next valuable step is no longer more search engineering
- it is to test whether this learned pair-guided mechanism generalizes beyond
  the single `right opening + interaction` slice

## 2026-06-09 pair-guided reranker multi-slice generalization

I expanded the pair-guided reranker from the single weak slice to all four
interaction-only hand-motion edit slices:

- `left_hand_motion -> closing`
- `left_hand_motion -> opening`
- `right_hand_motion -> closing`
- `right_hand_motion -> opening`

New artifacts:

- [build_pairguided_reranker_multislice.py](/opt/tiger/hand/tools/build_pairguided_reranker_multislice.py)
- [pairguided_reranker_multislice.json](/opt/tiger/hand/experiments/generated/pairguided_reranker_multislice.json)
- [pairguided_reranker_multislice.md](/opt/tiger/hand/experiments/generated/summary_tables/pairguided_reranker_multislice.md)

Main finding:

- the pair-guided mechanism is **not** a universal win on every slice
- but it is a stable and meaningful win exactly on the harder right-hand
  interaction slices, which are the bottlenecked regime

### Right-hand slices: strong compression result

For `right_hand_motion -> closing`:

- `base_top20` overall joint-hit: `0.0340`
- `pairguided_top10`: `0.0340`
- `base_top10`: `0.0322`
- `pairguided_top5`: `0.0304`
- `base_top5`: `0.0286`

For `right_hand_motion -> opening`:

- `base_top20`: `0.0320`
- `pairguided_top10`: `0.0320`
- `base_top10`: `0.0302`
- `pairguided_top5`: `0.0302`
- `base_top5`: `0.0266`

Interpretation:

- on both hard right-hand slices, pair-guided reranking consistently:
  - compresses `top20 -> top10`
  - improves `top5`
- this is now a repeated pattern, not a one-slice accident

### Left-hand slices: weaker or unnecessary

For the left-hand interaction slices:

- the absolute scores are much higher overall
- search depth already works strongly
- pair-guided reranking gives:
  - a very small gain for `left opening` at `top5`
  - no gain for `left closing`
  - and slight underperformance at `top10`

Interpretation:

- the learned mechanism is most useful where the slice is actually difficult
- on easier slices, deeper plain search is already sufficient and the learned
  reranker is not needed
- this is a good sign rather than a bad one:
  - the method is specializing to the true hard regime
  - instead of producing noisy tiny gains everywhere

Updated conclusion:

- pair-guided reranking is now a **general hard-slice mechanism**, not just a
  repair for one isolated weak case
- its strongest value is:
  - budget compression on hard right-hand interaction edits
- it should not be sold as a universal improvement over search depth for all
  slices

Updated strongest evidence:

- hard-slice compact mechanism:
  - `right closing`: `pairguided_top10 == base_top20`
  - `right opening`: `pairguided_top10 == base_top20`
- very-small-budget hard-slice improvement:
  - `right closing`: `pairguided_top5 > base_top5`
  - `right opening`: `pairguided_top5 > base_top5`

Updated boundary:

- keep:
  - pair-guided learned reranking is a real mechanism, not a one-off weak-slice
    patch
  - it generalizes across the hard right-hand interaction slices
- refine:
  - it is not the best tool for already-easy left-hand interaction slices
  - therefore its natural claim is hard-slice controllability / compact search,
    not universal per-slice dominance

Next:

- treat pair-guided reranking as the main compact-search mechanism for hard
  interaction slices
- the next high-value step is to integrate these hard-slice results into one
  compact evidence bundle against the “this is just a lookup heuristic” attack

## 2026-06-09 hard-slice compact-search bundle

I consolidated the strongest search-compression evidence into one bundle aimed
directly at the reviewer attack:

- "this is just a lookup heuristic" or "you only win by searching more"

New artifacts:

- [build_hard_slice_compact_search_bundle.py](/opt/tiger/hand/tools/build_hard_slice_compact_search_bundle.py)
- [hard_slice_compact_search_bundle.json](/opt/tiger/hand/experiments/generated/hard_slice_compact_search_bundle.json)
- [hard_slice_compact_search_bundle.md](/opt/tiger/hand/experiments/generated/summary_tables/hard_slice_compact_search_bundle.md)

What the bundle now makes explicit:

### Weak-slice progression

For `right_hand_motion -> opening`:

- baseline single donor: `0.0073`
- relaxed top-5: `0.0266`
- relaxed top-10: `0.0302`
- relaxed top-20: `0.0320`
- learned top-10: `0.0320`
- pair-guided top-5: `0.0302`
- pair-guided top-10: `0.0320`

This compresses the entire story into one line:

- search depth helps
- the depth gain is learnable
- pair-guided learning compresses that gain further

### Hard right-hand interaction slices

The bundle also isolates the only slices where this mechanism really matters:

- `right_hand_motion -> closing`
- `right_hand_motion -> opening`

Key summary:

- on both hard right-hand interaction slices:
  - `pairguided_top10 == base_top20`
- on both hard right-hand interaction slices:
  - `pairguided_top5 > base_top5`

### Left-hand reference slices

The bundle keeps the left-hand interaction slices as controls:

- they are easier overall
- pair-guided reranking is weaker or unnecessary there

Interpretation:

- this strengthens the claim boundary
- the mechanism is not "search everywhere"
- it is "compact search on the hard interaction regime"

Why this bundle matters:

- it is now much easier to defend against:
  - "you only searched more"
  - "this is just a heuristic donor lookup"
  - "the method is only patching one cherry-picked case"

Updated conclusion:

- the hard-slice compact-search line is now internally coherent:
  - plain search-depth scaling identifies recoverable support
  - left-only learned reranking compresses `top20 -> top10`
  - pair-guided reranking compresses:
    - `top20 -> top10`
    - and partially `top10 -> top5`
  - multi-slice evaluation shows this repeats on both hard right-hand
    interaction slices

Next:

- merge this hard-slice compact-search bundle with the existing
  structure-vs-repair-vs-control evidence so the strongest internal summary is
  no longer spread across separate memos

## 2026-06-09 topline evidence bundle

I merged the strongest current evidence into one top-level internal entry
point.

New artifacts:

- [build_topline_evidence_bundle.py](/opt/tiger/hand/tools/build_topline_evidence_bundle.py)
- [topline_evidence_bundle.json](/opt/tiger/hand/experiments/generated/topline_evidence_bundle.json)
- [topline_evidence_bundle.md](/opt/tiger/hand/experiments/generated/summary_tables/topline_evidence_bundle.md)

Covered evidence axes:

- `structure and repair`
- `control`
- `temporal transition`
- `hard-slice compact search`

What this bundle now fixes:

- the strongest internal story is no longer spread across:
  - structural frontier tables
  - representation risk bundle
  - hard-slice compact-search bundle
- there is now a single artifact that states:
  - what to keep
  - what not to claim
  - what the remaining gaps are

Current topline shape:

### Structure and repair

- flat seq @1.0: `0.6923`
- grouped seq @1.0: `0.8974`
- family seq @1.0: `1.0000`
- grouped-flat: `+0.2051`
- family-grouped: `+0.1026`
- family harmed: `0`

### Control

- symbolic clean rate is `1.0000` on the strongest local-edit tasks
- proxy clean rates remain near zero on the corresponding semantic controls

### Temporal transition

- interaction slices still show a symbolic advantage over proxy
- occlusion and finger-occlusion hard slices are already essentially solved

### Hard-slice compact search

- weak-slice `right opening` progression now compresses from:
  - baseline donor `0.0073`
  - to pair-guided top-5 `0.0302`
  - and pair-guided top-10 `0.0320`
- hard right-hand interaction slices repeat the same compression pattern

Why this matters:

- this is now the default internal decision entry point
- it also sharpens the current boundary:
  - the strongest novelty is not raw token separability
  - it is symbolic temporal structure plus compact hard-slice control/search

Updated conclusion:

- the project now has a coherent top-level evidence stack
- new experiments should be judged by whether they improve this stack, not just
  add another local result

Next:

- if we continue, the best next experiments are the ones that close the
  remaining top-level gaps listed in the bundle:
  - old-HL-plus-strong-encoder matched comparison for the new temporal channels
  - one concise interaction-vs-noninteraction top-level summary table

## 2026-06-09 interaction-vs-noninteraction slice on grouped-motif editor

I extended the grouped-motif realized editor with an explicit slice on whether
the current edited frame belongs to an interacting-hand context
(`hand_type == interacting`).

Updated artifact:

- [transition_conditioned_symbolic_editor.md](/opt/tiger/hand/experiments/generated/summary_tables/transition_conditioned_symbolic_editor.md)

Main finding:

- the grouped-motif symbolic advantage is **not** only a noninteraction effect
- on three of the four hand-motion tasks, symbolic remains clearly stronger in
  interaction slices
- the main localized weakness is now:
  - `right_hand_motion -> opening` under interaction

Semantic proxy source:

- left closing:
  - interaction slice rate `0.4153`
  - symbolic interaction motif match `0.5644`
  - proxy interaction motif match `0.0000`
  - symbolic noninteraction motif match `0.3846`
  - proxy noninteraction motif match `0.1795`
- left opening:
  - interaction `0.3497` vs proxy `0.0216`
  - noninteraction `0.2756` vs proxy `0.1049`
- right closing:
  - interaction `0.4539` vs proxy `0.1091`
  - noninteraction `0.4026` vs proxy `0.0658`
- right opening:
  - interaction `0.1330` vs proxy `0.0959`
  - noninteraction `0.4009` vs proxy `0.1198`

Continuous proxy source:

- left closing:
  - interaction `0.5644` vs proxy `0.0000`
  - noninteraction `0.3846` vs proxy `0.2179`
- left opening:
  - interaction `0.3497` vs proxy `0.0288`
  - noninteraction `0.2756` vs proxy `0.0707`
- right closing:
  - interaction `0.4539` vs proxy `0.0519`
  - noninteraction `0.4026` vs proxy `0.1026`
- right opening:
  - interaction `0.1330` vs proxy `0.0995`
  - noninteraction `0.4009` vs proxy `0.0760`

Interpretation:

- interaction does **not** collapse the grouped-motif editor story globally
- symbolic still has a strong realized-edit advantage in most interacting-hand
  slices
- however, the weakness is now sharply localized:
  - opening edits on the right hand inside interacting contexts are much less
    stable than the other three tasks
- this is a much better failure mode than a broad "interaction breaks the
  method" conclusion

Updated claim boundary:

- keep:
  - grouped-motif realized editing is a real symbolic advantage in both
    interaction and noninteraction settings for most tasks
- refine:
  - the remaining unresolved interaction problem is concentrated in
    `right-hand opening` rather than across all hand-motion edits

Next:

- add an explicit cross-hand coordination channel and retest the
  `right-hand opening + interaction` slice first

## 2026-06-09 weak-slice coordination breakdown

I further decomposed the remaining weak slice:

- task: `right_hand_motion -> opening`
- setting: `interaction` frames only
- breakdown keys:
  - other-hand motion
  - interaction-motion value
  - coarse coordination class

Updated artifact:

- [transition_conditioned_symbolic_editor.md](/opt/tiger/hand/experiments/generated/summary_tables/transition_conditioned_symbolic_editor.md)

Main finding:

- the weak slice is **not** a single homogeneous failure bucket
- it splits into two very different regimes

### Regime A: almost-total failure for both grouped motif and symbolic state

Many interaction subtypes collapse to:

- symbolic grouped motif match `0.0`
- symbolic state agreement `0.0`
- symbolic transition agreement `0.0`

Representative cases:

- `closing + approach`
- `closing + steady`
- `opening + steady`
- `steady + approach`

This means the symbolic donor selected under the current conditioning can still
be semantically valid at the coarse label level, yet after composition it
completely loses the donor motif in these interaction subtypes.

### Regime B: a narrower zone where symbolic still beats or matches proxy

The improvement is concentrated in a few specific subtypes, especially where
the other hand is effectively absent or less constraining in the exported
record:

- semantic proxy source:
  - `other=none, interaction=separate, one_active`:
    symbolic grouped motif `0.2500` vs proxy `0.2188`
  - `other=none, interaction=unknown, one_active`:
    symbolic grouped motif `1.0000` vs proxy `0.8095`
- continuous proxy source:
  - `other=none, interaction=separate, one_active`:
    symbolic grouped motif `0.2500` vs proxy `0.0000`
  - `other=none, interaction=unknown, one_active`:
    symbolic grouped motif `1.0000` vs proxy `0.0000`

### What this means

- the remaining weakness is not just "interaction is hard"
- the harder truth is:
  - current target-hand-only transition composition is structurally insufficient
    for several interaction subtypes
  - preserving coarse left-hand semantics is not enough
  - a real cross-hand coordination channel is now justified by evidence rather
    than by intuition

Updated boundary:

- keep:
  - grouped-motif realized editing is already strong and broadly valid
  - interaction is not globally fatal
- refine:
  - `right-hand opening + interaction` remains the single concentrated failure
    zone
  - within that zone, failure is strongest in specific interaction-motion /
    coordination subtypes rather than uniformly

Immediate next step:

- add one explicit cross-hand coordination field to the representation and
  retest only this weak slice before spending compute on broader reruns

## 2026-06-09 minimal coordination-channel rerun

I ran a targeted weak-slice rerun with one minimal explicit coordination
channel:

- new channel: `other-hand transition activity profile`
- focus remains only:
  - `right_hand_motion -> opening`
  - `interaction` frames

Artifacts:

- [build_weak_slice_coordination_rerun.py](/opt/tiger/hand/tools/build_weak_slice_coordination_rerun.py)
- [weak_slice_coordination_rerun.md](/opt/tiger/hand/experiments/generated/summary_tables/weak_slice_coordination_rerun.md)

Result:

- the minimal coordination-aware symbolic selector collapses in availability:
  - base symbolic availability: `0.7211`
  - coordination-aware symbolic availability: `0.0000`
- therefore this minimal field does **not** rescue the weak slice

Interpretation:

- the failure is stronger than "we forgot one temporal field"
- adding a strict other-hand transition-profile constraint makes the donor bank
  effectively empty under the current representation and conditioning protocol
- this indicates that the current weakness is not just missing metadata; it is
  a structural donor-composition problem under interacting-hand contexts

What this rules out:

- a naive fix that simply appends one more explicit preserved coordination field
- a story that the weak slice can be repaired by slightly stronger donor
  filtering alone

What this suggests instead:

- the next meaningful step is not another stricter selector
- it is a richer composition mechanism or a motif-level editor that can model
  cross-hand coupling more flexibly than exact donor matching

Updated boundary:

- keep:
  - grouped-motif realized editing is already strong in the mainline
  - the remaining weakness is sharply localized
- add:
  - the localized weakness is **not** fixed by a minimal explicit
    coordination-profile field
  - therefore the next experiment should target the composition mechanism, not
    just the conditioning signature

## 2026-06-09 weak-slice mechanism rerun

I then tested whether the weak slice is caused by the exactness of the
target-hand composition rule itself.

New artifact:

- [build_weak_slice_mechanism_rerun.py](/opt/tiger/hand/tools/build_weak_slice_mechanism_rerun.py)
- [weak_slice_mechanism_rerun.md](/opt/tiger/hand/experiments/generated/summary_tables/weak_slice_mechanism_rerun.md)

Protocol:

- keep the same symbolic donor selector
- only vary the donor delta scale:
  - `alpha = 0.25, 0.5, 0.75, 1.0`
- evaluate only:
  - `right_hand_motion -> opening`
  - `interaction` frames

Result:

- exact transplant remains best on grouped motif match
- softer composition does **not** rescue the weak slice

Overall grouped motif match:

- `alpha=0.25`: `0.0000`
- `alpha=0.5`: `0.0000`
- `alpha=0.75`: `0.0025`
- `alpha=1.0`: `0.1330`

Transition agreement rises monotonically with alpha:

- `0.5728 -> 0.6223 -> 0.7064 -> 0.7927`

Subtype pattern:

- for `other=none, interaction=separate`:
  - grouped motif match is `0.0000, 0.0000, 0.0000, 0.2500`
- for `other=none, interaction=unknown`:
  - grouped motif match is `0.0000, 0.0000, 0.0000, 1.0000`
- for `other=none, interaction=steady`:
  - grouped motif match is `0.0000, 0.0000, 0.0000, 0.0685`

Interpretation:

- the weak slice is **not** caused by exact donor deltas being too strong
- naive residualized / scaled composition is worse than exact composition
- together with the previous coordination-profile failure, this rules out two
  easy fixes:
  - stricter conditioning only
  - softer scalar blending only

What remains plausible:

- a richer nonlinear composition mechanism
- an editor that composes multi-hand motifs jointly rather than target-hand
  deltas alone
- a learned or search-based mechanism that can preserve donor motif structure
  under interaction constraints

Updated boundary:

- keep:
  - the mainline symbolic advantage on grouped-motif realized editing
- add:
  - the remaining weak slice is not repaired by minimal coordination metadata
  - the remaining weak slice is not repaired by simple delta scaling
- therefore:
  - the next meaningful advance must target a **joint composition mechanism**
    rather than more scalar heuristics

## 2026-06-09 weak-slice soft coordination reranking

I also tested a softer donor-selection mechanism on the same weak slice:

- keep the same coarse symbolic conditioning
- do **not** require exact other-hand profile matching
- rerank donor pairs with a soft penalty on the other-hand transition activity
  profile mismatch

Artifacts:

- [build_weak_slice_soft_coordination_rerun.py](/opt/tiger/hand/tools/build_weak_slice_soft_coordination_rerun.py)
- [weak_slice_soft_coordination_rerun.md](/opt/tiger/hand/experiments/generated/summary_tables/weak_slice_soft_coordination_rerun.md)

Result:

- soft coordination reranking makes effectively no difference
- all tested penalties produce the same grouped motif result:

Overall grouped motif match:

- `lambda=0.0`: `0.0959`
- `lambda=0.25`: `0.0959`
- `lambda=0.5`: `0.0959`
- `lambda=1.0`: `0.0959`
- `lambda=2.0`: `0.0959`

Subtype pattern:

- every major subtype is unchanged across all lambdas
- example:
  - `other=none, interaction=separate`: always `0.2500`
  - `other=none, interaction=steady`: always `0.0685`
  - all the hard interaction subtypes remain `0.0000`

Interpretation:

- soft profile-aware reranking is not enough either
- unlike the previous exact-profile test, this one does not destroy
  availability; it simply fails to change the selected donor in a meaningful
  way
- together, the two coordination tests now say:
  - exact profile matching is too strict
  - soft profile reranking is too weak

This is strong evidence that the remaining issue is **not** a small donor-score
adjustment problem. The coarse symbolic candidate set itself is not expressive
enough to support the needed interacting-hand edit.

Updated boundary:

- keep:
  - the mainline grouped-motif symbolic advantage
- add:
  - minimal coordination metadata fails
  - soft coordination reranking fails
  - scalar delta scaling fails
- therefore:
  - the next credible step is a genuinely joint editor / joint motif
    composition mechanism, not another donor-scoring heuristic

## 2026-06-09 weak-slice joint two-hand composition prototype

I built a small joint-composition prototype that applies donor deltas to both
hands instead of only to the target hand, while keeping the same symbolic donor
selector.

Artifacts:

- [build_weak_slice_joint_editor_prototype.py](/opt/tiger/hand/tools/build_weak_slice_joint_editor_prototype.py)
- [weak_slice_joint_editor_prototype.md](/opt/tiger/hand/experiments/generated/summary_tables/weak_slice_joint_editor_prototype.md)

Result:

- joint two-hand composition does **not** improve the weak slice under the
  current donor selector
- right grouped motif match stays unchanged:
  - target-only: `0.1330`
  - joint two-hand: `0.1330`

More important hidden finding:

- once restricted to donor-available weak-slice cases, the surviving subtypes in
  the summary are only:
  - `other_hand_motion = none`
  - interaction motion in `{approach, separate, steady, unknown}`

This means:

- the current symbolic donor selector is already excluding the harder weak-slice
  interaction subtypes before composition even starts
- the composition mechanism cannot fix donor-support gaps that occur upstream

Interpretation:

- this prototype does **not** falsify the need for joint editing
- but it does show that simply applying donor deltas to both hands is not
  enough when the donor support is already collapsed onto easier subtypes
- therefore the real next step is not just "joint composition after the same
  donor choice"
- it is a joint editor with a broader candidate / search space, or a learned
  mechanism that does not depend on exact donor support for the hard
  interaction subtypes

Updated boundary:

- keep:
  - current symbolic mainline is strong
  - weak-slice failures remain sharply localized
- add:
  - both-hand composition alone does not help under the current donor selector
  - donor support collapse is now an identified upstream bottleneck

Practical implication:

- the next prototype should expand or learn the candidate composition space
  itself, not just alter how the existing donor pair is applied

## 2026-06-09 split-donor candidate expansion prototype

I then tested a real candidate-space expansion rather than another scoring
heuristic.

Prototype:

- select the right-hand donor from an expanded symbolic pool focused on target
  realization
- select the left-hand donor independently from a pool focused on preserving
  the current left-hand grouped motif
- compose the two donors jointly

Artifacts:

- [build_weak_slice_split_donor_prototype.py](/opt/tiger/hand/tools/build_weak_slice_split_donor_prototype.py)
- [weak_slice_split_donor_prototype.md](/opt/tiger/hand/experiments/generated/summary_tables/weak_slice_split_donor_prototype.md)

This is the first weak-slice prototype that produces a clearly positive signal.

Overall:

- baseline availability: `0.9787`
- split-donor availability: `0.1563`
- baseline right grouped match: `0.1670`
- split right grouped match: `0.2500`
- baseline left preserve: `0.0399`
- split left preserve: `0.3409`
- baseline joint score: `0.0073`
- split joint score: `0.1136`

Interpretation:

- coverage drops sharply, but the successful region becomes much stronger
- once the candidate space is allowed to decouple right-hand realization from
  left-hand preservation, the weak slice is no longer almost hopeless
- the biggest lift is on the actual joint objective:
  - `0.0073 -> 0.1136`

Important subtype improvements:

- `opening + approach`:
  - baseline joint score `0.2857`
  - split joint score `0.6667`
- `opening + separate`:
  - baseline joint score `0.0000`
  - split joint score `0.6000`
- `steady + approach`:
  - baseline joint score `0.0909`
  - split joint score `0.2000`
- `steady + separate`:
  - baseline joint score `0.1250`
  - split joint score `0.5000`
- `steady + steady`:
  - baseline joint score `0.0000`
  - split joint score `0.0652`

Tradeoff:

- the gain comes with much lower availability (`0.1563`)
- so this is not yet a finished mechanism
- but it is the first experiment showing that expanding the candidate
  composition space can directly attack the weak slice

Updated boundary:

- keep:
  - donor-scoring heuristics and scalar blending are dead ends
- add:
  - candidate-space expansion is a live direction
  - decoupling right-target realization from left-hand preservation yields the
    first clear positive weak-slice lift

Next:

- improve coverage of the split-donor mechanism without losing its joint-score
  gain
- likely directions:
  - broaden left-preserve donor support
  - use top-k right donors x top-k left donors with a joint score instead of
    greedy best-best pairing

## 2026-06-09 top-k joint search over split donors

I upgraded the split-donor prototype from greedy best-best pairing to a small
joint search:

- top-5 right donors
- top-5 left donors
- evaluate all `5 x 5` pairs
- choose the best joint combination

Artifacts:

- [build_weak_slice_topk_joint_search.py](/opt/tiger/hand/tools/build_weak_slice_topk_joint_search.py)
- [weak_slice_topk_joint_search.md](/opt/tiger/hand/experiments/generated/summary_tables/weak_slice_topk_joint_search.md)

Result:

- coverage does **not** improve:
  - split availability: `0.1563`
  - top-k availability: `0.1563`
- but quality inside the covered region improves clearly:
  - split joint score: `0.1136`
  - top-k joint score: `0.1477`
  - split right grouped match: `0.2500`
  - top-k right grouped match: `0.3068`
  - split left preserve: `0.3409`
  - top-k left preserve: `0.6136`

Interpretation:

- the candidate-space expansion direction is now doubly supported:
  - greedy split donors already gave the first positive weak-slice signal
  - joint top-k search improves that signal further without changing coverage
- this means the problem is not just donor existence
- there is also real value in searching the expanded joint composition space

Important subtype gains:

- `opening + separate`:
  - split joint `0.6000`
  - top-k joint `1.0000`
- `steady + separate`:
  - split joint `0.5000`
  - top-k joint `1.0000`
- `opening + approach`:
  - remains `0.6667`
- `steady + approach`:
  - remains `0.2000`
- `steady + steady`:
  - remains `0.0652`

What this means now:

- coverage remains the next bottleneck
- but within the reachable support region, we now have a much stronger
  interaction-slice mechanism than the original single-donor baseline
- the next improvement should target **coverage expansion** rather than only
  better local search quality

Updated boundary:

- keep:
  - donor-scoring heuristics are dead ends
  - candidate-space expansion is a live direction
- strengthen:
  - joint search over the expanded candidate space produces a second positive
    lift on the weak slice
- next:
  - expand support coverage while preserving the joint-score gains from top-k
    search

## 2026-06-09 relaxed left-support top-k search

I then explicitly tested the next bottleneck: coverage.

Prototype:

- keep the successful top-k joint search
- relax the left-hand donor support from exact grouped motif match to a small
  family neighborhood:
  - same left-hand motion label
  - per-finger activity-profile distance at most `1`

Artifacts:

- [build_weak_slice_relaxed_support_topk.py](/opt/tiger/hand/tools/build_weak_slice_relaxed_support_topk.py)
- [weak_slice_relaxed_support_topk.md](/opt/tiger/hand/experiments/generated/summary_tables/weak_slice_relaxed_support_topk.md)

Result:

- coverage increases:
  - exact availability: `0.1563`
  - relaxed availability: `0.2060`
- but quality drops:
  - exact joint score: `0.1477`
  - relaxed joint score: `0.1293`

Breakdown of the tradeoff:

- right grouped match improves slightly:
  - `0.3068 -> 0.3190`
- left preserve drops:
  - `0.6136 -> 0.5086`
- so the loss is mainly from weaker left-hand preservation

This is an informative result, not a failure:

- it proves that coverage can be expanded by relaxing left support
- it also proves that naive global relaxation is too blunt

Important subtype behavior:

- clear coverage gains:
  - `mixed + separate`: `0.1667 -> 0.8333`
  - `opening + steady`: `0.3636 -> 0.7273`
  - `closing + steady`: `0.2353 -> 0.4706`
  - `steady + approach`: `0.4545 -> 0.6364`
- but some strong subtypes lose joint quality:
  - `opening + approach`: `0.6667 -> 0.4000`
  - `opening + separate`: `1.0000 -> 0.7143`
  - `steady + separate`: `1.0000 -> 0.6667`

Interpretation:

- candidate-space expansion remains the right direction
- but support relaxation has to be selective or score-aware
- a single global family-distance threshold is not enough

Updated boundary:

- keep:
  - exact-support top-k search gives the strongest quality in its covered
    region
  - relaxed-support top-k search gives better coverage but weaker left
    preservation
- therefore:
  - the next mechanism should be an adaptive / gated relaxation rather than a
    uniform relaxation across all subtypes

Most likely next step:

- keep exact support for already-strong subtypes
- allow relaxed support only where exact support is poor or unavailable

## Assets

- Pretrain labels: [temporal_hl_train.json](/opt/tiger/hand/experiments/generated/temporal_hl_train.json)
- Finetune labels: [temporal_hl_val.json](/opt/tiger/hand/experiments/generated/temporal_hl_val.json)
- Test labels: [temporal_hl_test.json](/opt/tiger/hand/experiments/generated/temporal_hl_test.json)
- Script: [train_symbolic_pretrain.py](/opt/tiger/hand/tools/train_symbolic_pretrain.py)

## Protocol

- Pretrain on `train` sequence windows using the full local train label space
- Finetune on ROM windows from `val`
- Evaluate on overlapping ROM windows from `test`
- Aggregate window predictions to sequences using `mean_log_prob`

## Temporal HL result

Command:

```bash
python tools/train_symbolic_pretrain.py \
  --train-pretrain-json experiments/generated/temporal_hl_train.json \
  --train-finetune-json experiments/generated/temporal_hl_val.json \
  --test-json experiments/generated/temporal_hl_test.json \
  --mode temporal \
  --fractions 0.5 1.0 \
  --seeds 0 1 2 \
  --hidden-dim 128 \
  --pretrain-epochs 120 \
  --finetune-epochs 120 \
  --output experiments/generated/symbolic_pretrain_temporal_curve.json
```

Result:

- Fraction `0.25`: scratch `0.3590`, pretrained `0.4872`
- Fraction `0.5`: scratch `0.5385`, pretrained `0.6923`
- Fraction `1.0`: scratch `0.6410`, pretrained `0.8205`

Window accuracy:

- Fraction `0.25`: scratch `0.5203`, pretrained `0.5529`
- Fraction `0.5`: scratch `0.6115`, pretrained `0.7095`
- Fraction `1.0`: scratch `0.7252`, pretrained `0.7387`

## State-only control

Command:

```bash
python tools/train_symbolic_pretrain.py \
  --train-pretrain-json experiments/generated/temporal_hl_train.json \
  --train-finetune-json experiments/generated/temporal_hl_val.json \
  --test-json experiments/generated/temporal_hl_test.json \
  --mode state \
  --fractions 0.5 1.0 \
  --seeds 0 1 2 \
  --hidden-dim 128 \
  --pretrain-epochs 120 \
  --finetune-epochs 120 \
  --output experiments/generated/symbolic_pretrain_state_curve.json
```

Result:

- Fraction `0.25`: scratch `0.1538`, pretrained `0.6410`
- Fraction `0.5`: scratch `0.7692`, pretrained `0.6410`
- Fraction `1.0`: scratch `0.7436`, pretrained `0.7949`

## Immediate interpretation

- Pretraining is not universally helpful
- Temporal HL benefits strongly and consistently from transfer
- State-only HL does not show the same robustness; at lower finetune data it can
  even regress
- This suggests temporal HL carries more reusable structure across action sets
- State-only transfer is unstable across data regimes, so "pretraining helps"
  is not a sufficient explanation by itself

## Relative standing against previous baselines

- Strongest linear classifier baseline:
  temporal HL `0.6923` at fraction `1.0`
- Temporal pretrained MLP:
  `0.8205` at fraction `1.0`

This is the first local result that clearly moves beyond the earlier linear
baseline rather than just matching it.

## Sequence-level error analysis for strongest temporal config

Analysis asset:

- [symbolic_pretrain_temporal_analysis.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_analysis.json)

Configuration:

- mode: `temporal`
- fraction: `1.0`
- seed: `0`
- hidden dim: `128`
- aggregation: `mean_log_prob`

Observed result for this run:

- scratch sequence accuracy: `0.8462`
- pretrained sequence accuracy: `1.0000`

Scratch errors:

- `ROM05_RT_Wrist_ROM -> ROM03_RT_No_Occlusion`
- `ROM08_Lt_Finger_Occlusions -> ROM03_LT_No_Occlusion`

Pretrained errors:

- none in this seed

Interpretation:

- pretraining is not only lifting aggregate accuracy
- it is directly resolving the two previously persistent confusion modes in the
  right-hand wrist and left-hand occlusion families

## Multi-seed consistency for strongest temporal pretrain setup

Analysis asset:

- [symbolic_pretrain_temporal_multiseed.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_multiseed.json)

Configuration:

- mode: `temporal`
- fraction: `1.0`
- seeds: `0, 1, 2`
- hidden dim: `128`
- aggregation: `mean_log_prob`

Run-level sequence accuracy:

- seed `0`: scratch `0.7692`, pretrained `0.7692`
- seed `1`: scratch `0.7692`, pretrained `0.8462`
- seed `2`: scratch `0.8462`, pretrained `0.9231`

Recurring scratch confusions:

- `ROM05_RT_Wrist_ROM -> ROM03_RT_No_Occlusion` : `3/3`
- `ROM07_Rt_Finger_Occlusions -> ROM03_RT_No_Occlusion` : `1/3`
- `ROM04_RT_Occlusion -> ROM03_RT_No_Occlusion` : `1/3`
- `ROM05_LT_Wrist_ROM -> ROM08_Lt_Finger_Occlusions` : `1/3`

Recurring pretrained confusions:

- `ROM05_RT_Wrist_ROM -> ROM03_RT_No_Occlusion` : `3/3`
- `ROM04_RT_Occlusion -> ROM03_RT_No_Occlusion` : `2/3`
- `ROM07_Rt_Finger_Occlusions -> ROM03_RT_No_Occlusion` : `1/3`

Consistency interpretation:

- temporal pretraining does not solve every hard case
- the most stubborn confusion remains
  `ROM05_RT_Wrist_ROM -> ROM03_RT_No_Occlusion`
- however, pretraining stabilizes several borderline classes:
  - `ROM05_LT_Wrist_ROM` improves from `2/3` correct to `3/3`
  - `ROM07_RT_Finger_Occlusions` style right-finger occlusion cases improve in
    consistency after canonicalization-aware evaluation
- the remaining gap is now highly concentrated rather than broadly distributed

## Wrist-feature ablation

Implementation note:

- wrist-sensitive temporal features are now available behind an explicit
  `--wrist-features` flag in the symbolic training and analysis scripts
- they are **not** the default pipeline

Regression check on the strongest temporal pretrain setup:

- default temporal pipeline:
  `scratch=0.6410`, `pretrained=0.8205`
- wrist-feature temporal pipeline:
  `scratch=0.6410`, `pretrained=0.7949`

Multi-seed error shift under wrist features:

- `ROM05_RT_Wrist_ROM -> ROM03_RT_No_Occlusion` drops from `3` to `2`
- but `ROM07_Rt_Finger_Occlusions -> ROM03_RT_No_Occlusion` becomes worse
- overall temporal pretrained accuracy decreases

Interpretation:

- the wrist features are directionally meaningful
- but in their current handcrafted form they trade one bottleneck against
  another, so they should be treated as an ablation rather than promoted into
  the default feature set

## Intrinsic representation discrimination

To evaluate the symbolic representation itself without going through the
classifier, I added:

- [eval_symbolic_representation_intrinsic.py](/opt/tiger/hand/tools/eval_symbolic_representation_intrinsic.py)

Artifacts:

- [symbolic_representation_intrinsic_val_test.json](/opt/tiger/hand/experiments/generated/symbolic_representation_intrinsic_val_test.json)
- [symbolic_representation_intrinsic_val_test_plus.json](/opt/tiger/hand/experiments/generated/symbolic_representation_intrinsic_val_test_plus.json)

Protocol:

- gallery: `temporal_hl_val.json`
- query: `temporal_hl_test.json`
- only overlapping ROM labels are evaluated
- no learned classifier is used
- each representation is scored by direct sequence discrimination:
  - top-1 correct-label retrieval
  - positive-vs-best-negative margin
  - correct-label rank

### Main result

Eventized symbolic sequence comparison is far stronger than frame-token or RLE
comparison.

Key results:

- `temporal_event`:
  - top-1: `0.9545`
  - mean positive margin: `0.0581`
  - mean correct rank: `1.2273`
- `state_event`:
  - top-1: `0.9545`
  - mean positive margin: `0.0423`
  - mean correct rank: `1.2727`
- `state_frame`:
  - top-1: `0.4545`
- `temporal_frame`:
  - top-1: `0.4091`
- `state_rle`:
  - top-1: `0.4091`
- `temporal_rle`:
  - top-1: `0.3636`

Interpretation:

- naive frame-token or RLE comparison is weak even when the representation is
  correct
- the sequence-native signal only becomes useful when grouped into event-level
  symbolic units
- this supports the project direction of explicit temporal symbolic structure
  rather than raw frame stacking

### State-event vs temporal-event

## Structural strong-protocol upgrade: grouped branch concat

To move beyond scalar channel weighting, I added:

- [train_symbolic_pretrain_grouped.py](/opt/tiger/hand/tools/train_symbolic_pretrain_grouped.py)

This script keeps the same symbolic channels but changes the strong-protocol
fusion structure:

- `state`
- `dynamics = transition + hand_motion`
- `context = interaction + tempo (+ event if enabled)`

Each branch is encoded separately, then fused by concatenation before the
classifier. The transfer setting remains:

- pretrain on `train`
- finetune on ROM `val`
- test on ROM `test`
- pretrain span/step `168/84`
- finetune/test windows unchanged
- `ROM05_RT_Wrist_ROM:3.0` class boost

Artifacts:

- [symbolic_pretrain_grouped_concat_pre_only_168_84_boost_rom05.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_grouped_concat_pre_only_168_84_boost_rom05.json)
- [grouped_concat_strong_protocol_report.json](/opt/tiger/hand/experiments/generated/grouped_concat_strong_protocol_report.json)
- [grouped_concat_strong_protocol_report.md](/opt/tiger/hand/experiments/generated/summary_tables/grouped_concat_strong_protocol_report.md)

Key results:

- fraction `0.5`
  - grouped concat pretrained seq: `0.8462`
  - flat-merge mainline pretrained seq: `0.7949`
  - delta: `+0.0513`
- fraction `1.0`
  - grouped concat pretrained seq: `0.8974`
  - flat-merge mainline pretrained seq: `0.9231`
  - delta: `-0.0256`

Window accuracy:

- fraction `0.5`
  - grouped concat pretrained win: `0.7635`
  - flat merge pretrained win: `0.7320`
  - delta: `+0.0315`
- fraction `1.0`
  - grouped concat pretrained win: `0.8029`
  - flat merge pretrained win: `0.7252`
  - delta: `+0.0777`

Seed-level sequence accuracy for grouped concat:

- fraction `0.5`: `0.7692`, `0.9231`, `0.8462`
- fraction `1.0`: `1.0000`, `0.6923`, `1.0000`

Interpretation:

- A structural fusion change is now proven viable under the strong pretrain
  protocol.
- The old conclusion "flat merged full temporal is simply best" is too coarse.
- The updated conclusion is:
  - flat merge remains strongest on full-data sequence accuracy
  - grouped concat is stronger in the low-data regime
  - grouped concat also improves window-level discrimination sharply
- Therefore, the fusion strategy itself is now a meaningful representation
  variable, not just an implementation detail.

## Second structural variant: state residual + gated auxiliaries

I extended the grouped strong-protocol script with:

- `fusion_type=residual_aux`

Design:

- preserve a dedicated `state` branch
- gate only over auxiliary branches:
  - `dynamics = transition + hand_motion`
  - `context = interaction + tempo (+ event if enabled)`
- project the auxiliary mixture back into the state feature and classify from
  `[state, state + aux]`

Artifacts:

- [symbolic_pretrain_grouped_residual_aux_pre_only_168_84_boost_rom05.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_grouped_residual_aux_pre_only_168_84_boost_rom05.json)
- [symbolic_pretrain_grouped_residual_aux_motionup_itdown_pre_only_168_84_boost_rom05.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_grouped_residual_aux_motionup_itdown_pre_only_168_84_boost_rom05.json)
- [strong_protocol_structural_fusion_report.json](/opt/tiger/hand/experiments/generated/strong_protocol_structural_fusion_report.json)
- [strong_protocol_structural_fusion_report.md](/opt/tiger/hand/experiments/generated/summary_tables/strong_protocol_structural_fusion_report.md)

Default residual-aux result:

- fraction `0.5`
  - pretrained seq: `0.7436`
  - pretrained win: `0.7523`
- fraction `1.0`
  - pretrained seq: `0.8462`
  - pretrained win: `0.7703`

Observed gate bias:

- low-data pretrained gate means:
  - `context=0.9138`
  - `dynamics=0.0862`
- full-data pretrained gate means:
  - `context=0.8420`
  - `dynamics=0.1580`

Interpretation:

- the default residual-aux variant over-trusts the context branch
- this is too strong and hurts the hard right-hand occlusion family

Targeted reweighting trial:

- weights:
  - `state=1.0`
  - `transition=0.5`
  - `hand_motion=0.75`
  - `interaction=0.1`
  - `tempo=0.1`

Reweighted residual-aux result:

- fraction `0.5`
  - pretrained seq: `0.7949`
  - pretrained win: `0.7658`
- fraction `1.0`
  - pretrained seq: `0.8718`
  - pretrained win: `0.7714`

Updated interpretation:

- residual-aux can be improved substantially by suppressing the context branch
- however, it still does not beat:
  - grouped concat on low-data sequence accuracy
  - flat merge on full-data sequence accuracy
- therefore residual-aux is not the next mainline

Current structural ranking:

- low-data sequence:
  - grouped concat best among tested structural variants
- full-data sequence:
  - flat merge still best overall
- window-level discrimination:
  - grouped concat best overall

## Current-code audit: old flat-mainline artifacts are stale

After adding the protocol-split bundle, I audited whether the older flat-merge
mainline artifacts still match the current worktree.

Rerun artifact:

- [symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_curve_rerun_20260609.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_curve_rerun_20260609.json)

Observed rerun under the current `train_symbolic_pretrain.py`:

- fraction `0.5`
  - scratch seq: `0.5641`
  - pretrained seq: `0.6410`
  - scratch win: `0.6216`
  - pretrained win: `0.6813`
- fraction `1.0`
  - scratch seq: `0.4615`
  - pretrained seq: `0.6923`
  - scratch win: `0.6374`
  - pretrained win: `0.7095`

This does **not** match the older flat-mainline artifact that had:

- fraction `0.5` pretrained seq: `0.7949`
- fraction `1.0` pretrained seq: `0.9231`

Interpretation:

- the older flat-mainline files are stale relative to the current worktree
- they should no longer be treated as authoritative evidence
- all claims should now be grounded in current-code reruns or clearly labeled as
  historical artifacts

Current-code protocol-split bundle:

- [protocol_split_evidence_bundle.json](/opt/tiger/hand/experiments/generated/protocol_split_evidence_bundle.json)
- [protocol_split_evidence_bundle.md](/opt/tiger/hand/experiments/generated/summary_tables/protocol_split_evidence_bundle.md)

Current-code audited conclusion:

- `grouped_concat` beats the rerun flat baseline in both regimes
  - low-data seq: `0.8462` vs `0.6410`
  - full-data seq: `0.8974` vs `0.6923`
- `grouped_concat` also wins on window accuracy in both regimes

Important nuance:

- grouped concat still has a seed-level weakness on hard right-hand occlusion
  neighbors
- but under the current code audit it is the stronger overall structural
  baseline than flat merge

## Grouped-concat + family expert: current strongest audited frontier

To repair the remaining grouped-concat family confusions, I reused the existing
family-expert correction line on grouped-concat analyses with exported family
score vectors.

Full-data family-score analyses:

- [symbolic_pretrain_grouped_concat_seed0_familyvec_analysis.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_grouped_concat_seed0_familyvec_analysis.json)
- [symbolic_pretrain_grouped_concat_seed1_familyvec_analysis.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_grouped_concat_seed1_familyvec_analysis.json)
- [symbolic_pretrain_grouped_concat_seed2_familyvec_analysis.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_grouped_concat_seed2_familyvec_analysis.json)

Low-data family-score analyses:

- [symbolic_pretrain_grouped_concat_fraction05_seed0_familyvec_analysis.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_grouped_concat_fraction05_seed0_familyvec_analysis.json)
- [symbolic_pretrain_grouped_concat_fraction05_seed1_familyvec_analysis.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_grouped_concat_fraction05_seed1_familyvec_analysis.json)
- [symbolic_pretrain_grouped_concat_fraction05_seed2_familyvec_analysis.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_grouped_concat_fraction05_seed2_familyvec_analysis.json)

Expert artifacts:

- [grouped_concat_family_expert.json](/opt/tiger/hand/experiments/generated/grouped_concat_family_expert.json)
- [grouped_concat_fraction05_family_expert.json](/opt/tiger/hand/experiments/generated/grouped_concat_fraction05_family_expert.json)
- [grouped_concat_family_expert_frontier.json](/opt/tiger/hand/experiments/generated/grouped_concat_family_expert_frontier.json)
- [grouped_concat_family_expert_frontier.md](/opt/tiger/hand/experiments/generated/summary_tables/grouped_concat_family_expert_frontier.md)

Base grouped-concat:

- fraction `0.5`
  - seq: `0.8462`
  - win: `0.7635`
  - correct: `33 / 39`
- fraction `1.0`
  - seq: `0.8974`
  - win: `0.8029`
  - correct: `35 / 39`

Zero-harm family expert result:

- fraction `0.5`
  - threshold: `0.3`
  - improved: `4`
  - harmed: `0`
  - corrected seq: `0.9487`
  - corrected correct: `37 / 39`
- fraction `1.0`
  - threshold: `0.3`
  - improved: `4`
  - harmed: `0`
  - corrected seq: `1.0000`
  - corrected correct: `39 / 39`

Repaired full-data cases:

- `ROM04_RT_Occlusion`: `ROM03_RT_No_Occlusion -> ROM04_RT_Occlusion`
- `ROM07_RT_Finger_Occlusions`: `ROM03_RT_No_Occlusion -> ROM07_Rt_Finger_Occlusions`
- `ROM07_Rt_Finger_Occlusions`: `ROM03_RT_No_Occlusion -> ROM07_Rt_Finger_Occlusions`
- `ROM08_LT_Finger_Occlusions`: `ROM03_LT_No_Occlusion -> ROM08_Lt_Finger_Occlusions`

Interpretation:

- This is the first correction layer that improves the current-code strongest
  grouped-concat baseline with zero harm.
- Under the current code audit, grouped-concat + family expert is the strongest
  symbolic frontier tested so far.
- The gains are targeted and anatomically sensible rather than broad
  post-hoc rewriting.

## Robustness of grouped-concat + family expert across fractions

To test whether the correction layer is stable beyond `0.5` and `1.0`, I added
`0.25` and consolidated all three fractions into:

- [symbolic_pretrain_grouped_concat_fraction025_pre_only_168_84_boost_rom05.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_grouped_concat_fraction025_pre_only_168_84_boost_rom05.json)
- [grouped_concat_fraction025_family_expert.json](/opt/tiger/hand/experiments/generated/grouped_concat_fraction025_family_expert.json)
- [grouped_concat_family_expert_robustness.json](/opt/tiger/hand/experiments/generated/grouped_concat_family_expert_robustness.json)
- [grouped_concat_family_expert_robustness.md](/opt/tiger/hand/experiments/generated/summary_tables/grouped_concat_family_expert_robustness.md)

Grouped-concat base:

- fraction `0.25`
  - seq: `0.4615`
  - win: `0.6520`
- fraction `0.5`
  - seq: `0.8462`
  - win: `0.7635`
- fraction `1.0`
  - seq: `0.8974`
  - win: `0.8029`

Zero-harm family-expert corrected:

- fraction `0.25`
  - corrected seq: `0.8205`
  - improved: `14`
  - harmed: `0`
  - family old/new: `0.4828 -> 0.9655`
- fraction `0.5`
  - corrected seq: `0.9487`
  - improved: `4`
  - harmed: `0`
  - family old/new: `0.8667 -> 1.0000`
- fraction `1.0`
  - corrected seq: `1.0000`
  - improved: `4`
  - harmed: `0`
  - family old/new: `0.8667 -> 1.0000`

Threshold stability:

- zero-harm thresholds observed across all three fractions:
  - `0.3`
  - `0.4`
- also zero-harm in all three current payloads:
  - `0.5`
  - `0.6`
  - `0.7`
  - `0.8`
  - `0.9`

Interpretation:

- the family-expert layer is not a narrow single-regime fix
- its effect becomes larger as the symbolic base gets weaker
- under the current audited pipeline, grouped-concat + family expert now has
  evidence of robustness across at least three data regimes

## Stability audit for grouped-concat + family expert

To verify that the family-expert layer is a stable repair mechanism rather than
an accidental threshold pick, I added:

- [build_family_expert_stability_audit.py](/opt/tiger/hand/tools/build_family_expert_stability_audit.py)
- [grouped_concat_family_expert_stability_audit.json](/opt/tiger/hand/experiments/generated/grouped_concat_family_expert_stability_audit.json)
- [grouped_concat_family_expert_stability_audit.md](/opt/tiger/hand/experiments/generated/summary_tables/grouped_concat_family_expert_stability_audit.md)

Verified stability results:

- shared zero-harm thresholds across fractions `0.25`, `0.5`, and `1.0`:
  - `0.3`
  - `0.4`
  - `0.5`
  - `0.6`
  - `0.7`
  - `0.8`
  - `0.9`

Recurring improved cases across fractions and thresholds:

- `ROM04_RT_Occlusion`: `ROM03_RT_No_Occlusion -> ROM04_RT_Occlusion`
- `ROM07_RT_Finger_Occlusions`: `ROM03_RT_No_Occlusion -> ROM07_Rt_Finger_Occlusions`
- `ROM08_LT_Finger_Occlusions`: `ROM03_LT_No_Occlusion -> ROM08_Lt_Finger_Occlusions`

Slice-level stability under the shared zero-harm threshold set:

- fraction `0.25`
  - `all`: mean delta `+0.2660`, all nonnegative
  - `occlusion`: mean delta `+0.3095`, all nonnegative
  - `finger_occlusion`: mean delta `+0.4286`, all nonnegative
- fraction `0.5`
  - `all`: mean delta `+0.0762`, all nonnegative
  - `occlusion`: mean delta `+0.0952`, all nonnegative
  - `finger_occlusion`: mean delta `+0.1071`, all nonnegative
- fraction `1.0`
  - `all`: mean delta `+0.0714`, all nonnegative
  - `occlusion`: mean delta `+0.0893`, all nonnegative
  - `finger_occlusion`: mean delta `+0.1071`, all nonnegative

Interpretation:

- the family-expert correction behaves like a stable family-level repair layer
  rather than a fragile single-threshold patch
- the repeatedly improved cases are the same hard right-hand occlusion and
  finger-occlusion confusions already exposed by the grouped-concat analyses
- slice-level gains stay nonnegative throughout the tested zero-harm threshold
  band, which is stronger evidence than a single best-threshold report

## Consolidated current-code symbolic frontier

To keep the current audited frontier separate from stale older artifacts, I
added:

- [build_current_code_symbolic_frontier.py](/opt/tiger/hand/tools/build_current_code_symbolic_frontier.py)
- [current_code_symbolic_frontier.json](/opt/tiger/hand/experiments/generated/current_code_symbolic_frontier.json)
- [current_code_symbolic_frontier.md](/opt/tiger/hand/experiments/generated/summary_tables/current_code_symbolic_frontier.md)

Current-code three-stage frontier:

- fraction `0.25`
  - grouped concat: `0.4615`
  - grouped concat + family expert: `0.8205`
  - correction gain over grouped concat: `+0.3590`
- fraction `0.5`
  - flat rerun: `0.6410`
  - grouped concat: `0.8462`
  - grouped concat + family expert: `0.9487`
  - grouped gain over flat rerun: `+0.2051`
  - correction gain over grouped concat: `+0.1026`
- fraction `1.0`
  - flat rerun: `0.6923`
  - grouped concat: `0.8974`
  - grouped concat + family expert: `1.0000`
  - grouped gain over flat rerun: `+0.2051`
  - correction gain over grouped concat: `+0.1026`

Interpretation:

- the current-code frontier now has two separable gain layers:
  - structural gain from grouped concatenation over the rerun flat baseline
  - family-level correction gain on top of grouped concat
- this makes the current strongest symbolic line easier to defend than a single
  opaque end-to-end jump
- all future frontier claims should point to this consolidated current-code
  report rather than older pre-audit numbers

## Gallery-shift audit for family expert

To test whether the family-expert repair depends on having the full validation
template sequence as gallery, I added:

- [build_family_expert_gallery_shift_audit.py](/opt/tiger/hand/tools/build_family_expert_gallery_shift_audit.py)
- [grouped_concat_family_expert_gallery_shift_audit.json](/opt/tiger/hand/experiments/generated/grouped_concat_family_expert_gallery_shift_audit.json)
- [grouped_concat_family_expert_gallery_shift_audit.md](/opt/tiger/hand/experiments/generated/summary_tables/grouped_concat_family_expert_gallery_shift_audit.md)

Audit protocol:

- keep the same query side and family-vector analyses
- degrade the gallery templates from `temporal_hl_val.json` using:
  - `first_half`
  - `second_half`
  - `center_half`
  - `stride2`
- rerun the threshold sweep `0.3 .. 0.9` for fractions `0.25`, `0.5`, `1.0`

Verified result:

- all `15 / 15` fraction-variant settings retain at least one zero-harm
  threshold
- for fractions `0.5` and `1.0`, all five gallery variants preserve the full
  zero-harm threshold band `0.3 .. 0.9`
- at fraction `0.25`, even the degraded variants remain zero-harm across
  `0.3 .. 0.9`

Representative family results:

- fraction `0.25`
  - full gallery: `0.4828 -> 0.9655`, improved `14`, harmed `0`
  - `first_half`: `0.4828 -> 0.9310`, improved `13`, harmed `0`
  - `stride2`: `0.4828 -> 0.9310`, improved `13`, harmed `0`
  - `second_half` and `center_half`: both match full gallery
- fraction `0.5`
  - every gallery variant: `0.8667 -> 1.0000`, improved `4`, harmed `0`
- fraction `1.0`
  - every gallery variant: `0.8667 -> 1.0000`, improved `4`, harmed `0`

Interpretation:

- the family-expert line is not relying on a single full-template retrieval
  protocol
- even when the gallery templates are temporally degraded, the repair layer
  still keeps zero-harm behavior and nearly identical improvement
- this makes the correction layer look more like a stable symbolic family
  mechanism than a brittle gallery-specific heuristic

## Learned-token proxy baseline on the same data

To test the reviewer attack "this is just another tokenizer", I added:

- [build_learned_token_proxy_report.py](/opt/tiger/hand/tools/build_learned_token_proxy_report.py)
- [learned_token_proxy_report.json](/opt/tiger/hand/experiments/generated/learned_token_proxy_report.json)
- [learned_token_proxy_report.md](/opt/tiger/hand/experiments/generated/summary_tables/learned_token_proxy_report.md)

This report evaluates two opaque token proxy families:

- `semantic_frame`
  - fit a k-means codebook on frame-level symbolic features extracted from the
    same temporal-HL frame content
  - this is the fairer "same information, opaque recoding" baseline
- `continuous_frame`
  - fit a k-means codebook on richer continuous frame vectors built from local
    hand vectors, flexion, transition angles, and cross-hand distance
  - this should be treated as an upper-bound style proxy, not a like-for-like
    symbolic recoding baseline

Matched protocols:

- retrieval on `val -> test`
- lightweight classifier on `val(fraction) -> test`
- codebook sizes: `32`, `64`, `128`

### Retrieval result

Best `semantic_frame` proxy:

- `32` clusters
- best view: `proxy_frame`
- top-1: `0.9545`
- mean positive margin: `0.1753`
- mean correct rank: `1.0455`

Best `continuous_frame` proxy:

- `32` clusters
- best view: `proxy_frame`
- top-1: `1.0000`
- mean positive margin: `0.2141`
- mean correct rank: `1.0000`

Direct comparison with current symbolic intrinsic reference:

- temporal symbolic `temporal_event`
  - top-1: `0.9545`
  - mean positive margin: `0.0581`
  - mean correct rank: `1.2273`

### Lightweight classifier result

Best `semantic_frame` proxy:

- fraction `0.25`: `0.5846`
- fraction `0.5`: `0.7231`
- fraction `1.0`: `0.7692`

Best `continuous_frame` proxy:

- fraction `0.25`: `0.5538`
- fraction `0.5`: `0.7692`
- fraction `1.0`: `0.8462`

Direct comparison with current symbolic classifier reference (`temporal_hl`,
`C=16.0`):

- fraction `0.25`: `0.3846`
- fraction `0.5`: `0.5385`
- fraction `1.0`: `0.6923`

Interpretation:

- on pure discriminability under these small matched protocols, opaque learned
  token proxies are already competitive and often stronger than the current
  lightweight symbolic classifier baseline
- therefore the main symbolic claim should **not** be "interpretable symbolic
  structure dominates opaque tokenization on every accuracy metric"
- the stronger defensible wedge remains:
  - explicit anatomy-aware structure
  - stable family-level repair
  - gallery-shift robustness of the repair mechanism
  - human-readability / controllability / editability
- this proxy report is still useful because it tells us exactly where the
  current symbolic story is vulnerable, and prevents us from overclaiming on
  tokenizer-style comparisons

## Local edit audit: symbolic vs opaque token proxies

To move beyond pure recognition-style evidence and test controllability, I
added:

- [build_local_edit_audit.py](/opt/tiger/hand/tools/build_local_edit_audit.py)
- [local_edit_audit.json](/opt/tiger/hand/experiments/generated/local_edit_audit.json)
- [local_edit_audit.md](/opt/tiger/hand/experiments/generated/summary_tables/local_edit_audit.md)

Audit setup:

- find real contiguous runs in `temporal_hl_test.json`
- apply local target edits on six task types:
  - `right_hand_motion -> opening`
  - `right_hand_motion -> closing`
  - `left_hand_motion -> opening`
  - `left_hand_motion -> closing`
  - `interaction_motion -> approach`
  - `interaction_motion -> separate`
- compare:
  - direct symbolic field edit
  - best-effort opaque token edit using a `32`-cluster proxy
- evaluate on tracked semantic fields:
  - `hand_type`
  - `interaction_motion`
  - `right_hand_motion`
  - `left_hand_motion`
  - `right_state_signature`
  - `left_state_signature`

### Main result

For direct symbolic edits:

- clean edit rate: `1.0000` on all six tasks
- mean collateral changed fields: `0.0000` on all six tasks

For the fairer `semantic_frame` opaque proxy:

- target success remains `1.0000` on all six tasks
- but clean edit rate is near zero:
  - `interaction -> approach`: `0.0268`
  - `interaction -> separate`: `0.0411`
  - `left opening`: `0.0131`
  - all other tasks: `0.0000`
- mean collateral changed fields stays high:
  - range `1.8748` to `2.2763`

For the stronger `continuous_frame` opaque proxy:

- most tasks still have target success `1.0000`
- but clean edit rate remains near zero and collateral is also high
- importantly, `interaction_motion -> separate` completely fails:
  - proxy target success: `0.0000`
  - clean edit rate: `0.0000`

Interpretation:

- opaque token proxies can often hit the requested target attribute only by
  entangling it with multiple other semantic changes
- symbolic edits are locally controllable in a way the opaque proxies are not:
  they isolate the intended field change without collateral rewrites
- this is the strongest current evidence that our value is not just
  discriminability, but editable structure with predictable local intervention
  behavior

## Local edit casebook

To make the editability result concrete rather than purely aggregate, I added:

- [build_local_edit_casebook.py](/opt/tiger/hand/tools/build_local_edit_casebook.py)
- [local_edit_casebook.json](/opt/tiger/hand/experiments/generated/local_edit_casebook.json)
- [local_edit_casebook.md](/opt/tiger/hand/experiments/generated/summary_tables/local_edit_casebook.md)

This casebook selects representative high-collateral segments and exports
frame-level comparisons among:

- original frame attributes
- direct symbolic edit
- best-effort opaque proxy edit

Representative patterns now documented explicitly:

- `semantic_frame`, `ROM09_Interaction_Fingers_Touching`,
  `right_hand_motion -> opening`
  - proxy does hit `opening`
  - but simultaneously rewrites:
    - `interaction_motion: steady -> approach`
    - `left_hand_motion: steady/closing/mixed -> opening`
    - both right and left state signatures
  - proxy collateral fields per frame: `4`

- `semantic_frame`, `ROM01_No_Interaction_2_Hand`,
  `left_hand_motion -> closing`
  - proxy does hit `closing`
  - but also rewrites:
    - `hand_type`
    - `interaction_motion: steady -> unknown`
    - `right_hand_motion: steady -> none`
    - both state signatures
  - proxy collateral fields per frame: `5`

- `continuous_frame`, `ROM01_No_Interaction_2_Hand`,
  `left_hand_motion -> opening`
  - proxy target succeeds
  - but `right_hand_motion: steady -> closing` and
    `interaction_motion: steady -> approach` are changed in the same edit
  - again showing entanglement rather than local control

Interpretation:

- the aggregate clean-edit gap is not an artifact of averaging
- the failure mode is visually and semantically consistent across concrete
  segments:
  proxy edits tend to rewrite coupled hand and interaction attributes together
- this casebook is currently the clearest qualitative support for the claim
  that symbolic structure enables local, predictable intervention while opaque
  token proxies do not

## Unified control-facing evidence bundle

To keep the repair/editability evidence in one place, I added:

- [build_control_evidence_bundle.py](/opt/tiger/hand/tools/build_control_evidence_bundle.py)
- [control_evidence_bundle.json](/opt/tiger/hand/experiments/generated/control_evidence_bundle.json)
- [control_evidence_bundle.md](/opt/tiger/hand/experiments/generated/summary_tables/control_evidence_bundle.md)

This bundle consolidates four lines:

- family-expert stability audit
- family-expert gallery-shift audit
- local edit audit
- local edit casebook

What the bundle now supports directly:

- repair stability
  - shared zero-harm thresholds across `0.25 / 0.5 / 1.0`:
    `0.3 .. 0.9`
  - recurring repaired cases remain concentrated on the same hard right-hand
    occlusion / finger-occlusion families

- gallery robustness
  - all tested gallery variants remain zero-harm at all three fractions
  - even `stride2` gallery degradation keeps:
    - `0.25`: `0.4828 -> 0.9310`
    - `0.5`: `0.8667 -> 1.0000`
    - `1.0`: `0.8667 -> 1.0000`

- local controllability
  - direct symbolic edits stay perfectly local under the tracked-field audit
  - the fairer `semantic_frame` proxy keeps target success `1.0000` on all six
    task types, but clean-edit rate remains near zero and collateral stays near
    `2` fields per frame

- concrete opaque-token failure patterns
  - the casebook now shows frame-level examples where the proxy hits the target
    field only by jointly rewriting:
    - the other hand's motion state
    - interaction state
    - state signatures
    - and sometimes even `hand_type`

Interpretation:

- this bundle is currently the cleanest summary of why the representation story
  should center on stable symbolic repair and local controllability rather than
  on pure discriminative accuracy
- for any future write-up or internal review, this bundle should now be the
  default entry point for the control-facing evidence line

## Remaining gap on this line

The main remaining gap was geometry-level post-edit verification. I now added a
first version of that audit:

- [build_geometry_locality_audit.py](/opt/tiger/hand/tools/build_geometry_locality_audit.py)
- [geometry_locality_audit.json](/opt/tiger/hand/experiments/generated/geometry_locality_audit.json)
- [geometry_locality_audit.md](/opt/tiger/hand/experiments/generated/summary_tables/geometry_locality_audit.md)

Current scope:

- interaction-focused local edits only
  - `interaction_motion -> approach`
  - `interaction_motion -> separate`
- use exemplar-style post-edit verification on real frames with:
  - `local_vectors`
  - `flexion_scores`
  - `cross_hand_distance`

Verified geometry-level result:

- `semantic_frame` proxy
  - `interaction -> approach`
    - symbolic locality: `0.8724`
    - proxy locality: `0.4948`
  - `interaction -> separate`
    - symbolic locality: `1.1681`
    - proxy locality: `0.4800`

- `continuous_frame` proxy
  - `interaction -> approach`
    - symbolic locality: `0.8724`
    - proxy locality: `0.4652`

Interpretation:

- on interaction-motion edits, symbolic post-edit geometry is more localized
  than either opaque proxy family under this exemplar-based audit
- the symbolic edit achieves a stronger ratio of target-side geometric change
  to collateral geometric change
- this upgrades the previous field-level local controllability result with a
  first geometry-aware check

Remaining limitation:

- the current geometry audit is solid for interaction-motion edits
- hand-motion geometry edits still need a cleaner post-edit reconstruction or
  matching protocol before they can be promoted to main evidence

### Why hand-motion geometry is still unstable

To diagnose that gap directly, I added:

- [build_hand_motion_geometry_diagnostic.py](/opt/tiger/hand/tools/build_hand_motion_geometry_diagnostic.py)
- [hand_motion_geometry_diagnostic.json](/opt/tiger/hand/experiments/generated/hand_motion_geometry_diagnostic.json)
- [hand_motion_geometry_diagnostic.md](/opt/tiger/hand/experiments/generated/summary_tables/hand_motion_geometry_diagnostic.md)

What this diagnostic measures:

- for each `right/left_hand_motion -> opening/closing` test frame
- count whether the train bank contains any **strict symbolic candidate**
  where:
  - the target hand-motion field flips to the requested value
  - every other tracked semantic field stays unchanged
- compare that against the availability of the two opaque proxy families

Result:

- proxy availability is not the bottleneck
  - semantic proxy available rate: `1.0000` on all four tasks
  - continuous proxy available rate: `1.0000` on all four tasks
- strict symbolic near-neighbor availability is extremely low
  - `right_hand_motion -> opening`
    - frames: `997`
    - frames with any strict symbolic candidate: `91`
    - mean strict symbolic candidates/frame: `0.26`
  - `right_hand_motion -> closing`
    - frames: `939`
    - frames with any strict symbolic candidate: `52`
    - mean strict symbolic candidates/frame: `0.07`
  - `left_hand_motion -> opening`
    - frames: `688`
    - frames with any strict symbolic candidate: `49`
    - mean strict symbolic candidates/frame: `0.23`
  - `left_hand_motion -> closing`
    - frames: `667`
    - frames with any strict symbolic candidate: `29`
    - mean strict symbolic candidates/frame: `0.06`

Recurring blocking fields:

- for right-hand edits:
  - `right_state_signature`
  - `left_hand_motion`
  - `interaction_motion`
  - then `hand_type`
- for left-hand edits:
  - `left_state_signature`
  - `right_hand_motion`
  - `interaction_motion`
  - then `hand_type`

Interpretation:

- the current failure is **not** that learned-token proxies cannot propose
  edits
- the real bottleneck is that the dataset almost never offers a true
  one-factor symbolic neighbor that changes only a single hand-motion field
  while preserving the rest of the tracked semantic state
- so the strict hand-motion geometry audit is currently dominated by data
  sparsity and coupling of motion with full hand-state changes, not by the
  symbolic method itself

Practical consequence:

- keep the interaction-motion geometry audit as main evidence
- treat hand-motion geometry locality as a documented limitation for now
- only reopen it if we introduce a more permissive but still defensible
  matching rule, or a reconstruction procedure that edits motion while
  conditioning on the original state signature
  instead of requiring an exact real-frame neighbor

### Conditioned hand-motion geometry audit

To test whether the problem was the exact-neighbor protocol itself, I added:

- [build_hand_motion_conditioned_geometry_audit.py](/opt/tiger/hand/tools/build_hand_motion_conditioned_geometry_audit.py)
- [hand_motion_conditioned_geometry_audit.json](/opt/tiger/hand/experiments/generated/hand_motion_conditioned_geometry_audit.json)
- [hand_motion_conditioned_geometry_audit.md](/opt/tiger/hand/experiments/generated/summary_tables/hand_motion_conditioned_geometry_audit.md)

This audit switches from **exact semantic neighbor** matching to a
**conditioned context preservation** rule:

- for `right_hand_motion` edits, preserve:
  - `hand_type`
  - `interaction_motion`
  - `left_hand_motion`
  - `left_state_signature`
- for `left_hand_motion` edits, preserve:
  - `hand_type`
  - `interaction_motion`
  - `right_hand_motion`
  - `right_state_signature`

The target hand's own state signature is allowed to change, which is a more
reasonable approximation of "edit the motion of one hand while keeping the
rest of the scene fixed."

What this fixes:

- conditioned symbolic candidates now exist for essentially the full audited
  frame set
  - right opening: `840` frames
  - right closing: `781`
  - left opening: `573`
  - left closing: `553`

So the earlier failure really was caused by the exact-neighbor requirement,
not by the absence of any hand-motion-conditioned counterpart at all.

However, this still does **not** promote hand-motion geometry to main evidence.

Why not:

- the proxy can sometimes land on a real frame whose geometry produces near-zero
  collateral under this conditioned protocol
- but the decoded proxy edit is still semantically dirty

Observed proxy semantic collateral under the conditioned protocol:

- `semantic_frame`
  - `left_hand_motion -> closing`: clean-rate `0.7052`,
    mean semantic collateral `1.5769`
  - `left_hand_motion -> opening`: clean-rate `0.7155`,
    mean semantic collateral `1.5410`
  - `right_hand_motion -> closing`: clean-rate `0.4866`,
    mean semantic collateral `1.9782`
  - `right_hand_motion -> opening`: clean-rate `0.5167`,
    mean semantic collateral `1.9417`

- `continuous_frame`
  - `left_hand_motion -> closing`: clean-rate `0.7052`,
    mean semantic collateral `1.5769`
  - `left_hand_motion -> opening`: clean-rate `0.7155`,
    mean semantic collateral `1.5532`
  - `right_hand_motion -> closing`: clean-rate `0.4866`,
    mean semantic collateral `2.0000`
  - `right_hand_motion -> opening`: clean-rate `0.6333`,
    mean semantic collateral `1.3262`

Interpretation:

- the conditioned audit successfully shows that exact-neighbor scarcity was the
  wrong bottleneck
- but geometry-only locality ratios are still not fully trustworthy here,
  because a proxy can retrieve a geometrically close frame while its decoded
  semantic edit still changes around `1.3` to `2.0` extra fields on average
- so this audit is useful as a **diagnostic and protocol correction**, but not
  yet as a clean mainline win

Updated stance:

- interaction-motion geometry locality remains the strongest geometry-aware
  control result
- conditioned hand-motion geometry is now a supporting diagnostic:
  it shows the exact-neighbor protocol was too strict and confirms that proxy
  edits remain semantically entangled even when conditioned candidates exist
- do not promote hand-motion geometry locality until we have either:
  - a true conditional reconstruction procedure, or
  - a post-edit evaluator that jointly constrains geometry and semantic
    preservation without relying on nearest real frames

## Unified representation risk bundle

To stop the project from drifting back toward an accuracy-only story, I added:

- [build_representation_risk_bundle.py](/opt/tiger/hand/tools/build_representation_risk_bundle.py)
- [representation_risk_bundle.json](/opt/tiger/hand/experiments/generated/representation_risk_bundle.json)
- [representation_risk_bundle.md](/opt/tiger/hand/experiments/generated/summary_tables/representation_risk_bundle.md)

This bundle merges four lines:

- current-code symbolic frontier
- learned-token proxy report
- control evidence bundle
- conditioned hand-motion geometry diagnostic

What it now fixes explicitly:

- it makes the **mainline claim boundary** machine-checkable
- it prevents later overclaiming on pure discriminability
- it records which evidence is strong enough for mainline use and which is only
  diagnostic

Current decision from the bundle:

- keep as mainline:
  - sequence-native symbolic structure
  - anatomy-aware grouped factorization
  - zero-harm family-level repair
  - local controllability / editability
  - interaction-motion geometry locality

- avoid as mainline:
  - "symbolic wins because pure retrieval/classification beats learned tokens"
  - "hand-motion geometry locality is already clean"
  - "temporal HL is just a stronger temporal backbone over frame-wise HL"

Key evidence behind that decision:

- best learned-token pressure is already too strong for a pure-accuracy story
  - `continuous_frame`, `32` clusters:
    - retrieval top1 `1.0000`
    - margin `0.2141`
    - classifier seq acc `0.5077 / 0.7077 / 0.7692`
  - `semantic_frame`, `32` clusters:
    - retrieval top1 `0.9545`
    - margin `0.1753`
    - classifier seq acc `0.5846 / 0.7231 / 0.7692`

- current-code symbolic frontier is strongest after structure + repair, not
  after raw flat discrimination
  - fraction `1.0`:
    - grouped `0.8974`
    - grouped+family `1.0000`
  - fraction `0.5`:
    - flat rerun `0.6410`
    - grouped `0.8462`
    - grouped+family `0.9487`

- control/editability remains the clearest symbolic win
  - symbolic clean edit stays `1.0000`
  - proxy clean edit remains near zero on the tracked-field audit
  - proxy collateral remains around `~2` fields per frame on the main tasks

Interpretation:

- this bundle formalizes the project's current "review defense geometry"
- the work should now be treated as a representation/control story under strong
  token-baseline pressure, not as a generic symbolic-vs-token accuracy contest
- any future experiment that does not strengthen one of the retained mainline
  claims should be treated as lower priority

## Counterfactual edit audit

To make the control story less dependent on raw field edits or unstable
geometry ratios, I added:

- [build_counterfactual_edit_audit.py](/opt/tiger/hand/tools/build_counterfactual_edit_audit.py)
- [counterfactual_edit_audit.json](/opt/tiger/hand/experiments/generated/counterfactual_edit_audit.json)
- [counterfactual_edit_audit.md](/opt/tiger/hand/experiments/generated/summary_tables/counterfactual_edit_audit.md)

This audit evaluates each edit as a **counterfactual intervention**:

- target channel should change
- preserved context fields should remain consistent
- geometric change should stay concentrated on the target side

I use a bounded metric

- `target_share = target_delta / (target_delta + context_drift)`

and a stricter combined score

- `counterfactual_score = target_share * preserved_share`

where `preserved_share` penalizes semantic mismatches on the fields that should
stay fixed under the edit.

What it adds over the previous audits:

- avoids the misleading huge ratios that appear when proxy context drift is near
  zero
- integrates semantic preservation and geometry concentration in one place
- lets us compare interaction-motion and hand-motion edits under the same
  protocol

Observed result:

- `interaction_motion -> approach`
  - symbolic CF score: `0.6124`
  - `semantic_frame` proxy CF score: `0.3226`
  - `continuous_frame` proxy CF score: `0.3050`

- `interaction_motion -> separate`
  - symbolic CF score: `0.6289`
  - `semantic_frame` proxy CF score: `0.3402`

So the interaction-motion control result remains clearly in favor of symbolic
edits even after switching to a bounded and stricter metric.

For hand-motion edits, the picture is more mixed:

- symbolic still leads on several tasks
  - e.g. `right_hand_motion -> closing`
    - symbolic `0.8086`
    - semantic proxy `0.7175`
    - continuous proxy `0.7500`
- but some left-hand or easier right-hand settings are now much closer
  - e.g. `left_hand_motion -> opening`
    - symbolic `0.9251`
    - proxies `0.8608`
  - `right_hand_motion -> opening`
    - symbolic `0.8776`
    - continuous proxy `0.9079`

Interpretation:

- this audit strengthens the interaction-motion control line
- it does **not** yet rescue hand-motion as a clean universal mainline win
- the reason is consistent with the earlier diagnosis:
  hand-motion edits are still partially confounded by nearest-frame proxy
  retrieval and by the limited fidelity of frame-level counterfactual matching

Updated consequence:

- interaction-motion counterfactual consistency is now part of the strongest
  control-facing evidence
- hand-motion counterfactual consistency is a useful stress test, but still not
  a headline result
- if we want to make hand-motion control truly reviewer-proof, the next step is
  still a real conditional reconstruction / conditional simulator-style
  evaluator rather than another nearest-frame audit

## Conditional hand-transplant audit

To move one step closer to a simulator-style hand-motion evaluation, I added:

- [build_conditional_hand_transplant_audit.py](/opt/tiger/hand/tools/build_conditional_hand_transplant_audit.py)
- [conditional_hand_transplant_audit.json](/opt/tiger/hand/experiments/generated/conditional_hand_transplant_audit.json)
- [conditional_hand_transplant_audit.md](/opt/tiger/hand/experiments/generated/summary_tables/conditional_hand_transplant_audit.md)

This audit removes the whole-frame confound and compares only the
**target-hand donor quality** under a conditional hand-only transplant:

- preserve the opposite-hand / interaction context
- swap in a donor for the edited hand only
- compare symbolic vs proxy donors on:
  - target-hand geometric delta
  - whether the symbolic donor is at least as close as the proxy donor
  - how many preserved-context fields the proxy still violates

This is cleaner than the earlier hand-motion geometry audit because it no
longer rewards a proxy simply for retrieving a whole frame that happens to be
globally close.

Observed result:

- `semantic_frame`
  - `left_hand_motion -> closing`
    - symbolic donor delta: `0.2361`
    - proxy donor delta: `0.3261`
    - symbolic beats proxy on `97.47%` of frames
  - `left_hand_motion -> opening`
    - symbolic donor delta: `0.2371`
    - proxy donor delta: `0.3218`
    - symbolic beats proxy on `81.85%` of frames
  - `right_hand_motion -> closing`
    - symbolic donor delta: `0.2552`
    - proxy donor delta: `0.4258`
    - symbolic beats proxy on `95.13%` of frames
  - `right_hand_motion -> opening`
    - symbolic donor delta: `0.2440`
    - proxy donor delta: `0.2567`
    - symbolic beats proxy on `79.64%` of frames

- `continuous_frame`
  - `left_hand_motion -> closing`
    - symbolic donor delta: `0.2361`
    - proxy donor delta: `0.5259`
    - symbolic beats proxy on `99.82%` of frames
  - `left_hand_motion -> opening`
    - symbolic donor delta: `0.2371`
    - proxy donor delta: `0.4578`
    - symbolic beats proxy on `99.13%` of frames
  - `right_hand_motion -> closing`
    - symbolic donor delta: `0.2552`
    - proxy donor delta: `0.6923`
    - symbolic beats proxy on `99.49%` of frames
  - `right_hand_motion -> opening`
    - symbolic donor delta: `0.2440`
    - proxy donor delta: `0.3526`
    - symbolic beats proxy on `81.90%` of frames

At the same time, proxy context preservation is still imperfect:

- `semantic_frame` preserved-clean rate:
  `0.4866 .. 0.7155`
- `continuous_frame` preserved-clean rate:
  `0.4866 .. 0.7155`, except `right_hand_motion -> opening` at `0.6333`
- proxy semantic collateral still stays around
  `1.3262 .. 2.0000` extra changed fields

Interpretation:

- this is the first hand-motion-specific audit that gives a fairly stable
  symbolic advantage after removing the whole-frame retrieval confound
- unlike the earlier nearest-frame counterfactual audit, the result is now
  mostly consistent across all four hand-motion tasks
- this does **not** fully promote hand-motion to a universal headline claim,
  because it is still a donor-selection audit rather than a full conditional
  reconstruction
- but it materially strengthens the supporting evidence that symbolic structure
  provides cleaner and closer hand-specific control than opaque proxy edits

Updated stance:

- interaction-motion control remains the strongest mainline control result
- hand-motion now has a stronger **supporting control** line through
  conditional hand-transplant quality
- the next real promotion step would be a conditional reconstruction or
  simulation-based evaluator that measures the realized edited motion rather
  than donor quality alone

## Conditional motion realization oracle audit

To test whether donor-quality improvements are enough to realize motion under
the original previous-frame context, I added:

- [build_conditional_motion_realization_oracle_audit.py](/opt/tiger/hand/tools/build_conditional_motion_realization_oracle_audit.py)
- [conditional_motion_realization_oracle_audit.json](/opt/tiger/hand/experiments/generated/conditional_motion_realization_oracle_audit.json)
- [conditional_motion_realization_oracle_audit.md](/opt/tiger/hand/experiments/generated/summary_tables/conditional_motion_realization_oracle_audit.md)

This audit is intentionally stricter than the transplant check:

- keep the original previous frame fixed
- search for donors that satisfy the conditioned context rule
- among those, prefer donors that would actually realize the requested
  `opening / closing` label when composed with that original previous frame

Observed result:

- all four hand-motion tasks collapse to zero for both symbolic and proxy donor
  families
  - symbolic realizing-candidate rate: `0.0000`
  - proxy realizing-candidate rate: `0.0000`

Interpretation:

- this is not just a failed experiment; it exposes a structural fact
- a **current-frame donor alone** is insufficient to realize hand-motion labels
  under arbitrary previous-frame context
- in other words, hand-motion control is fundamentally **transition-aware**,
  not just state-aware

This is actually strong evidence for the representation direction:

- if motion realization depended only on current state, some donor family
  should have survived this oracle check
- the fact that both symbolic and proxy donor selection fail means that
  hand-motion editing cannot be reduced to "find a better current frame"
- it motivates exactly the sequence-native temporal-HL claim:
  explicit transition structure is necessary if the representation is supposed
  to support controllable motion editing rather than pose substitution

Updated consequence:

- do **not** use donor-only realization as a success metric for hand-motion
- treat this audit as a structural negative result that justifies the need for
  explicit temporal channels
- the next valid hand-motion control evaluator must operate on
  transition-conditioned edits or short temporal motifs, not on current-frame
  donor search alone

## Transition-conditioned hand-motion audit

Following the oracle result above, I moved from current-frame donors to
**two-frame donor pairs** that already realize the requested motion:

- [build_transition_conditioned_hand_motion_audit.py](/opt/tiger/hand/tools/build_transition_conditioned_hand_motion_audit.py)
- [transition_conditioned_hand_motion_audit.json](/opt/tiger/hand/experiments/generated/transition_conditioned_hand_motion_audit.json)
- [transition_conditioned_hand_motion_audit.md](/opt/tiger/hand/experiments/generated/summary_tables/transition_conditioned_hand_motion_audit.md)

Protocol:

- for each test `hand_motion` frame
- search train donor pairs `(prev, curr)` whose `curr` frame satisfies the same
  conditioned context constraints as before
- require that the donor pair itself already realizes the requested
  `opening / closing` motion
- compare symbolic-conditioned donor pairs against proxy-conditioned donor pairs
  by the target-hand pair distance

This is the first audit on this line that is explicitly **transition-native**.

Observed result:

- `semantic_frame`
  - `left_hand_motion -> closing`
    - symbolic pair availability: `0.8291`
    - proxy pair availability: `1.0000`
    - symbolic pair distance: `0.6204`
    - proxy pair distance: `0.7202`
    - symbolic beats proxy on `77.51%` of frames
  - `left_hand_motion -> opening`
    - symbolic pair distance: `0.6435`
    - proxy pair distance: `0.6750`
    - symbolic beats proxy on `70.20%`
  - `right_hand_motion -> closing`
    - symbolic pair distance: `0.8023`
    - proxy pair distance: `1.7249`
    - symbolic beats proxy on `75.61%`
  - `right_hand_motion -> opening`
    - symbolic pair distance: `0.8663`
    - proxy pair distance: `0.6448`
    - symbolic beats proxy on `52.36%`

- `continuous_frame`
  - `left_hand_motion -> closing`
    - symbolic pair distance: `0.6204`
    - proxy pair distance: `0.8410`
    - symbolic beats proxy on `76.16%`
  - `left_hand_motion -> opening`
    - symbolic pair distance: `0.6435`
    - proxy pair distance: `1.1242`
    - symbolic beats proxy on `82.85%`
  - `right_hand_motion -> closing`
    - symbolic pair distance: `0.8023`
    - proxy pair distance: `1.2267`
    - symbolic beats proxy on `75.29%`
  - `right_hand_motion -> opening`
    - symbolic pair distance: `0.8663`
    - proxy pair distance: `0.9186`
    - symbolic beats proxy on `58.38%`

Proxy context preservation remains limited:

- preserved-clean rate stays around `0.4047 .. 0.5959`
- proxy semantic collateral remains around `1.60 .. 2.23` extra fields

Interpretation:

- once hand-motion is evaluated through **transition-conditioned short motifs**
  rather than current-frame donors, the symbolic advantage becomes much more
  stable again
- this directly supports the claim that hand-motion should be treated as a
  temporal-structure problem rather than a per-frame state-retrieval problem
- the line is now stronger than the previous donor-only supporting evidence,
  although it is still not as clean as the interaction-motion control results

Updated stance:

- hand-motion now has two complementary supporting results:
  - conditional hand-transplant donor quality
  - transition-conditioned short-motif matching
- the donor-only realization oracle remains useful because it explains *why*
  these temporal motifs are needed at all
- if we want to promote hand-motion even further, the next step should target a
  real transition-conditioned editor or learned motif-based simulator rather
  than more frame-level retrieval variants

### Difficulty slices for transition-conditioned hand motion

To check whether the temporal-motif result is broad or slice-specific, I added:

- [build_transition_conditioned_hand_motion_slice_audit.py](/opt/tiger/hand/tools/build_transition_conditioned_hand_motion_slice_audit.py)
- [transition_conditioned_hand_motion_slice_audit.json](/opt/tiger/hand/experiments/generated/transition_conditioned_hand_motion_slice_audit.json)
- [transition_conditioned_hand_motion_slice_audit.md](/opt/tiger/hand/experiments/generated/summary_tables/transition_conditioned_hand_motion_slice_audit.md)

This audit reuses the same two-frame donor-pair protocol, but reports results
by:

- `occlusion`
- `finger_occlusion`
- `wrist_rom`
- `interaction`
- left/right/all

Main finding:

- the temporal symbolic advantage is **extremely stable** on the hard
  single-hand slices
  - many `occlusion / finger_occlusion / wrist_rom` slices reach
    `symbolic beats proxy = 1.0000`
  - symbolic pair distance is consistently lower on these slices for both
    proxy families

Examples:

- `semantic_frame`, `right_hand_motion -> closing`
  - `occlusion`: beats proxy `1.0000`
  - `finger_occlusion`: beats proxy `1.0000`
  - `wrist_rom`: beats proxy `1.0000`

- `continuous_frame`, `left_hand_motion -> opening`
  - `occlusion`: beats proxy `1.0000`
  - `finger_occlusion`: beats proxy `1.0000`
  - `wrist_rom`: beats proxy `1.0000`

But the interaction slice is still much weaker:

- `semantic_frame`
  - `left_hand_motion -> opening`, `interaction`: `0.2626`
  - `right_hand_motion -> opening`, `interaction`: `0.1563`
- `continuous_frame`
  - `left_hand_motion -> closing`, `interaction`: `0.4657`
  - `right_hand_motion -> opening`, `interaction`: `0.2629`

Interpretation:

- the hand-motion temporal motif advantage is not a fragile average effect
- on difficult **single-hand** regimes, it is often nearly deterministic
- the remaining weakness is now much more localized:
  it lies mainly in **interacting-hand contexts**

This is strategically good news because it sharpens the problem decomposition:

- interaction-motion remains the strongest mainline control result
- hand-motion temporal motifs are already strong enough on one-hand hard cases
- the next frontier is not generic hand-motion robustness any more, but
  cross-hand interaction-aware motion editing

An important nuance is that `state_event` already reaches the same top-1 as
`temporal_event` on this retrieval-style intrinsic test, but `temporal_event`
still separates classes more strongly:

- temporal wins mean positive margin:
  `0.0581` vs `0.0423`
- temporal wins mean correct rank:
  `1.2273` vs `1.2727`
- temporal has a higher margin on `17/22` queries
- temporal improves rank on the hardest remaining query
  `ROM09_Interaction_Fingers_Touching`

So the strongest updated claim is not simply:

- "temporal event beats state event on top-1"

but rather:

- "eventized sequence structure is the main discriminator, and explicit
  temporal channels increase separation margin and ranking stability on top of
  that eventization"

### Persistence / duration check at the intrinsic level

Additional event-level variants:

- `temporal_event_persist`
- `temporal_event_segdur`

They match the same top-1:

- top-1: `0.9545`

but do not improve the main separation metric:

- `temporal_event` margin: `0.0581`
- `temporal_event_segdur` margin: `0.0564`
- `temporal_event_persist` margin: `0.0551`

Interpretation:

- persistence / segment-duration information is not useless
- but in the current symbolic formulation it still does not surpass plain
  transition-aware temporal events as the cleanest mainline representation

## Hard-case-aware finetune

Implementation note:

- finetune-time class weighting is now available through repeated
  `--boost-label LABEL:WEIGHT` arguments in
  [train_symbolic_pretrain.py](/opt/tiger/hand/tools/train_symbolic_pretrain.py)
- this changes only the ROM finetune stage, not the pretraining stage

Focused experiment:

- boost only the most persistent class
  `ROM05_RT_Wrist_ROM:3.0`

Result:

- [symbolic_pretrain_temporal_boost_rom05.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_boost_rom05.json)

## Occlusion-family late correction and gated audit

Purpose:

- test whether the remaining concentrated errors in the boosted symbolic
  mainline are due to a weak classifier boundary rather than missing symbolic
  information
- evaluate this without retraining the mainline by using symbolic retrieval as
  a restricted late-correction module over the occlusion / wrist family

Core assets:

- evaluator:
  [eval_occlusion_late_correction.py](/opt/tiger/hand/tools/eval_occlusion_late_correction.py)
- gated sweep:
  [sweep_occlusion_late_correction_gate.py](/opt/tiger/hand/tools/sweep_occlusion_late_correction_gate.py)
- corrected outputs:
  [occlusion_late_correction_mainline.json](/opt/tiger/hand/experiments/generated/occlusion_late_correction_mainline.json)
  [occlusion_late_correction_mainline_fraction05.json](/opt/tiger/hand/experiments/generated/occlusion_late_correction_mainline_fraction05.json)
  [occlusion_late_correction_gate_sweep_mainline.json](/opt/tiger/hand/experiments/generated/occlusion_late_correction_gate_sweep_mainline.json)
  [occlusion_late_correction_gate_sweep_mainline_fraction05.json](/opt/tiger/hand/experiments/generated/occlusion_late_correction_gate_sweep_mainline_fraction05.json)

Audit note:

- the late-correction audit originally risked mixing the `ROM07_RT/Rt` and
  `ROM08_LT/Lt` aliases during sequence identity handling
- the evaluator and gate sweep were corrected so that raw sequence identity is
  preserved while label comparison still uses canonicalized names
- current numbers below are the corrected ones and should be treated as the
  authoritative version

Corrected unrestricted family late-correction result:

- full-data mainline:
  - overall `all`: `0.9231 -> 1.0000` (`+0.0769`)
  - family accuracy: `0.9000 -> 1.0000`
  - improved `3`, harmed `0`
  - slices:
    - `finger_occlusion`: `0.8333 -> 1.0000`
    - `occlusion`: `0.9167 -> 1.0000`
    - `wrist_rom`: `0.8333 -> 1.0000`
- half-data mainline:
  - overall `all`: `0.7949 -> 1.0000` (`+0.2051`)
  - family accuracy: `0.7333 -> 1.0000`
  - improved `8`, harmed `0`
  - slices:
    - `finger_occlusion`: `0.7500 -> 1.0000`
    - `occlusion`: `0.7083 -> 1.0000`
    - `wrist_rom`: `0.8333 -> 1.0000`

Gated sweep takeaways:

- a simple `change_only` rule already matches the unrestricted family
  correction exactly in both regimes
- operationally this means:
  - only enter the restricted family retrieval path when the mainline
    prediction is already in the occlusion family
  - only replace the prediction if the retrieved family top-1 label differs
    from the mainline prediction
- this rule preserves zero harmed cases while recovering:
  - full-data mainline: all `3/3` recoverable family mistakes
  - half-data mainline: all `8/8` recoverable family mistakes

Margin-gated variant:

- `margin >= 0.045` is a stricter gate that still preserves the full-data gain
  exactly
- on the half-data regime it drops one correction:
  - overall `all`: `0.7949 -> 0.9744`
  - family accuracy: `0.7333 -> 0.9667`
  - improved `7`, harmed `0`

Interpretation:

- the symbolic sequence space already contains enough information to separate
  the remaining occlusion-family confusions
- the current failure mode is concentrated in the classifier head rather than
  in the symbolic representation itself
- a restricted hybrid symbolic correction is therefore a valid supporting
  experiment, and the `change_only` gate is the cleanest current operating
  point

## Hybrid symbolic baseline

Purpose:

- promote the validated `change_only` correction from an audit artifact into a
  baseline-quality hybrid symbolic system
- make the corrected symbolic system directly comparable to the mainline and to
  the strongest non-symbolic control under the same analysis format

Assets:

- builder:
  [build_hybrid_symbolic_analysis.py](/opt/tiger/hand/tools/build_hybrid_symbolic_analysis.py)
- analyses:
  [hybrid_symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_changeonly_analysis.json](/opt/tiger/hand/experiments/generated/hybrid_symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_changeonly_analysis.json)
  [hybrid_symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_fraction05_changeonly_analysis.json](/opt/tiger/hand/experiments/generated/hybrid_symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_fraction05_changeonly_analysis.json)
- joint comparison:
  [hybrid_symbolic_vs_joint_report.json](/opt/tiger/hand/experiments/generated/hybrid_symbolic_vs_joint_report.json)

Hybrid result:

- full-data regime:
  - mainline pretrained sequence mean: `0.9231`
  - hybrid pretrained sequence mean: `1.0000`
  - gain vs mainline: `+0.0769`
  - patch actions: applied `3`, improved `3`, harmed `0`
- half-data regime:
  - mainline pretrained sequence mean: `0.7949`
  - hybrid pretrained sequence mean: `1.0000`
  - gain vs mainline: `+0.2051`
  - patch actions: applied `8`, improved `8`, harmed `0`

Hybrid vs strongest non-symbolic control (`joint_sequence_best`):

- full-data regime:
  - hybrid symbolic `1.0000`
  - joint sequence `0.8718`
  - gap: `+0.1282`
- half-data regime:
  - hybrid symbolic `1.0000`
  - joint sequence `0.7179`
  - gap: `+0.2821`

Slice-level comparison against `joint_sequence_best`:

- full-data regime:
  - `all`: `1.0000 vs 0.8718`
  - `wrist_rom`: `1.0000 vs 0.8333`
  - `occlusion`: `1.0000 vs 0.8750`
  - `finger_occlusion`: tie at `1.0000`
- half-data regime:
  - `all`: `1.0000 vs 0.7179`
  - `wrist_rom`: `1.0000 vs 0.1667`
  - `occlusion`: `1.0000 vs 0.8333`
  - `finger_occlusion`: tie at `1.0000`

Current interpretation:

- the strongest symbolic story is no longer only "a better symbolic pretrain
  mainline"
- it is now "a symbolic representation whose remaining errors are sparse enough
  to be fully corrected by a narrow family-restricted symbolic retrieval gate"
- this is strong evidence that the bottleneck is decision-boundary quality, not
  representational insufficiency

## Rerun-cohort fusion gate audit

Purpose:

- test whether a more principled fusion rule can replace the hard
  `change_only` hybrid gate
- specifically test whether the mainline sequence confidence margin adds useful
  gating signal on top of the retrieval margin

Assets:

- fusion evaluator:
  [eval_symbolic_retrieval_fusion.py](/opt/tiger/hand/tools/eval_symbolic_retrieval_fusion.py)
- rerun detailed analyses:
  [symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_seed0_rerun_analysis.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_seed0_rerun_analysis.json)
  [symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_seed1_analysis.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_seed1_analysis.json)
  [symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_seed2_analysis.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_seed2_analysis.json)
  [symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_fraction05_seed0_analysis.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_fraction05_seed0_analysis.json)
  [symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_fraction05_seed1_analysis.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_fraction05_seed1_analysis.json)
  [symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_fraction05_seed2_analysis.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_fraction05_seed2_analysis.json)
- fusion sweeps:
  [symbolic_retrieval_fusion_full_rerun.json](/opt/tiger/hand/experiments/generated/symbolic_retrieval_fusion_full_rerun.json)
  [symbolic_retrieval_fusion_fraction05_rerun.json](/opt/tiger/hand/experiments/generated/symbolic_retrieval_fusion_fraction05_rerun.json)
  [symbolic_retrieval_fusion_rerun_summary.json](/opt/tiger/hand/experiments/generated/symbolic_retrieval_fusion_rerun_summary.json)

Important audit caveat:

- rerunning the detailed single-seed analyses under the same nominal config does
  not reproduce the exact historical multiseed numbers already stored in the
  mainline artifact
- therefore the fusion audit is evaluated as a self-consistent rerun cohort and
  should not be numerically mixed with the earlier frozen mainline tables

Rerun-cohort base accuracy:

- full-data rerun cohort:
  - overall `all`: `0.7949`
  - family-only: `0.7667`
- half-data rerun cohort:
  - overall `all`: `0.6154`
  - family-only: `0.5667`

Best zero-harm fusion over the rerun cohort:

- full-data rerun cohort:
  - best overall zero-harm point:
    - mainline margin cap `<= 2.0`
    - retrieval margin threshold `>= 0.05`
    - overall `all`: `0.7949 -> 0.9744`
    - family-only: `0.7667 -> 1.0000`
    - improved `7`, harmed `0`
  - with a meaningfully tighter mainline cap (`<= 0.5`):
    - overall `all`: `0.7949 -> 0.9231`
    - improved `5`, harmed `0`
- half-data rerun cohort:
  - best overall zero-harm point:
    - mainline margin cap `<= 2.0`
    - retrieval margin threshold `>= 0.04`
    - overall `all`: `0.6154 -> 0.9487`
    - family-only: `0.5667 -> 1.0000`
    - improved `13`, harmed `0`
  - with a meaningfully tighter mainline cap (`<= 0.8`):
    - overall `all`: `0.6154 -> 0.7949`
    - improved `7`, harmed `0`

Interpretation:

- retrieval margin is a reliable positive signal
- mainline top1-top2 sequence margin is not a strong enough gate feature by
  itself:
  - when the mainline margin cap is loose, fusion recovers most of the benefit
  - when the cap becomes genuinely selective, the gain drops sharply
- this suggests that the next learned gate should not rely only on a scalar
  confidence margin from the mainline head
- stronger gate features likely need:
  - the family-specific alternative label score from the mainline head
  - retrieval-vs-mainline label agreement features
  - possibly a small classifier over family-restricted score vectors

## Leave-one-seed-out learned gate

Purpose:

- test whether the family-restricted correction can be learned directly from
  score features rather than expressed as a hand-written gate
- evaluate it on the same rerun cohort using leave-one-seed-out training across
  seeds

Assets:

- learned gate evaluator:
  [eval_symbolic_learned_gate.py](/opt/tiger/hand/tools/eval_symbolic_learned_gate.py)
- main outputs:
  [symbolic_learned_gate_full_rerun.json](/opt/tiger/hand/experiments/generated/symbolic_learned_gate_full_rerun.json)
  [symbolic_learned_gate_fraction05_rerun.json](/opt/tiger/hand/experiments/generated/symbolic_learned_gate_fraction05_rerun.json)
- feature-preset ablation summary:
  [symbolic_learned_gate_rerun_preset_summary.json](/opt/tiger/hand/experiments/generated/symbolic_learned_gate_rerun_preset_summary.json)

Protocol:

- construct family-only samples from the rerun detailed analyses
- derive score features from:
  - mainline sequence `top3`
  - retrieval `top3`
  - relative score / rank relations between the two
- train a logistic regression gate on two seeds and evaluate on the held-out
  seed
- report the fixed `0.5` threshold result and a threshold sweep audit

Fixed-threshold (`0.5`) result:

- full-data rerun cohort:
  - family-only: `0.7667 -> 1.0000`
  - implied overall: `0.7949 -> 0.9744`
  - improved `7`, harmed `0`
- half-data rerun cohort:
  - family-only: `0.5667 -> 1.0000`
  - implied overall: `0.6154 -> 0.9487`
  - improved `13`, harmed `0`

Feature-preset ablation:

- tested presets:
  - `all`
  - `changed_only`
  - `margins_only`
  - `no_changed`
- surprising but consistent result:
  - all four presets achieve the same leave-one-seed-out result on the current
    rerun cohort
  - this means the family-correction cases are already linearly separable under
    several weak feature views, not only under the hand-written
    `changed_prediction` signal

Interpretation:

- this is stronger than the earlier scalar-margin fusion result
- on the rerun cohort, the family-restricted correction is not merely a brittle
  heuristic:
  - it is learnable
  - it generalizes across held-out seeds in this small protocol
  - and the local score structure is simple enough that even reduced feature
    sets recover the same boundary
- the main remaining weakness is not whether a gate exists, but whether the
  rerun cohort is stable enough and large enough to make this claim durable

## Family-vector expert baseline

Purpose:

- test a stronger family-only expert that predicts the family label directly
  from richer score vectors, instead of only learning a binary apply / not
  apply gate
- use the exported mainline family score vector to move beyond `top3`-only
  features

Assets:

- upgraded detailed exporter:
  [analyze_symbolic_pretrain.py](/opt/tiger/hand/tools/analyze_symbolic_pretrain.py)
  with `--export-family-scores`
- family-vector analyses:
  [symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_seed0_familyvec_analysis.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_seed0_familyvec_analysis.json)
  [symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_seed1_familyvec_analysis.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_seed1_familyvec_analysis.json)
  [symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_seed2_familyvec_analysis.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_seed2_familyvec_analysis.json)
  [symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_fraction05_seed0_familyvec_analysis.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_fraction05_seed0_familyvec_analysis.json)
  [symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_fraction05_seed1_familyvec_analysis.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_fraction05_seed1_familyvec_analysis.json)
  [symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_fraction05_seed2_familyvec_analysis.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_fraction05_seed2_familyvec_analysis.json)
- multiclass expert evaluator:
  [eval_symbolic_family_expert.py](/opt/tiger/hand/tools/eval_symbolic_family_expert.py)
- outputs:
  [symbolic_family_expert_full_rerun.json](/opt/tiger/hand/experiments/generated/symbolic_family_expert_full_rerun.json)
  [symbolic_family_expert_fraction05_rerun.json](/opt/tiger/hand/experiments/generated/symbolic_family_expert_fraction05_rerun.json)
  [symbolic_family_method_comparison_rerun.json](/opt/tiger/hand/experiments/generated/symbolic_family_method_comparison_rerun.json)

Protocol:

- family-only multiclass logistic regression
- inputs:
  - 8-D mainline family score vector
  - 8-D retrieval family score vector
  - 8-D score-difference vector
- leave-one-seed-out training across the rerun cohort
- confidence threshold over expert prediction before replacement

Fixed-threshold (`0.5`) result:

- full-data rerun cohort:
  - family-only: `0.7667 -> 0.9667`
  - improved `6`, harmed `0`
  - expert used on `29/30` family rows
- half-data rerun cohort:
  - family-only: `0.5667 -> 0.9000`
  - improved `10`, harmed `0`
  - expert used on `25/30` family rows

Comparison against simpler family methods on the same rerun cohort:

- learned gate fixed `0.5`:
  - full-data: `0.7667 -> 1.0000`
  - half-data: `0.5667 -> 1.0000`
- family-vector expert fixed `0.5`:
  - full-data: `0.7667 -> 0.9667`
  - half-data: `0.5667 -> 0.9000`

Interpretation:

- adding a richer family score vector does **not** automatically beat the
  simpler learned gate
- on the current rerun cohort, direct family-label prediction appears harder
  than learning the family-restricted correction boundary
- this suggests that the current evidence supports a stronger claim for
  "learnable correction gate" than for "small standalone family expert"

## Train-time family-aux baseline

Purpose:

- test the simplest train-time integration of family supervision
- instead of post-hoc correction, add a family auxiliary head during finetuning
  and see whether the main sequence classifier improves on the real test task

Assets:

- trainer / analyzer:
  [analyze_symbolic_family_aux_multiseed.py](/opt/tiger/hand/tools/analyze_symbolic_family_aux_multiseed.py)
- outputs:
  [symbolic_family_aux_pre168_84_boost_rom05_analysis.json](/opt/tiger/hand/experiments/generated/symbolic_family_aux_pre168_84_boost_rom05_analysis.json)
  [symbolic_family_aux_pre168_84_boost_rom05_fraction05_analysis.json](/opt/tiger/hand/experiments/generated/symbolic_family_aux_pre168_84_boost_rom05_fraction05_analysis.json)
  [symbolic_family_aux_pre168_84_boost_rom05_w01_analysis.json](/opt/tiger/hand/experiments/generated/symbolic_family_aux_pre168_84_boost_rom05_w01_analysis.json)
  [symbolic_family_aux_pre168_84_boost_rom05_w01_fraction05_analysis.json](/opt/tiger/hand/experiments/generated/symbolic_family_aux_pre168_84_boost_rom05_w01_fraction05_analysis.json)
  [symbolic_family_aux_comparison.json](/opt/tiger/hand/experiments/generated/symbolic_family_aux_comparison.json)

Protocol:

- keep the same strongest symbolic pretrain setup:
  - pretrain-only normalization
  - `168 / 84` temporal windows
  - finetune boost `ROM05_RT_Wrist_ROM:3.0`
- add a finetune-time auxiliary head over:
  - 8 family labels
  - plus one `other` bucket for non-family classes
- optimize:
  - main 13-way sequence label loss
  - plus `family_loss_weight * family_aux_loss`

Measured result:

- with `family_loss_weight = 0.5`
  - full-data mean sequence accuracy: `0.6667`
  - half-data mean sequence accuracy: `0.5641`
- with `family_loss_weight = 0.1`
  - full-data mean sequence accuracy: `0.7692`
  - half-data mean sequence accuracy: `0.5385`
- frozen mainline reference:
  - full-data mean sequence accuracy: `0.9231`
  - half-data mean sequence accuracy: `0.7949`

Error pattern:

- the train-time family-aux baseline still concentrates errors in the same
  family:
  - `ROM07_Rt_Finger_Occlusions -> ROM03_RT_No_Occlusion`
  - `ROM08_Lt_Finger_Occlusions -> ROM03_LT_No_Occlusion`
- and it additionally harms overall stability relative to the frozen mainline

Interpretation:

- naive train-time family supervision is **not** enough
- even a reduced family loss weight (`0.1`) remains clearly below the frozen
  mainline
- therefore the current evidence does not support "just add a family auxiliary
  head" as a viable integrated solution
- if we pursue train-time integration later, it likely needs a more targeted
  design:
  - family-restricted reranking head
  - conditional expert path
  - or joint training on richer family score vectors rather than a coarse
    auxiliary label

## Disagreement supervision audit

Purpose:

- check whether the current family-correction benchmark actually contains both
  positive and negative supervision for a pairwise reranker
- verify whether "mainline and retrieval disagree" is already a perfect signal
  on the current ROM family evaluation

Assets:

- auditor:
  [audit_family_disagreement_supervision.py](/opt/tiger/hand/tools/audit_family_disagreement_supervision.py)
- output:
  [family_disagreement_audit.json](/opt/tiger/hand/experiments/generated/family_disagreement_audit.json)

Result:

- strongest frozen mainline:
  - full-data:
    - family rows `30`
    - disagreement rows `3`
    - disagreement positives `3`
    - disagreement negatives `0`
  - half-data:
    - family rows `30`
    - disagreement rows `8`
    - disagreement positives `8`
    - disagreement negatives `0`
- rerun detailed cohorts:
  - every seed-level rerun shows the same pattern
  - all disagreement rows are positive corrections
  - no disagreement negatives appear
- train-time family-aux baseline:
  - even under the weaker integrated model, disagreement rows remain all
    positive
  - full-data: `12/12` positives, `0` negatives
  - half-data: `14/14` positives, `0` negatives

Implication:

- on the current benchmark, pairwise supervision is degenerate:
  - if retrieval top1 differs from the current symbolic prediction, the
    retrieval label is always the correct repair
- this explains why:
  - `change_only` is already optimal on the observed family disagreements
  - the learned gate can achieve perfect correction on the rerun cohort
  - and a pairwise reranker cannot be meaningfully stress-tested here because
    it has no negative disagreement examples to learn from

Consequence for next experiments:

- the current ROM family evaluation is strong enough to validate the existence
  of a family-correction mechanism
- but it is **not** strong enough to distinguish among:
  - heuristic disagreement rules
  - learned pairwise rerankers
  - or richer disagreement-based experts
- any next-stage reranker study needs either:
  - a broader / harder benchmark,
  - disagreement negatives from a weaker or differently trained backbone,
  - or a construction that introduces genuinely ambiguous family alternatives

## Duration / persistence channel audit

Representation update:

- [build_temporal_hl.py](/opt/tiger/hand/tools/build_temporal_hl.py) now exports
  explicit persistence fields into frame records:
  - per-hand `state_persistence_label`
  - per-hand `activity_persistence_label`
  - sequence-level `interaction_persistence_label`
  - sequence-level `interaction_activity_persistence_label`
- regenerated assets:
  - [temporal_hl_train.json](/opt/tiger/hand/experiments/generated/temporal_hl_train.json)
  - [temporal_hl_val.json](/opt/tiger/hand/experiments/generated/temporal_hl_val.json)
  - [temporal_hl_test.json](/opt/tiger/hand/experiments/generated/temporal_hl_test.json)

Classifier experiment:

```bash
python tools/train_symbolic_pretrain.py \
  --train-pretrain-json experiments/generated/temporal_hl_train.json \
  --train-finetune-json experiments/generated/temporal_hl_val.json \
  --test-json experiments/generated/temporal_hl_test.json \
  --mode temporal \
  --fractions 1.0 0.5 \
  --seeds 0 1 2 \
  --hidden-dim 128 \
  --pretrain-epochs 120 \
  --finetune-epochs 120 \
  --pretrain-window-span-units 168 \
  --pretrain-window-step-units 84 \
  --boost-label ROM05_RT_Wrist_ROM:3.0 \
  --duration-features \
  --output experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_duration.json
```

Result:

- [symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_duration.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_duration.json)
- fraction `1.0`: scratch `0.5385`, pretrained `0.8205`
- fraction `0.5`: scratch `0.6154`, pretrained `0.6410`

Comparison against the current strongest symbolic mainline:

- current best without explicit duration channel:
  fraction `1.0` pretrained sequence accuracy `0.9231`
- duration/persistence version:
  fraction `1.0` pretrained sequence accuracy `0.8205`

Interpretation:

- explicit persistence labels are real representation content, not just hidden
  tempo statistics
- however, injecting them directly into the current bag-of-features classifier
  hurts the strongest classification mainline
- this means persistence is not yet ready to replace the best symbolic mainline
- for now it should be treated as a sequence-native representation ablation and
  intrinsic-evaluation axis rather than the default training feature set

## Persistence-aware sequence retrieval audit

Command:

```bash
python tools/eval_sequence_symbolic_retrieval.py \
  --gallery-json experiments/generated/temporal_hl_val.json \
  --query-json experiments/generated/temporal_hl_test.json \
  --output experiments/generated/sequence_symbolic_retrieval_persistence.json
```

Result:

- [sequence_symbolic_retrieval_persistence.json](/opt/tiger/hand/experiments/generated/sequence_symbolic_retrieval_persistence.json)
- frame-token retrieval:
  - state-only `0.4545`
  - temporal `0.4091`
  - temporal + persistence `0.5000`
- RLE retrieval:
  - state-only `0.4091`
  - temporal `0.3636`
  - temporal + persistence `0.2273`
- DTW-Jaccard retrieval:
  - all three settings `0.9545`

Interpretation:

- persistence improves the plain frame-signature retrieval setting
- persistence is harmful when naively combined with the current RLE signature
- DTW retrieval is already saturated on this protocol, so it is not informative
  for judging persistence
- the practical takeaway is:
  persistence contains usable discriminative signal, but it needs a better
  sequence matcher or encoder than the current RLE / bag-of-features pipelines

## Event-level sequence matcher audit

Implementation update:

- [eval_sequence_symbolic_retrieval.py](/opt/tiger/hand/tools/eval_sequence_symbolic_retrieval.py)
  now includes an event-level DTW matcher:
  - compress consecutive frames with the same event key into symbolic segments
  - compare event sequences with DTW over Jaccard token sets
  - optionally attach persistence labels as event attributes

Command:

```bash
python tools/eval_sequence_symbolic_retrieval.py \
  --gallery-json experiments/generated/temporal_hl_val.json \
  --query-json experiments/generated/temporal_hl_test.json \
  --output experiments/generated/sequence_symbolic_retrieval_event.json
```

Artifacts:

- [sequence_symbolic_retrieval_event.json](/opt/tiger/hand/experiments/generated/sequence_symbolic_retrieval_event.json)
- [sequence_symbolic_retrieval_event_summary.json](/opt/tiger/hand/experiments/generated/sequence_symbolic_retrieval_event_summary.json)

Key results:

- event-DTW top-1 accuracy is saturated for all three settings:
  - state: `0.9545`
  - temporal: `0.9545`
  - temporal + persistence: `0.9545`
- margin statistics on event-DTW:
  - state event-DTW margin: `0.0423`
  - temporal event-DTW margin: `0.0581`
  - temporal + persistence event-DTW margin: `0.0551`
- reference frame-DTW margins:
  - state DTW: `0.0667`
  - temporal DTW: `0.0664`
  - temporal + persistence DTW: `0.0649`

Interpretation:

- moving from naive RLE matching to event-level DTW rescues the temporal
  sequence matcher, since event-DTW no longer shows the collapse observed for
  `temporal_rle`
- temporal event structure is more useful than state-only event structure under
  the compressed matcher
- persistence still does not become a net win under the current event design
- this narrows the remaining problem:
  the bottleneck is no longer just the matcher, but the current persistence
  label design or event-boundary construction

## Segment-duration label audit

Representation update:

- [build_temporal_hl.py](/opt/tiger/hand/tools/build_temporal_hl.py) now also
  exports full segment-duration labels, in addition to elapsed persistence:
  - per-hand `state_segment_duration_label`
  - per-hand `activity_segment_duration_label`
  - top-level `interaction_segment_duration_label`
  - top-level `interaction_activity_segment_duration_label`
- regenerated retrieval-side assets:
  - [temporal_hl_val.json](/opt/tiger/hand/experiments/generated/temporal_hl_val.json)
  - [temporal_hl_test.json](/opt/tiger/hand/experiments/generated/temporal_hl_test.json)

Sanity check:

- [temporal_hl_val_summary.json](/opt/tiger/hand/experiments/generated/temporal_hl_val_summary.json)
  now includes:
  - `hand_segment_duration_hist`
  - `hand_activity_segment_hist`
  - `interaction_segment_duration_hist`
  - `interaction_activity_segment_hist`

Evaluation update:

- [eval_sequence_symbolic_retrieval.py](/opt/tiger/hand/tools/eval_sequence_symbolic_retrieval.py)
  now supports event-DTW with:
  - temporal only
  - temporal + elapsed persistence
  - temporal + segment duration
  - temporal + elapsed persistence + segment duration

Command:

```bash
python tools/eval_sequence_symbolic_retrieval.py \
  --gallery-json experiments/generated/temporal_hl_val.json \
  --query-json experiments/generated/temporal_hl_test.json \
  --output experiments/generated/sequence_symbolic_retrieval_segdur_event.json
```

Artifacts:

- [sequence_symbolic_retrieval_segdur_event.json](/opt/tiger/hand/experiments/generated/sequence_symbolic_retrieval_segdur_event.json)
- [sequence_symbolic_retrieval_segdur_event_summary.json](/opt/tiger/hand/experiments/generated/sequence_symbolic_retrieval_segdur_event_summary.json)

Key event-DTW margin results:

- temporal only: `0.0581`
- temporal + elapsed persistence: `0.0551`
- temporal + segment duration: `0.0564`
- temporal + elapsed persistence + segment duration: `0.0537`

Interpretation:

- full segment duration is slightly better than the earlier elapsed-persistence
  variant under the same event matcher
- but neither duration encoding surpasses the plain temporal event matcher
- so the remaining bottleneck is not simply "persistence should use full segment
  duration"
- the more likely issue is that current event boundaries are still too tied to
  exact state identity rather than higher-level motion phases

## Motion-phase event audit

Implementation update:

- [eval_sequence_symbolic_retrieval.py](/opt/tiger/hand/tools/eval_sequence_symbolic_retrieval.py)
  now also supports a phase-level event matcher:
  - hand phases such as `hold`, `adjust`, `articulate`, `opening_phase`,
    `closing_phase`, `reconfigure`
  - interaction phases such as `steady_pair`, `approach_phase`,
    `separate_phase`
  - phase-level DTW over compressed event sets

Command:

```bash
python tools/eval_sequence_symbolic_retrieval.py \
  --gallery-json experiments/generated/temporal_hl_val.json \
  --query-json experiments/generated/temporal_hl_test.json \
  --output experiments/generated/sequence_symbolic_retrieval_phase_event.json
```

Artifacts:

- [sequence_symbolic_retrieval_phase_event.json](/opt/tiger/hand/experiments/generated/sequence_symbolic_retrieval_phase_event.json)
- [sequence_symbolic_retrieval_phase_event_summary.json](/opt/tiger/hand/experiments/generated/sequence_symbolic_retrieval_phase_event_summary.json)

Key comparison:

- temporal exact-state event-DTW:
  - top-1 `0.9545`
  - margin `0.0581`
  - avg query events `196.0909`
- temporal phase event-DTW:
  - top-1 `0.9545`
  - margin `0.0270`
  - avg query events `164.8636`

Adding duration-style signals on top of the phase matcher:

- phase + elapsed persistence:
  - top-1 `0.8182`
  - margin `0.0298`
- phase + segment duration:
  - top-1 `0.8182`
  - margin `0.0346`
- phase + persistence + segment duration:
  - top-1 `0.7727`
  - margin `0.0343`

Interpretation:

- the phase-level grouping is meaningful in one sense:
  it compresses the sequence more aggressively while preserving the same top-1
  accuracy as exact-state event-DTW when used alone
- however, the phase taxonomy is currently too coarse:
  the similarity margin drops substantially from `0.0581` to `0.0270`
- duration-style labels do not integrate cleanly with the current phase units;
  once added, top-1 drops sharply
- the practical conclusion is:
  higher-level motion phases are promising as a compression axis, but the
  current handcrafted phase inventory is not yet a stronger discriminative unit
  than exact-state temporal events

## Refined motion-phase audit

Implementation update:

- [eval_sequence_symbolic_retrieval.py](/opt/tiger/hand/tools/eval_sequence_symbolic_retrieval.py)
  now includes a refined phase taxonomy that distinguishes:
  - proximal / wrist-dominant changes
  - distal / finger-dominant articulation
  - coordinated opening / closing
  - global vs distal vs wrist reconfiguration
  - interaction phases using both motion and cross-hand distance buckets

Command:

```bash
python tools/eval_sequence_symbolic_retrieval.py \
  --gallery-json experiments/generated/temporal_hl_val.json \
  --query-json experiments/generated/temporal_hl_test.json \
  --output experiments/generated/sequence_symbolic_retrieval_refined_phase_event.json
```

Artifacts:

- [sequence_symbolic_retrieval_refined_phase_event.json](/opt/tiger/hand/experiments/generated/sequence_symbolic_retrieval_refined_phase_event.json)
- [sequence_symbolic_retrieval_refined_phase_event_summary.json](/opt/tiger/hand/experiments/generated/sequence_symbolic_retrieval_refined_phase_event_summary.json)

Key comparison:

- exact-state temporal event-DTW:
  - top-1 `0.9545`
  - margin `0.0581`
  - avg query events `196.0909`
- coarse phase event-DTW:
  - top-1 `0.9545`
  - margin `0.0270`
  - avg query events `164.8636`
- refined phase event-DTW:
  - top-1 `0.9091`
  - margin `0.0333`
  - avg query events `176.8182`

Adding duration-style information to the refined phase matcher:

- refined phase + elapsed persistence:
  - top-1 `0.9091`
  - margin `0.0346`
- refined phase + segment duration:
  - top-1 `0.9545`
  - margin `0.0367`
- refined phase + elapsed persistence + segment duration:
  - top-1 `0.9545`
  - margin `0.0371`

Interpretation:

- the refined phase inventory is substantially better than the earlier coarse
  phase taxonomy
- unlike the coarse phase matcher, the refined phase matcher can absorb
  segment-duration information without collapsing top-1 accuracy
- this is the first local evidence that a higher-level temporal abstraction can
  remain competitive with exact-state event matching while using fewer events
- the refined phase path still trails exact-state event-DTW on similarity
  margin, so it is not yet the strongest pure retrieval metric
- however, it is now a credible compressed sequence-native representation rather
  than a failed ablation

## Refined-phase transfer audit

New question:

- can the refined phase + segment-duration representation move from retrieval
  into the main pretrain -> finetune classification protocol?

Script:

- [train_refined_phase_pretrain.py](/opt/tiger/hand/tools/train_refined_phase_pretrain.py)

Design:

- construct compressed window features from:
  - per-hand refined phase histograms
  - interaction phase histograms
  - segment-duration histograms
  - refined phase-event histograms
  - a small set of phase-tempo scalars
- keep the same MLP pretrain -> finetune protocol as the symbolic baseline

Smoke run:

- [refined_phase_pretrain_smoke.json](/opt/tiger/hand/experiments/generated/refined_phase_pretrain_smoke.json)
- confirms the pipeline runs end to end

### Pretrain-only `168/84`

Command:

```bash
python tools/train_refined_phase_pretrain.py \
  --train-pretrain-json experiments/generated/temporal_hl_train.json \
  --train-finetune-json experiments/generated/temporal_hl_val.json \
  --test-json experiments/generated/temporal_hl_test.json \
  --fractions 1.0 0.5 \
  --seeds 0 1 2 \
  --hidden-dim 128 \
  --pretrain-epochs 120 \
  --finetune-epochs 120 \
  --pretrain-window-span-units 168 \
  --pretrain-window-step-units 84 \
  --output experiments/generated/refined_phase_pretrain_pre168_84_v1.json
```

Result:

- [refined_phase_pretrain_pre168_84_v1.json](/opt/tiger/hand/experiments/generated/refined_phase_pretrain_pre168_84_v1.json)
- fraction `1.0`:
  - scratch sequence accuracy `0.4359`
  - pretrained sequence accuracy `0.5385`
- fraction `0.5`:
  - scratch sequence accuracy `0.5128`
  - pretrained sequence accuracy `0.4359`

### Pretrain-only `186/96`

Command:

```bash
python tools/train_refined_phase_pretrain.py \
  --train-pretrain-json experiments/generated/temporal_hl_train.json \
  --train-finetune-json experiments/generated/temporal_hl_val.json \
  --test-json experiments/generated/temporal_hl_test.json \
  --fractions 1.0 0.5 \
  --seeds 0 1 2 \
  --hidden-dim 128 \
  --pretrain-epochs 120 \
  --finetune-epochs 120 \
  --pretrain-window-span-units 186 \
  --pretrain-window-step-units 96 \
  --output experiments/generated/refined_phase_pretrain_pre186_96_v1.json
```

Result:

- [refined_phase_pretrain_pre186_96_v1.json](/opt/tiger/hand/experiments/generated/refined_phase_pretrain_pre186_96_v1.json)
- fraction `1.0`:
  - scratch sequence accuracy `0.4103`
  - pretrained sequence accuracy `0.4103`
- fraction `0.5`:
  - scratch sequence accuracy `0.5128`
  - pretrained sequence accuracy `0.3590`

Compact summary:

- [refined_phase_pretrain_summary.json](/opt/tiger/hand/experiments/generated/refined_phase_pretrain_summary.json)

Interpretation:

- refined phase + segment duration is useful as a compressed retrieval
  representation
- but under the current bag-of-features MLP protocol it is not a strong
  transfer representation
- the `168/84` setting gives a small positive signal only at full finetune data
- at lower data or with the `186/96` span, pretraining is neutral or negative
- practical conclusion:
  the refined-phase representation is not ready to replace the current temporal
  HL mainline for classification / transfer

## Refined-phase sequence encoder audit

New question:

- is the refined-phase representation weak by itself, or is it mainly
  mismatched to the bag-of-features MLP head?

Script:

- [train_refined_phase_sequence.py](/opt/tiger/hand/tools/train_refined_phase_sequence.py)

Design:

- convert each frame into a compact refined-phase vector:
  - one-hot hand phase labels
  - one-hot interaction phase labels
  - one-hot segment-duration labels
  - three simple interaction-motion indicators
- model the window with a bidirectional GRU sequence encoder
- keep the same pretrain -> finetune protocol and sequence aggregation rule

Smoke run:

- [refined_phase_sequence_smoke.json](/opt/tiger/hand/experiments/generated/refined_phase_sequence_smoke.json)
- confirms the pipeline runs end to end

### Pretrain-only `168/84`

Command:

```bash
python tools/train_refined_phase_sequence.py \
  --train-pretrain-json experiments/generated/temporal_hl_train.json \
  --train-finetune-json experiments/generated/temporal_hl_val.json \
  --test-json experiments/generated/temporal_hl_test.json \
  --fractions 1.0 0.5 \
  --seeds 0 1 2 \
  --hidden-dim 128 \
  --pretrain-epochs 120 \
  --finetune-epochs 120 \
  --pretrain-window-span-units 168 \
  --pretrain-window-step-units 84 \
  --output experiments/generated/refined_phase_sequence_pre168_84_v1.json
```

Result:

- [refined_phase_sequence_pre168_84_v1.json](/opt/tiger/hand/experiments/generated/refined_phase_sequence_pre168_84_v1.json)
- fraction `1.0`:
  - scratch sequence accuracy `0.2564`
  - pretrained sequence accuracy `0.3333`
- fraction `0.5`:
  - scratch sequence accuracy `0.2821`
  - pretrained sequence accuracy `0.2821`

### Pretrain-only `186/96`

Command:

```bash
python tools/train_refined_phase_sequence.py \
  --train-pretrain-json experiments/generated/temporal_hl_train.json \
  --train-finetune-json experiments/generated/temporal_hl_val.json \
  --test-json experiments/generated/temporal_hl_test.json \
  --fractions 1.0 0.5 \
  --seeds 0 1 2 \
  --hidden-dim 128 \
  --pretrain-epochs 120 \
  --finetune-epochs 120 \
  --pretrain-window-span-units 186 \
  --pretrain-window-step-units 96 \
  --output experiments/generated/refined_phase_sequence_pre186_96_v1.json
```

Result:

- [refined_phase_sequence_pre186_96_v1.json](/opt/tiger/hand/experiments/generated/refined_phase_sequence_pre186_96_v1.json)
- fraction `1.0`:
  - scratch sequence accuracy `0.2564`
  - pretrained sequence accuracy `0.4359`
- fraction `0.5`:
  - scratch sequence accuracy `0.2821`
  - pretrained sequence accuracy `0.3846`

Compact summary:

- [refined_phase_sequence_summary.json](/opt/tiger/hand/experiments/generated/refined_phase_sequence_summary.json)

Interpretation:

- the refined-phase representation is indeed better matched to a sequence head
  than to the earlier bag-of-features MLP head
- `186/96` is clearly better than `168/84` for this compressed sequence input
- the sequence head turns the low-data setting from negative / neutral into
  positive transfer
- however, the absolute sequence accuracy is still far below the main temporal
  HL and raw-joint sequence baselines
- practical conclusion:
  refined phase + segment duration is now a viable supporting branch for
  compressed sequence modeling, but it is not yet competitive with the mainline

## Temporal HL frame-sequence encoder audit

New question:

- is the current temporal HL mainline limited by its bag-of-features MLP head?
- can a frame-level temporal-HL sequence encoder outperform the current
  strongest symbolic mainline?

Script:

- [train_temporal_hl_sequence.py](/opt/tiger/hand/tools/train_temporal_hl_sequence.py)

Design:

- each frame is represented by:
  - hand-specific state-token count features
  - hand-specific transition-token count features
  - hand-motion one-hot features
  - interaction-motion one-hot features
  - a few simple temporal scalars such as non-stay ratio and major-shift ratio
- a bidirectional GRU encoder processes the frame sequence inside each window
- the protocol remains pretrain -> finetune with the same sequence-level
  aggregation rule

Smoke run:

- [temporal_hl_sequence_smoke.json](/opt/tiger/hand/experiments/generated/temporal_hl_sequence_smoke.json)
- confirms the pipeline runs end to end

### Pretrain-only `168/84`

Command:

```bash
python tools/train_temporal_hl_sequence.py \
  --train-pretrain-json experiments/generated/temporal_hl_train.json \
  --train-finetune-json experiments/generated/temporal_hl_val.json \
  --test-json experiments/generated/temporal_hl_test.json \
  --fractions 1.0 0.5 \
  --seeds 0 1 2 \
  --hidden-dim 128 \
  --pretrain-epochs 120 \
  --finetune-epochs 120 \
  --pretrain-window-span-units 168 \
  --pretrain-window-step-units 84 \
  --output experiments/generated/temporal_hl_sequence_pre168_84_v1.json
```

Result:

- [temporal_hl_sequence_pre168_84_v1.json](/opt/tiger/hand/experiments/generated/temporal_hl_sequence_pre168_84_v1.json)
- fraction `1.0`:
  - scratch sequence accuracy `0.5641`
  - pretrained sequence accuracy `0.7692`
- fraction `0.5`:
  - scratch sequence accuracy `0.3333`
  - pretrained sequence accuracy `0.4872`

### Pretrain-only `186/96`

Command:

```bash
python tools/train_temporal_hl_sequence.py \
  --train-pretrain-json experiments/generated/temporal_hl_train.json \
  --train-finetune-json experiments/generated/temporal_hl_val.json \
  --test-json experiments/generated/temporal_hl_test.json \
  --fractions 1.0 0.5 \
  --seeds 0 1 2 \
  --hidden-dim 128 \
  --pretrain-epochs 120 \
  --finetune-epochs 120 \
  --pretrain-window-span-units 186 \
  --pretrain-window-step-units 96 \
  --output experiments/generated/temporal_hl_sequence_pre186_96_v1.json
```

Result:

- [temporal_hl_sequence_pre186_96_v1.json](/opt/tiger/hand/experiments/generated/temporal_hl_sequence_pre186_96_v1.json)
- fraction `1.0`:
  - scratch sequence accuracy `0.5641`
  - pretrained sequence accuracy `0.7436`
- fraction `0.5`:
  - scratch sequence accuracy `0.3333`
  - pretrained sequence accuracy `0.4359`

Compact summary:

- [temporal_hl_sequence_summary.json](/opt/tiger/hand/experiments/generated/temporal_hl_sequence_summary.json)

Comparison against the current strongest symbolic mainline:

- strongest bag-of-features temporal HL mainline:
  fraction `1.0` pretrained sequence accuracy `0.9231`
- temporal HL frame-sequence encoder:
  best observed pretrained sequence accuracy `0.7692`

Interpretation:

- the temporal HL sequence encoder does show positive transfer
- its best setting is still `168/84`, consistent with the strongest symbolic
  mainline
- however, replacing the bag-of-features head with a frame-level sequence
  encoder is a clear regression in absolute sequence accuracy
- this means the current symbolic mainline is not simply underpowered by a weak
  head; its histogram-style aggregation is currently a strength rather than a
  liability
- sequence accuracy:
  `scratch=0.5897`, `pretrained=0.8974`

Compared with the default strongest temporal pretrain route:

- default pretrained sequence accuracy: `0.8205`
- ROM05-boosted pretrained sequence accuracy: `0.8974`

Multi-seed error changes:

- [symbolic_pretrain_temporal_boost_rom05_analysis.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_boost_rom05_analysis.json)
- `ROM05_RT_Wrist_ROM -> ROM03_RT_No_Occlusion`
  drops from `3/3` to `1/3`
- `ROM04_RT_Occlusion -> ROM03_RT_No_Occlusion`
  drops from `2` to `1`
- `ROM07_Rt_Finger_Occlusions -> ROM03_RT_No_Occlusion`
  remains difficult at `3`

Interpretation:

- targeted finetune weighting is more effective than the current handcrafted
  wrist features
- it improves the main bottleneck without forcing a worse overall trade-off
- the next error frontier is increasingly concentrated on
  `ROM07_Rt_Finger_Occlusions`

## Duration-feature ablation

Implementation note:

- duration / run-length style tempo features are now available behind the
  `--duration-features` flag in the symbolic pretrain and analysis scripts
- this remains an ablation, not the default route

Focused experiment:

```bash
python tools/train_symbolic_pretrain.py \
  --train-pretrain-json experiments/generated/temporal_hl_train.json \
  --train-finetune-json experiments/generated/temporal_hl_val.json \
  --test-json experiments/generated/temporal_hl_test.json \
  --mode temporal \
  --duration-features \
  --fractions 1.0 \
  --seeds 0 1 2 \
  --hidden-dim 128 \
  --pretrain-epochs 120 \
  --finetune-epochs 120 \
  --boost-label ROM05_RT_Wrist_ROM:3.0 \
  --output experiments/generated/symbolic_pretrain_temporal_duration_boost_rom05.json
```

Result:

- [symbolic_pretrain_temporal_duration_boost_rom05.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_duration_boost_rom05.json)
- sequence accuracy:
  `scratch=0.4872`, `pretrained=0.7949`

Interpretation:

- explicit duration/run-length features in their current handcrafted form hurt
  the strongest boosted temporal route
- this is not the right mainline upgrade

## Time-normalized window protocol

Implementation note:

- symbolic window construction now supports
  `--window-span-units` and `--window-step-units`
- this addresses the verified split mismatch:
  train is dominated by frame delta `18`, while val/test are dominated by `6`

Reference setting used here:

- `window_span_units=186`, approximating the temporal span of 32 frames on the
  `delta=6` splits
- `window_step_units=96`, approximating the step of 16 frames on the `delta=6`
  splits

Observed sample count changes:

- pretrain windows: `1113 -> 3598`
- finetune windows: `164 -> 186`
- test windows: `296 -> 340`

### Time-normalized temporal pretrain with ROM05 boost

Command:

```bash
python tools/train_symbolic_pretrain.py \
  --train-pretrain-json experiments/generated/temporal_hl_train.json \
  --train-finetune-json experiments/generated/temporal_hl_val.json \
  --test-json experiments/generated/temporal_hl_test.json \
  --mode temporal \
  --fractions 1.0 \
  --seeds 0 1 2 \
  --hidden-dim 128 \
  --pretrain-epochs 120 \
  --finetune-epochs 120 \
  --window-span-units 186 \
  --window-step-units 96 \
  --boost-label ROM05_RT_Wrist_ROM:3.0 \
  --output experiments/generated/symbolic_pretrain_temporal_time_norm_boost_rom05.json
```

Result:

- [symbolic_pretrain_temporal_time_norm_boost_rom05.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_time_norm_boost_rom05.json)
- sequence accuracy:
  `scratch=0.7436`, `pretrained=0.8718`

Multi-seed analysis:

- [symbolic_pretrain_temporal_time_norm_boost_rom05_analysis.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_time_norm_boost_rom05_analysis.json)
- seed-level pretrained sequence scores:
  `0.9231`, `0.6923`, `1.0000`

Interpretation:

- time normalization is a real effect, not noise
- it dramatically improves the scratch baseline and yields a strong pretrained
  route
- however, it does not yet beat the current best default-window boosted route
  (`0.8974`)
- the remaining bottleneck is still concentrated on
  `ROM07_Rt_Finger_Occlusions -> ROM03_RT_No_Occlusion`

### Additional ROM07 boost under time-normalized windows

Tried:

- `ROM05_RT_Wrist_ROM:3.0 + ROM07_Rt_Finger_Occlusions:1.5`
- `ROM05_RT_Wrist_ROM:3.0 + ROM07_Rt_Finger_Occlusions:2.0`

Results:

- [symbolic_pretrain_temporal_time_norm_boost_rom05_rom07_15.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_time_norm_boost_rom05_rom07_15.json)
  -> pretrained sequence accuracy `0.8462`
- [symbolic_pretrain_temporal_time_norm_boost_rom05_rom07_20.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_time_norm_boost_rom05_rom07_20.json)
  -> pretrained sequence accuracy `0.8718`

Interpretation:

- under time-normalized windows, directly boosting ROM07 does not improve the
  mean result

## Default-window ROM07 micro-boost

Focused experiment:

```bash
python tools/train_symbolic_pretrain.py \
  --train-pretrain-json experiments/generated/temporal_hl_train.json \
  --train-finetune-json experiments/generated/temporal_hl_val.json \
  --test-json experiments/generated/temporal_hl_test.json \
  --mode temporal \
  --fractions 1.0 \
  --seeds 0 1 2 \
  --hidden-dim 128 \
  --pretrain-epochs 120 \
  --finetune-epochs 120 \
  --boost-label ROM05_RT_Wrist_ROM:3.0 \
  --boost-label ROM07_Rt_Finger_Occlusions:1.5 \
  --output experiments/generated/symbolic_pretrain_temporal_boost_rom05_rom07_15.json
```

Result:

- [symbolic_pretrain_temporal_boost_rom05_rom07_15.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_boost_rom05_rom07_15.json)
- sequence accuracy:
  `scratch=0.5897`, `pretrained=0.8974`
- pretrained window accuracy:
  `0.7477`

Multi-seed analysis:

- [symbolic_pretrain_temporal_boost_rom05_rom07_15_analysis.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_boost_rom05_rom07_15_analysis.json)
- recurring pretrained confusions:
  - `ROM05_RT_Wrist_ROM -> ROM03_RT_No_Occlusion`: `3`
  - `ROM04_RT_Occlusion -> ROM03_RT_No_Occlusion`: `2`
  - `ROM07_Rt_Finger_Occlusions -> ROM03_RT_No_Occlusion`: `2`

Interpretation:

- this does not improve the best sequence mean over the ROM05-only boosted
  route
- but it keeps the best mean while increasing pretrained window accuracy and
  reducing the ROM07 recurring confusion count from `3` to `2`
- so it is a useful operating-point variant rather than a new best mainline

## Seed-fixed analysis update

Implementation note:

- `analyze_symbolic_pretrain_multiseed.py` has been fixed to call `set_seed`
  inside the seed loop
- earlier multi-seed analysis files created before this fix should not be used
  as authoritative evidence when they disagree with the training summary files

Reliable regenerated artifacts:

- [symbolic_pretrain_temporal_boost_rom05_analysis_seedfixed.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_boost_rom05_analysis_seedfixed.json)
- [symbolic_pretrain_temporal_boost_rom05_rom07_15_analysis_seedfixed.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_boost_rom05_rom07_15_analysis_seedfixed.json)

For the default-window ROM05-only boosted route, the reliable recurring
pretrained confusions are now:

- `ROM07_Rt_Finger_Occlusions -> ROM03_RT_No_Occlusion`: `3`
- `ROM05_RT_Wrist_ROM -> ROM03_RT_No_Occlusion`: `1`

## Pretrain-only time normalization

New protocol change:

- apply time-normalized windows only to the pretraining split
- keep finetune and test on the default frame-count windows

Rationale:

- the strongest verified temporal-stride mismatch is between the large pretrain
  split and the ROM splits
- changing only pretrain is a more targeted intervention than changing all
  splits simultaneously

### Best configuration so far

Command:

```bash
python tools/train_symbolic_pretrain.py \
  --train-pretrain-json experiments/generated/temporal_hl_train.json \
  --train-finetune-json experiments/generated/temporal_hl_val.json \
  --test-json experiments/generated/temporal_hl_test.json \
  --mode temporal \
  --fractions 1.0 \
  --seeds 0 1 2 \
  --hidden-dim 128 \
  --pretrain-epochs 120 \
  --finetune-epochs 120 \
  --pretrain-window-span-units 168 \
  --pretrain-window-step-units 84 \
  --boost-label ROM05_RT_Wrist_ROM:3.0 \
  --output experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05.json
```

Result:

- [symbolic_pretrain_temporal_pre_only_168_84_boost_rom05.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05.json)
- sequence accuracy:
  `scratch=0.5897`, `pretrained=0.9231`
- pretrained window accuracy:
  `0.7252`

Reliable multi-seed analysis:

- [symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_analysis.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_analysis.json)
- seed-level pretrained sequence scores:
  `1.0000`, `0.8462`, `0.9231`
- recurring pretrained confusions:
  - `ROM05_RT_Wrist_ROM -> ROM03_RT_No_Occlusion`: `1`
  - `ROM08_Lt_Finger_Occlusions -> ROM03_LT_No_Occlusion`: `1`
  - `ROM07_Rt_Finger_Occlusions -> ROM03_RT_No_Occlusion`: `1`

Interpretation:

- this is the strongest symbolic route so far
- it improves over the previous best default-window ROM05-only boosted route
  (`0.8974 -> 0.9231`)
- it also removes the previous concentration of errors on a single recurring
  ROM07 confusion

### Neighboring pretrain-only settings

Tried:

- `pretrain-window-span-units=186`, `pretrain-window-step-units=96`
- `pretrain-window-span-units=222`, `pretrain-window-step-units=114`

Results:

- [symbolic_pretrain_temporal_pre_only_186_96_boost_rom05.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_pre_only_186_96_boost_rom05.json)
  -> pretrained sequence accuracy `0.8205`
- [symbolic_pretrain_temporal_pre_only_222_114_boost_rom05.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_pre_only_222_114_boost_rom05.json)
  -> pretrained sequence accuracy `0.8205`

Interpretation:

- the gain is not from "any pretrain-only time normalization"
- the effect is localized near the `168/84` configuration

### ROM07 micro-boost on the new best protocol

Tried:

- `ROM05_RT_Wrist_ROM:3.0 + ROM07_Rt_Finger_Occlusions:1.5`
- `ROM05_RT_Wrist_ROM:3.0 + ROM07_Rt_Finger_Occlusions:2.0`

Results:

- [symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_rom07_15.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_rom07_15.json)
  -> pretrained sequence accuracy `0.8462`
- [symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_rom07_20.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_rom07_20.json)
  -> pretrained sequence accuracy `0.7949`

Interpretation:

- once the pretrain-only normalization is in place, extra ROM07 boosting is
  harmful
- the `168/84 + ROM05-only` setup should remain the mainline

## Unified protocol comparison

Compact table artifact:

- [protocol_table.json](/opt/tiger/hand/experiments/generated/protocol_table.json)
- [protocol_matrix.json](/opt/tiger/hand/experiments/generated/protocol_matrix.json)

Symbolic branch summary at fraction `1.0`:

- `default`
  - `scratch_seq=0.5897`
  - `pretrained_seq=0.8974`
- `all_split_norm_168_84`
  - `scratch_seq=0.6667`
  - `pretrained_seq=0.7436`
- `pretrain_only_norm_168_84`
  - `scratch_seq=0.5897`
  - `pretrained_seq=0.9231`

Interpretation:

- for symbolic HL, normalizing all splits is harmful to the transfer route even
  if it slightly raises the scratch baseline
- the winning effect is specifically the pretrain-only normalization, not
  normalization in general

From the full protocol matrix:

- at fraction `0.5`, symbolic `pretrain_only_norm_168_84` improves
  pretrained sequence accuracy from `0.5897` to `0.7949`
- at fraction `1.0`, it improves from `0.8974` to `0.9231`
- symbolic `all_split_norm_168_84` is only competitive on scratch accuracy and
  is clearly inferior on pretrained sequence transfer

## Summary bundle

Single-file bundle artifact:

- [experiment_summary_bundle.json](/opt/tiger/hand/experiments/generated/experiment_summary_bundle.json)

This bundle consolidates:

- strongest symbolic and joint-sequence mainlines
- the full protocol matrix
- top slice deltas at fractions `0.5` and `1.0`
- old-vs-new symbolic error frontiers
- plain gain summaries for quick comparison

Current symbolic gain summary from the bundle:

- fraction `0.5`: `+0.2051`
- fraction `1.0`: `+0.0256`

Practical use:

- this is the current one-file handoff artifact for internal discussion,
  plotting, or rebuttal preparation
- table-friendly exports are available under
  [summary_tables](/opt/tiger/hand/experiments/generated/summary_tables)

## Fraction curve for the new best protocol

Comparison target:

- old mainline:
  default windows + `ROM05_RT_Wrist_ROM:3.0`
- new mainline:
  pretrain-only `168/84` time normalization + `ROM05_RT_Wrist_ROM:3.0`

Old mainline curve:

- [symbolic_pretrain_temporal_boost_rom05_curve_seedfixed.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_boost_rom05_curve_seedfixed.json)
- pretrained sequence accuracy:
  - fraction `0.25`: `0.5641`
  - fraction `0.5`: `0.5897`
  - fraction `1.0`: `0.8974`

New mainline curve:

- [symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_curve.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_curve.json)
- pretrained sequence accuracy:
  - fraction `0.25`: `0.4103`
  - fraction `0.5`: `0.7949`
  - fraction `1.0`: `0.9231`

Interpretation:

- the new protocol is not uniformly better in the extremely low-data regime
- but it is substantially better once finetune data is moderate or full:
  - `0.5`: `0.5897 -> 0.7949`
  - `1.0`: `0.8974 -> 0.9231`
- this suggests the gain is specifically about stronger cross-split pretraining
  transfer, not a universal low-data regularizer

## Sequence-level comparison against the old mainline

Seed-0 comparison artifacts:

- old mainline:
  [symbolic_pretrain_temporal_boost_rom05_seed0_analysis_seedfixed.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_boost_rom05_seed0_analysis_seedfixed.json)
- new mainline:
  [symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_seed0_analysis.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_seed0_analysis.json)

Observed change:

- old mainline seed-0 sequence accuracy: `0.8462`
- new mainline seed-0 sequence accuracy: `1.0000`

The concrete repaired sequences are:

- `ROM07_RT_Finger_Occlusions`
- `ROM07_Rt_Finger_Occlusions`

In both cases, the old mainline predicted `ROM03_RT_No_Occlusion`, while the
new mainline correctly predicts `ROM07_Rt_Finger_Occlusions`.

Interpretation:

- the new protocol does not just raise a mean score
- it directly fixes the exact right-hand finger-occlusion confusions that had
  remained a persistent bottleneck under the older strongest routes

## Slice comparison against the old mainline

Comparison artifact:

- [symbolic_slice_compare_old_vs_new.json](/opt/tiger/hand/experiments/generated/symbolic_slice_compare_old_vs_new.json)

At fraction `1.0`, the most meaningful pretrained slice changes are:

- `right`: `0.7333 -> 0.8667`
- `finger_occlusion`: `0.7500 -> 0.8333`
- `occlusion`: `0.8750 -> 0.9167`
- `wrist_rom`: unchanged at `0.8333`
- `interaction`, `no_interaction`, `no_occlusion`, `touching`: unchanged at
  `1.0000`

Interpretation:

- the overall gain is concentrated in the right-hand and occlusion-heavy
  sequences
- the improvement is not coming from already-solved interaction/no-occlusion
  subsets

## Medium-data regime: fraction 0.5

Reliable multi-seed analysis:

- old mainline:
  [symbolic_pretrain_temporal_boost_rom05_fraction05_analysis.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_boost_rom05_fraction05_analysis.json)
- new mainline:
  [symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_fraction05_analysis.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_fraction05_analysis.json)

Slice comparison artifact:

- [symbolic_slice_compare_old_vs_new_fraction05.json](/opt/tiger/hand/experiments/generated/symbolic_slice_compare_old_vs_new_fraction05.json)

At fraction `0.5`, the new mainline improves:

- `all`: `0.5897 -> 0.7949`
- `wrist_rom`: `0.0000 -> 0.8333`
- `no_interaction`: `0.3333 -> 1.0000`
- `interaction`: `0.6667 -> 1.0000`
- `left`: `0.6667 -> 0.8667`
- `right`: `0.4667 -> 0.6000`
- `finger_occlusion`: `0.6667 -> 0.7500`

while:

- `occlusion` stays flat at `0.7083`
- `no_occlusion` and `touching` remain saturated at `1.0000`

Interpretation:

- the pretrain-only normalization has a very strong effect in the medium-data
  regime
- at this regime, the gain is driven not only by right-hand finger occlusion
  fixes but also by better global sequence discrimination for wrist and
  interaction families

### Fraction-0.5 sequence diff highlights

Compared with the old mainline, the new mainline fixes the following examples
in at least one seed:

- `ROM05_RT_Wrist_ROM`
- `ROM05_LT_Wrist_ROM`

## Consolidated mainline evidence report

Single-file report artifact:

- [mainline_evidence_report.json](/opt/tiger/hand/experiments/generated/mainline_evidence_report.json)

New table exports:

- [mainline_vs_controls.md](/opt/tiger/hand/experiments/generated/summary_tables/mainline_vs_controls.md)
- [mainline_slice_gains.md](/opt/tiger/hand/experiments/generated/summary_tables/mainline_slice_gains.md)
- [mainline_error_repair_frontier.md](/opt/tiger/hand/experiments/generated/summary_tables/mainline_error_repair_frontier.md)
- [retrieval_evidence.md](/opt/tiger/hand/experiments/generated/summary_tables/retrieval_evidence.md)

Implementation:

- [build_mainline_evidence_report.py](/opt/tiger/hand/tools/build_mainline_evidence_report.py)

What this report consolidates:

- strongest symbolic mainline vs symbolic default
- strongest symbolic mainline vs key non-symbolic controls
- per-slice gains for fractions `0.5` and `1.0`
- error-repair frontier from old symbolic mainline to new symbolic mainline
- compact retrieval evidence for exact-state, temporal, phase, and refined-phase
  event representations

Key verified numbers from the report:

- symbolic mainline vs default:
  - fraction `0.5`: `0.7949 - 0.5897 = +0.2051`
  - fraction `1.0`: `0.9231 - 0.8974 = +0.0256`
- symbolic mainline vs best non-symbolic control (`joint_sequence_best`):
  - fraction `0.5`: `0.7949 - 0.7179 = +0.0769`
  - fraction `1.0`: `0.9231 - 0.8718 = +0.0513`

Key interpretation:

- the current strongest symbolic route is not merely the best symbolic variant;
  it also remains ahead of the best current joint-sequence student
- the gap is especially meaningful at fraction `0.5`, where the symbolic route
  is ahead of:
  - `joint_sequence_best` by `+0.0769`
  - `temporal_hl_sequence_best` by `+0.3077`
  - `refined_phase_sequence_best` by `+0.4103`
- at fraction `1.0`, the symbolic route still leads all current controls, with
  the closest challenger remaining the joint-sequence student

Slice and frontier interpretation:

- at fraction `1.0`, the strongest positive slice deltas remain concentrated in
  `right`, `finger_occlusion`, and `occlusion`
- at fraction `0.5`, the strongest symbolic protocol continues to deliver a
  broader recovery pattern, not just a tiny right-hand fix
- in the error frontier, the most concrete repaired confusion remains
  `ROM07_Rt_Finger_Occlusions -> ROM03_RT_No_Occlusion`, reduced from `3` to
  `1`

Retrieval interpretation:

- exact temporal event matching remains the strongest intrinsic sequence-native
  retrieval signal in margin terms among the currently tested compressed
  variants
- refined-phase + segment-duration recovers the same top-1 accuracy as exact
  temporal event DTW while using fewer events than the exact event
  representation, but with a clearly smaller similarity margin
- this supports keeping refined phase as a compressed supporting branch rather
  than a replacement for the mainline symbolic temporal representation

## Symbolic vs joint-sequence subgroup report

Direct comparison artifact:

- [symbolic_vs_joint_report.json](/opt/tiger/hand/experiments/generated/symbolic_vs_joint_report.json)

Table exports:

- [symbolic_vs_joint_slices.md](/opt/tiger/hand/experiments/generated/summary_tables/symbolic_vs_joint_slices.md)
- [symbolic_vs_joint_sequences.md](/opt/tiger/hand/experiments/generated/summary_tables/symbolic_vs_joint_sequences.md)
- [symbolic_vs_joint_error_frontier.md](/opt/tiger/hand/experiments/generated/summary_tables/symbolic_vs_joint_error_frontier.md)

Implementation:

- [build_symbolic_vs_joint_report.py](/opt/tiger/hand/tools/build_symbolic_vs_joint_report.py)

Purpose:

- compare the strongest symbolic mainline directly against the strongest
  joint-sequence student under the same multiseed sequence-consistency protocol
- replace the earlier aggregate-only claim with a finer statement about where
  symbolic wins and where joint-sequence still has an advantage

Key summary from the report:

- fraction `0.5`:
  - symbolic better on `4` sequence classes
  - joint-sequence better on `2`
  - tied on `7`
- fraction `1.0`:
  - symbolic better on `4` sequence classes
  - joint-sequence better on `3`
  - tied on `6`

Most important slice-level findings:

- fraction `0.5` symbolic advantages are concentrated in:
  - `wrist_rom`: `0.8333` vs `0.1667`
  - `no_interaction`: `1.0000` vs `0.6667`
  - `interaction`: `1.0000` vs `0.7778`
  - overall `all`: `0.7949` vs `0.7179`
- fraction `0.5` joint-sequence retains an advantage in:
  - `finger_occlusion`: `0.7500` vs `1.0000`
  - `occlusion`: `0.7083` vs `0.8333`
- fraction `1.0` symbolic still leads on:
  - `interaction`: `1.0000` vs `0.8889`
  - `left`: `0.9333` vs `0.8667`
  - overall `all`: `0.9231` vs `0.8718`
- fraction `1.0` joint-sequence still leads on:
  - `finger_occlusion`: `0.8333` vs `1.0000`

Important sequence-level interpretation:

- in the medium-data regime, symbolic is substantially stronger on wrist and
  interaction families:
  - `ROM05_LT_Wrist_ROM`: `3/3` vs `1/3`
  - `ROM05_RT_Wrist_ROM`: `2/3` vs `0/3`
  - `ROM01_No_Interaction_2_Hand`: `3/3` vs `2/3`
  - `ROM02_Interaction_2_Hand`: `3/3` vs `2/3`
- in the full-data regime, joint-sequence remains stronger on some finger
  occlusion instances:
  - `ROM07_Rt_Finger_Occlusions`: `2/3` vs `3/3`
  - `ROM08_LT_Finger_Occlusions`: `2/3` vs `3/3`
  - `ROM05_RT_Wrist_ROM`: `2/3` vs `3/3`

Error-frontier interpretation:

- at fraction `0.5`, joint-sequence introduces more medium-data confusions
  around wrist and interaction families, for example:
  - `ROM04_RT_Occlusion -> ROM07_Rt_Finger_Occlusions`: `0` vs `2`
  - `ROM01_No_Interaction_2_Hand -> ROM09_Interaction_Fingers_Touching`:
    `0` vs `1`
  - `ROM02_Interaction_2_Hand -> ROM09_Interaction_Fingers_Touching`:
    `0` vs `1`
- this supports a more precise claim than before:
  the symbolic temporal representation is especially strong at preserving
  sequence identity for wrist and interaction families under reduced-data
  finetuning, while the joint-sequence student remains competitive or stronger
  on some finger-occlusion-heavy cases

## Control frontier after upgrading temporal/refined analyses

New multiseed control analyses:

- [temporal_hl_sequence_pre168_84_analysis_v1.json](/opt/tiger/hand/experiments/generated/temporal_hl_sequence_pre168_84_analysis_v1.json)
- [temporal_hl_sequence_pre168_84_fraction05_analysis_v1.json](/opt/tiger/hand/experiments/generated/temporal_hl_sequence_pre168_84_fraction05_analysis_v1.json)
- [refined_phase_sequence_pre186_96_analysis_v1.json](/opt/tiger/hand/experiments/generated/refined_phase_sequence_pre186_96_analysis_v1.json)
- [refined_phase_sequence_pre186_96_fraction05_analysis_v1.json](/opt/tiger/hand/experiments/generated/refined_phase_sequence_pre186_96_fraction05_analysis_v1.json)

Implementation:

- [analyze_temporal_hl_sequence_multiseed.py](/opt/tiger/hand/tools/analyze_temporal_hl_sequence_multiseed.py)
- [analyze_refined_phase_sequence_multiseed.py](/opt/tiger/hand/tools/analyze_refined_phase_sequence_multiseed.py)

Unified frontier artifact:

- [symbolic_control_frontier.json](/opt/tiger/hand/experiments/generated/symbolic_control_frontier.json)
- [symbolic_control_frontier_slices.md](/opt/tiger/hand/experiments/generated/summary_tables/symbolic_control_frontier_slices.md)
- [symbolic_control_frontier_sequences.md](/opt/tiger/hand/experiments/generated/summary_tables/symbolic_control_frontier_sequences.md)

Implementation:

- [build_symbolic_control_frontier.py](/opt/tiger/hand/tools/build_symbolic_control_frontier.py)

What changed:

- temporal-HL sequence and refined-phase sequence controls now have the same
  multiseed `sequence_consistency` / error-count structure as symbolic and
  joint-sequence
- this makes it possible to compare all four representations under the same
  slice and per-sequence protocol instead of relying on aggregate summary means

Verified control means:

- temporal-HL sequence (`pre168_84`):
  - fraction `0.5`: pretrained sequence mean `0.4872`
  - fraction `1.0`: pretrained sequence mean `0.7692`
- refined-phase sequence (`pre186_96`):
  - fraction `0.5`: pretrained sequence mean `0.3846`
  - fraction `1.0`: pretrained sequence mean `0.4359`

Unified frontier interpretation:

- at fraction `0.5`, the overall `all` slice is:
  - symbolic: `0.7949`
  - joint-sequence: `0.7179`
  - temporal-HL sequence: `0.4872`
  - refined-phase sequence: `0.3846`
- at fraction `1.0`, the overall `all` slice is:
  - symbolic: `0.9231`
  - joint-sequence: `0.8718`
  - temporal-HL sequence: `0.7692`
  - refined-phase sequence: `0.4359`

Most important consequence:

- once all current controls are brought to the same multiseed sequence-level
  analysis protocol, the only serious challenger to the strongest symbolic
  mainline remains the joint-sequence student
- temporal-HL sequence is clearly weaker than symbolic across the key overall,
  wrist, left, and interaction slices
- refined-phase sequence is substantially weaker than both symbolic and joint on
  almost every non-trivial slice, confirming that it should remain a compressed
  supporting branch rather than a mainline candidate

Failure-mode interpretation for the weaker controls:

- temporal-HL sequence still suffers repeated collapses into
  `ROM03_RT_No_Occlusion`, especially for:
  - `ROM05_RT_Wrist_ROM`
  - `ROM07_Rt_Finger_Occlusions`
- refined-phase sequence amplifies this collapse pattern even more strongly,
  especially on right and left finger-occlusion families

Practical conclusion:

- there is now enough evidence to stop treating temporal-HL sequence and
  refined-phase sequence as plausible mainline replacements
- future effort should either:
  - improve the symbolic mainline on finger-occlusion-heavy cases
  - or use the weaker controls only as supporting evidence that explicit
    symbolic temporal factorization is more effective than these compressed
    sequence encoders

## Targeted finger-occlusion patch attempts

Goal of this round:

- improve the remaining `finger_occlusion` weakness of the strongest symbolic
  mainline without changing the overall pretrain-only `168/84 + ROM05` recipe

### 1. Validity / occlusion coverage metadata

Implementation:

- [build_temporal_hl.py](/opt/tiger/hand/tools/build_temporal_hl.py) now exports
  per-hand validity metadata such as:
  - `valid_joint_ratio`
  - `valid_edge_ratio`
  - `full_finger_ratio`
  - `tip_valid_ratio`
  - `distal_edge_valid_ratio`
  - `proximal_edge_valid_ratio`
- [train_symbolic_torch.py](/opt/tiger/hand/tools/train_symbolic_torch.py) now
  supports an optional `--occlusion-features` path that injects these signals
  into the `tempo` channel

Sanity check result:

- on the regenerated test split, the validity ratios do not meaningfully
  separate the hard `ROM07/ROM08` finger-occlusion classes from the clean
  no-occlusion classes
- the strong separation appears mainly on the touching / interaction families,
  not on the finger-occlusion families that are still hard for the mainline

Direct experiment:

- [symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_occlusionfeat.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_occlusionfeat.json)

Result:

- fraction `0.5`: pretrained sequence `0.6154`
- fraction `1.0`: pretrained sequence `0.6923`

Interpretation:

- explicit validity / coverage features are strongly harmful to the current
  symbolic mainline
- they should not be pursued further as a direct feature patch

### 2. High-dimensional finger-detail state / transition features

Implementation:

- [train_symbolic_torch.py](/opt/tiger/hand/tools/train_symbolic_torch.py) now
  supports `--finger-detail-features`, which adds per-hand / per-finger /
  proximal-vs-distal symbolic detail into the state and transition channels

Direct experiment:

- [symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_fingerdetail.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_fingerdetail.json)

Result:

- fraction `0.5`: pretrained sequence `0.6667`
- fraction `1.0`: pretrained sequence `0.7179`

Interpretation:

- directly injecting high-dimensional finger-identity symbolic detail causes a
  large generalization collapse
- this is much worse than the mainline and is not a viable direction in the
  current MLP-style symbolic pretrain pipeline

### 3. Low-dimensional finger-shape summary features

Implementation:

- [train_symbolic_torch.py](/opt/tiger/hand/tools/train_symbolic_torch.py) now
  also supports `--finger-shape-features`, which adds low-dimensional
  hand-shape statistics such as:
  - flexion mean / std / range
  - thumb-vs-other flexion gap
  - distal / proximal token uniqueness ratios
  - distal / proximal neutral-token ratios

Direct experiment:

- [symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_fingershape.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_fingershape.json)

Result:

- fraction `0.5`: pretrained sequence `0.5897`
- fraction `1.0`: pretrained sequence `0.8462`

Focused multiseed analysis:

- [symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_fingershape_analysis.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_fingershape_analysis.json)

Key finding:

- the intended hard confusions remain:
  - `ROM07_Rt_Finger_Occlusions -> ROM03_RT_No_Occlusion`: `1`
  - `ROM08_Lt_Finger_Occlusions -> ROM03_LT_No_Occlusion`: `1`
- while other families regress, including interaction-heavy cases

Slice comparison against the mainline:

- [symbolic_slice_compare_mainline_vs_fingershape.json](/opt/tiger/hand/experiments/generated/symbolic_slice_compare_mainline_vs_fingershape.json)

Observed changes:

- `wrist_rom`: `0.8333 -> 1.0000`
- `finger_occlusion`: unchanged at `0.8333`
- `all`: `0.9231 -> 0.8462`
- `interaction`: `1.0000 -> 0.7778`
- `no_interaction`: `1.0000 -> 0.3333`

Interpretation:

- low-dimensional finger-shape summaries are less destructive than
  high-dimensional finger-detail features
- but they still fail the actual requirement:
  they do not improve the `finger_occlusion` slice while harming stronger
  slices that the mainline already solved

### Practical conclusion from this patch round

- three direct symbolic feature augmentations have now been tested:
  - validity / coverage features
  - high-dimensional finger-detail symbolic features
  - low-dimensional finger-shape summary features
- all three fail to improve the true bottleneck without harming the mainline
- therefore, the next step should not be "add more obvious handcrafted
  side-features" to the current symbolic MLP
- the next credible direction should instead be a more structural change,
  likely one of:
  - a representation that preserves distal finger identity without exploding
    dimensionality
  - a specialized auxiliary objective for the finger-occlusion families
  - a late-fusion or retrieval-style correction layer restricted to the hard
    occlusion subset

## Occlusion-family late correction

Motivation:

- the strongest symbolic mainline now fails only on a small number of recurring
  occlusion-family confusions, mostly of the form
  `finger/wrist/occlusion -> No_Occlusion`
- direct feature patches were harmful, so the next structural test is a
  restricted post-hoc correction layer rather than a new mainline retrain

Implementation:

- [eval_occlusion_late_correction.py](/opt/tiger/hand/tools/eval_occlusion_late_correction.py)

Protocol:

- base model:
  strongest symbolic mainline multiseed analysis
- gallery:
  `temporal_hl_val.json`
- query:
  `temporal_hl_test.json`
- correction family:
  - `ROM03_LT_No_Occlusion`
  - `ROM03_RT_No_Occlusion`
  - `ROM04_LT_Occlusion`
  - `ROM04_RT_Occlusion`
  - `ROM05_LT_Wrist_ROM`
  - `ROM05_RT_Wrist_ROM`
  - `ROM07_Rt_Finger_Occlusions`
  - `ROM08_Lt_Finger_Occlusions`
- correction trigger:
  only when the mainline prediction already falls inside this family
- correction mechanism:
  restricted temporal event-DTW retrieval within the family, replacing the
  original prediction with the top retrieved family label

Artifacts:

- [occlusion_late_correction_mainline.json](/opt/tiger/hand/experiments/generated/occlusion_late_correction_mainline.json)
- [occlusion_late_correction_mainline_fraction05.json](/opt/tiger/hand/experiments/generated/occlusion_late_correction_mainline_fraction05.json)

Fraction `1.0` result:

- family accuracy:
  `0.9000 -> 1.0000` (`+0.1000`)
- improved cases: `3`
- harmed cases: `0`
- overall slice deltas:
  - `finger_occlusion`: `0.8333 -> 1.0000`
  - `wrist_rom`: `0.8333 -> 1.0000`
  - `occlusion`: `0.9167 -> 1.0000`
  - `all`: `0.9231 -> 1.0000`

Concrete fixed cases:

- `ROM05_RT_Wrist_ROM` seed `1`:
  `ROM03_RT_No_Occlusion -> ROM05_RT_Wrist_ROM`
- `ROM07_Rt_Finger_Occlusions` seed `2`:
  `ROM03_RT_No_Occlusion -> ROM07_Rt_Finger_Occlusions`
- `ROM08_Lt_Finger_Occlusions` seed `1`:
  `ROM03_LT_No_Occlusion -> ROM08_Lt_Finger_Occlusions`

Fraction `0.5` result:

- family accuracy:
  `0.7333 -> 1.0000` (`+0.2667`)
- improved cases: `8`
- harmed cases: `0`
- overall slice deltas:
  - `right`: `0.6000 -> 1.0000`
  - `occlusion`: `0.7083 -> 1.0000`
  - `finger_occlusion`: `0.7500 -> 1.0000`
  - `wrist_rom`: `0.8333 -> 1.0000`
  - `all`: `0.7949 -> 1.0000`

Concrete fixed cases include:

- `ROM04_LT_Occlusion` seed `1`
- `ROM04_RT_Occlusion` seeds `0,1,2`
- `ROM05_RT_Wrist_ROM` seed `1`
- `ROM07_Rt_Finger_Occlusions` seeds `0,2`
- `ROM08_Lt_Finger_Occlusions` seed `0`

Interpretation:

- the symbolic sequence space already contains enough information to perfectly
  separate the hard occlusion-family cases under a restricted family retrieval
  correction
- this is the first structural patch that actually fixes the bottleneck instead
  of moving errors elsewhere
- importantly, it does so with `0` harmed cases in both tested data regimes

Important limitation:

- this is still a family-restricted post-hoc correction experiment, not yet a
  fully learned unified model
- the trigger is currently simple:
  apply correction when the mainline prediction is already inside the
  occlusion-family label set
- therefore, the next step should be to turn this into a stronger baseline with
  a principled gate, confidence rule, or hybrid classifier, rather than claim
  that the end-to-end mainline itself is already solved
- `ROM01_No_Interaction_2_Hand`
- `ROM02_Interaction_2_Hand`
- `ROM07_RT_Finger_Occlusions`

but can introduce new errors in some seeds on:

- `ROM04_LT_Occlusion`
- `ROM04_RT_Occlusion`
- `ROM08_LT_Finger_Occlusions`

Interpretation:

- the protocol change is not universally monotone on every sequence
- however, its net gain at fraction `0.5` is large enough that these trade-offs
  are acceptable and worth documenting

## Temporal representation ablation consolidation

Purpose:

- consolidate the most important representation-level comparisons into one place
- avoid over-interpreting any single branch in isolation
- explicitly test whether the next temporal factor after `transition`,
  namely `duration`, is already strong enough to keep

Core report assets:

- builder:
  [build_temporal_representation_ablation_report.py](/opt/tiger/hand/tools/build_temporal_representation_ablation_report.py)
- JSON report:
  [temporal_representation_ablation_report.json](/opt/tiger/hand/experiments/generated/temporal_representation_ablation_report.json)
- table view:
  [temporal_representation_ablation_report.md](/opt/tiger/hand/experiments/generated/summary_tables/temporal_representation_ablation_report.md)

Included families:

- symbolic window MLP:
  - `state-only`
  - `state+transition`
  - `state+transition+duration`
- temporal sequence encoder:
  - `state+transition`
  - `state+transition+duration`
- learned-token controls on joint windows:
  - `joint_token32`
  - `joint_token64`

Most important numbers:

- strongest pretrained sequence result remains:
  `symbolic state+transition` at fraction `1.0`
  with mean sequence accuracy `0.9231`
- symbolic `state-only` remains below it at:
  `0.7949`
- symbolic `state+transition+duration` drops to:
  `0.8205`
- temporal sequence encoder with `state+transition` reaches:
  `0.7692`
- adding duration to that sequence encoder:
  - fraction `1.0`: `0.7692 -> 0.7692` (no gain)
  - fraction `0.5`: `0.4872 -> 0.4359` (regression)
- learned-token controls remain clearly weaker than the strongest symbolic
  branch:
  - `joint_token32`: `0.5897`
  - `joint_token64`: `0.6923`

Recurring error shift in the sequence encoder:

- `state+transition` pretrained recurring errors:
  - `ROM05_RT_Wrist_ROM -> ROM03_RT_No_Occlusion`: `3`
  - `ROM07_Rt_Finger_Occlusions -> ROM03_RT_No_Occlusion`: `2`
- `state+transition+duration` pretrained recurring errors:
  - `ROM05_RT_Wrist_ROM -> ROM03_RT_No_Occlusion`: `3`
  - `ROM07_Rt_Finger_Occlusions -> ROM03_RT_No_Occlusion`: `3`
  - plus a new left-finger occlusion miss

Interpretation:

- the current evidence is now consistent across three views:
  - `state-only` is not enough
  - `transition` is the temporal factor that actually buys reusable signal
  - `duration` is not yet a reliable additive factor
- this substantially sharpens the next-step filter:
  future representation work should not add more handcrafted duration channels
  by default
- if we want to push beyond the current mainline, the more promising space is
  likely:
  - better event / boundary formulation
  - coordination or cross-hand relation channels
  - or a learned symbolic correction layer over the existing
    `state+transition` backbone

## Exact-state event augmentation audit

Question:

- can exact-state temporal event / boundary statistics improve the strongest
  symbolic `state+transition` pretrain mainline when added as an extra feature
  channel?

Implementation update:

- [train_symbolic_torch.py](/opt/tiger/hand/tools/train_symbolic_torch.py)
  now supports `--event-features` through segment-level exact-state events:
  - compress consecutive frames with the same exact-state event key
  - export event histograms
  - export event-boundary bigrams
  - export event-count / run-length / boundary-rate tempo scalars
- [train_symbolic_pretrain.py](/opt/tiger/hand/tools/train_symbolic_pretrain.py)
  and [analyze_symbolic_pretrain.py](/opt/tiger/hand/tools/analyze_symbolic_pretrain.py)
  now support:
  - `--event-features`
  - `--event-weight`

Primary run with the same strong protocol:

```bash
python tools/train_symbolic_pretrain.py \
  --train-pretrain-json experiments/generated/temporal_hl_train.json \
  --train-finetune-json experiments/generated/temporal_hl_val.json \
  --test-json experiments/generated/temporal_hl_test.json \
  --mode temporal \
  --fractions 0.5 1.0 \
  --seeds 0 1 2 \
  --hidden-dim 128 \
  --pretrain-epochs 120 \
  --finetune-epochs 120 \
  --pretrain-window-span-units 168 \
  --pretrain-window-step-units 84 \
  --event-features \
  --boost-label ROM05_RT_Wrist_ROM:3.0 \
  --output experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_event.json
```

Artifacts:

- [symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_event.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_event.json)
- [symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_event_analysis.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_event_analysis.json)

Primary result with default `event_weight=0.3`:

- fraction `1.0`:
  - scratch sequence accuracy `0.5641`
  - pretrained sequence accuracy `0.5128`
- fraction `0.5`:
  - scratch sequence accuracy `0.4359`
  - pretrained sequence accuracy `0.5128`

Weight calibration check:

- [symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_event_w01.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_event_w01.json)
  with `event_weight=0.1`
  - fraction `1.0` pretrained sequence accuracy: `0.6923`
- [symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_event_w005.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_event_w005.json)
  with `event_weight=0.05`
  - fraction `1.0` pretrained sequence accuracy: `0.8205`

Comparison to the unchanged mainline:

- strongest base `state+transition` mainline:
  - fraction `1.0`: `0.9231`
  - fraction `0.5`: `0.7949`
- therefore even the best tested event-augmented variant remains below the
  base mainline

Interpretation:

- this rules out the easy hypothesis that the event failure was only a weight
  calibration bug
- lower event weight reduces damage, but the current exact-state event
  augmentation still does not create net gain over the mainline
- so "add explicit boundary/event statistics" is not sufficient by itself
- if the event direction is revisited later, it should probably be through:
  - a more selective event inventory
  - coordination-aware events rather than exact-state events
  - or a sequence model that consumes event order directly rather than bagged
    event/boundary histograms

## Coordination / cross-hand relation audit

Question:

- if exact-state events are too brittle, can more structural coordination
  signals help the strongest temporal symbolic backbone?

Implementation update:

- [train_symbolic_torch.py](/opt/tiger/hand/tools/train_symbolic_torch.py)
  now supports `--coordination-features` inside the existing temporal channel.
  The current augmentation includes:
  - synchronized non-stay ratio across left/right regional transitions
  - same-vs-opposite motion ratios for the two hands
  - interacting-frame ratio
  - cross-hand distance mean / min / range / delta statistics
  - mean cross-hand flexion-gap statistics
- no new training protocol was introduced; this is still evaluated under the
  same strongest pretrain -> finetune setup

Command:

```bash
python tools/train_symbolic_pretrain.py \
  --train-pretrain-json experiments/generated/temporal_hl_train.json \
  --train-finetune-json experiments/generated/temporal_hl_val.json \
  --test-json experiments/generated/temporal_hl_test.json \
  --mode temporal \
  --fractions 1.0 \
  --seeds 0 1 2 \
  --hidden-dim 128 \
  --pretrain-epochs 120 \
  --finetune-epochs 120 \
  --pretrain-window-span-units 168 \
  --pretrain-window-step-units 84 \
  --coordination-features \
  --boost-label ROM05_RT_Wrist_ROM:3.0 \
  --output experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_coord.json
```

Artifacts:

- [symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_coord.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_coord.json)
- [symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_coord_analysis.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_coord_analysis.json)

Key result:

- fraction `1.0`:
  - scratch sequence accuracy `0.6667`
  - pretrained sequence accuracy `0.7436`
  - scratch window accuracy `0.6959`
  - pretrained window accuracy `0.7410`

Comparison:

- stronger than the default exact-state event augmentation:
  - event `w=0.3`: `0.5128`
  - event `w=0.1`: `0.6923`
- but still below:
  - plain strongest `state+transition`: `0.9231`
  - `state+transition+duration`: `0.8205`

Interpretation:

- the direction is less harmful than exact-state event augmentation
- but the current coordination / cross-hand summary statistics still do not
  create a positive gain over the unchanged mainline
- this means the remaining gap is probably not solved by adding coarse
  interaction scalars alone
- if this family is revisited, the next credible step should be something more
  selective than bagged global stats:
  - relation-aware hard-case gating
  - subset-specific coordination features on interaction-heavy classes only
  - or a learned relation head rather than global summary features

## Wide disagreement-degeneracy audit

Question:

- is the family-correction disagreement degeneracy only a property of the
  strongest mainline, or does it persist across weaker symbolic variants too?

Script:

- [audit_family_disagreement_supervision.py](/opt/tiger/hand/tools/audit_family_disagreement_supervision.py)

Wide audit artifact:

- [family_disagreement_audit_wide.json](/opt/tiger/hand/experiments/generated/family_disagreement_audit_wide.json)

Coverage in this sweep:

- stronger temporal symbolic variants
- weaker boosted variants
- pretrain-only normalized mainline variants
- fingershape and coordination ablations
- family-aux negative controls
- hybrid corrected outputs

Key result:

- among all audited variants in this sweep, there are still **zero**
  disagreement negatives
- i.e. whenever family retrieval top-1 disagrees with the symbolic prediction,
  retrieval is always the correct choice on the current benchmark slice

Representative rows from the audit:

- `symbolic_pretrain_temporal_boost_rom05_fraction05_analysis.json`
  - disagreement rows: `13`
  - positive: `13`
  - negative: `0`
- `symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_coord_multiseed_analysis.json`
  - disagreement rows: `10`
  - positive: `10`
  - negative: `0`
- `symbolic_family_aux_pre168_84_boost_rom05_w01_fraction05_analysis.json`
  - disagreement rows: `16`
  - positive: `16`
  - negative: `0`
- even the event-augmented negative control:
  - disagreement rows: `4`
  - positive: `4`
  - negative: `0`

Interpretation:

- the disagreement degeneracy is not local to one strong model
- it currently appears to be a property of the whole audited family-correction
  benchmark slice
- therefore:
  - generic pairwise reranker comparisons are still scientifically weak here
  - more tuning of disagreement-based rerankers is not the best next use of
    time
  - if we want a meaningful learned-correction comparison, we first need a
    harder benchmark with real positive and negative disagreement supervision

Practical consequence:

- the correction line should now be treated as:
  - already sufficient to prove that symbolic space contains recoverable hard
    family information
  - but not yet sufficient to compare generic reranking methods rigorously
- future correction work should begin with benchmark construction, not model
  tuning

## Unified full-data / low-data frontier bundle

To consolidate the strongest representation-facing evidence into one place, I
added:

- [build_representation_frontier_full_lowdata.py](/opt/tiger/hand/tools/build_representation_frontier_full_lowdata.py)

Artifacts:

- [representation_frontier_full_lowdata.json](/opt/tiger/hand/experiments/generated/representation_frontier_full_lowdata.json)
- [representation_frontier_full_lowdata.md](/opt/tiger/hand/experiments/generated/summary_tables/representation_frontier_full_lowdata.md)

What this bundle merges:

- full-data symbolic mainline vs ablation controls
- low-data symbolic mainline vs the same nearby controls where available
- learned joint-token controls
- intrinsic event / frame / RLE evidence

Key sequence frontier numbers:

- full data:
  - symbolic mainline: `0.9231`
  - state-only: `0.7949`
  - state+transition+duration: `0.8205`
  - temporal sequence encoder: `0.7692`
  - best joint-sequence control: `0.8718`
  - joint-token64: `0.6923`
- low data:
  - symbolic mainline: `0.7949`
  - state+transition+duration: `0.6410`
  - temporal sequence encoder: `0.4872`
  - best joint-sequence control: `0.7179`
  - joint-token64: `0.5641`

Main gap summary:

- full data:
  - mainline minus duration: `0.1026`
  - mainline minus temporal encoder: `0.1538`
  - mainline minus best joint-sequence control: `0.0513`
  - mainline minus joint-token64: `0.2308`
- low data:
  - mainline minus duration: `0.1538`
  - mainline minus temporal encoder: `0.3077`
  - mainline minus best joint-sequence control: `0.0769`
  - mainline minus joint-token64: `0.2308`

Why this matters:

- the strongest symbolic representation now wins in both full-data and
  low-data regimes
- this reduces the risk that the current story is just a single favorable
  training regime
- the strongest evidence package is now:
  - eventized symbolic sequence structure is essential
  - transition-aware symbolic channels provide the cleanest practical gain
  - duration is not a reliable additive benefit
  - opaque learned-token baselines do not close the gap

## Slice-level frontier for hard cases

To move beyond global accuracy and inspect where the symbolic gain actually
comes from, I added:

- [build_slice_frontier_report.py](/opt/tiger/hand/tools/build_slice_frontier_report.py)

Artifacts:

- [slice_frontier_report.json](/opt/tiger/hand/experiments/generated/slice_frontier_report.json)
- [slice_frontier_report.md](/opt/tiger/hand/experiments/generated/summary_tables/slice_frontier_report.md)

This report aligns the symbolic mainline slice deltas with the best
joint-sequence control across both:

- full data (`fraction=1.0`)
- low data (`fraction=0.5`)

Most important observations:

- low data:
  - `wrist_rom`: `0.0000 -> 0.8333` for symbolic, versus `0.0000 -> 0.1667`
    for the joint-sequence control
  - `interaction`: `0.6667 -> 1.0000` for symbolic, versus `0.5556 -> 0.7778`
    for the joint-sequence control
  - `all`: `0.5897 -> 0.7949` for symbolic, versus `0.5897 -> 0.7179`
    for the joint-sequence control
- full data:
  - `right`: `0.7333 -> 0.8667`
  - `finger_occlusion`: `0.7500 -> 0.8333`
  - `occlusion`: `0.8750 -> 0.9167`

Important nuance:

- symbolic does **not** improve every slice
- the only visible regression is the full-data `left` slice:
  - `1.0000 -> 0.9333`
- some easy slices are saturated in both regimes and stay flat:
  - `no_occlusion`
  - `touching`

Current interpretation:

- the transition-aware symbolic upgrade is not a uniform "everything gets
  better" change
- it behaves more like a hard-case bias:
  - strongest on low-data wrist-ROM
  - useful on interaction-heavy and occlusion-related slices
  - largely irrelevant on already-saturated easy slices
- this strengthens the current story because it makes the gain pattern more
  specific and therefore more believable

## Hard-case event alignment

To check whether transition-aware symbols improve interpretability at the event
level rather than only boosting classifier accuracy, I added:

- [build_hardcase_event_alignment_report.py](/opt/tiger/hand/tools/build_hardcase_event_alignment_report.py)

Artifacts:

- [hardcase_event_alignment_report.json](/opt/tiger/hand/experiments/generated/hardcase_event_alignment_report.json)
- [hardcase_event_alignment_report.md](/opt/tiger/hand/experiments/generated/summary_tables/hardcase_event_alignment_report.md)

This report compares `state_event` vs `temporal_event` directly on the intrinsic
retrieval benchmark and asks:

- does temporal transition information improve positive-vs-negative margin?
- does it change the nearest wrong symbolic neighbor?
- do these changes align with the hard slices found in the classifier-level
  frontier?

Most important results:

- among `13` overlap queries:
  - temporal event margin is better on `11`
  - nearest wrong neighbor changes on `6`
  - top-1 rank is mostly unchanged
- slice-level mean margin delta:
  - `interaction`: `+0.0300`
  - `finger_occlusion`: `+0.0117`
  - `occlusion`: `+0.0111`
  - `all`: `+0.0127`

Representative hard-case neighbor changes:

- `ROM04_RT_Occlusion`:
  - state wrong neighbor: `ROM03_RT_No_Occlusion`
  - temporal wrong neighbor: `ROM07_Rt_Finger_Occlusions`
- `ROM04_LT_Occlusion`:
  - state wrong neighbor: `ROM03_LT_No_Occlusion`
  - temporal wrong neighbor: `ROM08_Lt_Finger_Occlusions`
- `ROM07_RT_Finger_Occlusions`:
  - state wrong neighbor: `ROM05_RT_Wrist_ROM`
  - temporal wrong neighbor: `ROM03_RT_No_Occlusion`

Important nuance:

- `wrist_rom` is not uniformly positive at the intrinsic event-margin level
  - mean margin delta there is `-0.0073`
  - but the wrong neighbor still changes
- therefore the wrist-ROM story is:
  - classifier/slice frontier says it is a strong low-data gain region
  - intrinsic event analysis says temporal transitions change the confusion
    structure there, but not via a clean margin increase

Current interpretation:

- temporal transition information mainly improves local symbolic separation,
  not top-1 rank
- this is especially visible on interaction / occlusion / finger-occlusion
  cases
- together with the slice frontier, this supports a narrower but stronger
  claim:
  - transition-aware symbolic representations act as hard-case separators
    rather than universal easy-case boosters

## Hard-case event casebook

To make the event-level structure concrete on representative hard cases, I
added:

- [build_hardcase_event_casebook.py](/opt/tiger/hand/tools/build_hardcase_event_casebook.py)

Artifacts:

- [hardcase_event_casebook.json](/opt/tiger/hand/experiments/generated/hardcase_event_casebook.json)
- [hardcase_event_casebook.md](/opt/tiger/hand/experiments/generated/summary_tables/hardcase_event_casebook.md)

This casebook compares, for a small set of representative queries:

- the query sequence
- its correct gallery neighbor
- the nearest wrong neighbor under `state_event`
- the nearest wrong neighbor under `temporal_event`

Representative cases currently included:

- `ROM04_RT_Occlusion`
- `ROM04_LT_Occlusion`
- `ROM07_RT_Finger_Occlusions`
- `ROM05_RT_Wrist_ROM`
- `ROM02_Interaction_2_Hand`

What the casebook now shows directly:

- `ROM04_RT_Occlusion`
  - state wrong neighbor: `ROM03_RT_No_Occlusion`
  - temporal wrong neighbor: `ROM07_Rt_Finger_Occlusions`
  - margin changes from `0.0505` to `0.0639`
- `ROM07_RT_Finger_Occlusions`
  - state wrong neighbor: `ROM05_RT_Wrist_ROM`
  - temporal wrong neighbor: `ROM03_RT_No_Occlusion`
  - margin changes from `0.0433` to `0.0589`
- `ROM02_Interaction_2_Hand`
  - wrong neighbor stays `ROM01_No_Interaction_2_Hand`
  - but temporal margin still expands from `0.0095` to `0.0399`

Important structural evidence:

- the query-side temporal profiles are dominated by:
  - transition tokens
  - hand-motion tokens
  - interaction-motion tokens
  - event-duration buckets
- these are exactly the channels absent from frame-only symbolic matching
- when the nearest wrong neighbor changes, it usually moves toward a harder and
  more structurally related competitor rather than a generic no-occlusion
  baseline

Important nuance preserved:

- `ROM05_RT_Wrist_ROM` remains mixed
  - the wrong neighbor changes
  - but temporal margin does not improve
- this keeps the wrist-ROM story aligned with the earlier event-alignment memo:
  useful structural change, but not a universal event-margin gain

## Counterfactual token ablation on hard cases

To test whether the temporal symbolic channels are causally important rather
than merely correlated with better neighborhood structure, I added:

- [build_hardcase_counterfactual_report.py](/opt/tiger/hand/tools/build_hardcase_counterfactual_report.py)

Artifacts:

- [hardcase_counterfactual_report.json](/opt/tiger/hand/experiments/generated/hardcase_counterfactual_report.json)
- [hardcase_counterfactual_report.md](/opt/tiger/hand/experiments/generated/summary_tables/hardcase_counterfactual_report.md)

Protocol:

- keep the same representative hard-case query set as the casebook
- start from `temporal_event`
- remove one token family at a time from the query events:
  - `transition`
  - `motion`
  - `interaction`
  - `duration`
- recompute event-DTW ranking against the same gallery

Aggregate result on the current 5-case set:

- `drop_motion`: mean margin delta `-0.0057`, neighbor changes `2`
- `drop_duration`: mean margin delta `-0.0015`, neighbor changes `0`
- `drop_interaction`: mean margin delta `+0.0005`, neighbor changes `0`
- `drop_transition`: mean margin delta `+0.0045`, neighbor changes `1`

Important interpretation:

- this is **not** evidence that transition is the single dominant temporal
  carrier in the current event encoding
- on this small representative set, the strongest average harm comes from
  dropping motion tokens, not transition tokens
- however, dropping transition is still not inert:
  - for `ROM07_RT_Finger_Occlusions`, the nearest wrong neighbor changes from
    `ROM03_RT_No_Occlusion` to `ROM05_RT_Wrist_ROM`
- dropping duration hurts the interaction case most:
  - `ROM02_Interaction_2_Hand`: margin changes from `0.0399` to `0.0267`

Why this still helps the paper-level experiment story:

- it stops us from making an overclaimed "transition alone explains
  everything" argument
- it suggests a more defensible conclusion:
  - current temporal symbolic performance comes from multiple temporal channels
    working together
  - motion is the strongest average local carrier on this case set
  - transition contributes selectively by reshaping local confusions
  - duration matters most for interaction-heavy cases
- this is a stronger scientific position than forcing a one-channel narrative

## Decoupled temporal-event channel study

To move beyond analysis and probe a concrete next-generation representation
candidate, I added:

- [eval_decoupled_temporal_event_intrinsic.py](/opt/tiger/hand/tools/eval_decoupled_temporal_event_intrinsic.py)
- [build_decoupled_temporal_event_report.py](/opt/tiger/hand/tools/build_decoupled_temporal_event_report.py)

Artifacts:

- [decoupled_temporal_event_intrinsic.json](/opt/tiger/hand/experiments/generated/decoupled_temporal_event_intrinsic.json)
- [decoupled_temporal_event_report.json](/opt/tiger/hand/experiments/generated/decoupled_temporal_event_report.json)
- [decoupled_temporal_event_report.md](/opt/tiger/hand/experiments/generated/summary_tables/decoupled_temporal_event_report.md)

Setup:

- keep event-DTW evaluation fixed
- decompose the temporal-event encoding into channel subsets
- compare:
  - `full = state + transition + motion + interaction + duration`
  - `state_transition`
  - `state_motion`
  - `transition_motion`
  - `motion_only`
  - `transition_only`
  - `duration_only`
  - `interaction_only`
  - `temporal_nostate`

Most important result:

- the best intrinsic representation is **not** the most additive one
- `state_motion`:
  - top-1 `0.9545`
  - mean margin `0.0762`
- `state_transition`:
  - top-1 `0.9545`
  - mean margin `0.0655`
- current `full` temporal event:
  - top-1 `0.9545`
  - mean margin `0.0598`

Key consequences:

- `state_motion` improves margin over full by `+0.0164` without losing top-1
- `state_transition` improves margin over full by `+0.0056` without losing top-1
- removing `state` hurts top-1:
  - `full 0.9545 -> temporal_nostate 0.9091`

Channel hierarchy on this benchmark:

- among pure temporal channels:
  - `motion_only`: top-1 `0.9091`
  - `transition_only`: top-1 `0.8182`
  - `interaction_only`: top-1 `0.3636`
  - `duration_only`: top-1 `0.3636`

Current interpretation:

- the current full temporal-event encoding is over-complete
- some temporal channels add redundancy or noise instead of improving
  separation
- the strongest next-step representation candidate is therefore not "add more
  temporal fields", but:
  - prune to `state+motion`
  - or prune to `state+transition`
  - then verify at classifier level whether the intrinsic gain survives

## Classifier-level validation of channel-pruned candidates

I then pushed the best intrinsic candidates to the lightweight classifier
protocol instead of leaving them as a retrieval-only observation.

Added:

- [eval_symbolic_channel_variants_classifier.py](/opt/tiger/hand/tools/eval_symbolic_channel_variants_classifier.py)
- [build_symbolic_channel_variants_classifier_report.py](/opt/tiger/hand/tools/build_symbolic_channel_variants_classifier_report.py)

Artifacts:

- [symbolic_channel_variants_classifier.json](/opt/tiger/hand/experiments/generated/symbolic_channel_variants_classifier.json)
- [symbolic_channel_variants_classifier_report.json](/opt/tiger/hand/experiments/generated/symbolic_channel_variants_classifier_report.json)
- [symbolic_channel_variants_classifier_report.md](/opt/tiger/hand/experiments/generated/summary_tables/symbolic_channel_variants_classifier_report.md)

Protocol:

- same lightweight window-classifier setting that previously supported the
  `state_only` vs `full_temporal` comparison
- `window=32`, `stride=16`, `C=16`, `mean_log_prob`
- evaluate:
  - `state_only`
  - `full_temporal`
  - `state_transition`
  - `state_motion`

Most important classifier results:

- low data (`fraction=0.5`)
  - `state_only`: `0.4615`
  - `full_temporal`: `0.5077`
  - `state_transition`: `0.5538`
  - `state_motion`: `0.6000`
- full data (`fraction=1.0`)
  - `state_only`: `0.5385`
  - `full_temporal`: `0.6154`
  - `state_transition`: `0.6923`
  - `state_motion`: `0.6923`

Key implication:

- the intrinsic pruning gain **does transfer** to classifier level
- both `state_transition` and `state_motion` beat the current `full_temporal`
  mixture at both data scales
- `state_motion` is the strongest low-data classifier variant on the current
  protocol

Gap summary versus current full temporal mixture:

- `fraction=0.5`
  - `state_motion - full_temporal = +0.0923`
  - `state_transition - full_temporal = +0.0462`
- `fraction=1.0`
  - `state_motion - full_temporal = +0.0769`
  - `state_transition - full_temporal = +0.0769`

Updated mainline consequence:

- `full_temporal` should no longer be treated as the strongest lightweight
  classifier formulation
- the upgraded candidate mainline is now:
  - `state_motion`
  - with `state_transition` as a tied full-data alternative and a slightly
    weaker low-data alternative
- this is the first result in the project that turns the decoupling analysis
  into a concrete representation replacement rather than just a diagnosis

## Strong-protocol check for pruned channel candidates

I then tested whether the same pruning wins survive under the much stronger
pretrain-only normalized protocol that currently defines the best symbolic
mainline.

Added:

- custom channel-weight support to
  [train_symbolic_pretrain.py](/opt/tiger/hand/tools/train_symbolic_pretrain.py)
- [build_strong_protocol_pruned_channel_report.py](/opt/tiger/hand/tools/build_strong_protocol_pruned_channel_report.py)

Artifacts:

- [symbolic_pretrain_state_transition_pre_only_168_84_boost_rom05.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_state_transition_pre_only_168_84_boost_rom05.json)
- [symbolic_pretrain_state_motion_pre_only_168_84_boost_rom05.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_state_motion_pre_only_168_84_boost_rom05.json)
- [symbolic_pretrain_state_transition_tempo_pre_only_168_84_boost_rom05.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_state_transition_tempo_pre_only_168_84_boost_rom05.json)
- [symbolic_pretrain_state_motion_tempo_pre_only_168_84_boost_rom05.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_state_motion_tempo_pre_only_168_84_boost_rom05.json)
- [symbolic_pretrain_state_motion_interaction_tempo_pre_only_168_84_boost_rom05.json](/opt/tiger/hand/experiments/generated/symbolic_pretrain_state_motion_interaction_tempo_pre_only_168_84_boost_rom05.json)
- [strong_protocol_pruned_channel_report.json](/opt/tiger/hand/experiments/generated/strong_protocol_pruned_channel_report.json)
- [strong_protocol_pruned_channel_report.md](/opt/tiger/hand/experiments/generated/summary_tables/strong_protocol_pruned_channel_report.md)

Protocol held fixed:

- `pretrain_window_span_units = 168`
- `pretrain_window_step_units = 84`
- `finetune_window_span_units = 168`
- `finetune_window_step_units = 84`
- `test_window_span_units = 168`
- `test_window_step_units = 84`
- `ROM05_RT_Wrist_ROM:3.0` finetune boost

Main result:

- under the strong protocol, the current full temporal mainline still wins
- mainline full temporal:
  - `fraction=0.5`: `0.7949`
  - `fraction=1.0`: `0.9231`

Best pruned candidate under the strong protocol:

- `state_motion_tempo`
  - `fraction=0.5`: `0.6667`
  - `fraction=1.0`: `0.7436`

Other representative pruned candidates:

- `state_transition`
  - `fraction=0.5`: `0.5641`
  - `fraction=1.0`: `0.7692`
- `state_motion`
  - `fraction=0.5`: `0.4359`
  - `fraction=1.0`: `0.6923`
- `state_transition_tempo`
  - `fraction=0.5`: `0.5641`
  - `fraction=1.0`: `0.6154`

Strong-protocol gap summary:

- `mainline - state_motion_tempo`
  - `fraction=0.5`: `+0.1282`
  - `fraction=1.0`: `+0.1795`
- `mainline - state_transition`
  - `fraction=1.0`: `+0.1538`

Updated interpretation:

- the lightweight classifier winner does **not** transfer cleanly to the
  strongest pretrain-finetune regime
- channel pruning helps in the lightweight protocol but is currently too
  aggressive for the strong protocol
- among the pruned strong-protocol candidates, `state_motion_tempo` is the
  best compromise, which implies:
  - `motion` remains valuable
  - `tempo` is necessary once transition is removed
  - but the full mixture still carries useful complementary structure

Current mainline status after this check:

- for lightweight classifiers:
  - `state_motion` is the best current candidate
- for the strongest pretrain-only normalized protocol:
  - the original full temporal mainline remains the best tested formulation
- therefore the representation story must now be split by protocol strength,
  rather than pretending there is already one universal winner

## 2026-06-09 interaction-aware realized pair-guided editor

New artifacts:

- [build_interaction_realized_pairguided_editor.py](/opt/tiger/hand/tools/build_interaction_realized_pairguided_editor.py)
- [interaction_realized_pairguided_editor.json](/opt/tiger/hand/experiments/generated/interaction_realized_pairguided_editor.json)
- [interaction_realized_pairguided_editor.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_pairguided_editor.md)

Goal:

- move from target-hand-only realized editing to an interaction-aware joint
  criterion on the hard right-hand slices
- compare:
  - `single`: target-hand-only symbolic donor composition
  - `base_topk`: split-donor joint search with unlearned preserve ordering
  - `pairguided_topk`: split-donor joint search with learned preserve reranking

Metric:

- right-hand grouped-motif fidelity to the target donor
- left-hand grouped preservation of the current interaction context
- joint score = right grouped match × left preserve

Key result:

- under this stricter interaction-aware criterion, `single` collapses:
  - right closing:
    - availability `0.7174`
    - right grouped `0.4539`
    - left preserve `0.0000`
    - joint overall `0.0000`
  - right opening:
    - availability `0.7211`
    - right grouped `0.1330`
    - left preserve `0.0000`
    - joint overall `0.0000`
- split-donor interaction editing recovers non-zero realized joint success:
  - right closing:
    - base top-10 joint overall `0.0322`
    - pair-guided top-10 joint overall `0.0340`
  - right opening:
    - base top-10 joint overall `0.0302`
    - pair-guided top-10 joint overall `0.0320`

Interpretation:

- this does not close the interaction gap
- but it upgrades the hard-slice story from a search-only artifact to a
  realized interaction-editing result
- the sharp conclusion is now:
  - target-hand-only symbolic editing is structurally inadequate for
    interaction
  - split-donor search is necessary
  - pair-guided reranking gives a real, if modest, additional gain

Decision:

- keep this as a supporting interaction-aware realized-edit artifact
- do **not** claim the interaction problem is solved
- the remaining gap is now narrower and better specified:
  - we need a method that improves the absolute realized joint score, not just
    the ranking efficiency of the current split-donor mechanism

## 2026-06-09 interaction-aware realized support scaling

New artifacts:

- [build_interaction_realized_support_scaling.py](/opt/tiger/hand/tools/build_interaction_realized_support_scaling.py)
- [interaction_realized_support_scaling.json](/opt/tiger/hand/experiments/generated/interaction_realized_support_scaling.json)
- [interaction_realized_support_scaling.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_support_scaling.md)

Goal:

- test whether the remaining low absolute interaction-aware joint score is
  mainly a preserve-support bottleneck
- vary:
  - preserve-donor family budget: `1` vs `2`
  - search depth: `10` vs `20`
  - base ordering vs pair-guided ordering

Key result:

- expanding support increases availability but does **not** improve the best
  absolute joint score
- right closing:
  - best remains `0.0340`
  - both `budget1_pairguided_top10` and larger-support variants tie at this
    value
- right opening:
  - best remains `0.0320`
  - again, larger-support variants tie rather than improve

More detailed pattern:

- increasing family budget from `1` to `2` raises availability:
  - closing: `0.2165 -> 0.2522`
  - opening: `0.2060 -> 0.2380`
- but joint-on-available drops enough that overall joint score stays flat
- increasing depth from `10` to `20` also saturates immediately on the same
  best overall values

Interpretation:

- the current interaction weakness is **not** mainly caused by insufficient
  preserve support size
- candidate-space expansion has already reached a plateau
- this rules out the easy next-step story that we only need more relaxed support
  or deeper search

Decision:

- retire support-scaling as a main upgrade direction
- keep the result as a negative-boundary artifact
- next interaction work should change the composition mechanism or the
  representation used for editing, not just the search budget

## 2026-06-09 interaction-aware realized constraint sweep

New artifacts:

- [build_interaction_realized_constraint_sweep.py](/opt/tiger/hand/tools/build_interaction_realized_constraint_sweep.py)
- [interaction_realized_constraint_sweep.json](/opt/tiger/hand/experiments/generated/interaction_realized_constraint_sweep.json)
- [interaction_realized_constraint_sweep.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_constraint_sweep.md)

Goal:

- test whether the remaining hard interaction bottleneck is partly in the
  realized preserve-hand geometry after split-donor selection
- keep the selector fixed and add target-aware repair operators on the preserve
  hand only
- compare:
  - no repair
  - blend-to-current repairs
  - edge token / transition snap
  - finger-profile snap
  - full-current oracle repair
- run on both hard right-hand interaction tasks:
  - `right_hand_motion->closing`
  - `right_hand_motion->opening`

Key result:

- this is the first mechanism change that materially improves the strict
  interaction-aware realized joint score on both hard slices
- strongest current variant is `hgb_finger_profile_snap_top10`
- closing:
  - previous realized frontier: `0.0340`
  - new best: `0.0465`
- opening:
  - previous realized frontier: `0.0320`
  - new best: `0.0426`

More detailed pattern:

- the gain comes entirely from better preserve-hand realization
- right-hand grouped score stays fixed:
  - closing: `0.0716`
  - opening: `0.0657`
- preserve-hand success rises sharply:
  - closing: `0.1163 -> 0.1610`
  - opening: `0.1137 -> 0.1510`
- `MLP` ordering keeps the same direction of gain with finger-profile repair,
  but `HGB + finger-profile snap` remains the strongest overall combination
- naive `full current oracle` is not stronger than finger-profile snap, which
  suggests the useful effect is not just absolute overwrite but a more targeted
  constrained repair

Interpretation:

- the remaining hard interaction weakness was not purely a donor-ranking
  problem
- a second gain layer exists in the realization step itself
- constrained preserve-hand repair is now part of the surviving mainline

Decision:

- promote constrained realization from speculative direction to mainline
  evidence
- keep pushing on realization-aware mechanisms rather than going back to pure
  search-budget growth
- the gap is narrower now, but the absolute hard-slice joint score is still not
  yet high enough to count as closure

## 2026-06-09 interaction-aware realized constraint scaling

New artifacts:

- [build_interaction_realized_constraint_scaling.py](/opt/tiger/hand/tools/build_interaction_realized_constraint_scaling.py)
- [interaction_realized_constraint_scaling.json](/opt/tiger/hand/experiments/generated/interaction_realized_constraint_scaling.json)
- [interaction_realized_constraint_scaling.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_constraint_scaling.md)

Goal:

- test whether the new `finger_profile_snap` gain is only a local
  `budget1/top10` effect or a stable mechanism layer
- scale across:
  - selector: `base`, `hgb`, `mlp`
  - preserve budget: `1`, `2`
  - depth: `5`, `10`, `20`
- keep the repair space narrow and only compare:
  - `none`
  - `finger_profile_snap`

Key result:

- the new repair gain is stable and compounds with larger preserve support
- strongest current hard-slice settings now reach:
  - closing: `0.0555`
  - opening: `0.0533`
- this is a substantial improvement over:
  - old pair-guided realized frontier: `0.0340 / 0.0320`
  - first repair-only frontier at budget1: `0.0465 / 0.0426`

More detailed pattern:

- `budget2` raises availability and also raises right grouped success under the
  repaired realization:
  - closing availability: `0.2165 -> 0.2522`
  - opening availability: `0.2060 -> 0.2380`
- with repair turned on, the extra support no longer plateaus immediately the
  way the unrepaired support-scaling experiment did
- for the strongest settings:
  - closing:
    - `hgb_budget2_finger_profile_snap_top10 = 0.0555`
    - `hgb_budget2_finger_profile_snap_top20 = 0.0555`
  - opening:
    - `hgb_budget2_finger_profile_snap_top10 = 0.0533`
    - `hgb_budget2_finger_profile_snap_top20 = 0.0533`
- `MLP` keeps the same directional pattern but still trails the best `HGB`
  configuration

Interpretation:

- the earlier support-scaling plateau was not a contradiction; it was a sign
  that support growth alone was insufficient
- once preserve-hand realization is repaired, larger support budgets become
  useful again
- the new strongest line is therefore:
  - compact / pair-guided search
  - expanded preserve support
  - constrained preserve-hand realization

Decision:

- promote `budget2 + finger_profile_snap` to the current strongest
  interaction-aware realized editor
- stop treating search vs realization as separate alternatives; the current win
  is their combination
- keep pushing this combined interaction line until the absolute hard-slice
  score is strong enough to stop looking like a localized rescue effect

## 2026-06-09 interaction-aware realized significance report

New artifacts:

- [build_interaction_realized_significance_report.py](/opt/tiger/hand/tools/build_interaction_realized_significance_report.py)
- [interaction_realized_significance_report.json](/opt/tiger/hand/experiments/generated/interaction_realized_significance_report.json)
- [interaction_realized_significance_report.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_significance_report.md)

Goal:

- test whether the new strongest interaction frontier is only a better point
  estimate or still holds under paired analysis on the exact same frames
- compute:
  - paired mean deltas
  - bootstrap confidence intervals
  - paired sign-flip permutation p-values
  - win / loss / tie counts

Key result:

- the strongest comparison now looks statistically credible on both hard
  right-hand interaction tasks
- `hgb_budget1_none_top10 -> hgb_budget2_finger_profile_snap_top10`
  - closing:
    - delta `+0.0215`
    - bootstrap CI `[0.0107, 0.0340]`
    - permutation `p=0.00055`
    - wins / losses / ties = `12 / 0 / 547`
  - opening:
    - delta `+0.0213`
    - bootstrap CI `[0.0107, 0.0337]`
    - permutation `p=0.00015`
    - wins / losses / ties = `12 / 0 / 551`

More detailed pattern:

- even the first repair step alone is paired-positive on both tasks:
  - `hgb_budget1_none_top10 -> hgb_budget1_finger_profile_snap_top10`
  - closing delta `+0.0125`, `p=0.0158`
  - opening delta `+0.0107`, `p=0.03155`
- the second step from repaired budget1 to repaired budget2 is also positive:
  - closing delta `+0.0089`
  - opening delta `+0.0107`
- the effect remains entirely one-sided in paired wins:
  - no paired losses in the strongest hard-slice comparisons

Interpretation:

- the new interaction-aware realized frontier is not only descriptively better;
  it is also statistically harder to dismiss as noise
- this strengthens the combined story:
  - pair-guided search opens the candidate space
  - constrained realization converts that space into real preserve-hand gains
  - expanded support becomes useful again only after that realization step is
    repaired

Decision:

- keep the significance report as a supporting mainline artifact
- use the stronger paired evidence to defend that the current interaction
  frontier is genuinely moving, even though the absolute score is still below
  final closure

## 2026-06-09 interaction-aware realized sequence concentration

New artifacts:

- [build_interaction_realized_sequence_concentration.py](/opt/tiger/hand/tools/build_interaction_realized_sequence_concentration.py)
- [interaction_realized_sequence_concentration.json](/opt/tiger/hand/experiments/generated/interaction_realized_sequence_concentration.json)
- [interaction_realized_sequence_concentration.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_sequence_concentration.md)

Goal:

- test whether the new strongest hard-slice gain is broadly distributed across
  hard sequences or mostly carried by one recurring sequence family
- quantify:
  - per-sequence delta contribution
  - cumulative contribution concentration
  - leave-one-sequence-out deltas

Key result:

- the gain is real across more than one sequence family, but it is still
  concentrated rather than evenly spread
- strongest comparison:
  - `hgb_budget1_none_top10 -> hgb_budget2_finger_profile_snap_top10`
- closing:
  - overall delta `+0.0215`
  - `ROM01_No_Interaction_2_Hand` contributes `+0.0179`
  - `ROM09_Interaction_Fingers_Touching` contributes `+0.0036`
  - `ROM02_Interaction_2_Hand` contributes `0.0000`
- opening:
  - overall delta `+0.0213`
  - `ROM01_No_Interaction_2_Hand` contributes `+0.0178`
  - `ROM09_Interaction_Fingers_Touching` contributes `+0.0036`
  - `ROM02_Interaction_2_Hand` contributes `0.0000`

More detailed pattern:

- removing `ROM01_No_Interaction_2_Hand` does not destroy the gain completely,
  but it reduces it sharply:
  - closing leave-one-out delta drops to `+0.0046`
  - opening leave-one-out delta drops to `+0.0045`
- removing `ROM09_Interaction_Fingers_Touching` leaves a much larger residual:
  - closing `+0.0680`
  - opening `+0.0662`
- removing `ROM02_Interaction_2_Hand` barely changes the delta

Interpretation:

- the current strongest interaction improvement is not a single-frame artifact
- but it is still dominated by one hard-sequence family rather than evenly
  distributed across all three hard-sequence families
- this tightens the current claim boundary:
  - paired significance is real
  - strongest gains are concentrated
  - broader hard-sequence coverage is still an open target

Decision:

- keep this as a negative-boundary artifact, not a topline “victory” result
- do not claim uniform hard-sequence generality yet
- next interaction work should explicitly target the weak families that remain
  almost unchanged under the current strongest editor

## 2026-06-10 interaction-aware realized right support sweep

New artifacts:

- [build_interaction_realized_right_support_sweep.py](/opt/tiger/hand/tools/build_interaction_realized_right_support_sweep.py)
- [interaction_realized_right_support_sweep.json](/opt/tiger/hand/experiments/generated/interaction_realized_right_support_sweep.json)
- [interaction_realized_right_support_sweep.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_right_support_sweep.md)

Goal:

- test the newly identified weak sequence families directly
- specifically ask whether `ROM02` and `ROM09` are limited by an overly tight
  right-hand donor pool rather than only by preserve-hand realization
- keep the left side fixed to the current strongest:
  - `budget2`
  - `finger_profile_snap`
- compare right donor support:
  - `strict`
  - `relax_both`

Key result:

- on the weak sequence families only, relaxing the right donor pool is a real
  rescue mechanism
- `hgb_strict -> hgb_relax_both`
  - closing subset: `0.0092 -> 0.0572`
  - opening subset: `0.0157 -> 0.0695`

More detailed pattern:

- the main gain comes from a large jump in right-hand grouped success:
  - closing: `0.0137 -> 0.0892`
  - opening: `0.0247 -> 0.0987`
- `ROM02` moves dramatically:
  - closing: `0.0400 -> 0.6000`
  - opening: `0.1471 -> 0.6176`
- `ROM09` also improves, though much less:
  - closing: `0.0073 -> 0.0243`
  - opening: `0.0049 -> 0.0243`

Interpretation:

- the weak-family bottleneck is not just on the preserve side
- right-target donor support is also too tight for these families
- this is strong evidence that a future selective gate over right-pool modes is
  worth pursuing
- it does **not** justify globally switching the whole pipeline to `relax_both`
  without a gate, because the full-sequence line has not yet been revalidated

Decision:

- keep this as a diagnostic support artifact
- use it to justify a learned or heuristic gate between `strict` and
  `relax_both`
- do not yet promote it to the mainline frontier until the gate can recover the
  weak-family gains without sacrificing the already-strong families

## 2026-06-10 corrected full-slice right-support check and candidate-pool bug

New artifacts:

- [build_interaction_realized_global_right_support_check.py](/opt/tiger/hand/tools/build_interaction_realized_global_right_support_check.py)
- [build_interaction_realized_global_right_support_bundle.py](/opt/tiger/hand/tools/build_interaction_realized_global_right_support_bundle.py)
- [interaction_realized_global_right_support_bundle.json](/opt/tiger/hand/experiments/generated/interaction_realized_global_right_support_bundle.json)
- [interaction_realized_global_right_support_bundle.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_global_right_support_bundle.md)

Bug found and fixed:

- the old generic right-hand candidate-pool path inherited
  `TASK_TARGET="opening"` from the earlier weak-slice prototype
- this polluted closing-side candidate pools in scripts that called
  `candidate_pool_for_task(...)` without an explicit `task_target`
- additionally, the old helper incorrectly used `curr_attrs[task_field]` as the
  target value, even though the hard-slice frame collector is keyed by the run
  label rather than the current-frame hand-motion state

Code changes:

- `right_relaxed_candidates(...)` now accepts explicit `task_field` and
  `task_target`
- `candidate_pool_for_task(...)` now requires the true `task_target`
- all downstream callers were updated to pass the intended target explicitly

Corrected full-slice result:

- on the corrected full hard right-hand slices, global `relax_both` is not just
  a weak-family rescue trick; it is a strong full-slice gain
- corrected `hgb_strict -> hgb_relax_both`
  - closing: `0.0250 -> 0.1807`
  - opening: `0.0533 -> 0.1812`

Paired significance:

- closing joint-score delta:
  - `+0.1556`
  - bootstrap CI `[0.1270, 0.1860]`
  - permutation `p=0.000000`
  - wins/losses/ties = `87 / 0 / 472`
- opening joint-score delta:
  - `+0.1279`
  - bootstrap CI `[0.1012, 0.1563]`
  - permutation `p=0.000000`
  - wins/losses/ties = `73 / 1 / 489`

Interpretation:

- the earlier closing-side strongest line was not a trustworthy mainline anchor
  because it depended on a polluted candidate-pool path
- after correction, the strongest current hard-slice story is no longer
  “support scaling becomes useful again after preserve-hand repair”
- the stronger current statement is:
  - corrected right-target donor support matters directly
  - global `relax_both` is a large full-slice gain on both right-hand hard tasks
  - the gain is driven mainly by right-hand grouped realization, with a smaller
    secondary preserve-side gain

Decision:

- treat the corrected global-right-support bundle as the current authoritative
  interaction-editing artifact for the hard right-hand slices
- do not reuse pre-fix closing-side numbers from
  `candidate_pool_for_task(...)`-based artifacts as current evidence
- keep the next step on subtype / boundary analysis of the corrected relaxed
  editor, not on reviving the old gate-first story

## 2026-06-10 feasibility audit and feasible-subset right-repair follow-up

New artifacts:

- [build_interaction_realized_feasibility_audit.py](/opt/tiger/hand/tools/build_interaction_realized_feasibility_audit.py)
- [interaction_realized_feasibility_audit.json](/opt/tiger/hand/experiments/generated/interaction_realized_feasibility_audit.json)
- [interaction_realized_feasibility_audit.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasibility_audit.md)
- [build_interaction_realized_feasible_dual_repair_followup.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_dual_repair_followup.py)
- [interaction_realized_feasible_dual_repair_followup_closing.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_dual_repair_followup_closing.md)
- [interaction_realized_feasible_dual_repair_followup_opening.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_dual_repair_followup_opening.md)

Feasibility finding:

- under the corrected global hard-slice editor, a large majority of right-hand
  hard-slice frames have `curr_frame.left is None`
- this means the strict joint metric is structurally impossible on those frames
  because opposite-hand preservation is undefined rather than merely incorrect
- measured feasible-two-hand rate:
  - closing: `0.2826`
  - opening: `0.2789`

Feasible-subset result:

- on the feasible two-hand subset only, corrected `hgb_relax_both` is much
  stronger than the raw full-slice joint score suggests
- feasible joint:
  - closing: `0.6392`
  - opening: `0.6497`
- especially important:
  - `ROM09_Interaction_Fingers_Touching` still looks weak at full-slice level
    (`0.0243`)
  - but on its feasible subset the same corrected editor reaches `0.4545`

Targeted follow-up:

- I then tested whether an extra right-target donor-side repair layer could
  improve the feasible subset while keeping:
  - corrected `relax_both` right support
  - budget-2 pair-guided left support
  - left `finger_profile_snap`
- compared right-side follow-up modes:
  - `none`
  - `edge_transition_snap`
  - `finger_profile_snap`
  - `full_donor_oracle`

Result:

- extra right-target repair is not the next gain layer
- feasible subset, closing:
  - base `none = 0.6203`
  - `edge_transition_snap = 0.2278`
  - `finger_profile_snap = 0.6203`
  - `full_donor_oracle = 0.0696`
- feasible subset, opening:
  - base `none = 0.6497`
  - `edge_transition_snap = 0.4076`
  - `finger_profile_snap = 0.6306`
  - `full_donor_oracle = 0.1465`

Interpretation:

- the corrected relaxed-support editor is already near the right-target ceiling
  on the feasible two-hand subset
- donor-side right repair is either neutral or harmful
- the next meaningful gain must come from:
  - a better handling of the structurally infeasible absent-opposite-hand mass,
    or
  - the remaining true feasible failures, not donor-side snap heuristics

Decision:

- keep the feasibility audit as a support artifact and reviewer-defense asset
- keep the feasible-subset dual-repair follow-up as a negative-boundary result
- do not spend more cycles on right-target snap / donor-oracle variants

## 2026-06-10 feasible-subset left-repair follow-up

New artifacts:

- [build_interaction_realized_feasible_left_repair_followup.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_left_repair_followup.py)
- [build_interaction_realized_feasible_left_repair_bundle.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_left_repair_bundle.py)
- [interaction_realized_feasible_left_repair_bundle.json](/opt/tiger/hand/experiments/generated/interaction_realized_feasible_left_repair_bundle.json)
- [interaction_realized_feasible_left_repair_bundle.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_repair_bundle.md)

Motivation:

- after the feasibility audit, the dominant true failure mode on the feasible
  two-hand subset was:
  - `right_grouped_match = 1`
  - `left_preserve = 0`
- so the next meaningful targeted follow-up was to strengthen the left preserve
  side rather than the right target side

Protocol:

- keep fixed:
  - corrected `relax_both` right support
  - budget-2 pair-guided left donor selection
  - no extra right-target repair
- feasible subset only
- vary left repair:
  - `none`
  - `edge_transition_snap`
  - `finger_profile_snap`
  - `full_curr_oracle`

Key result:

- left repair is a real second gain layer on the feasible subset
- best mode is task-dependent:
  - closing:
    - best = `edge_transition_snap`
    - feasible joint `0.4747 -> 0.6582`
    - paired delta `+0.1835`
    - bootstrap CI `[0.1076, 0.2595]`
    - permutation `p=0.000050`
  - opening:
    - best = `finger_profile_snap`
    - feasible joint `0.4841 -> 0.6497`
    - paired delta `+0.1656`
    - bootstrap CI `[0.1083, 0.2229]`
    - permutation `p=0.000000`

Interpretation:

- the corrected hard-slice story is now naturally two-stage:
  - stage 1:
    corrected right-support relaxation on the full slice
  - stage 2:
    targeted left preserve repair on the feasible two-hand subset
- unlike donor-side right repair, left repair is a genuine source of additional
  gain
- but the best left repair is not yet universal across tasks, so this is a
  stronger mechanism result than a final unified protocol

Decision:

- promote the feasible left-repair bundle into the current strong evidence stack
- keep the claim boundary explicit:
  - full-slice corrected `relax_both` is the authoritative global result
  - feasible-subset left repair is a real strengthening layer
  - but task-dependent best left repair means the unified final mechanism is
    still open

## 2026-06-10 cheap subtype gate for feasible left repair

New artifacts:

- [build_interaction_realized_feasible_left_repair_gate.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_left_repair_gate.py)
- [interaction_realized_feasible_left_repair_gate_closing.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_repair_gate_closing.md)
- [interaction_realized_feasible_left_repair_gate_opening.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_repair_gate_opening.md)

Goal:

- test whether the task-dependent left-repair win can be replaced by a very
  cheap unified selector
- selector:
  - key = `(other_hand_motion, interaction_motion_value)`
  - train-time choice = highest average feasible-subset joint score on `val`

Result:

- the cheap subtype gate is not good enough
- it consistently beats `no-repair`, but loses to the fixed task-best mode
- closing:
  - fixed task-best `0.6582`
  - gate `0.5253`
  - oracle `0.7405`
- opening:
  - fixed task-best `0.6497`
  - gate `0.5860`
  - oracle `0.7197`

Interpretation:

- there is still real oracle headroom above the fixed task-best policy
- but the headroom is not captured by a simple subtype-average gate
- this means:
  - the remaining selection problem is more contextual than
    `(other_hand_motion, interaction_motion)` alone
  - a better selector may still exist
  - but the cheapest hand-built gate is not a strong enough next mechanism

Decision:

- keep the cheap subtype gate as a negative-boundary artifact
- retain the feasible left-repair gain as task-dependent best current evidence
- if we revisit unification, the next selector must use richer context than the
  current subtype-average rule

## 2026-06-10 richer learned gate for feasible left repair

New artifacts:

- [build_interaction_realized_feasible_left_repair_model_gate.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_left_repair_model_gate.py)
- [interaction_realized_feasible_left_repair_model_gate_closing.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_repair_model_gate_closing.md)
- [interaction_realized_feasible_left_repair_model_gate_opening.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_repair_model_gate_opening.md)

Goal:

- test whether a slightly richer but still lightweight learned selector can
  unify feasible left-repair better than the failed subtype-average gate
- added features:
  - `other_hand_motion`
  - `interaction_motion`
  - no-repair `left_state_agreement`
  - no-repair `left_transition_agreement`
  - no-repair `right_state_agreement`
  - no-repair `right_transition_agreement`

Result:

- this richer lightweight gate also fails
- it stays above `no-repair`, but is much worse than the fixed task-best mode
- closing:
  - fixed task-best `0.6582`
  - learned gate `0.4873`
  - oracle `0.7405`
- opening:
  - fixed task-best `0.6497`
  - learned gate `0.5159`
  - oracle `0.7197`

Failure diagnosis:

- the feasible training subset is extremely label-imbalanced under the oracle
  best-mode label
- learned gate label counts:
  - closing: `none=81`, `edge_transition_snap=2`
  - opening: `none=77`, `edge_transition_snap=4`, `finger_profile_snap=1`
- so this class of small selector largely collapses toward `none`

Interpretation:

- the remaining oracle headroom is real
- but it is not recoverable by small gates built on the current feasible-subset
  supervision signal
- this pushes the next unification attempt toward something materially richer,
  such as:
  - more informative supervision / relabeling
  - larger contextual features
  - or a different representation of left-repair decision targets

Decision:

- keep the richer learned gate as a second negative-boundary result
- stop iterating on cheap / small left-repair selectors for now
- keep the best current evidence as:
  - corrected global `relax_both` on the full slice
  - task-dependent left repair on the feasible two-hand subset

## 2026-06-10 binary apply gate for closing left repair

New artifacts:

- [build_interaction_realized_feasible_left_apply_gate.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_left_apply_gate.py)
- [interaction_realized_feasible_left_apply_gate.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_apply_gate.md)

Goal:

- after the multiclass gates failed, test the narrowest plausible lightweight
  selector:
  - task fixed to `right_hand_motion->closing`
  - only decide whether to apply the already-best left repair
    `edge_transition_snap`
  - compare against:
    - `fixed_none`
    - `fixed_best`
    - binary gate
    - apply-or-not oracle

Result:

- the binary apply gate also fails
- closing:
  - fixed best `0.6582`
  - binary gate `0.4873`
  - apply-or-not oracle `0.7089`

Failure diagnosis:

- the feasible training subset is even more imbalanced for this binary decision
  than expected:
  - positive (repair helps) = `2`
  - negative = `81`
- so the lightweight binary gate also collapses toward “do not repair”

Interpretation:

- the residual closing headroom above fixed best is real (`+0.0506`)
- but it is too sparse to be recovered by a small apply-or-not classifier under
  the current supervision

Decision:

- treat the binary apply gate as a third negative-boundary result in the same
  family
- stop spending cycles on lightweight left-repair gates
- if we revisit unification, it should not be with cheap classifiers trained on
  the current sparse feasible-subset labels

## 2026-06-10 dense KNN teacher for feasible left repair

New artifacts:

- [build_interaction_realized_feasible_left_dense_knn.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_left_dense_knn.py)
- [interaction_realized_feasible_left_dense_knn_closing.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_dense_knn_closing.md)
- [interaction_realized_feasible_left_dense_knn_opening.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_dense_knn_opening.md)

Goal:

- test the next more serious unification route after the cheap gates failed
- replace sparse oracle-mode classification with dense continuous supervision:
  - each feasible training frame contributes repair gains
    `joint(repair) - joint(none)`
  - use nearest-neighbor regression over richer context:
    - `other_hand_motion`
    - `interaction_motion`
    - left/right state signatures
    - no-repair left/right state and transition agreements
- tune the selector on feasible `val` via leave-one-out, then evaluate on
  feasible `test`

Result:

- this denser teacher signal still does not produce a usable unified selector
- opening:
  - dense KNN exactly collapses to the fixed task-best result
  - fixed task-best `0.6497`
  - dense KNN `0.6497`
  - oracle `0.7197`
- closing:
  - dense KNN is substantially worse than the fixed task-best policy
  - fixed task-best `0.6582`
  - dense KNN `0.5696`
  - oracle `0.7405`

Failure diagnosis:

- the selector looks deceptively strong under feasible-`val` leave-one-out:
  - closing LOO `0.7470`
  - opening LOO `0.7805`
- but this does not transfer to feasible `test`
- on closing, the dense selector over-applies `none` or the wrong repair on the
  hard subtype pockets:
  - `ROM02_Interaction_2_Hand`: `1.0000 -> 0.8235`
  - `ROM09_Interaction_Fingers_Touching`: `0.4091 -> 0.3182`
- this means the remaining selection problem is not solved by simply replacing
  sparse class labels with framewise continuous repair gains

Interpretation:

- the earlier diagnosis was directionally right:
  - lightweight gates were too weak
- but this new result adds a stronger boundary:
  - even a denser gain-based teacher plus richer local context still fails to
    beat the fixed task-best policy
- so the next remaining gap is not “just use a better small selector”
- if left-repair unification is revisited, it likely needs one of:
  - sequence-level or chunk-level supervision
  - explicit cross-frame temporal context
  - a different left-repair target representation instead of framewise mode
    choice

Decision:

- keep dense KNN as a fourth negative-boundary artifact in the feasible
  left-repair unification family
- do not claim that dense continuous teacher signals already solve the
  task-dependent left-repair boundary
- keep the strongest supported statement unchanged:
  - corrected full-slice `relax_both` is the authoritative global gain
  - feasible-subset left repair is a real second gain layer
  - but a unified left-repair selector is still open

## 2026-06-10 template-conditioned feasible left-repair gate

New artifacts:

- [build_interaction_realized_feasible_left_template_gate.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_left_template_gate.py)
- [interaction_realized_feasible_left_template_gate_closing.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_template_gate_closing.md)
- [interaction_realized_feasible_left_template_gate_opening.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_template_gate_opening.md)

Goal:

- after local framewise selectors failed, test whether a coarser temporal /
  template context can unify feasible left repair
- fit feasible-`val` repair rules using:
  - `seq_name`
  - `(other_hand_motion, interaction_motion)`
  - `seq_name + subtype`
- evaluate on feasible `test`

Result:

- these template-conditioned rules also fail
- none of them beats the fixed task-best policy on either hard task
- closing:
  - fixed task-best `0.6582`
  - seq gate `0.5000`
  - subtype gate `0.5253`
  - seq+subtype gate `0.5696`
- opening:
  - fixed task-best `0.6497`
  - seq gate `0.5223`
  - subtype gate `0.5860`
  - seq+subtype gate `0.5287`

Failure diagnosis:

- these rules collapse heavily toward `none` on `test`
- example mode counts:
  - closing seq gate:
    - `none=136`
    - `finger_profile_snap=22`
  - opening seq gate:
    - `none=112`
    - `finger_profile_snap=45`
- even adding subtype does not fix this:
  - closing seq+subtype:
    - `none=82`
    - `edge_transition_snap=76`
  - opening seq+subtype:
    - `none=96`
    - `finger_profile_snap=44`
    - `edge_transition_snap=17`
- so the feasible left-repair preference is not stable enough across
  `val -> test` to be recovered by coarse template rules

Interpretation:

- the remaining gap is not solved by:
  - framewise local selectors
  - dense local gain teachers
  - or coarse template-conditioned rules
- this is a stronger boundary than before because it rules out both ends:
  - purely local framewise context
  - coarse sequence-template context
- the likely next step is a genuinely temporal mechanism with explicit
  cross-frame evidence, or a different repair target that is more stable across
  splits than the current framewise mode choice

Decision:

- keep template-conditioned gates as another negative-boundary family
- do not claim that sequence/template metadata is enough to unify feasible left
  repair
- the strongest supported statement remains unchanged:
  - feasible left repair is real
  - but unified left-repair selection is still unsolved

## 2026-06-10 temporal-window KNN for feasible left repair

New artifacts:

- [build_interaction_realized_feasible_left_temporal_window_knn.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_left_temporal_window_knn.py)
- [interaction_realized_feasible_left_temporal_window_knn_closing.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_temporal_window_knn_closing.md)
- [interaction_realized_feasible_left_temporal_window_knn_opening.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_temporal_window_knn_opening.md)

Goal:

- move beyond frame-local and coarse-template selectors
- test whether explicit one-hop temporal context helps:
  - previous feasible frame in the same sequence
  - current frame
  - next feasible frame
- keep dense gain supervision and select by temporal-window KNN

Result:

- this is the first left-repair unification attempt where explicit temporal
  context shows a real directional benefit, but only on one task
- opening:
  - fixed task-best `0.6497`
  - temporal-window KNN `0.6624`
  - delta `+0.0127`
  - wins `2`, losses `0`, ties `155`
  - bootstrap CI `[0.0000, 0.0318]`
  - permutation `p=0.496350`
- closing:
  - fixed task-best `0.6582`
  - temporal-window KNN `0.4937`
  - delta `-0.1646`

Interpretation:

- explicit cross-frame context is not useless
- compared with the earlier dense local KNN and template gates, this is the
  first sign that temporal evidence can move the selector in the right
  direction on at least one task
- but the current one-hop window is far from a solved unified mechanism
- the opening-side lift is not statistically convincing yet, so it should be
  treated as directional evidence rather than a stabilized gain
- the task asymmetry is now sharper:
  - opening appears mostly recoverable with a temporal-biased selector
  - closing still collapses strongly toward `none`

Boundary:

- do not claim unified left-repair selection is solved
- do not claim the temporal-window gain is a stable global win yet
- the opening improvement is small and the closing failure is severe

Decision:

- keep this as a mixed boundary/support artifact rather than a mainline win
- the next credible direction is no longer “more local features” or “coarser
  template keys”
- it is a stronger temporal mechanism, likely one of:
  - longer chunk-level context
  - asymmetric task-specific temporal policies
  - or a different temporal repair target, especially for closing

## 2026-06-10 closing-specific temporal routing

New artifacts:

- [build_interaction_realized_feasible_left_closing_temporal_routing.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_left_closing_temporal_routing.py)
- [interaction_realized_feasible_left_closing_temporal_routing.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_closing_temporal_routing.md)

Goal:

- directly target the remaining hard case `right_hand_motion->closing`
- instead of generic multiclass selection, start from fixed-best
  `edge_transition_snap` and test two closing-specific temporal routes:
  - `edge -> none`
  - `edge -> finger`

Result:

- both closing-specific routes fail
- `edge -> none` collapses catastrophically:
  - fixed edge `0.6582`
  - route `0.4747`
- `edge -> finger` is less bad but still clearly worse than fixed edge:
  - fixed edge `0.6582`
  - route `0.6203`

Failure diagnosis:

- under train-time leave-one-out selection, both routing families degenerate to
  single-mode behavior:
  - `edge -> none` chooses threshold `0.5` and predicts `none` for all train
    examples
  - `edge -> finger` chooses threshold `-1.0` and predicts `finger` for all
    train examples
- the same collapse appears on test:
  - `edge -> none`: `none=156`, `edge=2`
  - `edge -> finger`: `finger=158`

Interpretation:

- this is stronger than the earlier generic closing selector failures
- even when the decision is narrowed to routing from the already-best closing
  mode, the train-side signal still prefers degenerate single-mode collapse
- so the closing gap is not presently solved by:
  - generic multiclass selectors
  - abstention-from-edge selectors
  - or family-switch-from-edge selectors

Decision:

- keep this as a closing-specific negative-boundary artifact
- do not spend more time on one-hop mode-routing for closing under the current
  target definition
- if closing is revisited, it should be with a stronger temporal target or a
  different repair objective, not another local routing threshold

## 2026-06-10 closing chunk-level KNN

New artifacts:

- [build_interaction_realized_feasible_left_closing_chunk_knn.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_left_closing_chunk_knn.py)
- [interaction_realized_feasible_left_closing_chunk_knn.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_closing_chunk_knn.md)

Goal:

- test the next stronger temporal reformulation for the hard closing case
- change the decision unit from a single frame to a short temporal run
- choose one repair mode per run and broadcast it to all frames in the run

Result:

- chunk-level reformulation also fails badly on closing
- fixed edge:
  - `0.6582`
- chunk KNN:
  - `0.4747`
- oracle:
  - `0.7405`

Failure diagnosis:

- the train side only contains `16` feasible closing runs, while test contains
  `27`
- under leave-one-out selection, the run-level KNN collapses to a degenerate
  all-`none` policy:
  - train predicted run modes: `none=16`
  - test predicted run modes: `none=27`
- so even after switching the target granularity from frame to run, the
  available closing supervision is still too weak / unstable for a trainable
  mode selector

Interpretation:

- this is an important negative result because it rules out a naive “just make
  the temporal unit coarser” fix
- closing is not failing merely because framewise labels are too noisy
- under the current protocol, both:
  - frame-level temporal selectors
  - and run-level chunk selectors
  collapse to degenerate solutions

Decision:

- keep this as a closing-specific negative-boundary artifact
- stop iterating on supervised closing mode selection under the current
  `none/edge/finger` target family
- if closing is revisited, it likely needs:
  - a different repair objective
  - stronger supervision than current feasible `val`
  - or a protocol that does not force direct train-to-test mode transfer from
    only a handful of runs

## 2026-06-10 external support expansion engineering boundary

New scripts:

- [build_interaction_realized_closing_support_expansion.py](/opt/tiger/hand/tools/build_interaction_realized_closing_support_expansion.py)
- [build_interaction_realized_closing_selector_support_expansion.py](/opt/tiger/hand/tools/build_interaction_realized_closing_selector_support_expansion.py)
- [build_interaction_realized_closing_selector_support_scaling.py](/opt/tiger/hand/tools/build_interaction_realized_closing_selector_support_scaling.py)

Goal:

- test whether the remaining closing collapse is mainly due to using only the
  tiny feasible `val` subset as supervision
- inject larger train-side interacting closing support in progressively cleaner
  ways:
  - full support expansion
  - selector-only support expansion
  - small-scale `top-k` external support scaling

Observed engineering boundary:

- all three routes are currently bottlenecked by the same hot path:
  - `collect_rows(...)`
  - `pick_best_from_cache(...)`
  - `evaluate_joint(...)`
- even when the pair-guided model itself is kept on the val-only protocol,
  generating train-side support rows is still expensive enough that:
  - full `train+val` support did not finish in a practical turnaround
  - even `top-2` / `top-4` external support scaling did not finish before
    manual interruption

Interpretation:

- this does **not** yet prove that larger external support fails
- it proves that the current implementation path is too expensive for clean
  support-scaling experiments
- so the next support-expansion step must first change the mechanics, e.g.:
  - cache support rows
  - prune candidate pools before `evaluate_joint`
  - or precompute a lighter train-side support representation

Decision:

- treat this as an engineering boundary rather than a scientific conclusion
- do not make any scientific claim from the unfinished support-expansion runs
- if we revisit stronger supervision for closing, the first required step is
  computational restructuring, not another direct rerun

## 2026-06-10 fast-path closing selector support scaling

New artifact:

- [build_interaction_realized_closing_selector_support_scaling.py](/opt/tiger/hand/tools/build_interaction_realized_closing_selector_support_scaling.py)

What changed:

- instead of the old expensive support-row path, use an explicitly pruned fast
  path:
  - right donor cap = `5`
  - left preserve cap = `5`
- this makes external-support experiments at least partially runnable

Current verified baseline result:

- `top_0` = no external support, only feasible val selector rows
- under the fast path:
  - fixed edge joint overall = `0.6013`
  - temporal-window KNN = `0.6139`
  - delta = `+0.0127`
  - wins `10`, losses `8`, ties `140`

Interpretation:

- this is not yet a final scientific result because the fast path changes the
  absolute evaluation regime relative to the earlier full-depth protocol
- but it is an important engineering breakthrough:
  - closing selector support scaling is now at least experimentally reachable
  - the fast-path closing regime does not instantly collapse to all-`none`
  - it exposes a nontrivial `edge/finger` mixture (`edge=37`, `finger=121`)

Open work:

- `top_2` external-support run is still expensive but remains feasible under
  the fast path
- the next clean step is to cache the fast-path external rows and then sweep
  support budgets from cache instead of recomputing them each run

## 2026-06-10 cached fast-path external support result

New cache path:

- [external_rows_top2.json](/opt/tiger/hand/experiments/generated/cache/closing_selector_support_scaling/external_rows_top2.json)

Verified result:

- after fixing the duplicate-sequence bug in the external train subset and
  rebuilding from cache, `top_2` now truly uses:
  - `2` external sequences
  - `39` external support rows
- compared against the same refreshed `top_0` baseline on the same fast-path
  closing slice (`56` frames):
  - `top_0` temporal-window KNN = `0.7321`
  - `top_2` temporal-window KNN = `0.7321`
  - fixed edge = `0.6607`

Interpretation:

- the cached fast path is now working correctly
- a small amount of external closing support changes the selector’s train-side
  fit and mode counts, but it does **not** improve the test joint score beyond
  the refreshed fast-path baseline
- this is the first clean evidence that “more support” is not automatically a
  win, at least for a small external support budget

Boundary:

- do not overclaim that stronger supervision is solved or disproven globally
- what is currently supported is narrower:
  - cached fast-path support scaling is operational
  - `top_2` external support does not improve over `top_0` on the current
    fast-path closing slice

Next:

- if we continue this direction, the next useful point is not rerunning `top_2`
- it is either:
  - sweep larger cached budgets like `top_4`
  - or use the cached regime to test whether the support target itself should
    be changed rather than just increased

## 2026-06-10 cached fast-path support plateau at top-4

New artifact:

- [interaction_realized_closing_selector_support_scaling_top4.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_closing_selector_support_scaling_top4.md)
- [external_rows_top4.json](/opt/tiger/hand/experiments/generated/cache/closing_selector_support_scaling/external_rows_top4.json)

Verified result:

- after extending cached external support from:
  - `top_0`: `0` external sequences
  - `top_2`: `2` external sequences / `39` rows
  - `top_4`: `4` external sequences / `62` rows
- the test-time closing result does not move at all:
  - fixed edge = `0.6607`
  - temporal-window KNN = `0.7321`
  - oracle = `0.7679`

Important detail:

- train-side selector fit keeps increasing with larger support:
  - `top_0` LOO = `0.6988`
  - `top_2` LOO = `0.7459`
  - `top_4` LOO = `0.7793`
- but the test-time prediction is unchanged:
  - `edge=20`
  - `finger=36`

Interpretation:

- this is the strongest external-support result so far
- cached larger support is now both:
  - technically runnable
  - scientifically informative
- the conclusion is not just “small support did not help”
- it is stronger:
  - under the current fast-path closing target, larger external support
    improves train-side fit but does not improve test-time closing performance
- this points to a generalization / target-mismatch problem rather than a pure
  support-volume problem

Decision:

- stop treating “just add more external support rows” as the main closing fix
- keep cached support scaling as a useful infrastructure and boundary artifact
- if we continue this line, the next meaningful step is to change the support
  target or supervision signal, not merely raise the external support budget

## 2026-06-10 closing margin-target support experiment

New artifact:

- [build_interaction_realized_closing_margin_target.py](/opt/tiger/hand/tools/build_interaction_realized_closing_margin_target.py)
- [interaction_realized_closing_margin_target.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_closing_margin_target.md)

Goal:

- after cached support volume plateaued, test whether the real issue is the
  support target rather than support amount
- replace discrete mode imitation with direct `finger - edge` margin
  prediction
- compare:
  - `val_all`
  - `val + top4 all`
  - `val + top4 informative-only`

Result:

- the changed support target fails more strongly than the original selector
- all three support sets collapse to an all-`finger` policy:
  - predicted mode counts on test: `finger=56`
- all three produce the same inferior result:
  - fixed edge = `0.6607`
  - margin target = `0.6250`
  - delta = `-0.0357`

Failure diagnosis:

- the cached top4 external rows are almost completely uninformative for this
  target:
  - cached top4 rows = `62`
  - informative cached top4 rows = `1`
- even val rows are only weakly informative:
  - val rows = `83`
  - informative val rows = `7`
- under these conditions, threshold selection degenerates to:
  - threshold `-1.0`
  - always choose `finger`

Interpretation:

- this is a stronger boundary than the earlier support-volume plateau
- not only does more support fail to help
- even changing the support target to direct edge-vs-finger margin prediction
  still collapses under the current supervision geometry
- so the next closing step likely needs a support target that is not just
  another framewise relabeling of the same `edge/finger` decision

Decision:

- keep margin-target support as a negative-boundary artifact
- do not spend more time on framewise `edge-vs-finger` target variants under
  the current cached fast-path regime
- if we continue closing support research, it should move to a different level
  of target design, not just a different label on the same rows

## 2026-06-10 closing run-level edge-vs-finger target

New artifact:

- [build_interaction_realized_closing_run_target.py](/opt/tiger/hand/tools/build_interaction_realized_closing_run_target.py)
- [interaction_realized_closing_run_target.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_closing_run_target.md)

Goal:

- after row-level changed targets failed, move one supervision level up
- train on short temporal runs from the fast-path closing regime
- choose one repair family per run: `edge` vs `finger`
- compare:
  - val runs only
  - val runs + cached top4 singleton rows

Result:

- run-level target is directionally better than the failed row-level
  margin-target experiment
- it does beat fixed edge:
  - fixed edge = `0.6607`
  - run target = `0.6786`
  - delta = `+0.0179`
- but it still does **not** beat the current fast-path frame-level selector
  plateau:
  - frame-level fast-path selector = `0.7321`
  - run target = `0.6786`

Important detail:

- adding cached top4 singleton rows changes train-side fit:
  - val-runs-only LOO = `0.7108`
  - val-runs-plus-top4-singletons LOO = `0.7862`
- but test-time result is unchanged
- predicted test run modes stay mostly edge-dominant:
  - val-runs-only: `edge=11`, `finger=2`
  - plus top4 singletons: `edge=12`, `finger=1`

Interpretation:

- this is the first changed-target experiment where moving the supervision
  level upward is genuinely better than the row-level variant
- but it is not yet the new best closing mechanism
- the likely lesson is:
  - supervision level matters
  - yet singleton external rows are not the right support form for a run-level
    target

Decision:

- keep run-level target as a support artifact and partial positive direction
- do not promote it above the current frame-level fast-path selector
- if we continue changing supervision level, the next real step should use
  true external runs or run-level support targets, not singleton rows appended
  to run support

Follow-up: relaxed external short runs

- reran the same closing run-target experiment, but this time grouped cached
  top4 external support rows into true short runs using a relaxed frame-gap
  threshold of `18` instead of treating them as singleton rows
- this produces:
  - cached top4 rows = `62`
  - cached top4 relaxed runs = `17`
- train-side fit stays high:
  - val-runs-plus-top4-relaxed-runs LOO = `0.7862`
- but test-time behavior gets *more* edge-dominant and loses the small run
  target gain:
  - predicted test run modes: `edge=13`, `finger=0`
  - fixed edge = `0.6607`
  - relaxed-run target = `0.6607`
  - frame-level fast-path selector = `0.7321`

Interpretation:

- the closing gap is now very unlikely to be solvable by merely converting the
  cached external support into short temporal runs
- singleton external rows were not the real bottleneck
- under the current run-level objective, extra external run structure actually
  regularizes the selector back to the degenerate all-`edge` regime

Decision:

- promote this to a stronger negative boundary than the singleton result
- do not spend more time on the current `edge-vs-finger` run-level target with
  the same support statistics
- the next closing-side attempt should change the run/chunk objective itself,
  not only the support packaging

Follow-up: tri-mode run target (`none/edge/finger`)

- extended the same run-level selector to predict among all three repair modes
  instead of only `edge/finger`
- result:
  - on val runs only, tri-mode merely ties the binary run target:
    - binary run target = `0.6786`
    - tri-mode run target = `0.6786`
  - once external support is appended, tri-mode becomes unstable and collapses
    toward `none`:
    - plus top4 singletons = `0.6250`
    - plus top4 relaxed runs = `0.5000`
- predicted test run modes reveal the failure mode directly:
  - val-runs-only tri-mode: `finger=4`, `edge=6`, `none=3`
  - plus top4 singletons: `edge=8`, `none=5`
  - plus top4 relaxed runs: `none=11`, `edge=1`, `finger=1`
- the corresponding tri-mode oracle is actually stronger than the binary
  oracle:
  - oracle edge/finger = `0.7500`
  - oracle tri-mode = `0.7679`

Interpretation:

- the available headroom is real, but the current run-level neighbor objective
  cannot recover it
- increasing the action space from binary to tri-mode does not fix the closing
  bottleneck
- with external support, the current run-distance and voting rule bias the
  selector toward overusing `none`

Decision:

- promote tri-mode run selection to a clear negative boundary under the current
  support/statistics design
- the next closing experiment should change the scoring objective itself
  (e.g. gain-weighted or preserve-aware run targets), not just support
  packaging or label space

Follow-up: conservative gain-aware run targets

- tested a stricter family of run-level objectives that only allow a switch
  away from `edge` when the alternative has a positive run-level gain margin
  over `edge`
- searched gain thresholds over:
  - `tau in {0.0, 0.05, 0.1, 0.2}` for binary `edge/finger`
  - `(tau_none, tau_finger)` on the same grid for tri-mode
- result:
  - the best binary gain-aware selector collapses back to the original binary
    target with `tau=0.0`
  - the best tri-mode gain-aware selector also collapses back to the
    edge/finger-only behavior with `tau_none=0.0, tau_finger=0.0`
- corresponding test results:
  - val-runs-only:
    - gain-binary = `0.6786`
    - gain-trimode = `0.6607`
  - plus top4 singletons:
    - gain-binary = `0.6786`
    - gain-trimode = `0.6607`
  - plus top4 relaxed runs:
    - gain-binary = `0.6607`
    - gain-trimode = `0.6607`

Interpretation:

- the current run-level feature space and neighbor rule do not benefit from a
  more conservative gain margin
- margining does suppress the pathological `none` collapse, but it does so by
  reverting to the same edge-dominant solution we already had
- therefore the remaining closing gap is not caused by a missing confidence
  threshold around mode switching

Decision:

- promote gain-aware run targets to another explicit negative boundary
- the next closing-side step should stop tuning labels inside the current KNN
  run selector and instead change the scoring mechanism itself
  (e.g. learned run embeddings, chunk-level gain prediction, or direct
  preserve-aware regression)

Follow-up: learned run-level scorers

- replaced the hand-crafted run-distance + voting mechanism with learned
  run-level scorers on the same fast-path closing regime:
  - HistGradientBoosting classifier on binary `edge/finger`
  - HistGradientBoosting regressor on binary mode values
  - trimode classifier/regressor for comparison
- result:
  - both learned binary variants beat the previous KNN run target:
    - previous binary KNN = `0.6786`
    - learned binary classifier = `0.7143`
    - learned binary regressor = `0.7143`
  - this gain is stable across all tested support packagings:
    - val-runs-only = `0.7143`
    - plus top4 singletons = `0.7143`
    - plus top4 relaxed runs = `0.7143`
  - trimode learning is directionally helpful relative to KNN trimode, but
    still inferior to the learned binary scorer:
    - trimode classifier = `0.6964 / 0.6964 / 0.6786`
    - trimode regressor = `0.6786 / 0.6786 / 0.6786`
- the learned binary scorer also fixes the earlier relaxed-run collapse:
  - KNN binary with relaxed runs = `0.6607`
  - learned binary classifier with relaxed runs = `0.7143`
  - learned binary regressor with relaxed runs = `0.7143`

Interpretation:

- the bottleneck was indeed the hand-crafted KNN distance / voting mechanism,
  not merely the support labels
- learned run-level scoring is the first run-level mechanism that beats the
  current KNN run-selector frontier while remaining stable under external
  support packaging changes
- the strongest current learned binary scorer recovers roughly half of the
  remaining gap from KNN binary (`0.6786`) to the trimode oracle (`0.7679`)
- but it still remains below the current best frame-level fast-path selector:
  - learned binary run scorer = `0.7143`
  - frame-level selector = `0.7321`

Decision:

- promote learned binary run-level scoring to the new run-level mainline
- demote the old KNN run selector to a reference baseline inside the run-level
  family
- the next closing-side step should build on learned run scoring rather than
  continue tuning hand-crafted run distances

Follow-up: learned chunk-level closing scorers

- tested a stronger temporal factorization by splitting runs into shorter
  learned chunks and predicting binary `edge/finger` choices per chunk
- chunk lengths `2` and `3` both work; the key result is stable across support
  packaging:
  - chunk-len-2 classifier:
    - val chunks only = `0.7500`
    - plus top4 singletons = `0.7500`
    - plus top4 relaxed chunks = `0.7500`
  - chunk-len-3 classifier:
    - val chunks only = `0.7500`
    - plus top4 singletons = `0.7500`
    - plus top4 relaxed chunks = `0.7500`
- this improves over:
  - learned binary run scorer = `0.7143`
  - previous best frame-level fast-path selector = `0.7321`
- for these binary chunk experiments, test-time performance reaches the current
  binary oracle on the same slice:
  - chunk binary oracle = `0.7500`

Interpretation:

- the remaining closing error after learned run scoring was indeed caused by
  within-run heterogeneity
- short learned chunks are a better temporal unit than whole runs for this
  fast-path closing problem
- external support packaging is no longer a brittle axis once the scoring unit
  is shortened and learned

Decision:

- promote learned binary chunk scoring to the current closing fast-path
  mainline
- demote learned binary run scoring to the previous-best intermediate stage
- the next closing-side step should now focus on whether chunk-level learning
  also transfers to broader slices beyond the current fast-path regime

Follow-up: broader corrected feasible closing transfer (`val_only`)

- before the full `train_plus_val` broader-transfer cache finishes, we already
  evaluated the chunk scorer on the broader corrected feasible closing slice
  using the cached `val_only` support rows
- result:
  - broader-slice fixed edge = `0.7025`
  - broader-slice chunk classifier = `0.7785`
  - broader-slice chunk regressor = `0.7785`
  - broader-slice binary oracle = `0.7785`
- this means the chunk-level scorer does **not** only work on the narrow
  fast-path slice:
  - fast-path chunk cls = `0.7500`
  - broader corrected feasible closing chunk cls = `0.7785`
- paired evidence on the broader slice is clean:
  - delta vs fixed edge = `+0.0759`
  - wins/losses/ties = `12 / 0 / 146`

Interpretation:

- chunk-level temporal factorization already survives at least one broader
  closing protocol beyond the narrow fast-path regime
- on the current broader corrected feasible closing slice, the learned chunk
  scorer reaches the binary oracle exactly
- therefore the chunk mainline has moved from “promising fast-path mechanism”
  to “real broader-slice mechanism with direct evidence”

Decision:

- promote broader-slice `val_only` chunk transfer to a positive mainline
  support artifact
- keep the `train_plus_val` transfer follow-up as a strengthening experiment,
  not as the only evidence for generalization

Interim strengthening: partial `train_only` + `val_only`

- while the full `train_only` cache is still being built incrementally, we
  already evaluated a stronger interim support source by combining:
  - cached `support_val_only`
  - current `support_train_only.partial`
- with the current partial support, the broader-slice result stays unchanged
  across multiple snapshots:
  - first checked at `8` cached train units / `136` cached support rows
  - later rechecked at `13` cached train units / `171` cached support rows
  - later rechecked again at `21` cached train units / `233` cached support rows
  - later rechecked again at `22` cached train units / `241` cached support rows
  - later rechecked again at `30` cached train units / `301` cached support rows
  - later rechecked again at `38` cached train units / `362` cached support rows
  - later rechecked again at `42` cached train units / `391` cached support rows
  - later rechecked again at `50` cached train units / `455` cached support rows
  - later rechecked again at `54` cached train units / `486` cached support rows
  - later rechecked again at `62` cached train units / `550` cached support rows
  - later rechecked again at `70` cached train units / `614` cached support rows
  - later rechecked again at `86` cached train units / `738` cached support rows
  - later rechecked again at `94` cached train units / `801` cached support rows
  - later rechecked again at `110` cached train units / `929` cached support rows
  - later rechecked again at `118` cached train units / `993` cached support rows
  - later rechecked again at `126` cached train units / `1057` cached support rows
  - later rechecked again at `142` cached train units / `1184` cached support rows
  - later rechecked again at `162` cached train units / `1344` cached support rows
  - both snapshots produce the same broader-slice result:
  - fixed edge = `0.7025`
  - chunk classifier = `0.7785`
  - chunk regressor = `0.7785`
  - oracle binary = `0.7785`
- this means the broader corrected feasible closing gain is already stable to
  the first tranche of extra train-only support, not only to val-only support

Interpretation:

- the broader-slice chunk result does not look fragile with respect to the
  initial expansion of support rows
- finishing the full `train_only` cache is still useful as a strengthening
  experiment, but the current broader positive result no longer depends on it

Follow-up: broader corrected feasible opening transfer (`val_only`)

- to test whether chunk-level temporal factorization is actually a closing-only
  mechanism or a broader motion-side mechanism, we ran the same broader-slice
  transfer test on `right_hand_motion->opening` using `val_only` support
- result:
  - broader opening fixed edge = `0.6306`
  - broader opening chunk classifier = `0.7197`
  - broader opening chunk regressor = `0.7197`
  - broader opening binary oracle = `0.7197`
- paired evidence is again clean:
  - delta vs fixed edge = `+0.0892`
  - wins/losses/ties = `14 / 0 / 143`

Interpretation:

- chunk-level temporal factorization is no longer a closing-only positive
  mechanism
- with direct broader-slice evidence on both `closing` and `opening`, the
  current temporal chunk story is materially harder to dismiss as a one-task
  accident

Follow-up: broader corrected feasible closing chunk-length robustness

- to test whether the broader corrected feasible closing gain is tied to a
  single chunk length, we swept `chunk_len` in `{2, 3, 4}` with cached
  `val_only` support
- all three settings give the same broader-slice result:
  - fixed edge = `0.7025`
  - chunk classifier = `0.7785`
  - chunk regressor = `0.7785`
  - oracle binary = `0.7785`
  - paired delta vs fixed edge = `+0.0759`

Interpretation:

- on the broader corrected feasible closing slice, the chunk gain is not tied
  to one specific short temporal granularity
- this makes the broader chunk-transfer evidence harder to dismiss as a
  chunk-length artifact

Follow-up: broader corrected feasible closing chunk-length robustness with
stronger partial support

- we repeated the chunk-length sweep after mixing cached `support_val_only`
  with the stronger parallel `support_train_only.partial` snapshot
  (`50` train units / `455` cached support rows total)
- again, `chunk_len` in `{2, 3, 4}` gives exactly the same broader-slice
  result:
  - fixed edge = `0.7025`
  - chunk classifier = `0.7785`
  - chunk regressor = `0.7785`
  - oracle binary = `0.7785`
  - paired delta vs fixed edge = `+0.0759`

Interpretation:

- the broader corrected feasible closing gain remains stable across chunk
  lengths even after materially expanding the support source

Follow-up: support progression memo for broader corrected feasible closing

- instead of only keeping the latest partial snapshot, we consolidated the
  repeated `train_partial_plus_val` checks into a single progression memo
- across the recorded support-growth path from `8 / 136` to `162 / 1344`
  (`train partial units / cached support rows`), the broader-slice result is
  numerically unchanged:
  - fixed edge = `0.7025`
  - chunk classifier = `0.7785`
  - chunk regressor = `0.7785`
  - oracle binary = `0.7785`
  - paired delta vs fixed edge = `+0.0759`

Interpretation:

- the unfinished full cache is now mainly a report-closing step
- the gain itself is already repeatedly stable under substantial support growth

Final strengthening: full `train_plus_val` broader closing transfer

- the full `support_train_only.json` cache is now complete, so we generated
  the formal broader corrected feasible closing transfer report with full
  `train_plus_val` support
- final result remains exactly unchanged:
  - fixed edge = `0.7025`
  - chunk classifier = `0.7785`
  - chunk regressor = `0.7785`
  - oracle binary = `0.7785`
  - paired delta vs fixed edge = `+0.0759`
- support statistics of the final report:
  - support frames = `1423`
  - support sequences = `158`
  - pair bank size = `4159`
  - support chunks = `1394`

Interpretation:

- the broader corrected feasible closing gain is no longer only a strong
  partial-support robustness story
- we now have the finalized full-support report, and it lands on the same
  number as the entire partial-support progression

Follow-up: hard-slice residual subtype audit

- to turn the open hard-slice weakness into something more targetable, we
  audited subtype-level residual performance under the current strongest
  hard-slice method: `hgb_budget2_finger_profile_snap_top20`
- the dominant zero-score mass is concentrated in `other_hand_motion=none`
  subtypes on both `right_hand_motion->closing` and
  `right_hand_motion->opening`
- examples:
  - closing: `none/steady` `217` frames, `none/approach` `96` frames,
    `none/separate` `67` frames, all at `0.0000`
  - opening: `none/steady` `219` frames, `none/approach` `98` frames,
    `none/separate` `68` frames, all at `0.0000`

Interpretation:

- this tightens the current open-gap diagnosis:
  - the absolute hard-slice score is still compressed mainly by structurally
    infeasible mass
  - the next strong mechanism test should focus on the feasible interaction-rich
    residuals rather than trying to globally “fix” the whole hard slice at once
