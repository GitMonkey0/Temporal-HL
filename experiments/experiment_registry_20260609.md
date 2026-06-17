# Experiment Registry

Date: 2026-06-09

This is a research operations memo.
It is not paper text.

## Purpose

Provide one place to answer:

- what experiments exist
- which scripts generated them
- where the outputs live
- what conclusion each experiment currently supports
- whether the result is part of the mainline, only a reviewer-defense artifact,
  or retired

This is intentionally selective rather than exhaustive.
It tracks the experiments that currently matter for project decisions.

## Status Keys

- `mainline`: actively supports the strongest current thesis
- `support`: useful supporting evidence, but not the main wedge
- `negative-boundary`: important negative result that constrains claims
- `retired`: explored and learned from, but should not be promoted
- `open-gap`: identifies the remaining strongest weakness

## Current Mainline

As of 2026-06-09, the strongest thesis-supporting line is:

- sequence-native symbolic hand representation
- explicit temporal channels inside the representation
- anatomy-aware grouped factorization
- zero-harm family-level repair
- local editability and control
- transition-conditioned grouped-motif editing
- compact search / pair-guided reranking for hard right-hand interaction slices
- corrected full-slice right-support relaxation on the hard right-hand
  interaction slices

What is **not** the mainline:

- raw sequence classification
- generic retrieval wins
- old HL plus stronger temporal encoder
- generic motion tokenization

## A. Top-Level Entry Points

### 1. Topline bundle

- Status: `mainline`
- Why it matters:
  - single entry point for the strongest evidence stack
- Script:
  - [build_topline_evidence_bundle.py](/opt/tiger/hand/tools/build_topline_evidence_bundle.py)
- Outputs:
  - [topline_evidence_bundle.md](/opt/tiger/hand/experiments/generated/summary_tables/topline_evidence_bundle.md)
  - [topline_evidence_bundle.json](/opt/tiger/hand/experiments/generated/topline_evidence_bundle.json)
- Current use:
  - first file to open when checking whether the thesis still holds

### 2. Rolling experiment log

- Status: `support`
- Why it matters:
  - chronological narrative of what was tried, what failed, and why the
    current line survived
- File:
  - [symbolic_pretrain_notes.md](/opt/tiger/hand/experiments/symbolic_pretrain_notes.md)
- Current use:
  - audit trail for experiment history
  - source of “why did we pivot” explanations

### 3. Reviewer risk map

- Status: `support`
- Why it matters:
  - maps each likely reviewer attack to current counter-evidence and remaining
    gaps
- File:
  - [reviewer_risk_matrix_20260609.md](/opt/tiger/hand/experiments/reviewer_risk_matrix_20260609.md)
- Current use:
  - check whether a new experiment changes the claim boundary

## B. Structure And Repair

### 4. Grouped symbolic structure frontier

- Status: `mainline`
- Why it matters:
  - proves structure and repair are separate gains
- Script / source:
  - existing generated frontier files; surfaced through the topline bundle
- Outputs:
  - [current_code_symbolic_frontier.md](/opt/tiger/hand/experiments/generated/summary_tables/current_code_symbolic_frontier.md)
  - [current_code_symbolic_frontier.json](/opt/tiger/hand/experiments/generated/current_code_symbolic_frontier.json)
- Key current conclusion:
  - `flat seq @1.0 = 0.6923`
  - `grouped seq @1.0 = 0.8974`
  - `family seq @1.0 = 1.0000`
  - grouped structure is a real intrinsic gain
  - family repair is a second layer, not the only reason the method works

### 5. Representation risk bundle

- Status: `mainline`
- Why it matters:
  - consolidates structure, control, counterfactuals, and transition evidence
- Outputs:
  - [representation_risk_bundle.md](/opt/tiger/hand/experiments/generated/summary_tables/representation_risk_bundle.md)
  - [representation_risk_bundle.json](/opt/tiger/hand/experiments/generated/representation_risk_bundle.json)
- Current use:
  - best single bundle for “why symbolic structure is not just accuracy cleanup”

## C. Local Control And Editability

### 6. Control evidence bundle

- Status: `mainline`
- Why it matters:
  - strongest non-classification argument for symbolic structure
- Outputs:
  - [control_evidence_bundle.md](/opt/tiger/hand/experiments/generated/summary_tables/control_evidence_bundle.md)
  - [control_evidence_bundle.json](/opt/tiger/hand/experiments/generated/control_evidence_bundle.json)
- Key current conclusion:
  - symbolic local edits are clean where proxy edits cause collateral damage

### 7. Transition-conditioned symbolic editor

- Status: `mainline`
- Why it matters:
  - realized editing result, not only donor auditing
- Script:
  - [build_transition_conditioned_symbolic_editor.py](/opt/tiger/hand/tools/build_transition_conditioned_symbolic_editor.py)
- Outputs:
  - [transition_conditioned_symbolic_editor.md](/opt/tiger/hand/experiments/generated/summary_tables/transition_conditioned_symbolic_editor.md)
  - [transition_conditioned_symbolic_editor.json](/opt/tiger/hand/experiments/generated/transition_conditioned_symbolic_editor.json)
- Key current conclusion:
  - coarse opening/closing success is too weak
  - grouped-motif fidelity is the right realized-edit metric
  - symbolic grouped-motif edits consistently beat opaque proxies
- Current limitation:
  - interaction-aware realized editing is still the main unresolved frontier

### 8. Interaction vs noninteraction summary

- Status: `mainline`
- Why it matters:
  - compresses the grouped-motif editor result into a top-level slice view
- Script:
  - [build_interaction_vs_noninteraction_summary.py](/opt/tiger/hand/tools/build_interaction_vs_noninteraction_summary.py)
- Outputs:
  - [interaction_vs_noninteraction_summary.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_vs_noninteraction_summary.md)
  - [interaction_vs_noninteraction_summary.json](/opt/tiger/hand/experiments/generated/interaction_vs_noninteraction_summary.json)
- Key current conclusion:
  - symbolic edits beat proxies on both interaction and noninteraction slices
  - interaction remains the concentrated weakness

## D. Hard Right-Hand Interaction Slices

### 9. Pair-guided reranker multi-slice study

- Status: `mainline`
- Why it matters:
  - shows the compact search mechanism is not a one-off weak-slice trick
- Script:
  - [build_pairguided_reranker_multislice.py](/opt/tiger/hand/tools/build_pairguided_reranker_multislice.py)
- Outputs:
  - [pairguided_reranker_multislice.md](/opt/tiger/hand/experiments/generated/summary_tables/pairguided_reranker_multislice.md)
  - [pairguided_reranker_multislice.json](/opt/tiger/hand/experiments/generated/pairguided_reranker_multislice.json)
- Key current conclusion:
  - strongest gains are concentrated on:
    - `right_hand_motion->closing`
    - `right_hand_motion->opening`
  - should not be sold as universal across all slices

### 10. Hard-slice compact-search bundle

- Status: `mainline`
- Why it matters:
  - highest-level summary for the hard interaction bottleneck and the recovery
    mechanism
- Script:
  - [build_hard_slice_compact_search_bundle.py](/opt/tiger/hand/tools/build_hard_slice_compact_search_bundle.py)
- Outputs:
  - [hard_slice_compact_search_bundle.md](/opt/tiger/hand/experiments/generated/summary_tables/hard_slice_compact_search_bundle.md)
  - [hard_slice_compact_search_bundle.json](/opt/tiger/hand/experiments/generated/hard_slice_compact_search_bundle.json)
- Key current conclusion:
  - on `right_hand_motion->opening` interaction:
    - single donor overall joint hit `0.0073`
    - relaxed top-20 `0.0320`
    - pair-guided top-10 `0.0320`
    - pair-guided top-5 `0.0302`
  - compact search is a real recovery mechanism for the hard interaction case

### 11. Interaction realized pair-guided editor

- Status: `support`
- Why it matters:
  - converts the hard-slice interaction story from a search-only result into a
    realized interaction-editing result
- Script:
  - [build_interaction_realized_pairguided_editor.py](/opt/tiger/hand/tools/build_interaction_realized_pairguided_editor.py)
- Outputs:
  - [interaction_realized_pairguided_editor.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_pairguided_editor.md)
  - [interaction_realized_pairguided_editor.json](/opt/tiger/hand/experiments/generated/interaction_realized_pairguided_editor.json)
- Key current conclusion:
  - under a stricter joint interaction criterion, target-hand-only editing
    collapses to zero joint score
  - split-donor search recovers non-zero joint success
  - pair-guided reranking adds a further modest gain on the hard right-hand
    slices
- Why not mainline yet:
  - the absolute interaction-aware realized joint score is still low
  - this is evidence of necessity and partial recovery, not final closure

### 11b. Interaction realized support scaling

- Status: `negative-boundary`
- Why it matters:
  - tests whether the remaining interaction weakness is mainly a support-size
    bottleneck
- Script:
  - [build_interaction_realized_support_scaling.py](/opt/tiger/hand/tools/build_interaction_realized_support_scaling.py)
- Outputs:
  - [interaction_realized_support_scaling.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_support_scaling.md)
  - [interaction_realized_support_scaling.json](/opt/tiger/hand/experiments/generated/interaction_realized_support_scaling.json)
- Key current conclusion:
  - larger preserve-donor family budgets increase availability
  - deeper search also expands candidate coverage
  - but the best absolute joint interaction score stays flat
- Current implication:
- the remaining weakness is not mainly solved by more support depth or family
  budget
- the next step must change composition or editing mechanism

### 11c. Feasible interaction-rich residual audit

- Status: `open-gap`
- Why it matters:
  - turns the residual subtype diagnosis into a quantitative hard-slice
    decomposition
  - separates structurally infeasible `other_hand_motion=none` mass from the
    true feasible interaction-rich residual
- Script:
  - [build_interaction_realized_feasible_interaction_rich_residuals.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_interaction_rich_residuals.py)
- Outputs:
  - [interaction_realized_feasible_interaction_rich_residuals.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_interaction_rich_residuals.md)
  - [interaction_realized_feasible_interaction_rich_residuals.json](/opt/tiger/hand/experiments/generated/interaction_realized_feasible_interaction_rich_residuals.json)
- Key current conclusion:
  - on the full hard slice, best-vs-baseline repair gains stay near `+0.021`
  - on the feasible interaction-rich residual:
    - closing improves from `0.1203` to `0.1962` (`+0.0759`)
    - opening improves from `0.1146` to `0.1911` (`+0.0764`)
  - the feasible interaction-rich score is about `3.5x` the mixed full-slice
    score on both tasks
- Current implication:
  - the remaining hard-slice weakness is still real, but the next mechanism
    should be judged on true feasible residuals rather than on the
    structurally-infeasible `none` mass

## E. Chunk Transfer Beyond Fast-Path Closing

### 12. Opening chunk-transfer length robustness

- Status: `mainline`
- Why it matters:
  - removes the residual concern that the opening-side broader gain only
    appears at one lucky chunk length
- Script:
  - [build_interaction_realized_opening_chunk_transfer_length_sweep_val_only.py](/opt/tiger/hand/tools/build_interaction_realized_opening_chunk_transfer_length_sweep_val_only.py)
- Outputs:
  - [interaction_realized_opening_chunk_transfer_length_sweep_val_only.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_opening_chunk_transfer_length_sweep_val_only.md)
  - [interaction_realized_opening_chunk_transfer_length_sweep_val_only.json](/opt/tiger/hand/experiments/generated/interaction_realized_opening_chunk_transfer_length_sweep_val_only.json)
- Key current conclusion:
  - on broader corrected feasible opening with val-only support:
    - `chunk_len = 2 / 3 / 4` all give:
      - fixed edge `0.6306`
      - chunk `0.7197`
      - delta `+0.0892`
  - the opening-side broader transfer gain is therefore stable across
    chunk lengths, not a single-length artifact

## F. Feasible Left Routing Closure

### 13. Lightweight routing closure bundle

- Status: `negative-boundary`
- Why it matters:
  - consolidates the feasible-left routing attempts so the project stops
    revisiting the same weak family under slightly different names
- Scripts:
  - [build_interaction_realized_feasible_left_opening_chunk_knn.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_left_opening_chunk_knn.py)
  - [build_interaction_realized_feasible_rich_template_gate.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_rich_template_gate.py)
  - [build_interaction_realized_feasible_left_routing_closure.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_left_routing_closure.py)
- Outputs:
  - [interaction_realized_feasible_left_opening_chunk_knn.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_opening_chunk_knn.md)
  - [interaction_realized_feasible_rich_template_gate.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_rich_template_gate.md)
  - [interaction_realized_feasible_left_routing_closure.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_routing_closure.md)
- Key current conclusion:
  - closing:
    - fixed task-best stays at `0.6582`
    - all lightweight routing variants remain below it
    - best cheap alternatives top out around `0.6203` or `0.5696`
  - opening:
    - fixed task-best stays at `0.6497`
    - frame-level temporal-window KNN reaches `0.6624` (`+0.0127`)
    - dense KNN only ties
    - opening chunk KNN falls to `0.5350` (`-0.1146`)
  - framewise oracle headroom remains:
    - closing `0.7405`
    - opening `0.7197`
- Current implication:
  - lightweight routing is effectively closed as the next-step family
  - the remaining gap should be attacked with stronger supervision or changed
    objectives rather than more cheap gate variants

### 13b. Dense gain regressor follow-up

- Status: `negative-boundary`
- Why it matters:
  - tests a stronger version of the selector story where supervision becomes
    continuous gain regression rather than sparse mode labels
- Script:
  - [build_interaction_realized_feasible_left_gain_regressor.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_left_gain_regressor.py)
- Outputs:
  - [interaction_realized_feasible_left_gain_regressor.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_gain_regressor.md)
  - [interaction_realized_feasible_left_gain_regressor.json](/opt/tiger/hand/experiments/generated/interaction_realized_feasible_left_gain_regressor.json)
- Key current conclusion:
  - closing:
    - fixed task-best `0.6582`
    - gain regressor `0.5759`
    - delta `-0.0823`
  - opening:
    - fixed task-best `0.6497`
    - gain regressor `0.5541`
    - delta `-0.0955`
- Current implication:
  - even changing the supervision target from discrete mode labels to dense
    gains does not recover the feasible-left selector gap
  - the selector family is now closed more strongly than before

### 13c. Oracle gap audit and sparse override

- Status: `negative-boundary`
- Why it matters:
  - tests whether the structured oracle conflict can be captured by a sparse
    subtype override rather than a global selector
- Scripts:
  - [build_interaction_realized_feasible_left_oracle_gap_audit.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_left_oracle_gap_audit.py)
  - [build_interaction_realized_feasible_left_sparse_override.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_left_sparse_override.py)
- Outputs:
  - [interaction_realized_feasible_left_oracle_gap_audit.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_oracle_gap_audit.md)
  - [interaction_realized_feasible_left_sparse_override.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_sparse_override.md)
- Key current conclusion:
  - oracle headroom is structured:
    - closing conflict slice `+0.1074`
    - opening conflict slice `+0.0733`
  - but sparse subtype override still fails:
    - closing `0.6329 < 0.6582`
    - opening `0.6497 = 0.6497`
- Current implication:
  - the remaining headroom is not recoverable by a small whitelist of subtype
    overrides
  - the next promising redesign has to target a richer representation/objective
    of preserve-side success rather than more subtype-rule surgery

### 13d. Soft preserve-objective follow-up

- Status: `negative-boundary`
- Why it matters:
  - directly tests whether the current exact grouped preserve target is too
    harsh by replacing mode choice with a softer local agreement objective
- Script:
  - [build_interaction_realized_feasible_left_soft_objective.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_left_soft_objective.py)
- Outputs:
  - [interaction_realized_feasible_left_soft_objective.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_soft_objective.md)
  - [interaction_realized_feasible_left_soft_objective.json](/opt/tiger/hand/experiments/generated/interaction_realized_feasible_left_soft_objective.json)
- Key current conclusion:
  - best soft-objective policy still loses:
    - closing `0.5759 < 0.6582`
    - opening `0.5796 < 0.6497`
  - yet it improves left transition agreement beyond both fixed task-best and
    oracle on both tasks
- Current implication:
  - the remaining preserve-side gap is not just a consequence of the exact
    grouped objective being too strict
  - there is a real mismatch between local agreement targets and final
    preserve-side success

## G. Granularity Targeting

### 14. Feasible-left oracle granularity audit

- Status: `mainline`
- Why it matters:
  - identifies the temporal granularity where the remaining feasible-left
    headroom actually lives
- Script:
  - [build_interaction_realized_feasible_left_oracle_granularity.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_left_oracle_granularity.py)
- Outputs:
  - [interaction_realized_feasible_left_oracle_granularity.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_oracle_granularity.md)
  - [interaction_realized_feasible_left_oracle_granularity.json](/opt/tiger/hand/experiments/generated/interaction_realized_feasible_left_oracle_granularity.json)
- Key current conclusion:
  - closing:
    - fixed `0.6582`
    - sequence oracle `0.6646`
    - subtype oracle `0.6962`
    - run oracle `0.7089`
    - frame oracle `0.7405`
  - opening:
    - fixed `0.6497`
    - sequence oracle `0.6624`
    - subtype oracle `0.6752`
    - run oracle `0.6943`
    - frame oracle `0.7197`
- Current implication:
  - the useful redesign target is run/chunk-level, not sequence-level
  - there is still some true frame-level headroom, but a run-level target could
    already recover a substantial fraction of the missing gap if learned well

### 14a. Feasible-left run-level learned scorer

- Status: `negative-boundary`
- Why it matters:
  - tests whether the run-level granularity identified by the oracle audit can
    be turned into a real learned gain instead of remaining a diagnostic bound
- Script:
  - [build_interaction_realized_feasible_left_run_learned_scorer.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_left_run_learned_scorer.py)
- Outputs:
  - [interaction_realized_feasible_left_run_learned_scorer.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_run_learned_scorer.md)
  - [interaction_realized_feasible_left_run_learned_scorer.json](/opt/tiger/hand/experiments/generated/interaction_realized_feasible_left_run_learned_scorer.json)
- Key current conclusion:
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
- Current implication:
  - the run-level oracle headroom is real, but a simple run-summary
    classifier/regressor does not recover it robustly
  - `run_reg` only gives a modest opening gain and still loses badly on
    closing
  - therefore this is closure evidence for the preserve-side line, not a new
    mainline mechanism

### 11b2. Corrected global right-support bundle

- Status: `mainline`
- Why it matters:
  - provides the corrected authoritative result for full hard right-hand slices
    after fixing the candidate-pool target bug
- Scripts:
  - [build_interaction_realized_global_right_support_check.py](/opt/tiger/hand/tools/build_interaction_realized_global_right_support_check.py)
  - [build_interaction_realized_global_right_support_bundle.py](/opt/tiger/hand/tools/build_interaction_realized_global_right_support_bundle.py)
- Outputs:
  - [interaction_realized_global_right_support_bundle.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_global_right_support_bundle.md)
  - [interaction_realized_global_right_support_bundle.json](/opt/tiger/hand/experiments/generated/interaction_realized_global_right_support_bundle.json)
- Key current conclusion:
  - corrected `hgb_strict -> hgb_relax_both`
  - right closing: `0.0250 -> 0.1807`
  - right opening: `0.0533 -> 0.1812`
  - paired significance is decisive on both tasks
- Current use:
  - authoritative hard-slice interaction-editing entry point after the
    candidate-pool fix

### 11b3. Candidate-pool target bug

- Status: `negative-boundary`
- Why it matters:
  - invalidates older closing-side evidence that flowed through the generic
    right-hand candidate-pool helper
- Files changed:
  - [build_weak_slice_split_donor_prototype.py](/opt/tiger/hand/tools/build_weak_slice_split_donor_prototype.py)
  - [build_pairguided_reranker_multislice.py](/opt/tiger/hand/tools/build_pairguided_reranker_multislice.py)
  - downstream callers in the interaction-realized stack
- Key current conclusion:
  - the old generic right-hand candidate-pool path inherited
    `TASK_TARGET="opening"` from the weak-slice prototype
  - it also used `curr_attrs[task_field]` instead of the true task target
  - therefore pre-fix closing-side `candidate_pool_for_task(...)` outputs are
    polluted and must not be used as current mainline evidence

### 11b4. Feasibility audit

- Status: `support`
- Why it matters:
  - separates structurally infeasible absent-opposite-hand frames from true
    two-hand editing failures
- Script:
  - [build_interaction_realized_feasibility_audit.py](/opt/tiger/hand/tools/build_interaction_realized_feasibility_audit.py)
- Outputs:
  - [interaction_realized_feasibility_audit.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasibility_audit.md)
  - [interaction_realized_feasibility_audit.json](/opt/tiger/hand/experiments/generated/interaction_realized_feasibility_audit.json)
- Key current conclusion:
  - feasible two-hand rate is only about `0.28` on both hard right-hand tasks
  - corrected `hgb_relax_both` reaches feasible-subset joint:
    - closing `0.6392`
    - opening `0.6497`
  - therefore the raw full-slice joint score materially understates the editor
    on the frames where the strict two-hand criterion is actually feasible

### 11b5. Feasible-subset dual-repair follow-up

- Status: `negative-boundary`
- Why it matters:
  - tests whether a donor-side right-target repair layer is the next gain after
    corrected relaxed support
- Script:
  - [build_interaction_realized_feasible_dual_repair_followup.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_dual_repair_followup.py)
- Outputs:
  - [interaction_realized_feasible_dual_repair_followup_closing.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_dual_repair_followup_closing.md)
  - [interaction_realized_feasible_dual_repair_followup_opening.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_dual_repair_followup_opening.md)
- Key current conclusion:
  - on the feasible two-hand subset, the corrected relaxed editor with
    right-mode `none` is already strongest
  - `edge_transition_snap` is sharply harmful
  - `finger_profile_snap` is neutral on closing and slightly harmful on opening
  - `full_donor_oracle` is strongly harmful
- Current implication:
  - the next gain should not be pursued through donor-side right-hand snap or
    oracle repair

### 11b6. Feasible-subset left-repair bundle

- Status: `mainline`
- Why it matters:
  - identifies the dominant true failure mode on the feasible two-hand subset
    and shows a second real gain layer there
- Scripts:
  - [build_interaction_realized_feasible_left_repair_followup.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_left_repair_followup.py)
  - [build_interaction_realized_feasible_left_repair_bundle.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_left_repair_bundle.py)
- Outputs:
  - [interaction_realized_feasible_left_repair_bundle.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_repair_bundle.md)
  - [interaction_realized_feasible_left_repair_bundle.json](/opt/tiger/hand/experiments/generated/interaction_realized_feasible_left_repair_bundle.json)
- Key current conclusion:
  - after fixing right support, the dominant feasible-subset failure is
    `right=1, left=0`
  - task-specific left repair lifts feasible joint further:
    - closing: `0.4747 -> 0.6582`
    - opening: `0.4841 -> 0.6497`
  - paired significance supports both gains
- Current use:
  - second-layer strengthening artifact for the corrected hard right-hand line
  - important boundary: best left repair is task-dependent rather than unified

### 11b7. Cheap subtype gate for feasible left repair

- Status: `negative-boundary`
- Why it matters:
  - tests whether the task-dependent feasible left-repair gain can be replaced
    by a very cheap unified selector
- Script:
  - [build_interaction_realized_feasible_left_repair_gate.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_left_repair_gate.py)
- Outputs:
  - [interaction_realized_feasible_left_repair_gate_closing.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_repair_gate_closing.md)
  - [interaction_realized_feasible_left_repair_gate_opening.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_repair_gate_opening.md)
- Key current conclusion:
  - a subtype-average gate over `(other_hand_motion, interaction_motion)` beats
    `no-repair`, but is clearly worse than the fixed task-best left repair
  - there is still oracle headroom, but it is not captured by this cheapest
    selector
- Current implication:
  - do not sell a simple subtype gate as the unifying mechanism
  - if unification is revisited, it needs richer context than the current
    subtype-average rule

### 11b8. Richer learned gate for feasible left repair

- Status: `negative-boundary`
- Why it matters:
  - tests whether a lightweight learned selector with mismatch features can
    unify feasible left repair better than the failed subtype-average gate
- Script:
  - [build_interaction_realized_feasible_left_repair_model_gate.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_left_repair_model_gate.py)
- Outputs:
  - [interaction_realized_feasible_left_repair_model_gate_closing.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_repair_model_gate_closing.md)
  - [interaction_realized_feasible_left_repair_model_gate_opening.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_repair_model_gate_opening.md)
- Key current conclusion:
  - even with no-repair state/transition-agreement features, the lightweight
    learned gate is far below the fixed task-best left repair
  - the train feasible subset is extremely label-imbalanced toward `none`
  - the remaining oracle headroom is therefore not captured by this class of
    small selector
- Current implication:
  - stop iterating on cheap / lightweight left-repair selectors
  - any future unification attempt needs richer supervision or context

### 11b9. Binary apply gate for feasible closing repair

- Status: `negative-boundary`
- Why it matters:
  - tests the narrowest plausible lightweight gate after multiclass selectors
    failed
- Script:
  - [build_interaction_realized_feasible_left_apply_gate.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_left_apply_gate.py)
- Outputs:
  - [interaction_realized_feasible_left_apply_gate.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_apply_gate.md)
- Key current conclusion:
  - even a binary apply-or-not gate for the already-best closing repair is far
    below fixed task-best
  - the train feasible subset has only `2` positive apply examples against `81`
    negatives
  - the lightweight apply gate therefore collapses toward “do not repair”
- Current implication:
  - lightweight left-repair gates are exhausted as a direction
  - any future unification needs fundamentally richer supervision or context

### 11b10. Dense KNN teacher for feasible left repair

- Status: `negative-boundary`
- Why it matters:
  - tests a materially stronger unification attempt after the lightweight gates
    failed
  - replaces sparse best-mode labels with dense continuous repair-gain targets
    and richer state-signature-aware context
- Script:
  - [build_interaction_realized_feasible_left_dense_knn.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_left_dense_knn.py)
- Outputs:
  - [interaction_realized_feasible_left_dense_knn_closing.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_dense_knn_closing.md)
  - [interaction_realized_feasible_left_dense_knn_opening.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_dense_knn_opening.md)
- Key current conclusion:
  - dense gain-based supervision is still not enough to unify left repair
  - on `opening`, dense KNN only ties the fixed task-best policy
  - on `closing`, dense KNN is clearly worse than fixed task-best:
    - `0.6582 -> 0.5696`
  - there remains substantial oracle headroom on both tasks
- Current implication:
  - the next remaining gap is not just “replace sparse labels with denser local
    gains”
  - if unification is revisited, it should likely use sequence-level
    supervision, explicit temporal context, or a different repair target
    representation
  - do not spend more time on local context-only framewise selectors

### 11b11. Template-conditioned feasible left-repair gate

- Status: `negative-boundary`
- Why it matters:
  - tests whether coarse temporal/template context can unify feasible left
    repair after local framewise selectors failed
- Script:
  - [build_interaction_realized_feasible_left_template_gate.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_left_template_gate.py)
- Outputs:
  - [interaction_realized_feasible_left_template_gate_closing.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_template_gate_closing.md)
  - [interaction_realized_feasible_left_template_gate_opening.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_template_gate_opening.md)
- Key current conclusion:
  - sequence-template and template-plus-subtype rules also fail badly on
    `val -> test`
  - none of them beats the fixed task-best policy on either hard task
  - they collapse heavily toward `none` on test, showing that feasible
    left-repair preferences drift across splits
- Current implication:
  - the unification gap is not solved by either:
    - local framewise context
    - or coarse template metadata
  - if revisited, the next attempt should use explicit cross-frame temporal
    evidence or a more stable repair target than the current framewise mode
    label

### 11b12. Temporal-window KNN for feasible left repair

- Status: `support`
- Why it matters:
  - first explicit cross-frame selector for feasible left repair
  - establishes whether one-hop temporal evidence is genuinely informative
- Script:
  - [build_interaction_realized_feasible_left_temporal_window_knn.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_left_temporal_window_knn.py)
- Outputs:
  - [interaction_realized_feasible_left_temporal_window_knn_closing.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_temporal_window_knn_closing.md)
  - [interaction_realized_feasible_left_temporal_window_knn_opening.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_temporal_window_knn_opening.md)
- Key current conclusion:
  - explicit one-hop temporal context gives a small positive gain on
    `right_hand_motion->opening`:
    - `0.6497 -> 0.6624`
    - but this lift is not statistically convincing yet
  - but fails badly on `right_hand_motion->closing`:
    - `0.6582 -> 0.4937`
- Current implication:
  - temporal evidence is directionally useful, at least for opening
  - but weak temporal context is not enough to unify both tasks
  - the next step should focus on stronger temporal mechanisms or task-specific
    temporal repair policies rather than more local/template selectors

### 11b13. Closing-specific temporal routing

- Status: `negative-boundary`
- Why it matters:
  - isolates the hardest remaining feasible-left case and tests whether routing
    from the already-best closing mode is easier than generic multiclass
    selection
- Script:
  - [build_interaction_realized_feasible_left_closing_temporal_routing.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_left_closing_temporal_routing.py)
- Outputs:
  - [interaction_realized_feasible_left_closing_temporal_routing.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_closing_temporal_routing.md)
- Key current conclusion:
  - both closing-specific one-hop routing families fail:
    - `edge -> none`: `0.6582 -> 0.4747`
    - `edge -> finger`: `0.6582 -> 0.6203`
  - train-time routing degenerates to single-mode collapse
- Current implication:
  - closing is not currently a simple routing problem from the fixed edge mode
  - the next credible closing direction needs a stronger temporal target or a
    different repair objective, not another thresholded local routing variant

### 11b14. Closing chunk-level KNN

- Status: `negative-boundary`
- Why it matters:
  - tests the strongest straightforward temporal reformulation for the hard
    closing case by changing the decision unit from frame to short run
- Script:
  - [build_interaction_realized_feasible_left_closing_chunk_knn.py](/opt/tiger/hand/tools/build_interaction_realized_feasible_left_closing_chunk_knn.py)
- Outputs:
  - [interaction_realized_feasible_left_closing_chunk_knn.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_feasible_left_closing_chunk_knn.md)
- Key current conclusion:
  - run-level KNN also collapses to a degenerate all-`none` policy
  - it is far below fixed edge:
    - `0.6582 -> 0.4747`
  - current feasible closing supervision is too weak to support direct
    train-to-test run-level mode transfer
- Current implication:
  - closing is not rescued by simply coarsening the temporal unit
  - the next credible closing step needs a different repair objective,
    stronger supervision, or a protocol that avoids direct low-shot mode
    transfer

### 11b15. External support expansion engineering boundary

- Status: `open-gap`
- Why it matters:
  - stronger supervision is the next most plausible closing direction after the
    direct selector families failed
  - before claiming anything scientific, we need to know whether the current
    implementation can even run the required support-expansion experiments
- Scripts:
  - [build_interaction_realized_closing_support_expansion.py](/opt/tiger/hand/tools/build_interaction_realized_closing_support_expansion.py)
  - [build_interaction_realized_closing_selector_support_expansion.py](/opt/tiger/hand/tools/build_interaction_realized_closing_selector_support_expansion.py)
  - [build_interaction_realized_closing_selector_support_scaling.py](/opt/tiger/hand/tools/build_interaction_realized_closing_selector_support_scaling.py)
- Key current conclusion:
  - external support expansion is currently blocked by computation, not yet by
    negative scientific evidence
  - the hot path is train-side support-row generation through
    `collect_rows -> evaluate_joint`
  - even `top-2` / `top-4` external support scaling did not finish in a useful
    turnaround
- Current implication:
  - the next stronger-supervision step requires computational restructuring
    first:
    - caching
    - candidate pruning
    - or lighter precomputed train-side support representations

### 11b16. Fast-path closing selector support scaling

- Status: `open-gap`
- Why it matters:
  - converts the external-support direction from “too slow to test” into a
    partially runnable regime
- Script:
  - [build_interaction_realized_closing_selector_support_scaling.py](/opt/tiger/hand/tools/build_interaction_realized_closing_selector_support_scaling.py)
- Key current conclusion:
  - with explicit candidate caps (`right=5`, `left=5`), the no-external-support
    closing selector becomes runnable and no longer collapses to all `none`
  - verified `top_0` baseline:
    - fixed edge `0.6013`
    - temporal-window KNN `0.6139`
    - delta `+0.0127`
- Current implication:
  - fast-path pruning is sufficient to make closing support scaling reachable
  - the next required step is row caching so external-support budgets can be
    swept without recomputing the same expensive support rows

### 11b17. Cached external support scaling on closing

- Status: `support`
- Why it matters:
  - turns the external-support direction into a reproducible cached regime
  - gives the first clean scientific read on whether a small amount of
    additional closing support helps
- Artifacts:
  - [interaction_realized_closing_selector_support_scaling_top0.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_closing_selector_support_scaling_top0.md)
  - [interaction_realized_closing_selector_support_scaling_top2.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_closing_selector_support_scaling_top2.md)
  - [external_rows_top2.json](/opt/tiger/hand/experiments/generated/cache/closing_selector_support_scaling/external_rows_top2.json)
- Key current conclusion:
  - on the refreshed fast-path closing slice (`56` frames), the selector beats
    fixed edge:
    - fixed edge `0.6607`
    - temporal-window KNN `0.7321`
  - but adding a small external support budget (`top_2`) does not further
    improve over the refreshed `top_0` baseline
- Current implication:
  - cached support scaling now works technically
  - small external support alone is not yet enough to move the fast-path
    closing result
  - the next useful step is either a larger cached budget or a changed support
    target, not repeated `top_2` reruns

### 11b18. Cached support plateau at top-4

- Status: `support`
- Why it matters:
  - extends the cached external-support result far enough to distinguish
    between “not enough support yet” and “support volume may not be the main
    issue”
- Artifacts:
  - [interaction_realized_closing_selector_support_scaling_top4.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_closing_selector_support_scaling_top4.md)
  - [external_rows_top4.json](/opt/tiger/hand/experiments/generated/cache/closing_selector_support_scaling/external_rows_top4.json)
- Key current conclusion:
  - `top_0`, `top_2`, and `top_4` all produce the same test-time closing
    result on the refreshed fast-path slice:
    - fixed edge `0.6607`
    - temporal-window KNN `0.7321`
  - larger support improves train-side leave-one-out fit, but not test-time
    performance
- Current implication:
  - the main remaining closing gap is no longer best described as a simple
    support-volume shortage
  - cached external support is now infrastructure for changed-target
    experiments rather than evidence that more rows alone will solve closing

### 11b19. Closing margin-target support experiment

- Status: `negative-boundary`
- Why it matters:
  - tests the next plausible explanation after support-volume plateau:
    maybe the support target is wrong rather than the support amount
- Artifacts:
  - [interaction_realized_closing_margin_target.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_closing_margin_target.md)
- Key current conclusion:
  - direct `finger-edge` margin prediction collapses to an all-`finger` policy
    on every tested support set
  - it is worse than fixed edge:
    - fixed edge `0.6607`
    - margin target `0.6250`
  - cached top4 support is almost entirely uninformative for this target
- Current implication:
  - the remaining closing gap is not fixed by a simple framewise target
    relabeling from class labels to edge-vs-finger margins
  - the next changed-target step should likely move beyond framewise support
    rows rather than keep relabeling the same row-level supervision

### 11b20. Closing run-level edge-vs-finger target

- Status: `support`
- Why it matters:
  - first changed-target result that moves supervision one level above the
    row-level closing variants
- Artifacts:
  - [interaction_realized_closing_run_target.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_closing_run_target.md)
- Key current conclusion:
  - run-level target is better than the failed row-level margin target and
    does beat fixed edge:
    - fixed edge `0.6607`
    - run target `0.6786`
  - but it still remains below the current best fast-path frame-level selector
    result:
    - frame-level selector `0.7321`
- Current implication:
  - changing supervision level is directionally useful
  - but singleton external cached rows are not enough to make run-level support
    outperform the current frame-level fast-path regime

### 11b+. Interaction realized closing run target with relaxed external runs

- Status: `negative-boundary`
- Why it matters:
  - directly tests the strongest remaining closing hypothesis from the previous
    run-target result: whether cached external support helps only after being
    grouped into true short runs instead of singleton rows
- Script:
  - [build_interaction_realized_closing_run_target.py](/opt/tiger/hand/tools/build_interaction_realized_closing_run_target.py)
- Artifacts:
  - [interaction_realized_closing_run_target.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_closing_run_target.md)
- Key current conclusion:
  - relaxed external support does not lift the run-level target above the prior
    singleton result and does not move the fast-path closing frontier:
    - cached top4 rows `62`
    - cached top4 relaxed runs `17` at gap `18`
    - fixed edge `0.6607`
    - relaxed-run target `0.6607`
    - frame-level selector `0.7321`
- Current implication:
  - the closing-side gap is not explained by the earlier singleton packaging
    artifact alone
  - under the current `edge-vs-finger` run objective, stronger external run
    structure actually collapses the selector to all `edge`
  - the next productive closing step should change the run/chunk objective or
    supervision target itself rather than only repackaging support into runs

### 11b++. Interaction realized closing tri-mode run target

- Status: `negative-boundary`
- Why it matters:
  - tests whether the remaining closing gap is caused by an underspecified
    binary run label space rather than by support volume alone
- Script:
  - [build_interaction_realized_closing_run_target.py](/opt/tiger/hand/tools/build_interaction_realized_closing_run_target.py)
- Artifacts:
  - [interaction_realized_closing_run_target.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_closing_run_target.md)
- Key current conclusion:
  - expanding the run-level action space from `edge/finger` to
    `none/edge/finger` does not improve the best closing result:
    - val-runs-only binary `0.6786`
    - val-runs-only tri-mode `0.6786`
  - with external support, tri-mode collapses hard toward `none` and becomes
    clearly worse:
    - plus top4 singletons `0.6250`
    - plus top4 relaxed runs `0.5000`
  - yet the tri-mode oracle is stronger than the binary oracle:
    - oracle edge/finger `0.7500`
    - oracle tri-mode `0.7679`
- Current implication:
  - the headroom is not the issue; the current run-distance / vote objective is
    failing to recover it
  - the next productive closing step should change the run-level scoring target
    itself, likely toward gain-weighted or preserve-aware supervision, not just
    support packaging or label-space expansion

### 11b+++. Interaction realized closing conservative gain-aware run targets

- Status: `negative-boundary`
- Why it matters:
  - directly tests whether the closing-side run selector mainly needs a
    confidence margin before switching away from fixed edge
- Script:
  - [build_interaction_realized_closing_run_target.py](/opt/tiger/hand/tools/build_interaction_realized_closing_run_target.py)
- Artifacts:
  - [interaction_realized_closing_run_target.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_closing_run_target.md)
- Key current conclusion:
  - searching explicit gain margins does not produce a new best closing
    selector:
    - binary gain-aware best = original binary target with `tau=0.0`
    - tri-mode gain-aware best = effective edge/finger-only behavior with
      `tau_none=0.0, tau_finger=0.0`
  - test-time results confirm that gain margins add no new lift:
    - val-runs-only gain-binary `0.6786`
    - plus top4 singletons gain-binary `0.6786`
    - plus top4 relaxed runs gain-binary `0.6607`
    - gain-trimode never beats fixed edge and usually collapses back to it
- Current implication:
  - margining can suppress the pathological `none` collapse, but only by
    reverting to the same edge-dominant solution already reached before
  - the next productive closing step should move beyond label tuning inside the
    current KNN run selector and instead change the scoring mechanism itself

### 11b++++. Interaction realized closing learned run-level scorers

- Status: `mainline`
- Why it matters:
  - this is the first closing-side mechanism that cleanly beats the current
    KNN run-selector frontier on the fast-path regime
- Script:
  - [build_interaction_realized_closing_run_learned_scorer.py](/opt/tiger/hand/tools/build_interaction_realized_closing_run_learned_scorer.py)
- Artifacts:
  - [interaction_realized_closing_run_learned_scorer.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_closing_run_learned_scorer.md)
- Key current conclusion:
  - learned binary run-level scoring lifts closing joint beyond the previous
    KNN run target:
    - KNN binary `0.6786`
    - learned binary classifier `0.7143`
    - learned binary regressor `0.7143`
  - this improvement is stable across all tested support packagings:
    - val-runs-only `0.7143`
    - plus top4 singletons `0.7143`
    - plus top4 relaxed runs `0.7143`
  - learned trimode scoring is better than the old KNN trimode collapse, but
    still remains below the learned binary scorer:
    - trimode classifier `0.6964 / 0.6964 / 0.6786`
    - trimode regressor `0.6786 / 0.6786 / 0.6786`
- Current implication:
  - the hand-crafted KNN distance / vote rule was a real bottleneck
  - learned run-level scoring should replace KNN as the current run-level
    mainline on the fast-path slice
  - but it still remains below the current best frame-level fast-path selector
    (`0.7321`), so it is not yet the overall best closing mechanism
  - the next productive closing step should build on learned run scoring
    rather than continue tuning manual run-distance objectives

### 11b+++++. Interaction realized closing learned chunk-level scorers

- Status: `mainline`
- Why it matters:
  - this is the first closing-side mechanism on the fast-path slice that beats
    both the learned run-level scorer and the old frame-level fast-path
    selector
- Script:
  - [build_interaction_realized_closing_chunk_learned_scorer.py](/opt/tiger/hand/tools/build_interaction_realized_closing_chunk_learned_scorer.py)
- Artifacts:
  - [interaction_realized_closing_chunk_learned_scorer.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_closing_chunk_learned_scorer.md)
- Key current conclusion:
  - short learned binary chunks lift closing to `0.7500`, stable across tested
    support packagings:
    - val chunks only `0.7500`
    - plus top4 singletons `0.7500`
    - plus top4 relaxed chunks `0.7500`
  - this improves over both:
    - learned binary run scorer `0.7143`
    - previous best frame-level fast-path selector `0.7321`
  - chunk-len `2` and `3` both reach the current binary oracle on the same
    slice (`0.7500`)
- Current implication:
  - the remaining closing gap after run-level learning was primarily
    within-run heterogeneity rather than support packaging
  - chunk-level learned scoring is now the current best closing fast-path
    mechanism
  - the next productive step should test whether this chunk-level gain can be
    extended beyond the current fast-path slice

### 11b++++++. Interaction realized broader closing chunk transfer (`val_only`)

- Status: `mainline`
- Why it matters:
  - provides the first direct evidence that the learned closing chunk scorer
    survives beyond the narrow fast-path slice
- Script:
  - [build_interaction_realized_closing_chunk_transfer.py](/opt/tiger/hand/tools/build_interaction_realized_closing_chunk_transfer.py)
- Artifacts:
  - [interaction_realized_closing_chunk_transfer_val_only.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_closing_chunk_transfer_val_only.md)
- Key current conclusion:
  - on the broader corrected feasible closing slice with `val_only` support,
    the chunk-level scorer remains strong and reaches the binary oracle:
    - fixed edge `0.7025`
    - chunk classifier `0.7785`
    - chunk regressor `0.7785`
    - binary oracle `0.7785`
  - this is stronger than the current fast-path chunk score:
    - fast-path chunk classifier `0.7500`
    - broader corrected feasible closing chunk classifier `0.7785`
- Current implication:
  - chunk-level temporal factorization is no longer just a fast-path-specific
    positive result
  - the remaining broader-transfer work (`train_plus_val`) is now a
    strengthening follow-up, not the first proof of generalization

### 11b+++++++. Interaction realized broader closing chunk transfer (`train_partial_plus_val`)

- Status: `support`
- Why it matters:
  - checks whether the broader corrected feasible closing gain is already
    stable once early train-only support rows are mixed in, before the full
    train-only cache finishes
- Script:
  - [build_interaction_realized_closing_chunk_transfer.py](/opt/tiger/hand/tools/build_interaction_realized_closing_chunk_transfer.py)
- Artifacts:
  - [interaction_realized_closing_chunk_transfer_train_partial_plus_val.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_closing_chunk_transfer_train_partial_plus_val.md)
- Key current conclusion:
  - with the current partial train-only cache mixed into val-only support, the
    broader corrected feasible closing result stays unchanged across multiple
    partial-cache snapshots:
    - first checked at `8` train units / `136` cached support rows
    - rechecked at `13` train units / `171` cached support rows
    - rechecked again at `21` train units / `233` cached support rows
    - rechecked again at `22` train units / `241` cached support rows
    - rechecked again at `30` train units / `301` cached support rows
    - rechecked again at `38` train units / `362` cached support rows
    - rechecked again at `42` train units / `391` cached support rows
    - rechecked again at `50` train units / `455` cached support rows
    - rechecked again at `54` train units / `486` cached support rows
    - rechecked again at `62` train units / `550` cached support rows
    - rechecked again at `70` train units / `614` cached support rows
    - rechecked again at `86` train units / `738` cached support rows
    - rechecked again at `94` train units / `801` cached support rows
    - rechecked again at `110` train units / `929` cached support rows
    - rechecked again at `118` train units / `993` cached support rows
    - rechecked again at `126` train units / `1057` cached support rows
    - rechecked again at `142` train units / `1184` cached support rows
    - rechecked again at `162` train units / `1344` cached support rows
    - fixed edge `0.7025`
    - chunk classifier `0.7785`
    - chunk regressor `0.7785`
    - binary oracle `0.7785`
- Current implication:
  - the broader positive chunk result does not appear fragile to the first
    tranche of extra train-only support
  - finishing the full train-only cache remains a strengthening step, but the
    main generalization claim no longer depends on waiting for it

### 11b++++++++. Interaction realized broader opening chunk transfer (`val_only`)

- Status: `support`
- Why it matters:
  - tests whether learned chunk-level temporal factorization is a broader
    motion-side mechanism or only a closing-side accident
- Script:
  - [build_interaction_realized_opening_chunk_transfer_val_only.py](/opt/tiger/hand/tools/build_interaction_realized_opening_chunk_transfer_val_only.py)
- Artifacts:
  - [interaction_realized_opening_chunk_transfer_val_only.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_opening_chunk_transfer_val_only.md)
- Key current conclusion:
  - on the broader corrected feasible opening slice with `val_only` support:
    - fixed edge `0.6306`
    - chunk classifier `0.7197`
    - chunk regressor `0.7197`
    - binary oracle `0.7197`
  - paired delta vs fixed edge is `+0.0892` with wins/losses/ties `14 / 0 / 143`
- Current implication:
  - chunk-level temporal factorization now has broader-slice evidence on both
    `right_hand_motion->closing` and `right_hand_motion->opening`
  - this materially strengthens the claim boundary around temporal chunking as
    a representation-side mechanism rather than a closing-only patch

### 11b+++++++++. Interaction realized broader closing chunk transfer length sweep (`val_only`)

- Status: `support`
- Why it matters:
  - checks whether the broader corrected feasible closing gain depends on one
    specific short chunk length
- Artifacts:
  - [interaction_realized_closing_chunk_transfer_length_sweep_val_only.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_closing_chunk_transfer_length_sweep_val_only.md)
- Key current conclusion:
  - with cached `val_only` support, `chunk_len` in `{2, 3, 4}` gives the same
    broader corrected feasible closing result:
    - fixed edge `0.7025`
    - chunk classifier `0.7785`
    - chunk regressor `0.7785`
    - binary oracle `0.7785`
    - paired delta vs fixed edge `+0.0759`
- Current implication:
  - the broader corrected feasible closing gain is not an artifact of a single
    short chunk length

### 11b++++++++++. Interaction realized broader closing chunk transfer length sweep (`train_partial_plus_val`)

- Status: `support`
- Why it matters:
  - checks whether chunk-length robustness survives once the support source is
    materially expanded beyond `val_only`
- Artifacts:
  - [interaction_realized_closing_chunk_transfer_length_sweep_train_partial_plus_val.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_closing_chunk_transfer_length_sweep_train_partial_plus_val.md)
- Key current conclusion:
  - with `50` train units / `455` cached support rows, `chunk_len` in
    `{2, 3, 4}` still gives the same broader corrected feasible closing
    result:
    - fixed edge `0.7025`
    - chunk classifier `0.7785`
    - chunk regressor `0.7785`
    - binary oracle `0.7785`
    - paired delta vs fixed edge `+0.0759`
- Current implication:
  - the broader corrected feasible closing gain is stable across chunk lengths
    even after materially expanding the support source

### 11b+++++++++++. Interaction realized broader closing support progression (`train_partial_plus_val`)

- Status: `support`
- Why it matters:
  - compresses many repeated partial-cache checks into one direct robustness
    artifact that shows the gain remains stable as support grows
- Artifacts:
  - [interaction_realized_closing_chunk_transfer_support_progression.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_closing_chunk_transfer_support_progression.md)
- Key current conclusion:
  - across the recorded growth path from `8 / 136` to `162 / 1344`
    (`train partial units / cached support rows`), the broader corrected
    feasible closing result is unchanged:
    - fixed edge `0.7025`
    - chunk classifier `0.7785`
    - chunk regressor `0.7785`
    - binary oracle `0.7785`
    - paired delta vs fixed edge `+0.0759`
- Current implication:
  - the remaining unfinished full-cache work is now mainly about closing the
    final report, not establishing whether the gain exists

### 11b++++++++++++. Interaction realized broader closing chunk transfer (`train_plus_val`, final)

- Status: `mainline`
- Why it matters:
  - closes the broader corrected feasible closing transfer line with the full
    cached `train_plus_val` support source instead of partial snapshots
- Script:
  - [build_interaction_realized_closing_chunk_transfer.py](/opt/tiger/hand/tools/build_interaction_realized_closing_chunk_transfer.py)
- Artifacts:
  - [interaction_realized_closing_chunk_transfer.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_closing_chunk_transfer.md)
- Key current conclusion:
  - on the broader corrected feasible closing slice with full `train_plus_val`
    support:
    - fixed edge `0.7025`
    - chunk classifier `0.7785`
    - chunk regressor `0.7785`
    - binary oracle `0.7785`
    - paired delta vs fixed edge `+0.0759`
  - support statistics:
    - support frames `1423`
    - support sequences `158`
    - pair bank size `4159`
    - support chunks `1394`
- Current implication:
  - the finalized full-support report lands on exactly the same number as the
    entire partial-support robustness progression
  - the broader corrected feasible closing chunk gain is now closed as a
    full-support mainline result rather than only a partial-cache strengthening story

### 11c+. Hard-slice residual subtype audit

- Status: `support`
- Why it matters:
  - converts the open hard-slice weakness from a vague low-score complaint into
    a concrete subtype distribution that can guide the next mechanism experiment
- Artifacts:
  - [interaction_realized_hard_slice_residual_subtypes.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_hard_slice_residual_subtypes.md)
- Key current conclusion:
  - under the current strongest hard-slice method
    `hgb_budget2_finger_profile_snap_top20`, the dominant zero-score mass is
    concentrated in `other_hand_motion=none` subtypes for both closing and opening
  - examples:
    - closing: `none/steady` `217` frames, `none/approach` `96` frames,
      `none/separate` `67` frames, all at `0.0000`
    - opening: `none/steady` `219` frames, `none/approach` `98` frames,
      `none/separate` `68` frames, all at `0.0000`
- Current implication:
  - this supports the current diagnosis that the absolute hard-slice score is
    still compressed mainly by structurally infeasible mass
  - the next high-value mechanism test should target the feasible
    interaction-rich residuals rather than treating the whole hard slice as
    homogeneous

### 11c. Interaction realized mechanism sweep

- Status: `negative-boundary`
- Why it matters:
  - directly tests new mechanism classes instead of only expanding search
    budget
- Script:
  - [build_interaction_realized_mechanism_sweep.py](/opt/tiger/hand/tools/build_interaction_realized_mechanism_sweep.py)
- Outputs:
  - [interaction_realized_mechanism_sweep_seed0.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_mechanism_sweep_seed0.md)
  - [interaction_realized_mechanism_sweep_seed1.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_mechanism_sweep_seed1.md)
  - [interaction_realized_mechanism_sweep_seed0.json](/opt/tiger/hand/experiments/generated/interaction_realized_mechanism_sweep_seed0.json)
  - [interaction_realized_mechanism_sweep_seed1.json](/opt/tiger/hand/experiments/generated/interaction_realized_mechanism_sweep_seed1.json)
- Key current conclusion:
  - preserve-hand `absolute` state transplant is decisively worse than `delta`
    composition on both hard right-hand interaction tasks
  - switching the inner pair objective from binary to surrogate changes
    surrogate quality only slightly
  - GPU-trained MLP pair-guided ordering is stable across two seeds but does
    not improve the absolute joint interaction score beyond the existing
    `0.0340 / 0.0320` frontier
- Current implication:
  - the next gain will not come from naive absolute preserve transplants
  - a stronger breakthrough likely needs a different representation-level or
    constrained realization mechanism

### 11d. Interaction realized constraint sweep

- Status: `mainline`
- Why it matters:
  - this is the first realization-stage mechanism that materially lifts the
    strict joint interaction score on both hard right-hand slices
- Script:
  - [build_interaction_realized_constraint_sweep.py](/opt/tiger/hand/tools/build_interaction_realized_constraint_sweep.py)
- Outputs:
  - [interaction_realized_constraint_sweep.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_constraint_sweep.md)
  - [interaction_realized_constraint_sweep.json](/opt/tiger/hand/experiments/generated/interaction_realized_constraint_sweep.json)
- Key current conclusion:
  - target-aware preserve-hand repair improves the realized interaction editor
    beyond the previous pair-guided frontier
  - strongest current variant is `hgb_finger_profile_snap_top10`
  - hard-slice joint score improves from:
    - closing: `0.0340 -> 0.0465`
    - opening: `0.0320 -> 0.0426`
  - the right-hand grouped score stays fixed while the gain comes from better
    preserve-hand realization
- Current implication:
  - the bottleneck was not only donor search; it was also the realized
    preserve-hand geometry after composition
  - constrained realization is now part of the surviving mainline

### 11e. Interaction realized constraint scaling

- Status: `mainline`
- Why it matters:
  - hardens the new constrained-realization gain by showing it remains useful
    under larger preserve-support budgets and deeper search
- Script:
  - [build_interaction_realized_constraint_scaling.py](/opt/tiger/hand/tools/build_interaction_realized_constraint_scaling.py)
- Outputs:
  - [interaction_realized_constraint_scaling.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_constraint_scaling.md)
  - [interaction_realized_constraint_scaling.json](/opt/tiger/hand/experiments/generated/interaction_realized_constraint_scaling.json)
- Key current conclusion:
  - the new repair gain is not a fragile top-10 artifact
  - with `budget2 + finger_profile_snap`, the strict hard-slice joint frontier
    rises further to:
    - closing: `0.0555`
    - opening: `0.0533`
  - this improves substantially over both:
    - the pre-repair pair-guided frontier: `0.0340 / 0.0320`
    - the first repair-only frontier at budget1: `0.0465 / 0.0426`
- Current implication:
  - the strongest current interaction editor is now:
    - expanded preserve support
    - plus constrained preserve-hand realization
  - interaction progress is now coming from the combination of search and
    realization, not either one alone

### 11f. Interaction realized significance report

- Status: `support`
- Why it matters:
  - converts the new hard-slice interaction frontier from point estimates into
    paired statistical evidence on the exact same frames
- Script:
  - [build_interaction_realized_significance_report.py](/opt/tiger/hand/tools/build_interaction_realized_significance_report.py)
- Outputs:
  - [interaction_realized_significance_report.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_significance_report.md)
  - [interaction_realized_significance_report.json](/opt/tiger/hand/experiments/generated/interaction_realized_significance_report.json)
- Key current conclusion:
  - `hgb_budget1_none_top10 -> hgb_budget2_finger_profile_snap_top10` is
    paired-positive on both hard slices with zero paired losses
  - closing:
    - delta `+0.0215`
    - bootstrap CI `[0.0107, 0.0340]`
    - permutation `p=0.00055`
  - opening:
    - delta `+0.0213`
    - bootstrap CI `[0.0107, 0.0337]`
    - permutation `p=0.00015`
- Current implication:
  - the new strongest interaction frontier is not just a nicer point estimate
  - the gain is statistically credible under paired hard-slice analysis

### 11g. Interaction realized sequence concentration

- Status: `negative-boundary`
- Why it matters:
  - checks whether the new hard-slice gain is broadly sequence-distributed or
    mainly concentrated in one recurring sequence family
- Script:
  - [build_interaction_realized_sequence_concentration.py](/opt/tiger/hand/tools/build_interaction_realized_sequence_concentration.py)
- Outputs:
  - [interaction_realized_sequence_concentration.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_sequence_concentration.md)
  - [interaction_realized_sequence_concentration.json](/opt/tiger/hand/experiments/generated/interaction_realized_sequence_concentration.json)
- Key current conclusion:
  - the current strongest hard-slice gain is not coming from a single frame
    anomaly, but it is still concentrated in one dominant sequence family
  - for `hgb_budget1_none_top10 -> hgb_budget2_finger_profile_snap_top10`:
    - most of the delta comes from `ROM01_No_Interaction_2_Hand`
    - `ROM09_Interaction_Fingers_Touching` contributes a smaller positive gain
    - `ROM02_Interaction_2_Hand` is essentially unchanged
- Current implication:
  - we can defend that the gain is real and paired-positive
  - we cannot yet claim that the interaction improvement is evenly distributed
    across all hard-sequence families

### 11h. Interaction realized right support sweep

- Status: `support`
- Why it matters:
  - directly tests whether the remaining weak sequence families are limited by
    an overly constrained right-hand donor pool
- Script:
  - [build_interaction_realized_right_support_sweep.py](/opt/tiger/hand/tools/build_interaction_realized_right_support_sweep.py)
- Outputs:
  - [interaction_realized_right_support_sweep.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_right_support_sweep.md)
  - [interaction_realized_right_support_sweep.json](/opt/tiger/hand/experiments/generated/interaction_realized_right_support_sweep.json)
- Key current conclusion:
  - on the weak sequence families only (`ROM02`, `ROM09`), relaxing the right
    donor pool materially improves strict joint success
  - `hgb_strict -> hgb_relax_both`:
    - closing subset: `0.0092 -> 0.0572`
    - opening subset: `0.0157 -> 0.0695`
  - this is driven by large right-grouped gains:
    - closing right grouped: `0.0137 -> 0.0892`
    - opening right grouped: `0.0247 -> 0.0987`
- Current implication:
  - the weak-family bottleneck is not only preserve-hand realization
  - right-target donor support is also too tight on these families
  - the next useful mechanism is a selective gate, not a global always-relax
    rule

## E. Negative Results That Must Stay Visible

### 12. Matched old-HL vs temporal-HL report

- Status: `negative-boundary`
- Why it matters:
  - prevents the project from drifting back to a weak classification story
- Script:
  - [build_oldhl_temporal_matched_report.py](/opt/tiger/hand/tools/build_oldhl_temporal_matched_report.py)
- Outputs:
  - [oldhl_temporal_matched_report.md](/opt/tiger/hand/experiments/generated/summary_tables/oldhl_temporal_matched_report.md)
  - [oldhl_temporal_matched_report.json](/opt/tiger/hand/experiments/generated/oldhl_temporal_matched_report.json)
- Key current conclusion:
  - matched strong protocol does **not** show temporal HL beating old HL
  - this is a useful constraint, not a result to hide

### 13. Channel-pruning / over-complete temporal mixture evidence

- Status: `negative-boundary`
- Why it matters:
  - shows more temporal channels are not automatically better
- Outputs:
  - [decoupled_temporal_event_report.md](/opt/tiger/hand/experiments/generated/summary_tables/decoupled_temporal_event_report.md)
  - [strong_protocol_pruned_channel_report.md](/opt/tiger/hand/experiments/generated/summary_tables/strong_protocol_pruned_channel_report.md)
  - [strong_protocol_weight_search_report.md](/opt/tiger/hand/experiments/generated/summary_tables/strong_protocol_weight_search_report.md)
- Key current conclusion:
  - explicit temporal structure matters
  - but additive feature stuffing is not the right story

## F. Retired Or Non-Mainline Mechanisms

### 14. Joint two-hand same-donor composition prototype

- Status: `retired`
- Script:
  - [build_weak_slice_joint_editor_prototype.py](/opt/tiger/hand/tools/build_weak_slice_joint_editor_prototype.py)
- Outputs:
  - [weak_slice_joint_editor_prototype.md](/opt/tiger/hand/experiments/generated/summary_tables/weak_slice_joint_editor_prototype.md)
- Why retired:
  - same donor for both hands does not solve the hard interaction slice

### 15. Split-donor candidate expansion prototype

- Status: `retired`
- Script:
  - [build_weak_slice_split_donor_prototype.py](/opt/tiger/hand/tools/build_weak_slice_split_donor_prototype.py)
- Outputs:
  - [weak_slice_split_donor_prototype.md](/opt/tiger/hand/experiments/generated/summary_tables/weak_slice_split_donor_prototype.md)
- Why retired:
  - candidate expansion helps, but is too weak by itself

### 16. Top-k joint search and support-relaxation studies

- Status: `retired`
- Scripts:
  - [build_weak_slice_topk_joint_search.py](/opt/tiger/hand/tools/build_weak_slice_topk_joint_search.py)
  - [build_weak_slice_relaxed_support_topk.py](/opt/tiger/hand/tools/build_weak_slice_relaxed_support_topk.py)
  - [build_weak_slice_relaxed_search_scaling.py](/opt/tiger/hand/tools/build_weak_slice_relaxed_search_scaling.py)
  - [build_weak_slice_adaptive_support_topk.py](/opt/tiger/hand/tools/build_weak_slice_adaptive_support_topk.py)
- Outputs:
  - [weak_slice_topk_joint_search.md](/opt/tiger/hand/experiments/generated/summary_tables/weak_slice_topk_joint_search.md)
  - [weak_slice_relaxed_support_topk.md](/opt/tiger/hand/experiments/generated/summary_tables/weak_slice_relaxed_support_topk.md)
  - [weak_slice_relaxed_search_scaling.md](/opt/tiger/hand/experiments/generated/summary_tables/weak_slice_relaxed_search_scaling.md)
  - [weak_slice_adaptive_support_topk.md](/opt/tiger/hand/experiments/generated/summary_tables/weak_slice_adaptive_support_topk.md)
- Why retired:
  - these establish the search-space shape
  - but the stronger current mechanism is pair-guided reranking

### 17. Coordination-channel attempts

- Status: `retired`
- Scripts / outputs:
  - [build_weak_slice_coordination_rerank.py](/opt/tiger/hand/tools/build_weak_slice_coordination_rerank.py)
  - [build_weak_slice_coordination_rerun.py](/opt/tiger/hand/tools/build_weak_slice_coordination_rerun.py)
  - [build_weak_slice_soft_coordination_rerun.py](/opt/tiger/hand/tools/build_weak_slice_soft_coordination_rerun.py)
  - [weak_slice_coordination_rerank.md](/opt/tiger/hand/experiments/generated/summary_tables/weak_slice_coordination_rerank.md)
  - [weak_slice_coordination_rerun.md](/opt/tiger/hand/experiments/generated/summary_tables/weak_slice_coordination_rerun.md)
  - [weak_slice_soft_coordination_rerun.md](/opt/tiger/hand/experiments/generated/summary_tables/weak_slice_soft_coordination_rerun.md)
- Why retired:
  - explicit minimal coordination metadata did not help enough
  - soft coordination penalties also did not help

## G. Related-Work And Positioning Assets

### 18. Deep related-work refresh

- Status: `support`
- File:
  - [related_work_deep_refresh_20260609.md](/opt/tiger/hand/experiments/related_work_deep_refresh_20260609.md)
- Why it matters:
  - fixes the external identity boundary:
    - not plain temporal hand modeling
    - not generic tokenizer work
    - not sign-language notation identity

## H. Current Open Gap

### 19. Interaction-aware realized editing

- Status: `open-gap`
- What is missing:
  - a stronger realized editor that materially improves grouped-motif fidelity
    on the hard right-hand interaction slices
- Best current evidence around the gap:
  - [transition_conditioned_symbolic_editor.md](/opt/tiger/hand/experiments/generated/summary_tables/transition_conditioned_symbolic_editor.md)
  - [pairguided_reranker_multislice.md](/opt/tiger/hand/experiments/generated/summary_tables/pairguided_reranker_multislice.md)
  - [hard_slice_compact_search_bundle.md](/opt/tiger/hand/experiments/generated/summary_tables/hard_slice_compact_search_bundle.md)
  - [interaction_realized_pairguided_editor.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_pairguided_editor.md)
  - [interaction_realized_constraint_sweep.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_constraint_sweep.md)
  - [interaction_realized_constraint_scaling.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_constraint_scaling.md)
  - [interaction_realized_sequence_concentration.md](/opt/tiger/hand/experiments/generated/summary_tables/interaction_realized_sequence_concentration.md)
- Immediate implication:
  - interaction-aware realized editing now has a genuine mechanism gain
  - but the absolute hard-slice joint score is still not yet high enough to
    count as closure

## Operating Rule

When a new experiment is added, update three places:

1. [symbolic_pretrain_notes.md](/opt/tiger/hand/experiments/symbolic_pretrain_notes.md)
2. this registry
3. one of the top-level bundles if the claim boundary changes

If an experiment is negative but changes what we are allowed to claim, it still
belongs here.
