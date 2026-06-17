#!/usr/bin/env python3
"""Build the top-level strongest-evidence bundle for the project."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/opt/tiger/hand")
GEN = ROOT / "experiments/generated"
SUM = GEN / "summary_tables"


def load(name: str):
    return json.loads((GEN / name).read_text())


def fmt(x: float) -> str:
    return f"{x:.4f}"


def main():
    frontier = load("current_code_symbolic_frontier.json")
    risk = load("representation_risk_bundle.json")
    hard = load("hard_slice_compact_search_bundle.json")
    oldhl = load("oldhl_temporal_matched_report.json")
    interaction = load("interaction_vs_noninteraction_summary.json")
    global_support = load("interaction_realized_global_right_support_bundle.json")
    feasibility = load("interaction_realized_feasibility_audit.json")
    feasible_left = load("interaction_realized_feasible_left_repair_bundle.json")
    learned_run = load("interaction_realized_closing_run_learned_scorer.json")
    learned_chunk = load("interaction_realized_closing_chunk_learned_scorer.json")
    broader_chunk_val = load("interaction_realized_closing_chunk_transfer_val_only.json")
    broader_chunk_full = load("interaction_realized_closing_chunk_transfer.json")
    broader_chunk_len = load("interaction_realized_closing_chunk_transfer_length_sweep_val_only.json")
    broader_chunk_len_train_partial = load("interaction_realized_closing_chunk_transfer_length_sweep_train_partial_plus_val.json")
    broader_chunk_progression = load("interaction_realized_closing_chunk_transfer_support_progression.json")
    broader_opening_chunk_val = load("interaction_realized_opening_chunk_transfer_val_only.json")
    broader_opening_chunk_len = load("interaction_realized_opening_chunk_transfer_length_sweep_val_only.json")
    feasible_interaction_rich = load("interaction_realized_feasible_interaction_rich_residuals.json")
    feasible_left_routing = load("interaction_realized_feasible_left_routing_closure.json")
    feasible_left_run = load("interaction_realized_feasible_left_run_learned_scorer.json")

    grouped_row = next(row for row in frontier["rows"] if abs(float(row["fraction"]) - 1.0) < 1e-9)
    structure_block = {
        "flat_seq_1.0": grouped_row["flat_seq"],
        "grouped_seq_1.0": grouped_row["grouped_seq"],
        "family_seq_1.0": grouped_row["family_seq"],
        "grouped_minus_flat": grouped_row["delta_grouped_minus_flat"],
        "family_minus_grouped": grouped_row["delta_family_minus_grouped"],
        "family_harmed": grouped_row["family_harmed"],
    }

    control_block = {
        "interaction_approach_symbolic_clean_semantic": risk["control_advantage"]["local_editability"]["semantic_frame"]["interaction_motion->approach"]["symbolic_clean"],
        "interaction_approach_proxy_clean_semantic": risk["control_advantage"]["local_editability"]["semantic_frame"]["interaction_motion->approach"]["proxy_clean"],
        "interaction_separate_symbolic_clean_semantic": risk["control_advantage"]["local_editability"]["semantic_frame"]["interaction_motion->separate"]["symbolic_clean"],
        "interaction_separate_proxy_clean_semantic": risk["control_advantage"]["local_editability"]["semantic_frame"]["interaction_motion->separate"]["proxy_clean"],
        "right_opening_symbolic_clean_semantic": risk["control_advantage"]["local_editability"]["semantic_frame"]["right_hand_motion->opening"]["symbolic_clean"],
        "right_opening_proxy_clean_semantic": risk["control_advantage"]["local_editability"]["semantic_frame"]["right_hand_motion->opening"]["proxy_clean"],
    }

    transition_block = {
        "right_opening_interaction_beats_proxy_semantic": risk["transition_conditioned_slices"]["sources"]["semantic_frame"]["right_hand_motion->opening"]["interaction"]["symbolic_beats_proxy_rate"],
        "right_closing_interaction_beats_proxy_semantic": risk["transition_conditioned_slices"]["sources"]["semantic_frame"]["right_hand_motion->closing"]["interaction"]["symbolic_beats_proxy_rate"],
        "left_opening_interaction_beats_proxy_semantic": risk["transition_conditioned_slices"]["sources"]["semantic_frame"]["left_hand_motion->opening"]["interaction"]["symbolic_beats_proxy_rate"],
        "left_closing_interaction_beats_proxy_semantic": risk["transition_conditioned_slices"]["sources"]["semantic_frame"]["left_hand_motion->closing"]["interaction"]["symbolic_beats_proxy_rate"],
        "right_opening_occlusion_beats_proxy_semantic": risk["transition_conditioned_slices"]["sources"]["semantic_frame"]["right_hand_motion->opening"]["occlusion"]["symbolic_beats_proxy_rate"],
        "right_opening_finger_occlusion_beats_proxy_semantic": risk["transition_conditioned_slices"]["sources"]["semantic_frame"]["right_hand_motion->opening"]["finger_occlusion"]["symbolic_beats_proxy_rate"],
    }

    compact_search_block = {
        "right_opening_baseline_single_donor": hard["weak_opening_progression"]["baseline_single_donor_joint_overall"],
        "right_opening_relaxed_top20": hard["weak_opening_progression"]["relaxed_top20_joint_overall"],
        "right_opening_learned_top10": hard["weak_opening_progression"]["learned_top10_joint_overall"],
        "right_opening_pairguided_top5": hard["weak_opening_progression"]["pairguided_top5_joint_overall"],
        "right_opening_pairguided_top10": hard["weak_opening_progression"]["pairguided_top10_joint_overall"],
    }

    corrected_support_block = {
        "closing_strict_joint": global_support["task_results"]["right_hand_motion->closing"]["summary"]["hgb_strict"]["joint_score_overall"],
        "closing_relax_joint": global_support["task_results"]["right_hand_motion->closing"]["summary"]["hgb_relax_both"]["joint_score_overall"],
        "closing_delta": global_support["task_results"]["right_hand_motion->closing"]["paired_significance"]["joint_score"]["delta"],
        "closing_ci_low": global_support["task_results"]["right_hand_motion->closing"]["paired_significance"]["joint_score"]["bootstrap_ci_low"],
        "closing_ci_high": global_support["task_results"]["right_hand_motion->closing"]["paired_significance"]["joint_score"]["bootstrap_ci_high"],
        "closing_pvalue": global_support["task_results"]["right_hand_motion->closing"]["paired_significance"]["joint_score"]["permutation_pvalue"],
        "opening_strict_joint": global_support["task_results"]["right_hand_motion->opening"]["summary"]["hgb_strict"]["joint_score_overall"],
        "opening_relax_joint": global_support["task_results"]["right_hand_motion->opening"]["summary"]["hgb_relax_both"]["joint_score_overall"],
        "opening_delta": global_support["task_results"]["right_hand_motion->opening"]["paired_significance"]["joint_score"]["delta"],
        "opening_ci_low": global_support["task_results"]["right_hand_motion->opening"]["paired_significance"]["joint_score"]["bootstrap_ci_low"],
        "opening_ci_high": global_support["task_results"]["right_hand_motion->opening"]["paired_significance"]["joint_score"]["bootstrap_ci_high"],
        "opening_pvalue": global_support["task_results"]["right_hand_motion->opening"]["paired_significance"]["joint_score"]["permutation_pvalue"],
    }
    feasibility_block = {
        "closing_feasible_rate": feasibility["task_results"]["right_hand_motion->closing"]["summary"]["hgb_relax_both"]["feasible_rate"],
        "closing_feasible_joint": feasibility["task_results"]["right_hand_motion->closing"]["summary"]["hgb_relax_both"]["feasible_joint_overall"],
        "opening_feasible_rate": feasibility["task_results"]["right_hand_motion->opening"]["summary"]["hgb_relax_both"]["feasible_rate"],
        "opening_feasible_joint": feasibility["task_results"]["right_hand_motion->opening"]["summary"]["hgb_relax_both"]["feasible_joint_overall"],
    }
    feasible_left_block = {
        "closing_best_mode": feasible_left["task_results"]["right_hand_motion->closing"]["best_mode"],
        "closing_best_joint": feasible_left["task_results"]["right_hand_motion->closing"]["summary"][feasible_left["task_results"]["right_hand_motion->closing"]["best_mode"]]["joint_score_overall"],
        "closing_delta": feasible_left["task_results"]["right_hand_motion->closing"]["paired_significance"]["best_vs_none_joint_score"]["delta"],
        "closing_ci_low": feasible_left["task_results"]["right_hand_motion->closing"]["paired_significance"]["best_vs_none_joint_score"]["bootstrap_ci_low"],
        "closing_ci_high": feasible_left["task_results"]["right_hand_motion->closing"]["paired_significance"]["best_vs_none_joint_score"]["bootstrap_ci_high"],
        "closing_pvalue": feasible_left["task_results"]["right_hand_motion->closing"]["paired_significance"]["best_vs_none_joint_score"]["permutation_pvalue"],
        "opening_best_mode": feasible_left["task_results"]["right_hand_motion->opening"]["best_mode"],
        "opening_best_joint": feasible_left["task_results"]["right_hand_motion->opening"]["summary"][feasible_left["task_results"]["right_hand_motion->opening"]["best_mode"]]["joint_score_overall"],
        "opening_delta": feasible_left["task_results"]["right_hand_motion->opening"]["paired_significance"]["best_vs_none_joint_score"]["delta"],
        "opening_ci_low": feasible_left["task_results"]["right_hand_motion->opening"]["paired_significance"]["best_vs_none_joint_score"]["bootstrap_ci_low"],
        "opening_ci_high": feasible_left["task_results"]["right_hand_motion->opening"]["paired_significance"]["best_vs_none_joint_score"]["bootstrap_ci_high"],
        "opening_pvalue": feasible_left["task_results"]["right_hand_motion->opening"]["paired_significance"]["best_vs_none_joint_score"]["permutation_pvalue"],
    }
    learned_run_block = {
        "val_runs_only_knn_binary": learned_run["task_results"]["val_runs_only"]["summary"]["knn_binary"]["joint_score_overall"],
        "val_runs_only_learned_binary_cls": learned_run["task_results"]["val_runs_only"]["summary"]["learned_binary_cls"]["joint_score_overall"],
        "val_runs_only_learned_binary_reg": learned_run["task_results"]["val_runs_only"]["summary"]["learned_binary_reg"]["joint_score_overall"],
        "top4_singletons_knn_binary": learned_run["task_results"]["val_runs_plus_top4_singletons"]["summary"]["knn_binary"]["joint_score_overall"],
        "top4_singletons_learned_binary_cls": learned_run["task_results"]["val_runs_plus_top4_singletons"]["summary"]["learned_binary_cls"]["joint_score_overall"],
        "top4_singletons_learned_binary_reg": learned_run["task_results"]["val_runs_plus_top4_singletons"]["summary"]["learned_binary_reg"]["joint_score_overall"],
        "top4_relaxed_knn_binary": learned_run["task_results"]["val_runs_plus_top4_relaxed_runs"]["summary"]["knn_binary"]["joint_score_overall"],
        "top4_relaxed_learned_binary_cls": learned_run["task_results"]["val_runs_plus_top4_relaxed_runs"]["summary"]["learned_binary_cls"]["joint_score_overall"],
        "top4_relaxed_learned_binary_reg": learned_run["task_results"]["val_runs_plus_top4_relaxed_runs"]["summary"]["learned_binary_reg"]["joint_score_overall"],
        "oracle_trimode": learned_run["task_results"]["val_runs_plus_top4_relaxed_runs"]["summary"]["oracle_trimode"]["joint_score_overall"],
    }
    learned_chunk_block = {
        "chunk2_val_only_cls": learned_chunk["chunk_results"]["chunk_len_2"]["val_chunks_only"]["summary"]["chunk_cls"]["joint_score_overall"],
        "chunk2_top4_singletons_cls": learned_chunk["chunk_results"]["chunk_len_2"]["val_chunks_plus_top4_singletons"]["summary"]["chunk_cls"]["joint_score_overall"],
        "chunk2_top4_relaxed_cls": learned_chunk["chunk_results"]["chunk_len_2"]["val_chunks_plus_top4_relaxed_chunks"]["summary"]["chunk_cls"]["joint_score_overall"],
        "chunk3_val_only_cls": learned_chunk["chunk_results"]["chunk_len_3"]["val_chunks_only"]["summary"]["chunk_cls"]["joint_score_overall"],
        "chunk3_top4_singletons_cls": learned_chunk["chunk_results"]["chunk_len_3"]["val_chunks_plus_top4_singletons"]["summary"]["chunk_cls"]["joint_score_overall"],
        "chunk3_top4_relaxed_cls": learned_chunk["chunk_results"]["chunk_len_3"]["val_chunks_plus_top4_relaxed_chunks"]["summary"]["chunk_cls"]["joint_score_overall"],
        "oracle_binary": learned_chunk["chunk_results"]["chunk_len_3"]["val_chunks_plus_top4_relaxed_chunks"]["summary"]["oracle_binary"]["joint_score_overall"],
    }
    broader_chunk_block = {
        "fixed_edge": broader_chunk_val["result"]["summary"]["fixed_edge"]["joint_score_overall"],
        "chunk_cls": broader_chunk_val["result"]["summary"]["chunk_cls"]["joint_score_overall"],
        "chunk_reg": broader_chunk_val["result"]["summary"]["chunk_reg"]["joint_score_overall"],
        "oracle_binary": broader_chunk_val["result"]["summary"]["oracle_binary"]["joint_score_overall"],
    }
    broader_chunk_full_block = {
        "fixed_edge": broader_chunk_full["task_results"]["train_plus_val"]["chunk_lengths"]["chunk_len_2"]["summary"]["fixed_edge"]["joint_score_overall"],
        "chunk_cls": broader_chunk_full["task_results"]["train_plus_val"]["chunk_lengths"]["chunk_len_2"]["summary"]["chunk_cls"]["joint_score_overall"],
        "chunk_reg": broader_chunk_full["task_results"]["train_plus_val"]["chunk_lengths"]["chunk_len_2"]["summary"]["chunk_reg"]["joint_score_overall"],
        "oracle_binary": broader_chunk_full["task_results"]["train_plus_val"]["chunk_lengths"]["chunk_len_2"]["summary"]["oracle_binary"]["joint_score_overall"],
    }
    broader_opening_chunk_block = {
        "fixed_edge": broader_opening_chunk_val["result"]["summary"]["fixed_edge"]["joint_score_overall"],
        "chunk_cls": broader_opening_chunk_val["result"]["summary"]["chunk_cls"]["joint_score_overall"],
        "chunk_reg": broader_opening_chunk_val["result"]["summary"]["chunk_reg"]["joint_score_overall"],
        "oracle_binary": broader_opening_chunk_val["result"]["summary"]["oracle_binary"]["joint_score_overall"],
    }
    broader_opening_chunk_len_block = {
        chunk_name: result["summary"]["chunk_cls"]["joint_score_overall"]
        for chunk_name, result in broader_opening_chunk_len["results"].items()
    }
    feasible_interaction_rich_block = {
        "closing_all": feasible_interaction_rich["task_results"]["right_hand_motion->closing"]["all_subtypes"]["best_joint"],
        "closing_rich_only": feasible_interaction_rich["task_results"]["right_hand_motion->closing"]["feasible_interaction_rich_only"]["best_joint"],
        "closing_rich_delta_vs_baseline": feasible_interaction_rich["task_results"]["right_hand_motion->closing"]["deltas"]["best_minus_baseline_rich"],
        "opening_all": feasible_interaction_rich["task_results"]["right_hand_motion->opening"]["all_subtypes"]["best_joint"],
        "opening_rich_only": feasible_interaction_rich["task_results"]["right_hand_motion->opening"]["feasible_interaction_rich_only"]["best_joint"],
        "opening_rich_delta_vs_baseline": feasible_interaction_rich["task_results"]["right_hand_motion->opening"]["deltas"]["best_minus_baseline_rich"],
    }
    feasible_left_routing_block = {
        "closing_fixed_task_best": feasible_left_routing["task_results"]["right_hand_motion->closing"]["metrics"]["fixed_task_best"],
        "closing_best_lightweight_alt": max(
            value
            for key, value in feasible_left_routing["task_results"]["right_hand_motion->closing"]["metrics"].items()
            if key not in {"fixed_task_best", "oracle_framewise"}
        ),
        "closing_gain_regressor": feasible_left_routing["task_results"]["right_hand_motion->closing"]["metrics"]["gain_regressor"],
        "closing_sparse_override": feasible_left_routing["task_results"]["right_hand_motion->closing"]["metrics"]["sparse_override"],
        "closing_oracle": feasible_left_routing["task_results"]["right_hand_motion->closing"]["metrics"]["oracle_framewise"],
        "opening_fixed_task_best": feasible_left_routing["task_results"]["right_hand_motion->opening"]["metrics"]["fixed_task_best"],
        "opening_temporal_window_knn": feasible_left_routing["task_results"]["right_hand_motion->opening"]["metrics"]["temporal_window_knn"],
        "opening_gain_regressor": feasible_left_routing["task_results"]["right_hand_motion->opening"]["metrics"]["gain_regressor"],
        "opening_sparse_override": feasible_left_routing["task_results"]["right_hand_motion->opening"]["metrics"]["sparse_override"],
        "opening_chunk_knn": feasible_left_routing["task_results"]["right_hand_motion->opening"]["metrics"]["chunk_knn"],
        "opening_oracle": feasible_left_routing["task_results"]["right_hand_motion->opening"]["metrics"]["oracle_framewise"],
    }
    feasible_left_run_block = {
        "closing_fixed_task_best": feasible_left_run["task_results"]["right_hand_motion->closing"]["summary"]["fixed_task_best"]["joint_score_overall"],
        "closing_run_cls": feasible_left_run["task_results"]["right_hand_motion->closing"]["summary"]["run_cls"]["joint_score_overall"],
        "closing_run_reg": feasible_left_run["task_results"]["right_hand_motion->closing"]["summary"]["run_reg"]["joint_score_overall"],
        "closing_run_oracle": feasible_left_run["task_results"]["right_hand_motion->closing"]["summary"]["run_oracle"]["joint_score_overall"],
        "closing_frame_oracle": feasible_left_run["task_results"]["right_hand_motion->closing"]["summary"]["frame_oracle"]["joint_score_overall"],
        "opening_fixed_task_best": feasible_left_run["task_results"]["right_hand_motion->opening"]["summary"]["fixed_task_best"]["joint_score_overall"],
        "opening_run_cls": feasible_left_run["task_results"]["right_hand_motion->opening"]["summary"]["run_cls"]["joint_score_overall"],
        "opening_run_reg": feasible_left_run["task_results"]["right_hand_motion->opening"]["summary"]["run_reg"]["joint_score_overall"],
        "opening_run_oracle": feasible_left_run["task_results"]["right_hand_motion->opening"]["summary"]["run_oracle"]["joint_score_overall"],
        "opening_frame_oracle": feasible_left_run["task_results"]["right_hand_motion->opening"]["summary"]["frame_oracle"]["joint_score_overall"],
    }
    broader_closing_chunk_len_block = {
        chunk_name: result["chunk_cls"]
        for chunk_name, result in broader_chunk_len["results"].items()
    }
    broader_closing_chunk_len_train_partial_block = {
        chunk_name: result["chunk_cls"]
        for chunk_name, result in broader_chunk_len_train_partial["results"].items()
    }
    broader_closing_progression_block = {
        "num_snapshots": len(broader_chunk_progression["snapshots"]),
        "first_snapshot": broader_chunk_progression["snapshots"][0],
        "last_snapshot": broader_chunk_progression["snapshots"][-1],
    }

    hard_right_rows = hard["hard_right_hand_slices"]
    left_ref_rows = hard["left_hand_reference_slices"]
    oldhl_block = {
        "state_scratch": oldhl["summary"]["state"]["scratch_mean_sequence_accuracy"],
        "state_pretrained": oldhl["summary"]["state"]["pretrained_mean_sequence_accuracy"],
        "temporal_scratch": oldhl["summary"]["temporal"]["scratch_mean_sequence_accuracy"],
        "temporal_pretrained": oldhl["summary"]["temporal"]["pretrained_mean_sequence_accuracy"],
        "scratch_temporal_minus_state": oldhl["summary"]["delta"]["scratch_temporal_minus_state"],
        "pretrained_temporal_minus_state": oldhl["summary"]["delta"]["pretrained_temporal_minus_state"],
    }
    interaction_rollups = interaction["editor_rollups"]

    decision = {
        "keep_mainline": [
            "sequence-native symbolic structure",
            "explicit temporal channels inside the representation rather than only in the encoder",
            "anatomy-aware grouped factorization",
            "zero-harm family-level repair",
            "local controllability and editability",
            "transition-conditioned motif quality",
            "hard-slice compact search via pair-guided reranking",
            "corrected right-support relaxation is now a dominant gain on the hard right-hand slices",
            "learned run-level closing scorers beat the old KNN run selector inside the fast-path regime",
            "learned chunk-level closing scorers now beat both learned run scorers and the old frame-level fast-path selector",
            "chunk-level closing gains already survive on a broader corrected feasible closing slice with val-only support",
            "that broader corrected feasible closing gain remains identical under the finalized full train_plus_val support report",
            "that broader corrected feasible closing gain is also stable across chunk lengths 2-4 with val-only support",
            "the same broader corrected feasible closing gain remains stable across chunk lengths 2-4 after materially expanding support with train_partial_plus_val",
            "that broader corrected feasible closing gain also remains numerically unchanged across the recorded support-growth path from 8/136 to 162/1344",
            "chunk-level opening gains also survive on a broader corrected feasible opening slice with val-only support",
            "that broader corrected feasible opening gain is also stable across chunk lengths 2-4 with val-only support",
        ],
        "claim_boundary": [
            "do not sell raw retrieval/classification as the main symbolic wedge",
            "do not argue that temporal HL wins because the same strong encoder classifies it better",
            "do not claim universal gains on all interaction slices",
            "do not cite pre-2026-06-10 closing-side candidate-pool numbers as current evidence; they were polluted by a task-target bug",
            "do not claim interaction-aware realized editing is solved globally; even the corrected right-support win still leaves many zero-score subtypes",
            "do not claim uniform hard-sequence coverage beyond the corrected right-hand full-slice evidence",
            "do not claim right-target donor snap is the next gain layer; targeted feasible-subset follow-up shows it is neutral or harmful",
            "do not collapse the feasible-subset left-repair gain into a single universal repair rule yet; the best left repair is task-dependent",
            "do not claim a cheap subtype-average gate solves left-repair selection; it loses to the fixed task-best policy on both tasks",
            "do not claim a lightweight learned gate solves left-repair selection either; with current supervision it collapses toward `none` and remains far below task-best",
            "do not claim a binary apply gate solves closing repair either; the positive apply signal is too sparse under current feasible-subset supervision",
            "do not claim dense gain-based KNN unifies left-repair selection either; it only ties task-best on opening and degrades clearly on closing",
            "do not claim sequence/template-conditioned gates solve left-repair selection either; they still collapse toward `none` and remain below task-best on both hard tasks",
            "do not claim weak one-hop temporal context solves left-repair selection either; it helps opening slightly but still fails badly on closing",
            "do not claim dense gain regression solves feasible-left selector learning either; it remains clearly below fixed task-best on both closing and opening",
            "do not claim oracle-audit-guided sparse subtype override solves the feasible-left gap either; it degrades closing and does not move opening",
            "do not claim a softer preserve-side local agreement objective solves the feasible-left gap either; it raises transition agreement but still lowers final joint score on both tasks",
            "do not claim closing-specific one-hop routing solves the remaining closing gap either; both edge->none and edge->finger routes collapse to degenerate single-mode behavior and lose to fixed edge",
            "do not claim closing run-level chunk selection solves the remaining closing gap either; chunk KNN also collapses to all `none` under current supervision and loses badly to fixed edge",
            "do not claim opening run-level chunk selection solves the remaining opening gap either; opening chunk KNN also loses clearly to the fixed finger-profile policy",
            "do not claim simple run-level learned scorers solve feasible-left repair either; `run_reg` helps opening modestly but both learned run policies remain below fixed task-best on closing",
            "do not claim small external support automatically improves closing either; in the cached fast-path regime, `top_2` external support changes the selector fit but does not improve test joint over `top_0`",
            "do not claim larger external support volume solves closing either; in the cached fast-path regime, `top_0`, `top_2`, and `top_4` have identical test-time closing performance even though train-side fit keeps improving",
            "do not claim a simple changed support target solves closing either; direct framewise `edge-vs-finger` margin prediction collapses to all `finger` and underperforms fixed edge",
            "do not claim changing supervision level alone solves closing either; run-level edge-vs-finger targets beat fixed edge but still remain below the current best fast-path frame-level selector",
            "do not claim learned run-level closing scorers solve the full fast-path closing problem either; they beat KNN run selectors but still remain below the best frame-level selector",
            "do not claim chunk-level closing gains have fully saturated all broader support protocols beyond the currently validated broader corrected feasible slice family",
        ],
        "remaining_gap": [
            "the right-hand hard-slice editor is materially stronger after correcting the target-pool bug and relaxing support, but the overall absolute joint score is still compressed by a large structurally infeasible mass",
            "the new feasible interaction-rich residual audit now quantifies that compression directly: the current strongest hard-slice method reaches about 3.5x higher joint score on the true feasible residual than on the mixed full slice",
            "on the feasible two-hand subset, task-specific left repair lifts the editor further, so the next step should target remaining true feasible failures rather than absent-opposite-hand frames",
            "matched old-HL vs temporal-HL already rules out the simple classification story, so the mainline must stay on control/editability/search",
            "the next step should move beyond direct supervised mode selection if we try to solve feasible closing repair; explicit one-hop temporal context, closing-specific routing, and run-level chunk KNN all fail under current supervision, so a different closing-side repair target or stronger supervision is more plausible",
            "the preserve-side closure is now stronger: even at the correct run-level granularity, a simple learned scorer still fails to recover the closing gap robustly",
            "cached fast-path support scaling is now operational, but a small external support budget does not yet improve closing beyond the refreshed fast-path baseline, so larger budgets or changed support targets are the next stronger tests",
            "cached external support can now be scaled and reused, but the current `top_0 -> top_2 -> top_4` plateau suggests the next productive closing step is to change the support target or supervision signal rather than just add more support rows",
            "a simple framewise changed target is now also ruled out in the cached regime, so the next productive closing step likely needs a different supervision level or objective, not just a relabeling of row-level edge/finger outcomes",
            "chunk-level learned scorers already transfer to a broader corrected feasible closing slice with val-only support, so the next productive closing step is to finish the stronger train-plus-val support transfer and then push beyond the corrected feasible slice",
        ],
    }

    payload = {
        "focus": {
            "bundle": "top-level strongest-evidence bundle",
            "purpose": "single internal entry point for the current strongest thesis-supporting experiments",
        },
        "structure_and_repair": structure_block,
        "control": control_block,
        "temporal_transition": transition_block,
        "oldhl_temporal_matched": oldhl_block,
        "interaction_vs_noninteraction_rollups": interaction_rollups,
        "hard_slice_compact_search": compact_search_block,
        "corrected_global_right_support": corrected_support_block,
        "feasibility_audit": feasibility_block,
        "feasible_left_repair": feasible_left_block,
        "closing_run_level_learned_scorer": learned_run_block,
        "closing_chunk_level_learned_scorer": learned_chunk_block,
        "broader_closing_chunk_transfer_val_only": broader_chunk_block,
        "broader_closing_chunk_transfer_train_plus_val_final": broader_chunk_full_block,
        "broader_closing_chunk_transfer_length_sweep_val_only": broader_closing_chunk_len_block,
        "broader_closing_chunk_transfer_length_sweep_train_partial_plus_val": broader_closing_chunk_len_train_partial_block,
        "broader_closing_chunk_transfer_support_progression": broader_closing_progression_block,
        "broader_opening_chunk_transfer_val_only": broader_opening_chunk_block,
        "broader_opening_chunk_transfer_length_sweep_val_only": broader_opening_chunk_len_block,
        "feasible_interaction_rich_residuals": feasible_interaction_rich_block,
        "feasible_left_routing_closure": feasible_left_routing_block,
        "feasible_left_run_level_learned_scorer": feasible_left_run_block,
        "hard_right_hand_slices": hard_right_rows,
        "left_hand_reference_slices": left_ref_rows,
        "decision": decision,
        "source_artifacts": {
            "current_code_symbolic_frontier": str(GEN / "current_code_symbolic_frontier.json"),
            "representation_risk_bundle": str(GEN / "representation_risk_bundle.json"),
            "hard_slice_compact_search_bundle": str(GEN / "hard_slice_compact_search_bundle.json"),
            "interaction_realized_global_right_support_bundle": str(GEN / "interaction_realized_global_right_support_bundle.json"),
            "interaction_realized_feasibility_audit": str(GEN / "interaction_realized_feasibility_audit.json"),
            "interaction_realized_feasible_left_repair_bundle": str(GEN / "interaction_realized_feasible_left_repair_bundle.json"),
            "interaction_realized_closing_run_learned_scorer": str(GEN / "interaction_realized_closing_run_learned_scorer.json"),
            "interaction_realized_closing_chunk_learned_scorer": str(GEN / "interaction_realized_closing_chunk_learned_scorer.json"),
            "interaction_realized_closing_chunk_transfer_val_only": str(GEN / "interaction_realized_closing_chunk_transfer_val_only.json"),
            "interaction_realized_opening_chunk_transfer_length_sweep_val_only": str(GEN / "interaction_realized_opening_chunk_transfer_length_sweep_val_only.json"),
            "interaction_realized_feasible_interaction_rich_residuals": str(GEN / "interaction_realized_feasible_interaction_rich_residuals.json"),
            "interaction_realized_feasible_left_routing_closure": str(GEN / "interaction_realized_feasible_left_routing_closure.json"),
            "interaction_realized_feasible_left_run_learned_scorer": str(GEN / "interaction_realized_feasible_left_run_learned_scorer.json"),
            "oldhl_temporal_matched_report": str(GEN / "oldhl_temporal_matched_report.json"),
            "interaction_vs_noninteraction_summary": str(GEN / "interaction_vs_noninteraction_summary.json"),
        },
    }

    out_json = GEN / "topline_evidence_bundle.json"
    out_md = SUM / "topline_evidence_bundle.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Topline Evidence Bundle",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "A single entry point for the current strongest internal evidence.",
        "",
        "## Structure And Repair",
        "",
        "| axis | value |",
        "| --- | ---: |",
        f"| flat seq @1.0 | {fmt(structure_block['flat_seq_1.0'])} |",
        f"| grouped seq @1.0 | {fmt(structure_block['grouped_seq_1.0'])} |",
        f"| family seq @1.0 | {fmt(structure_block['family_seq_1.0'])} |",
        f"| grouped-flat | {fmt(structure_block['grouped_minus_flat'])} |",
        f"| family-grouped | {fmt(structure_block['family_minus_grouped'])} |",
        f"| family harmed | {structure_block['family_harmed']} |",
        "",
        "## Control",
        "",
        "| task | symbolic clean | proxy clean |",
        "| --- | ---: | ---: |",
        f"| interaction->approach | {fmt(control_block['interaction_approach_symbolic_clean_semantic'])} | {fmt(control_block['interaction_approach_proxy_clean_semantic'])} |",
        f"| interaction->separate | {fmt(control_block['interaction_separate_symbolic_clean_semantic'])} | {fmt(control_block['interaction_separate_proxy_clean_semantic'])} |",
        f"| right opening | {fmt(control_block['right_opening_symbolic_clean_semantic'])} | {fmt(control_block['right_opening_proxy_clean_semantic'])} |",
        "",
        "## Temporal Transition",
        "",
        "| slice | semantic beats proxy |",
        "| --- | ---: |",
        f"| left closing interaction | {fmt(transition_block['left_closing_interaction_beats_proxy_semantic'])} |",
        f"| left opening interaction | {fmt(transition_block['left_opening_interaction_beats_proxy_semantic'])} |",
        f"| right closing interaction | {fmt(transition_block['right_closing_interaction_beats_proxy_semantic'])} |",
        f"| right opening interaction | {fmt(transition_block['right_opening_interaction_beats_proxy_semantic'])} |",
        f"| right opening occlusion | {fmt(transition_block['right_opening_occlusion_beats_proxy_semantic'])} |",
        f"| right opening finger occlusion | {fmt(transition_block['right_opening_finger_occlusion_beats_proxy_semantic'])} |",
        "",
        "## Matched Old-HL vs Temporal-HL",
        "",
        "| model | scratch seq | pretrained seq |",
        "| --- | ---: | ---: |",
        f"| old HL (`state`) | {fmt(oldhl_block['state_scratch'])} | {fmt(oldhl_block['state_pretrained'])} |",
        f"| temporal HL (`temporal`) | {fmt(oldhl_block['temporal_scratch'])} | {fmt(oldhl_block['temporal_pretrained'])} |",
        "",
        "| delta | value |",
        "| --- | ---: |",
        f"| scratch temporal - state | {fmt(oldhl_block['scratch_temporal_minus_state'])} |",
        f"| pretrained temporal - state | {fmt(oldhl_block['pretrained_temporal_minus_state'])} |",
        "",
        "## Interaction vs Noninteraction",
        "",
        "| source | symbolic interaction | proxy interaction | delta | symbolic noninteraction | proxy noninteraction | delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in interaction_rollups:
        lines.append(
            f"| {row['source']} | {fmt(row['weighted_symbolic_interaction'])} | {fmt(row['weighted_proxy_interaction'])} | {fmt(row['weighted_interaction_delta'])} | "
            f"{fmt(row['weighted_symbolic_noninteraction'])} | {fmt(row['weighted_proxy_noninteraction'])} | {fmt(row['weighted_noninteraction_delta'])} |"
        )

    lines.extend(
        [
            "",
            "## Hard-Slice Compact Search",
            "",
            "| stage | overall joint hit |",
            "| --- | ---: |",
            f"| baseline single donor | {fmt(compact_search_block['right_opening_baseline_single_donor'])} |",
            f"| relaxed top-20 | {fmt(compact_search_block['right_opening_relaxed_top20'])} |",
            f"| learned top-10 | {fmt(compact_search_block['right_opening_learned_top10'])} |",
            f"| pair-guided top-5 | {fmt(compact_search_block['right_opening_pairguided_top5'])} |",
            f"| pair-guided top-10 | {fmt(compact_search_block['right_opening_pairguided_top10'])} |",
            "",
            "## Corrected Global Right Support",
            "",
            "| task | strict | relax-both |",
            "| --- | ---: | ---: |",
            f"| right closing | {fmt(corrected_support_block['closing_strict_joint'])} | {fmt(corrected_support_block['closing_relax_joint'])} |",
            f"| right opening | {fmt(corrected_support_block['opening_strict_joint'])} | {fmt(corrected_support_block['opening_relax_joint'])} |",
            "",
            "## Corrected Significance",
            "",
            "| task | delta | CI low | CI high | p-value |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| right closing | {fmt(corrected_support_block['closing_delta'])} | {fmt(corrected_support_block['closing_ci_low'])} | {fmt(corrected_support_block['closing_ci_high'])} | {corrected_support_block['closing_pvalue']:.6f} |",
            f"| right opening | {fmt(corrected_support_block['opening_delta'])} | {fmt(corrected_support_block['opening_ci_low'])} | {fmt(corrected_support_block['opening_ci_high'])} | {corrected_support_block['opening_pvalue']:.6f} |",
            "",
            "## Feasibility Audit",
            "",
            "| task | feasible rate | feasible joint (`relax_both`) |",
            "| --- | ---: | ---: |",
            f"| right closing | {fmt(feasibility_block['closing_feasible_rate'])} | {fmt(feasibility_block['closing_feasible_joint'])} |",
            f"| right opening | {fmt(feasibility_block['opening_feasible_rate'])} | {fmt(feasibility_block['opening_feasible_joint'])} |",
            "",
            "## Feasible Left Repair",
            "",
            "| task | best mode | best feasible joint | delta vs no left repair | p-value |",
            "| --- | --- | ---: | ---: | ---: |",
            f"| right closing | {feasible_left_block['closing_best_mode']} | {fmt(feasible_left_block['closing_best_joint'])} | {fmt(feasible_left_block['closing_delta'])} | {feasible_left_block['closing_pvalue']:.6f} |",
            f"| right opening | {feasible_left_block['opening_best_mode']} | {fmt(feasible_left_block['opening_best_joint'])} | {fmt(feasible_left_block['opening_delta'])} | {feasible_left_block['opening_pvalue']:.6f} |",
            "",
            "## Closing Run-Level Learned Scorer",
            "",
            "| support | knn binary | learned binary cls | learned binary reg |",
            "| --- | ---: | ---: | ---: |",
            f"| val runs only | {fmt(learned_run_block['val_runs_only_knn_binary'])} | {fmt(learned_run_block['val_runs_only_learned_binary_cls'])} | {fmt(learned_run_block['val_runs_only_learned_binary_reg'])} |",
            f"| plus top4 singletons | {fmt(learned_run_block['top4_singletons_knn_binary'])} | {fmt(learned_run_block['top4_singletons_learned_binary_cls'])} | {fmt(learned_run_block['top4_singletons_learned_binary_reg'])} |",
            f"| plus top4 relaxed runs | {fmt(learned_run_block['top4_relaxed_knn_binary'])} | {fmt(learned_run_block['top4_relaxed_learned_binary_cls'])} | {fmt(learned_run_block['top4_relaxed_learned_binary_reg'])} |",
            "",
            "## Closing Chunk-Level Learned Scorer",
            "",
            "| support | chunk-len-2 cls | chunk-len-3 cls | oracle binary |",
            "| --- | ---: | ---: | ---: |",
            f"| val only | {fmt(learned_chunk_block['chunk2_val_only_cls'])} | {fmt(learned_chunk_block['chunk3_val_only_cls'])} | {fmt(learned_chunk_block['oracle_binary'])} |",
            f"| plus top4 singletons | {fmt(learned_chunk_block['chunk2_top4_singletons_cls'])} | {fmt(learned_chunk_block['chunk3_top4_singletons_cls'])} | {fmt(learned_chunk_block['oracle_binary'])} |",
            f"| plus top4 relaxed chunks | {fmt(learned_chunk_block['chunk2_top4_relaxed_cls'])} | {fmt(learned_chunk_block['chunk3_top4_relaxed_cls'])} | {fmt(learned_chunk_block['oracle_binary'])} |",
            "",
        "## Broader Closing Chunk Transfer",
        "",
        "| support | fixed edge | chunk cls | chunk reg | oracle binary |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| val-only broader corrected feasible closing | {fmt(broader_chunk_block['fixed_edge'])} | {fmt(broader_chunk_block['chunk_cls'])} | {fmt(broader_chunk_block['chunk_reg'])} | {fmt(broader_chunk_block['oracle_binary'])} |",
        "",
        "## Broader Opening Chunk Transfer",
        "",
        "| support | fixed edge | chunk cls | chunk reg | oracle binary |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| val-only broader corrected feasible opening | {fmt(broader_opening_chunk_block['fixed_edge'])} | {fmt(broader_opening_chunk_block['chunk_cls'])} | {fmt(broader_opening_chunk_block['chunk_reg'])} | {fmt(broader_opening_chunk_block['oracle_binary'])} |",
        "",
        "## Broader Opening Chunk Transfer Length Sweep",
        "",
        "| chunk len | chunk cls |",
        "| --- | ---: |",
        f"| chunk_len_2 | {fmt(broader_opening_chunk_len_block['chunk_len_2'])} |",
        f"| chunk_len_3 | {fmt(broader_opening_chunk_len_block['chunk_len_3'])} |",
        f"| chunk_len_4 | {fmt(broader_opening_chunk_len_block['chunk_len_4'])} |",
        "",
        "## Feasible Interaction-Rich Residuals",
        "",
        "| task | all-subtype best | feasible-rich best | feasible-rich delta vs baseline |",
        "| --- | ---: | ---: | ---: |",
        f"| right closing | {fmt(feasible_interaction_rich_block['closing_all'])} | {fmt(feasible_interaction_rich_block['closing_rich_only'])} | {fmt(feasible_interaction_rich_block['closing_rich_delta_vs_baseline'])} |",
        f"| right opening | {fmt(feasible_interaction_rich_block['opening_all'])} | {fmt(feasible_interaction_rich_block['opening_rich_only'])} | {fmt(feasible_interaction_rich_block['opening_rich_delta_vs_baseline'])} |",
        "",
        "## Feasible Left Routing Closure",
        "",
        "| task | fixed task-best | selected lightweight point | oracle |",
        "| --- | ---: | ---: | ---: |",
        f"| right closing | {fmt(feasible_left_routing_block['closing_fixed_task_best'])} | {fmt(feasible_left_routing_block['closing_best_lightweight_alt'])} | {fmt(feasible_left_routing_block['closing_oracle'])} |",
        f"| right opening | {fmt(feasible_left_routing_block['opening_fixed_task_best'])} | {fmt(feasible_left_routing_block['opening_temporal_window_knn'])} | {fmt(feasible_left_routing_block['opening_oracle'])} |",
        "",
        f"Gain regressor also stays below fixed task-best: closing `{fmt(feasible_left_routing_block['closing_gain_regressor'])}` vs `{fmt(feasible_left_routing_block['closing_fixed_task_best'])}`, opening `{fmt(feasible_left_routing_block['opening_gain_regressor'])}` vs `{fmt(feasible_left_routing_block['opening_fixed_task_best'])}`.",
        "",
        f"Sparse subtype override also fails: closing `{fmt(feasible_left_routing_block['closing_sparse_override'])}` vs `{fmt(feasible_left_routing_block['closing_fixed_task_best'])}`, opening `{fmt(feasible_left_routing_block['opening_sparse_override'])}` vs `{fmt(feasible_left_routing_block['opening_fixed_task_best'])}`.",
        "",
        f"Opening chunk KNN stays below fixed task-best: `{fmt(feasible_left_routing_block['opening_chunk_knn'])}` vs `{fmt(feasible_left_routing_block['opening_fixed_task_best'])}`.",
        "",
        "## Feasible Left Run-Level Learned Scorer",
        "",
        "| task | fixed task-best | run cls | run reg | run oracle | frame oracle |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| right closing | {fmt(feasible_left_run_block['closing_fixed_task_best'])} | {fmt(feasible_left_run_block['closing_run_cls'])} | {fmt(feasible_left_run_block['closing_run_reg'])} | {fmt(feasible_left_run_block['closing_run_oracle'])} | {fmt(feasible_left_run_block['closing_frame_oracle'])} |",
        f"| right opening | {fmt(feasible_left_run_block['opening_fixed_task_best'])} | {fmt(feasible_left_run_block['opening_run_cls'])} | {fmt(feasible_left_run_block['opening_run_reg'])} | {fmt(feasible_left_run_block['opening_run_oracle'])} | {fmt(feasible_left_run_block['opening_frame_oracle'])} |",
        "",
        "This closes the simple run-summary path:",
        "",
        "- run-level oracle headroom is real on both tasks",
        "- but learned `run_cls` collapses badly on closing",
        "- learned `run_reg` gives only a modest opening gain and still remains below",
        "  fixed task-best on closing",
        "- therefore run-level learned scoring is a boundary artifact, not a mainline",
        "  preserve-side mechanism",
        "",
        "## Hard Right-Hand Slices",
            "",
            "| task | base top-5 | base top-10 | base top-20 | pair-guided top-5 | pair-guided top-10 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in hard_right_rows:
        lines.append(
            f"| {row['task']} | {fmt(row['base_top5'])} | {fmt(row['base_top10'])} | {fmt(row['base_top20'])} | "
            f"{fmt(row['pairguided_top5'])} | {fmt(row['pairguided_top10'])} |"
        )

    lines.extend(["", "## Claim Boundary", "", "- Keep mainline:"])
    for item in decision["keep_mainline"]:
        lines.append(f"- {item}")
    lines.extend(["", "- Limit claims:"])
    for item in decision["claim_boundary"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "- Matched negative result to keep explicit:",
            "- temporal HL does not beat old HL under the matched strong sequence-classification protocol",
            "- therefore temporal value has to be defended through transition-conditioned control, editability, and hard-slice compact search",
            "",
            "- Remaining gaps:",
        ]
    )
    for item in decision["remaining_gap"]:
        lines.append(f"- {item}")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
