#!/usr/bin/env python3
"""Template-conditioned gate on feasible interaction-rich residuals only.

This is an experiment memo, not paper text.
"""

from __future__ import annotations

import json

from tools.build_interaction_realized_feasible_left_template_gate import (
    LEFT_REPAIR_MODES,
    TASK_BEST,
    apply_rule,
    best_mode_for_items,
    build_frames,
    enrich_rows,
    fit_rule,
    fmt,
    paired_stats,
    row_key_seq,
    row_key_seq_subtype,
    row_key_subtype,
    summarize,
)
from tools.build_interaction_realized_feasible_left_repair_model_gate import collect_rows
from tools.build_interaction_realized_pairguided_editor import TASKS, train_pairguided_model
from tools.build_transition_conditioned_symbolic_editor import (
    GEN,
    SUM,
    build_pair_bank,
    build_semantic_frame_vocab,
    load_json,
    overlap_labels,
)


def rich_only(rows):
    return [row for row in rows if row["other_hand_motion"] != "none"]


def run_task(train_data, test_data, labels, pair_bank, task_target: str):
    train_frames = build_frames(train_data, labels, task_target)
    test_frames = build_frames(test_data, labels, task_target)
    pair_model, pair_stats = train_pairguided_model(train_frames, pair_bank, task_target)

    train_rows = rich_only(collect_rows(train_frames, pair_bank, task_target, pair_model))
    test_rows = rich_only(collect_rows(test_frames, pair_bank, task_target, pair_model))

    seq_map, seq_fit_stats = fit_rule(train_rows, row_key_seq)
    subtype_map, subtype_fit_stats = fit_rule(train_rows, row_key_subtype)
    seq_subtype_map, seq_subtype_fit_stats = fit_rule(train_rows, row_key_seq_subtype)

    fixed_mode = TASK_BEST[task_target]
    chosen_modes_by_method = {}
    chosen_modes_by_method["seq_gate"], seq_counts = apply_rule(test_rows, seq_map, row_key_seq, fixed_mode)
    chosen_modes_by_method["subtype_gate"], subtype_counts = apply_rule(test_rows, subtype_map, row_key_subtype, fixed_mode)
    chosen_modes_by_method["seq_subtype_gate"], seq_subtype_counts = apply_rule(
        test_rows, seq_subtype_map, row_key_seq_subtype, fixed_mode
    )

    rows = enrich_rows(test_rows, chosen_modes_by_method, fixed_mode)
    summary = {
        name: summarize(rows, name)
        for name in ("fixed_none", "fixed_task_best", "seq_gate", "subtype_gate", "seq_subtype_gate", "oracle")
    }
    paired = {
        "seq_gate_vs_fixed_task_best": paired_stats(rows, "fixed_task_best", "seq_gate"),
        "subtype_gate_vs_fixed_task_best": paired_stats(rows, "fixed_task_best", "subtype_gate"),
        "seq_subtype_gate_vs_fixed_task_best": paired_stats(rows, "fixed_task_best", "seq_subtype_gate"),
        "oracle_vs_fixed_task_best": paired_stats(rows, "fixed_task_best", "oracle"),
    }
    return {
        "training_stats": {
            "num_train_rows": len(train_rows),
            "num_test_rows": len(test_rows),
            "pair_model": pair_stats,
            "seq_gate": {"mapping": {str(k): v for k, v in seq_map.items()}, "fit_stats": seq_fit_stats},
            "subtype_gate": {"mapping": {str(k): v for k, v in subtype_map.items()}, "fit_stats": subtype_fit_stats},
            "seq_subtype_gate": {
                "mapping": {str(k): v for k, v in seq_subtype_map.items()},
                "fit_stats": seq_subtype_fit_stats,
            },
        },
        "test_pred_mode_counts": {
            "seq_gate": seq_counts,
            "subtype_gate": subtype_counts,
            "seq_subtype_gate": seq_subtype_counts,
        },
        "summary": summary,
        "paired": paired,
        "rows": rows,
    }


def main():
    train_data = load_json(GEN / "temporal_hl_val.json")
    test_data = load_json(GEN / "temporal_hl_test.json")
    labels = overlap_labels(train_data, test_data)
    semantic_vocab = build_semantic_frame_vocab(train_data, labels)
    pair_bank = build_pair_bank(train_data, labels, semantic_vocab)

    task_results = {}
    for _, task_target in TASKS:
        task_results[f"right_hand_motion->{task_target}"] = run_task(
            train_data,
            test_data,
            labels,
            pair_bank,
            task_target,
        )

    payload = {
        "artifacts": {"train_json": str(GEN / "temporal_hl_val.json"), "test_json": str(GEN / "temporal_hl_test.json")},
        "focus": {
            "goal": "template-conditioned selection on feasible interaction-rich residuals",
            "tasks": [f"right_hand_motion->{target}" for _, target in TASKS],
            "repair_modes": list(LEFT_REPAIR_MODES),
            "filter": "other_hand_motion != none",
        },
        "task_results": task_results,
    }

    out_json = GEN / "interaction_realized_feasible_rich_template_gate.json"
    out_md = SUM / "interaction_realized_feasible_rich_template_gate.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction Realized Feasible-Rich Template Gate",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Template-conditioned left-repair selection on feasible interaction-rich residuals only (`other_hand_motion != none`).",
        "",
    ]
    for task_name, result in task_results.items():
        lines.extend(
            [
                f"## {task_name}",
                "",
                f"Train rows: `{result['training_stats']['num_train_rows']}`; test rows: `{result['training_stats']['num_test_rows']}`",
                "",
                "| method | right grouped | left preserve | joint overall |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for method in ("fixed_none", "fixed_task_best", "seq_gate", "subtype_gate", "seq_subtype_gate", "oracle"):
            stats = result["summary"][method]
            lines.append(
                f"| {method} | {fmt(stats['right_grouped_match_overall'])} | {fmt(stats['left_preserve_overall'])} | {fmt(stats['joint_score_overall'])} |"
            )
        lines.extend(
            [
                "",
                "| comparison | delta | wins | losses | ties |",
                "| --- | ---: | ---: | ---: | ---: |",
                f"| seq_gate vs fixed_task_best | {fmt(result['paired']['seq_gate_vs_fixed_task_best']['delta'])} | {result['paired']['seq_gate_vs_fixed_task_best']['wins']} | {result['paired']['seq_gate_vs_fixed_task_best']['losses']} | {result['paired']['seq_gate_vs_fixed_task_best']['ties']} |",
                f"| subtype_gate vs fixed_task_best | {fmt(result['paired']['subtype_gate_vs_fixed_task_best']['delta'])} | {result['paired']['subtype_gate_vs_fixed_task_best']['wins']} | {result['paired']['subtype_gate_vs_fixed_task_best']['losses']} | {result['paired']['subtype_gate_vs_fixed_task_best']['ties']} |",
                f"| seq_subtype_gate vs fixed_task_best | {fmt(result['paired']['seq_subtype_gate_vs_fixed_task_best']['delta'])} | {result['paired']['seq_subtype_gate_vs_fixed_task_best']['wins']} | {result['paired']['seq_subtype_gate_vs_fixed_task_best']['losses']} | {result['paired']['seq_subtype_gate_vs_fixed_task_best']['ties']} |",
                f"| oracle vs fixed_task_best | {fmt(result['paired']['oracle_vs_fixed_task_best']['delta'])} | {result['paired']['oracle_vs_fixed_task_best']['wins']} | {result['paired']['oracle_vs_fixed_task_best']['losses']} | {result['paired']['oracle_vs_fixed_task_best']['ties']} |",
                "",
                "### Test Mode Counts",
                "",
                f"- `seq_gate`: {result['test_pred_mode_counts']['seq_gate']}",
                f"- `subtype_gate`: {result['test_pred_mode_counts']['subtype_gate']}",
                f"- `seq_subtype_gate`: {result['test_pred_mode_counts']['seq_subtype_gate']}",
                "",
            ]
        )

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
