#!/usr/bin/env python3
"""Support expansion for the hard feasible closing task.

This is an experiment memo, not paper text.

Hypothesis:
- closing-side selector collapse may come from using only the tiny feasible
  `val` subset as support

Test:
- compare `val_only` vs `train_plus_val` support sources
- under the same closing protocol, measure:
  - fixed `edge_transition_snap`
  - temporal-window KNN selector
"""

from __future__ import annotations

import json
from collections import Counter

from tools.build_interaction_realized_feasible_left_temporal_window_knn import (
    attach_context,
    build_frames,
    choose_cfg,
    predict_mode,
    summarize,
    paired_stats,
)
from tools.build_interaction_realized_feasible_left_repair_model_gate import collect_rows
from tools.build_interaction_realized_pairguided_editor import train_pairguided_model
from tools.build_pairguided_reranker_multislice import collect_slice_frames
from tools.build_transition_conditioned_symbolic_editor import (
    GEN,
    SUM,
    build_pair_bank,
    build_semantic_frame_vocab,
    canonical,
    load_json,
)


TASK_TARGET = "closing"


def fmt(x: float) -> str:
    return f"{x:.4f}"


def all_sequence_labels(data):
    return {canonical(seq["seq_name"]) for seq in data["sequences"]} | {seq["seq_name"] for seq in data["sequences"]}


def subset_to_closing_sequences(data):
    rows = collect_slice_frames(data, "right_hand_motion", TASK_TARGET)
    keep = {row["seq_name"] for row in rows}
    return {"sequences": [seq for seq in data["sequences"] if seq["seq_name"] in keep]}


def merge_data(*datasets):
    out = {"sequences": []}
    for dataset in datasets:
        out["sequences"].extend(dataset["sequences"])
    return out


def evaluate_support(train_support_data, test_data, support_name: str):
    labels = all_sequence_labels(train_support_data)
    semantic_vocab = build_semantic_frame_vocab(train_support_data, labels)
    pair_bank = build_pair_bank(train_support_data, labels, semantic_vocab)

    train_frames = [
        row
        for row in collect_slice_frames(train_support_data, "right_hand_motion", TASK_TARGET)
        if row["curr_frame"].get("left") is not None
    ]
    test_frames = build_frames(test_data, all_sequence_labels(test_data), TASK_TARGET)

    pair_model, pair_stats = train_pairguided_model(train_frames, pair_bank, TASK_TARGET)
    train_rows = attach_context(collect_rows(train_frames, pair_bank, TASK_TARGET, pair_model))
    test_rows = attach_context(collect_rows(test_frames, pair_bank, TASK_TARGET, pair_model))
    best = choose_cfg(train_rows)

    output_rows = []
    pred_modes = []
    for row in test_rows:
        pred_mode, scores = predict_mode(train_rows, row, best["cfg"])
        pred_modes.append(pred_mode)
        rec = dict(row)
        oracle_mode = max(("none", "edge_transition_snap", "finger_profile_snap"), key=lambda mode: float(rec[f"left_{mode}_joint_score"]))
        for name, mode in [
            ("fixed_edge", "edge_transition_snap"),
            ("temporal_window_knn", pred_mode),
            ("oracle", oracle_mode),
        ]:
            prefix = f"left_{mode}"
            for field in ("right_grouped_match", "left_preserve", "joint_score"):
                rec[f"{name}_{field}"] = rec[f"{prefix}_{field}"]
        rec["pred_mode"] = pred_mode
        rec["oracle_mode"] = oracle_mode
        rec["support_name"] = support_name
        output_rows.append(rec)

    return {
        "training_stats": {
            "pair_model": pair_stats,
            "temporal_window_knn": best,
            "num_support_frames": len(train_frames),
            "num_support_sequences": len(train_support_data["sequences"]),
            "pair_bank_size": len(pair_bank),
        },
        "summary": {name: summarize(output_rows, name) for name in ("fixed_edge", "temporal_window_knn", "oracle")},
        "paired": {
            "temporal_window_knn_vs_fixed_edge": paired_stats(output_rows, "fixed_edge", "temporal_window_knn"),
            "oracle_vs_fixed_edge": paired_stats(output_rows, "fixed_edge", "oracle"),
        },
        "pred_mode_counts": dict(Counter(pred_modes)),
        "rows": output_rows,
    }


def main():
    train_data = load_json(GEN / "temporal_hl_train.json")
    val_data = load_json(GEN / "temporal_hl_val.json")
    test_data = load_json(GEN / "temporal_hl_test.json")

    train_subset = subset_to_closing_sequences(train_data)
    val_subset = subset_to_closing_sequences(val_data)
    test_subset = subset_to_closing_sequences(test_data)

    task_results = {
        "val_only": evaluate_support(val_subset, test_subset, "val_only"),
        "train_plus_val": evaluate_support(merge_data(train_subset, val_subset), test_subset, "train_plus_val"),
    }

    payload = {
        "focus": {
            "goal": "support expansion for hard feasible closing repair",
            "task": "right_hand_motion->closing",
            "support_sources": ["val_only", "train_plus_val"],
        },
        "task_results": task_results,
    }

    out_json = GEN / "interaction_realized_closing_support_expansion.json"
    out_md = SUM / "interaction_realized_closing_support_expansion.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction Realized Closing Support Expansion",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Compare `val_only` vs `train_plus_val` support for the hard feasible closing task.",
        "",
    ]
    for support_name, result in task_results.items():
        lines.extend(
            [
                f"## {support_name}",
                "",
                f"Support frames: `{result['training_stats']['num_support_frames']}`",
                f"Support sequences: `{result['training_stats']['num_support_sequences']}`",
                f"Pair bank size: `{result['training_stats']['pair_bank_size']}`",
                "",
                "| method | right grouped | left preserve | joint overall |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for method in ("fixed_edge", "temporal_window_knn", "oracle"):
            stats = result["summary"][method]
            lines.append(
                f"| {method} | {fmt(stats['right_grouped_match_overall'])} | {fmt(stats['left_preserve_overall'])} | {fmt(stats['joint_score_overall'])} |"
            )
        paired = result["paired"]
        best = result["training_stats"]["temporal_window_knn"]
        lines.extend(
            [
                "",
                f"Selected temporal-window config: `{best['cfg']}`, leave-one-out joint `{fmt(best['leave_one_out_joint_score'])}`",
                "",
                "| comparison | delta | wins | losses | ties |",
                "| --- | ---: | ---: | ---: | ---: |",
                f"| temporal_window_knn vs fixed_edge | {fmt(paired['temporal_window_knn_vs_fixed_edge']['delta'])} | {paired['temporal_window_knn_vs_fixed_edge']['wins']} | {paired['temporal_window_knn_vs_fixed_edge']['losses']} | {paired['temporal_window_knn_vs_fixed_edge']['ties']} |",
                f"| oracle vs fixed_edge | {fmt(paired['oracle_vs_fixed_edge']['delta'])} | {paired['oracle_vs_fixed_edge']['wins']} | {paired['oracle_vs_fixed_edge']['losses']} | {paired['oracle_vs_fixed_edge']['ties']} |",
                "",
                f"Pred mode counts: {result['pred_mode_counts']}",
                "",
            ]
        )

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
