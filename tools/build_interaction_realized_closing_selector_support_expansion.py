#!/usr/bin/env python3
"""Selector-only support expansion for hard feasible closing.

This is an experiment memo, not paper text.

Keep the current val-trained pair-guided editor fixed, but expand the selector
supervision source from feasible `val` rows to feasible `train + val` rows.
This isolates whether the remaining closing collapse is mainly due to tiny
selector supervision rather than the editor itself.
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
from tools.build_transition_conditioned_symbolic_editor import (
    GEN,
    SUM,
    build_pair_bank,
    build_semantic_frame_vocab,
    load_json,
)


TASK_TARGET = "closing"


def fmt(x: float) -> str:
    return f"{x:.4f}"


def all_sequence_labels(data):
    return {seq["seq_name"] for seq in data["sequences"]}


def subset_to_closing_sequences(data):
    keep = {row["seq_name"] for row in build_frames(data, all_sequence_labels(data), TASK_TARGET)}
    return {"sequences": [seq for seq in data["sequences"] if seq["seq_name"] in keep]}


def merge_data(*datasets):
    out = {"sequences": []}
    for dataset in datasets:
        out["sequences"].extend(dataset["sequences"])
    return out


def evaluate_selector_support(selector_train_rows, test_rows):
    best = choose_cfg(selector_train_rows)
    output_rows = []
    pred_modes = []
    for row in test_rows:
        pred_mode, _ = predict_mode(selector_train_rows, row, best["cfg"])
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
        output_rows.append(rec)
    return {
        "selector_stats": best,
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
    # Keep the pair-guided model itself on the current val-only protocol.
    val_labels = all_sequence_labels(val_subset)
    val_semantic_vocab = build_semantic_frame_vocab(val_subset, val_labels)
    val_pair_bank = build_pair_bank(val_subset, val_labels, val_semantic_vocab)
    val_train_frames = build_frames(val_subset, all_sequence_labels(val_subset), TASK_TARGET)
    pair_model, pair_stats = train_pairguided_model(val_train_frames, val_pair_bank, TASK_TARGET)

    # Generate val/test rows with the current protocol; train rows come from a
    # separate train-only bank and act only as extra selector supervision.
    train_labels = all_sequence_labels(train_subset)
    train_semantic_vocab = build_semantic_frame_vocab(train_subset, train_labels)
    train_pair_bank = build_pair_bank(train_subset, train_labels, train_semantic_vocab)

    val_selector_rows = attach_context(collect_rows(val_train_frames, val_pair_bank, TASK_TARGET, pair_model))
    train_selector_rows = attach_context(
        collect_rows(build_frames(train_subset, all_sequence_labels(train_subset), TASK_TARGET), train_pair_bank, TASK_TARGET, pair_model)
    )
    test_rows = attach_context(
        collect_rows(build_frames(test_subset, all_sequence_labels(test_subset), TASK_TARGET), val_pair_bank, TASK_TARGET, pair_model)
    )

    task_results = {
        "val_only_rows": evaluate_selector_support(val_selector_rows, test_rows),
        "train_plus_val_rows": evaluate_selector_support(train_selector_rows + val_selector_rows, test_rows),
    }
    payload = {
        "focus": {
            "goal": "selector-only support expansion for hard feasible closing",
            "task": "right_hand_motion->closing",
            "editor_training": "val_only",
            "selector_support_sources": ["val_only_rows", "train_plus_val_rows"],
        },
        "training_stats": {
            "pair_model": pair_stats,
            "num_val_selector_rows": len(val_selector_rows),
            "num_train_selector_rows": len(train_selector_rows),
            "val_pair_bank_size": len(val_pair_bank),
            "train_pair_bank_size": len(train_pair_bank),
        },
        "task_results": task_results,
    }

    out_json = GEN / "interaction_realized_closing_selector_support_expansion.json"
    out_md = SUM / "interaction_realized_closing_selector_support_expansion.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction Realized Closing Selector Support Expansion",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Keep the editor on the current val-trained protocol, but expand selector supervision from feasible `val` rows to feasible `train + val` rows.",
        "",
        f"Val selector rows: `{len(val_selector_rows)}`",
        f"Train selector rows: `{len(train_selector_rows)}`",
        f"Val pair bank size: `{len(val_pair_bank)}`",
        f"Train pair bank size: `{len(train_pair_bank)}`",
        "",
    ]
    for name, result in task_results.items():
        lines.extend(
            [
                f"## {name}",
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
        best = result["selector_stats"]
        lines.extend(
            [
                "",
                f"Selected config: `{best['cfg']}`, leave-one-out joint `{fmt(best['leave_one_out_joint_score'])}`",
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
