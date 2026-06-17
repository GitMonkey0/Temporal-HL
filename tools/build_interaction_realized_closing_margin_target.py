#!/usr/bin/env python3
"""Closing support target change: edge-vs-finger margin prediction.

This is an experiment memo, not paper text.

Use the cached fast-path regime, but change the support target from discrete
mode imitation to informative edge-vs-finger margin prediction.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from tools.build_interaction_realized_closing_selector_support_scaling import (
    CACHE_DIR,
    TASK_TARGET,
    all_sequence_labels,
    collect_rows_fast,
    subset_to_closing_sequences,
)
from tools.build_interaction_realized_feasible_left_temporal_window_knn import (
    attach_context,
    build_frames,
    distance,
)
from tools.build_interaction_realized_pairguided_editor import train_pairguided_model
from tools.build_transition_conditioned_symbolic_editor import (
    GEN,
    SUM,
    build_pair_bank,
    build_semantic_frame_vocab,
    load_json,
)


ROOT = Path("/opt/tiger/hand")
THRESHOLDS = (-1.0, -0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0)
CFG = {"k": 40, "subtype_weight": 0.25, "agreement_weight": 0.5, "epsilon": 0.05}


def fmt(x: float) -> str:
    return f"{x:.4f}"


def load_cached_external_rows(top_n: int):
    path = CACHE_DIR / f"external_rows_top{top_n}.json"
    return json.loads(path.read_text())


def margin(row):
    return float(row["left_finger_profile_snap_joint_score"]) - float(row["left_edge_transition_snap_joint_score"])


def nearest_margin(train_rows, row):
    neighbors = sorted(((distance(train_row, row, CFG), train_row) for train_row in train_rows), key=lambda x: x[0])[: CFG["k"]]
    if not neighbors:
        return 0.0
    return sum(margin(train_row) for _, train_row in neighbors) / len(neighbors)


def choose_threshold(train_rows):
    pred_margin = []
    for idx, row in enumerate(train_rows):
        pred_margin.append(nearest_margin(train_rows[:idx] + train_rows[idx + 1 :], row))
    best = None
    for threshold in THRESHOLDS:
        modes = ["finger_profile_snap" if val > threshold else "edge_transition_snap" for val in pred_margin]
        score = sum(float(row[f"left_{mode}_joint_score"]) for row, mode in zip(train_rows, modes)) / len(train_rows)
        candidate = {
            "threshold": threshold,
            "leave_one_out_joint_score": score,
            "pred_mode_counts": dict(Counter(modes)),
        }
        if best is None or candidate["leave_one_out_joint_score"] > best["leave_one_out_joint_score"]:
            best = candidate
    return best


def summarize(rows, prefix: str):
    n = len(rows)
    return {
        "num_frames": n,
        "right_grouped_match_overall": sum(float(row[f"{prefix}_right_grouped_match"]) for row in rows) / n,
        "left_preserve_overall": sum(float(row[f"{prefix}_left_preserve"]) for row in rows) / n,
        "joint_score_overall": sum(float(row[f"{prefix}_joint_score"]) for row in rows) / n,
    }


def paired_stats(rows, a_prefix: str, b_prefix: str):
    a = [float(row[f"{a_prefix}_joint_score"]) for row in rows]
    b = [float(row[f"{b_prefix}_joint_score"]) for row in rows]
    diff = [y - x for x, y in zip(a, b)]
    return {
        "delta": sum(diff) / len(diff),
        "wins": int(sum(x > 0 for x in diff)),
        "losses": int(sum(x < 0 for x in diff)),
        "ties": int(sum(x == 0 for x in diff)),
    }


def evaluate_support(name: str, train_rows, test_rows):
    best = choose_threshold(train_rows)
    output_rows = []
    pred_modes = []
    for row in test_rows:
        pred = nearest_margin(train_rows, row)
        mode = "finger_profile_snap" if pred > best["threshold"] else "edge_transition_snap"
        pred_modes.append(mode)
        rec = dict(row)
        oracle_mode = max(("none", "edge_transition_snap", "finger_profile_snap"), key=lambda m: float(rec[f"left_{m}_joint_score"]))
        for out_name, out_mode in [
            ("fixed_edge", "edge_transition_snap"),
            ("margin_target", mode),
            ("oracle", oracle_mode),
        ]:
            prefix = f"left_{out_mode}"
            for field in ("right_grouped_match", "left_preserve", "joint_score"):
                rec[f"{out_name}_{field}"] = rec[f"{prefix}_{field}"]
        output_rows.append(rec)
    return {
        "support_name": name,
        "num_support_rows": len(train_rows),
        "num_informative_rows": int(sum(margin(r) != 0 for r in train_rows)),
        "threshold_stats": best,
        "summary": {key: summarize(output_rows, key) for key in ("fixed_edge", "margin_target", "oracle")},
        "paired": {
            "margin_target_vs_fixed_edge": paired_stats(output_rows, "fixed_edge", "margin_target"),
            "oracle_vs_fixed_edge": paired_stats(output_rows, "fixed_edge", "oracle"),
        },
        "pred_mode_counts": dict(Counter(pred_modes)),
    }


def main():
    val_data = load_json(GEN / "temporal_hl_val.json")
    test_data = load_json(GEN / "temporal_hl_test.json")

    val_subset = subset_to_closing_sequences(val_data)
    test_subset = subset_to_closing_sequences(test_data)
    val_labels = all_sequence_labels(val_subset)
    val_semantic_vocab = build_semantic_frame_vocab(val_subset, val_labels)
    val_pair_bank = build_pair_bank(val_subset, val_labels, val_semantic_vocab)
    val_frames = build_frames(val_subset, val_labels, TASK_TARGET)
    pair_model, pair_stats = train_pairguided_model(val_frames, val_pair_bank, TASK_TARGET)

    val_rows = attach_context(collect_rows_fast(val_frames, val_pair_bank, pair_model))
    test_rows = attach_context(collect_rows_fast(build_frames(test_subset, all_sequence_labels(test_subset), TASK_TARGET), val_pair_bank, pair_model))
    ext_top4 = load_cached_external_rows(4)["rows"]

    informative_val = [row for row in val_rows if margin(row) != 0]
    informative_ext = [row for row in ext_top4 if margin(row) != 0]

    task_results = {
        "val_all": evaluate_support("val_all", val_rows, test_rows),
        "val_plus_top4_all": evaluate_support("val_plus_top4_all", val_rows + ext_top4, test_rows),
        "val_plus_top4_informative": evaluate_support(
            "val_plus_top4_informative",
            informative_val + informative_ext,
            test_rows,
        ),
    }

    payload = {
        "focus": {
            "goal": "closing margin-target experiment on cached fast-path support",
            "task": "right_hand_motion->closing",
            "cfg": CFG,
        },
        "training_stats": {
            "pair_model": pair_stats,
            "num_val_rows": len(val_rows),
            "num_test_rows": len(test_rows),
            "num_cached_top4_rows": len(ext_top4),
            "num_cached_top4_informative_rows": len(informative_ext),
        },
        "task_results": task_results,
    }

    out_json = GEN / "interaction_realized_closing_margin_target.json"
    out_md = SUM / "interaction_realized_closing_margin_target.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction Realized Closing Margin Target",
        "",
        "This is an experiment memo, not paper text.",
        "",
        f"Val rows: `{len(val_rows)}`; test rows: `{len(test_rows)}`; cached top4 rows: `{len(ext_top4)}`; informative cached top4 rows: `{len(informative_ext)}`",
        "",
    ]
    for name, result in task_results.items():
        lines.extend(
            [
                f"## {name}",
                "",
                f"Support rows: `{result['num_support_rows']}`, informative rows: `{result['num_informative_rows']}`",
                f"Threshold stats: `{result['threshold_stats']}`",
                "",
                "| method | right grouped | left preserve | joint overall |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for method in ("fixed_edge", "margin_target", "oracle"):
            stats = result["summary"][method]
            lines.append(
                f"| {method} | {fmt(stats['right_grouped_match_overall'])} | {fmt(stats['left_preserve_overall'])} | {fmt(stats['joint_score_overall'])} |"
            )
        paired = result["paired"]
        lines.extend(
            [
                "",
                f"Pred mode counts: `{result['pred_mode_counts']}`",
                "",
                "| comparison | delta | wins | losses | ties |",
                "| --- | ---: | ---: | ---: | ---: |",
                f"| margin_target vs fixed_edge | {fmt(paired['margin_target_vs_fixed_edge']['delta'])} | {paired['margin_target_vs_fixed_edge']['wins']} | {paired['margin_target_vs_fixed_edge']['losses']} | {paired['margin_target_vs_fixed_edge']['ties']} |",
                f"| oracle vs fixed_edge | {fmt(paired['oracle_vs_fixed_edge']['delta'])} | {paired['oracle_vs_fixed_edge']['wins']} | {paired['oracle_vs_fixed_edge']['losses']} | {paired['oracle_vs_fixed_edge']['ties']} |",
                "",
            ]
        )
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
