#!/usr/bin/env python3
"""Temporal-window KNN selector for feasible left repair.

This is an experiment memo, not paper text.

Use explicit cross-frame temporal context rather than frame-local or coarse
template metadata:

- for each feasible frame, build a window feature with previous/current/next
  feasible rows from the same sequence when available
- keep dense continuous gain supervision (`joint(repair) - joint(none)`)
- select the repair mode by nearest-neighbor regression on the feasible `val`
  subset, then evaluate on feasible `test`
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from tools.build_interaction_realized_feasible_left_repair_model_gate import collect_rows
from tools.build_interaction_realized_pairguided_editor import TASKS, train_pairguided_model
from tools.build_pairguided_reranker_multislice import collect_slice_frames
from tools.build_transition_conditioned_symbolic_editor import (
    GEN,
    SUM,
    build_pair_bank,
    build_semantic_frame_vocab,
    canonical,
    load_json,
    overlap_labels,
)


LEFT_REPAIR_MODES = ("none", "edge_transition_snap", "finger_profile_snap")
TASK_BEST = {
    "closing": "edge_transition_snap",
    "opening": "finger_profile_snap",
}
OTHER_VALUES = ["none", "start", "opening", "closing", "mixed", "steady", "unknown", "__missing__"]
INTERACTION_VALUES = ["approach", "separate", "steady", "unknown", "__missing__"]
GRID = []
for k in (3, 5, 10, 20, 40):
    for subtype_weight in (0.25, 0.5, 1.0):
        for agreement_weight in (0.5, 1.0):
            GRID.append(
                {
                    "k": k,
                    "subtype_weight": subtype_weight,
                    "agreement_weight": agreement_weight,
                    "epsilon": 0.05,
                }
            )


def fmt(x: float) -> str:
    return f"{x:.4f}"


def canon_value(value: str | None, vocab: list[str]) -> str:
    if value is None:
        return "__missing__"
    return value if value in vocab else "unknown"


def build_frames(data, labels, task_target: str):
    return [
        row
        for row in collect_slice_frames(data, "right_hand_motion", task_target)
        if (canonical(row["seq_name"]) in labels or row["seq_name"] in labels) and row["curr_frame"].get("left") is not None
    ]


def attach_context(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["seq_name"]].append(row)
    out = []
    for seq_name, items in grouped.items():
        items = sorted(items, key=lambda x: int(x["frame_idx"]))
        for idx, row in enumerate(items):
            rec = dict(row)
            prev_row = items[idx - 1] if idx > 0 and int(row["frame_idx"]) - int(items[idx - 1]["frame_idx"]) <= 12 else None
            next_row = items[idx + 1] if idx + 1 < len(items) and int(items[idx + 1]["frame_idx"]) - int(row["frame_idx"]) <= 12 else None
            for prefix, source in (("prev", prev_row), ("curr", row), ("next", next_row)):
                if source is None:
                    rec[f"{prefix}_other_hand_motion"] = "__missing__"
                    rec[f"{prefix}_interaction_motion_value"] = "__missing__"
                    rec[f"{prefix}_none_left_state"] = 0.0
                    rec[f"{prefix}_none_left_transition"] = 0.0
                    rec[f"{prefix}_none_right_state"] = 0.0
                    rec[f"{prefix}_none_right_transition"] = 0.0
                else:
                    rec[f"{prefix}_other_hand_motion"] = canon_value(source["other_hand_motion"], OTHER_VALUES)
                    rec[f"{prefix}_interaction_motion_value"] = canon_value(source["interaction_motion_value"], INTERACTION_VALUES)
                    rec[f"{prefix}_none_left_state"] = float(source["left_none_left_state_agreement"])
                    rec[f"{prefix}_none_left_transition"] = float(source["left_none_left_transition_agreement"])
                    rec[f"{prefix}_none_right_state"] = float(source["left_none_right_state_agreement"])
                    rec[f"{prefix}_none_right_transition"] = float(source["left_none_right_transition_agreement"])
            out.append(rec)
    return out


def distance(a, b, cfg):
    total = 0.0
    for prefix in ("prev", "curr", "next"):
        total += cfg["subtype_weight"] * float(a[f"{prefix}_other_hand_motion"] != b[f"{prefix}_other_hand_motion"])
        total += cfg["subtype_weight"] * float(a[f"{prefix}_interaction_motion_value"] != b[f"{prefix}_interaction_motion_value"])
        total += cfg["agreement_weight"] * abs(a[f"{prefix}_none_left_state"] - b[f"{prefix}_none_left_state"])
        total += cfg["agreement_weight"] * abs(a[f"{prefix}_none_left_transition"] - b[f"{prefix}_none_left_transition"])
        total += cfg["agreement_weight"] * abs(a[f"{prefix}_none_right_state"] - b[f"{prefix}_none_right_state"])
        total += cfg["agreement_weight"] * abs(a[f"{prefix}_none_right_transition"] - b[f"{prefix}_none_right_transition"])
    return total


def predict_mode(train_rows, row, cfg):
    neighbors = sorted(((distance(train_row, row, cfg), train_row) for train_row in train_rows), key=lambda x: x[0])[: cfg["k"]]
    scores = {"none": float(row["left_none_joint_score"])}
    for mode in LEFT_REPAIR_MODES[1:]:
        weighted_delta = 0.0
        total_weight = 0.0
        for dist, train_row in neighbors:
            weight = 1.0 / (cfg["epsilon"] + dist)
            weighted_delta += weight * (
                float(train_row[f"left_{mode}_joint_score"]) - float(train_row["left_none_joint_score"])
            )
            total_weight += weight
        pred_delta = 0.0 if total_weight == 0.0 else weighted_delta / total_weight
        scores[mode] = float(row["left_none_joint_score"]) + pred_delta
    pred_mode = max(scores, key=scores.get)
    return pred_mode, scores


def evaluate_prediction(rows, pred_modes):
    return sum(float(row[f"left_{mode}_joint_score"]) for row, mode in zip(rows, pred_modes)) / len(rows)


def choose_cfg(train_rows):
    best = None
    for cfg in GRID:
        pred_modes = []
        for idx, row in enumerate(train_rows):
            pred_mode, _ = predict_mode(train_rows[:idx] + train_rows[idx + 1 :], row, cfg)
            pred_modes.append(pred_mode)
        score = evaluate_prediction(train_rows, pred_modes)
        candidate = {
            "cfg": cfg,
            "leave_one_out_joint_score": score,
            "pred_mode_counts": dict(Counter(pred_modes)),
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


def run_task(train_data, test_data, labels, pair_bank, task_target: str):
    train_frames = build_frames(train_data, labels, task_target)
    test_frames = build_frames(test_data, labels, task_target)
    pair_model, pair_stats = train_pairguided_model(train_frames, pair_bank, task_target)
    train_rows = attach_context(collect_rows(train_frames, pair_bank, task_target, pair_model))
    test_rows = attach_context(collect_rows(test_frames, pair_bank, task_target, pair_model))
    best = choose_cfg(train_rows)

    fixed_mode = TASK_BEST[task_target]
    pred_modes = []
    pred_scores = []
    for row in test_rows:
        pred_mode, scores = predict_mode(train_rows, row, best["cfg"])
        pred_modes.append(pred_mode)
        pred_scores.append(scores)

    output_rows = []
    for row, pred_mode, scores in zip(test_rows, pred_modes, pred_scores):
        rec = dict(row)
        oracle_mode = max(LEFT_REPAIR_MODES, key=lambda mode: float(rec[f"left_{mode}_joint_score"]))
        for name, mode in [
            ("fixed_none", "none"),
            ("fixed_task_best", fixed_mode),
            ("temporal_window_knn", pred_mode),
            ("oracle", oracle_mode),
        ]:
            prefix = f"left_{mode}"
            for field in ("right_grouped_match", "left_preserve", "joint_score"):
                rec[f"{name}_{field}"] = rec[f"{prefix}_{field}"]
        rec["pred_mode"] = pred_mode
        rec["pred_scores"] = scores
        rec["oracle_mode"] = oracle_mode
        output_rows.append(rec)

    summary = {name: summarize(output_rows, name) for name in ("fixed_none", "fixed_task_best", "temporal_window_knn", "oracle")}
    paired = {
        "temporal_window_knn_vs_fixed_task_best": paired_stats(output_rows, "fixed_task_best", "temporal_window_knn"),
        "temporal_window_knn_vs_fixed_none": paired_stats(output_rows, "fixed_none", "temporal_window_knn"),
        "oracle_vs_temporal_window_knn": paired_stats(output_rows, "temporal_window_knn", "oracle"),
    }
    return {
        "training_stats": {
            "pair_model": pair_stats,
            "temporal_window_knn": best,
        },
        "summary": summary,
        "paired": paired,
        "pred_mode_counts": dict(Counter(pred_modes)),
        "rows": output_rows,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-target",
        choices=[target for _, target in TASKS],
        default=None,
        help="Run only one right-hand hard-slice target.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_data = load_json(GEN / "temporal_hl_val.json")
    test_data = load_json(GEN / "temporal_hl_test.json")
    labels = overlap_labels(train_data, test_data)
    semantic_vocab = build_semantic_frame_vocab(train_data, labels)
    pair_bank = build_pair_bank(train_data, labels, semantic_vocab)

    selected_targets = [target for _, target in TASKS if args.task_target is None or args.task_target == target]
    selected_targets = sorted(set(selected_targets))
    task_results = {}
    for task_target in selected_targets:
        task_results[f"right_hand_motion->{task_target}"] = run_task(train_data, test_data, labels, pair_bank, task_target)

    payload = {
        "artifacts": {"train_json": str(GEN / "temporal_hl_val.json"), "test_json": str(GEN / "temporal_hl_test.json")},
        "focus": {
            "goal": "temporal-window KNN selector for feasible left repair",
            "tasks": [f"right_hand_motion->{target}" for target in selected_targets],
            "repair_modes": list(LEFT_REPAIR_MODES),
            "grid_size": len(GRID),
        },
        "task_results": task_results,
    }

    suffix = "" if args.task_target is None else f"_{args.task_target}"
    out_json = GEN / f"interaction_realized_feasible_left_temporal_window_knn{suffix}.json"
    out_md = SUM / f"interaction_realized_feasible_left_temporal_window_knn{suffix}.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction Realized Feasible Left Temporal Window KNN",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Temporal-window dense selector using previous/current/next feasible context from the same sequence.",
        "",
    ]
    for task_name, result in task_results.items():
        lines.extend(
            [
                f"## {task_name}",
                "",
                "| method | right grouped | left preserve | joint overall |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for method in ("fixed_none", "fixed_task_best", "temporal_window_knn", "oracle"):
            stats = result["summary"][method]
            lines.append(
                f"| {method} | {fmt(stats['right_grouped_match_overall'])} | {fmt(stats['left_preserve_overall'])} | {fmt(stats['joint_score_overall'])} |"
            )
        paired = result["paired"]
        cfg = result["training_stats"]["temporal_window_knn"]
        lines.extend(
            [
                "",
                f"Selected config: `k={cfg['cfg']['k']}`, leave-one-out joint `{fmt(cfg['leave_one_out_joint_score'])}`",
                "",
                "| comparison | delta | wins | losses | ties |",
                "| --- | ---: | ---: | ---: | ---: |",
                f"| temporal_window_knn vs fixed_task_best | {fmt(paired['temporal_window_knn_vs_fixed_task_best']['delta'])} | {paired['temporal_window_knn_vs_fixed_task_best']['wins']} | {paired['temporal_window_knn_vs_fixed_task_best']['losses']} | {paired['temporal_window_knn_vs_fixed_task_best']['ties']} |",
                f"| temporal_window_knn vs fixed_none | {fmt(paired['temporal_window_knn_vs_fixed_none']['delta'])} | {paired['temporal_window_knn_vs_fixed_none']['wins']} | {paired['temporal_window_knn_vs_fixed_none']['losses']} | {paired['temporal_window_knn_vs_fixed_none']['ties']} |",
                f"| oracle vs temporal_window_knn | {fmt(paired['oracle_vs_temporal_window_knn']['delta'])} | {paired['oracle_vs_temporal_window_knn']['wins']} | {paired['oracle_vs_temporal_window_knn']['losses']} | {paired['oracle_vs_temporal_window_knn']['ties']} |",
                "",
                f"Test mode counts: {result['pred_mode_counts']}",
                "",
            ]
        )

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
