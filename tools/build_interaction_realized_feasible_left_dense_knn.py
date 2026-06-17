#!/usr/bin/env python3
"""Dense KNN repair selector on feasible hard right-hand slices.

This is an experiment memo, not paper text.

Instead of learning sparse framewise oracle labels, use dense supervision:

- each feasible training frame provides continuous gain targets
  (`repair joint score - none joint score`) for each left repair mode
- test-time selection uses nearest-neighbor regression over a richer context:
  - opposite-hand motion
  - interaction motion
  - left/right state signatures
  - no-repair agreement statistics

Hyperparameters are tuned on the feasible `val` subset via leave-one-out
selection, then evaluated on the feasible `test` subset.
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


TASK_BEST = {
    "closing": "edge_transition_snap",
    "opening": "finger_profile_snap",
}
LEFT_REPAIR_MODES = ("none", "edge_transition_snap", "finger_profile_snap")
GRID = []
for k in (3, 5, 10, 20, 40):
    for left_sig_weight in (0.5, 1.0, 1.5, 2.0):
        for right_sig_weight in (0.5, 1.0, 1.5, 2.0):
            GRID.append(
                {
                    "k": k,
                    "other_motion_penalty": 0.5,
                    "interaction_penalty": 0.5,
                    "left_sig_weight": left_sig_weight,
                    "right_sig_weight": right_sig_weight,
                    "left_state_weight": 0.75,
                    "left_transition_weight": 0.75,
                    "right_state_weight": 0.75,
                    "right_transition_weight": 0.75,
                    "epsilon": 0.05,
                }
            )


def fmt(x: float) -> str:
    return f"{x:.4f}"


def parse_signature(signature: str) -> list[str]:
    return signature.split("|") if signature else []


def hamming_fraction(a: list[str], b: list[str]) -> float:
    if not a or not b or len(a) != len(b):
        return 1.0
    return sum(x != y for x, y in zip(a, b)) / len(a)


def attach_signatures(rows, frames):
    out = []
    for entry, row in zip(frames, rows):
        rec = dict(row)
        rec["left_state_signature"] = str(entry["curr_attrs"].get("left_state_signature", ""))
        rec["right_state_signature"] = str(entry["curr_attrs"].get("right_state_signature", ""))
        out.append(rec)
    return out


def convert_rows(rows):
    out = []
    for row in rows:
        rec = {
            "seq_name": row["seq_name"],
            "frame_idx": int(row["frame_idx"]),
            "other_hand_motion": str(row["other_hand_motion"]),
            "interaction_motion_value": str(row["interaction_motion_value"]),
            "left_sig": parse_signature(row["left_state_signature"]),
            "right_sig": parse_signature(row["right_state_signature"]),
            "joint_none": float(row["left_none_joint_score"]),
            "joint_edge_transition_snap": float(row["left_edge_transition_snap_joint_score"]),
            "joint_finger_profile_snap": float(row["left_finger_profile_snap_joint_score"]),
            "none_left_state": float(row["left_none_left_state_agreement"]),
            "none_left_transition": float(row["left_none_left_transition_agreement"]),
            "none_right_state": float(row["left_none_right_state_agreement"]),
            "none_right_transition": float(row["left_none_right_transition_agreement"]),
        }
        for mode in LEFT_REPAIR_MODES[1:]:
            rec[f"delta_{mode}"] = rec[f"joint_{mode}"] - rec["joint_none"]
        out.append(rec)
    return out


def distance(a, b, cfg):
    return (
        cfg["other_motion_penalty"] * float(a["other_hand_motion"] != b["other_hand_motion"])
        + cfg["interaction_penalty"] * float(a["interaction_motion_value"] != b["interaction_motion_value"])
        + cfg["left_sig_weight"] * hamming_fraction(a["left_sig"], b["left_sig"])
        + cfg["right_sig_weight"] * hamming_fraction(a["right_sig"], b["right_sig"])
        + cfg["left_state_weight"] * abs(a["none_left_state"] - b["none_left_state"])
        + cfg["left_transition_weight"] * abs(a["none_left_transition"] - b["none_left_transition"])
        + cfg["right_state_weight"] * abs(a["none_right_state"] - b["none_right_state"])
        + cfg["right_transition_weight"] * abs(a["none_right_transition"] - b["none_right_transition"])
    )


def predict_mode(train_rows, row, cfg):
    neighbors = sorted(((distance(train_row, row, cfg), train_row) for train_row in train_rows), key=lambda x: x[0])[: cfg["k"]]
    scores = {"none": row["joint_none"]}
    for mode in LEFT_REPAIR_MODES[1:]:
        weighted_delta = 0.0
        total_weight = 0.0
        for dist, train_row in neighbors:
            weight = 1.0 / (cfg["epsilon"] + dist)
            weighted_delta += weight * train_row[f"delta_{mode}"]
            total_weight += weight
        pred_delta = 0.0 if total_weight == 0.0 else weighted_delta / total_weight
        scores[mode] = row["joint_none"] + pred_delta
    pred_mode = max(scores, key=scores.get)
    return pred_mode, scores


def evaluate_prediction(rows, pred_modes):
    return sum(row[f"joint_{mode}"] for row, mode in zip(rows, pred_modes)) / max(len(rows), 1)


def summarize_method(rows, prefixes_by_row_key):
    n = len(rows)
    return {
        "num_frames": n,
        "right_grouped_match_overall": sum(float(row[prefixes_by_row_key["right_grouped_match"]]) for row in rows) / n,
        "left_preserve_overall": sum(float(row[prefixes_by_row_key["left_preserve"]]) for row in rows) / n,
        "joint_score_overall": sum(float(row[prefixes_by_row_key["joint_score"]]) for row in rows) / n,
    }


def paired_stats(rows, a_prefix: str, b_prefix: str):
    a = [float(row[f"{a_prefix}_joint_score"]) for row in rows]
    b = [float(row[f"{b_prefix}_joint_score"]) for row in rows]
    diff = [x2 - x1 for x1, x2 in zip(a, b)]
    return {
        "delta": sum(diff) / len(diff),
        "wins": int(sum(x > 0 for x in diff)),
        "losses": int(sum(x < 0 for x in diff)),
        "ties": int(sum(x == 0 for x in diff)),
    }


def sequence_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["seq_name"]].append(row)
    out = []
    for seq_name, items in sorted(grouped.items()):
        rec = {
            "seq_name": seq_name,
            "num_frames": len(items),
        }
        for method in ("fixed_none", "fixed_task_best", "dense_knn", "oracle"):
            rec[f"{method}_joint_score_overall"] = sum(float(item[f"{method}_joint_score"]) for item in items) / len(items)
        out.append(rec)
    return out


def subtype_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["other_hand_motion"], row["interaction_motion_value"])].append(row)
    out = []
    for key, items in sorted(grouped.items()):
        rec = {
            "other_hand_motion": key[0],
            "interaction_motion_value": key[1],
            "num_frames": len(items),
        }
        for method in ("fixed_none", "fixed_task_best", "dense_knn", "oracle"):
            rec[f"{method}_joint_score_overall"] = sum(float(item[f"{method}_joint_score"]) for item in items) / len(items)
        out.append(rec)
    return out


def build_frames(data, labels, task_target: str):
    return [
        row
        for row in collect_slice_frames(data, "right_hand_motion", task_target)
        if (canonical(row["seq_name"]) in labels or row["seq_name"] in labels) and row["curr_frame"].get("left") is not None
    ]


def choose_cfg(train_rows):
    best = None
    for cfg in GRID:
        preds = []
        for idx, row in enumerate(train_rows):
            pred_mode, _ = predict_mode(train_rows[:idx] + train_rows[idx + 1 :], row, cfg)
            preds.append(pred_mode)
        score = evaluate_prediction(train_rows, preds)
        candidate = {
            "cfg": cfg,
            "leave_one_out_joint_score": score,
            "pred_mode_counts": dict(Counter(preds)),
        }
        if best is None or candidate["leave_one_out_joint_score"] > best["leave_one_out_joint_score"]:
            best = candidate
    return best


def run_task(train_data, test_data, labels, pair_bank, task_target: str):
    train_frames = build_frames(train_data, labels, task_target)
    test_frames = build_frames(test_data, labels, task_target)
    pair_model, pair_stats = train_pairguided_model(train_frames, pair_bank, task_target)

    train_rows = attach_signatures(collect_rows(train_frames, pair_bank, task_target, pair_model), train_frames)
    test_rows = attach_signatures(collect_rows(test_frames, pair_bank, task_target, pair_model), test_frames)

    train_knn = convert_rows(train_rows)
    test_knn = convert_rows(test_rows)
    best = choose_cfg(train_knn)

    fixed_mode = TASK_BEST[task_target]
    pred_modes = []
    pred_scores = []
    for row in test_knn:
        pred_mode, scores = predict_mode(train_knn, row, best["cfg"])
        pred_modes.append(pred_mode)
        pred_scores.append(scores)

    output_rows = []
    for raw_row, pred_mode, scores in zip(test_rows, pred_modes, pred_scores):
        row = dict(raw_row)
        task_best_prefix = f"left_{fixed_mode}"
        dense_prefix = f"left_{pred_mode}"
        oracle_mode = max(LEFT_REPAIR_MODES, key=lambda mode: float(row[f"left_{mode}_joint_score"]))
        oracle_prefix = f"left_{oracle_mode}"
        for name, prefix in [
            ("fixed_none", "left_none"),
            ("fixed_task_best", task_best_prefix),
            ("dense_knn", dense_prefix),
            ("oracle", oracle_prefix),
        ]:
            for field in ("right_grouped_match", "left_preserve", "joint_score"):
                row[f"{name}_{field}"] = row[f"{prefix}_{field}"]
        row["pred_mode"] = pred_mode
        row["pred_scores"] = scores
        row["oracle_mode"] = oracle_mode
        output_rows.append(row)

    summary = {
        "fixed_none": summarize_method(
            output_rows,
            {"right_grouped_match": "fixed_none_right_grouped_match", "left_preserve": "fixed_none_left_preserve", "joint_score": "fixed_none_joint_score"},
        ),
        "fixed_task_best": summarize_method(
            output_rows,
            {"right_grouped_match": "fixed_task_best_right_grouped_match", "left_preserve": "fixed_task_best_left_preserve", "joint_score": "fixed_task_best_joint_score"},
        ),
        "dense_knn": summarize_method(
            output_rows,
            {"right_grouped_match": "dense_knn_right_grouped_match", "left_preserve": "dense_knn_left_preserve", "joint_score": "dense_knn_joint_score"},
        ),
        "oracle": summarize_method(
            output_rows,
            {"right_grouped_match": "oracle_right_grouped_match", "left_preserve": "oracle_left_preserve", "joint_score": "oracle_joint_score"},
        ),
    }
    paired = {
        "dense_knn_vs_fixed_task_best": paired_stats(output_rows, "fixed_task_best", "dense_knn"),
        "dense_knn_vs_fixed_none": paired_stats(output_rows, "fixed_none", "dense_knn"),
        "oracle_vs_dense_knn": paired_stats(output_rows, "dense_knn", "oracle"),
    }
    return {
        "training_stats": {
            "pair_model": pair_stats,
            "knn_selector": best,
        },
        "summary": summary,
        "paired": paired,
        "sequence_summary": sequence_summary(output_rows),
        "subtype_summary": subtype_summary(output_rows),
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
            "goal": "dense KNN selector for feasible left repair on corrected hard right-hand slices",
            "tasks": [f"right_hand_motion->{target}" for target in selected_targets],
            "grid_size": len(GRID),
            "repair_modes": list(LEFT_REPAIR_MODES),
        },
        "task_results": task_results,
    }

    suffix = "" if args.task_target is None else f"_{args.task_target}"
    out_json = GEN / f"interaction_realized_feasible_left_dense_knn{suffix}.json"
    out_md = SUM / f"interaction_realized_feasible_left_dense_knn{suffix}.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction Realized Feasible Left Dense KNN",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Dense feasible-left repair selection using continuous gain targets and state-signature-aware nearest-neighbor regression.",
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
        for method in ("fixed_none", "fixed_task_best", "dense_knn", "oracle"):
            stats = result["summary"][method]
            lines.append(
                f"| {method} | {fmt(stats['right_grouped_match_overall'])} | {fmt(stats['left_preserve_overall'])} | {fmt(stats['joint_score_overall'])} |"
            )
        cfg = result["training_stats"]["knn_selector"]
        paired = result["paired"]
        lines.extend(
            [
                "",
                f"Selected config: `k={cfg['cfg']['k']}`, leave-one-out joint `{fmt(cfg['leave_one_out_joint_score'])}`",
                "",
                "| comparison | delta | wins | losses | ties |",
                "| --- | ---: | ---: | ---: | ---: |",
                f"| dense_knn vs fixed_task_best | {fmt(paired['dense_knn_vs_fixed_task_best']['delta'])} | {paired['dense_knn_vs_fixed_task_best']['wins']} | {paired['dense_knn_vs_fixed_task_best']['losses']} | {paired['dense_knn_vs_fixed_task_best']['ties']} |",
                f"| dense_knn vs fixed_none | {fmt(paired['dense_knn_vs_fixed_none']['delta'])} | {paired['dense_knn_vs_fixed_none']['wins']} | {paired['dense_knn_vs_fixed_none']['losses']} | {paired['dense_knn_vs_fixed_none']['ties']} |",
                f"| oracle vs dense_knn | {fmt(paired['oracle_vs_dense_knn']['delta'])} | {paired['oracle_vs_dense_knn']['wins']} | {paired['oracle_vs_dense_knn']['losses']} | {paired['oracle_vs_dense_knn']['ties']} |",
                "",
                "### Sequence Summary",
                "",
                "| sequence | none | fixed task-best | dense knn | oracle |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in result["sequence_summary"]:
            lines.append(
                f"| {row['seq_name']} | {fmt(row['fixed_none_joint_score_overall'])} | {fmt(row['fixed_task_best_joint_score_overall'])} | "
                f"{fmt(row['dense_knn_joint_score_overall'])} | {fmt(row['oracle_joint_score_overall'])} |"
            )
        lines.extend(
            [
                "",
                "### Subtype Summary",
                "",
                "| other hand | interaction | frames | none | fixed task-best | dense knn | oracle |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in result["subtype_summary"]:
            lines.append(
                f"| {row['other_hand_motion']} | {row['interaction_motion_value']} | {row['num_frames']} | "
                f"{fmt(row['fixed_none_joint_score_overall'])} | {fmt(row['fixed_task_best_joint_score_overall'])} | "
                f"{fmt(row['dense_knn_joint_score_overall'])} | {fmt(row['oracle_joint_score_overall'])} |"
            )
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
