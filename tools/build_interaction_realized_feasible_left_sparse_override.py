#!/usr/bin/env python3
"""Sparse subtype override over fixed feasible-left policy.

This is an experiment memo, not paper text.
"""

from __future__ import annotations

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
GRID = []
for min_count in (3, 5, 8):
    for min_delta in (0.05, 0.1, 0.15, 0.2):
        GRID.append({"min_count": min_count, "min_delta": min_delta})


def fmt(x: float) -> str:
    return f"{x:.4f}"


def subtype_key(row):
    return (row["other_hand_motion"], row["interaction_motion_value"])


def build_frames(data, labels, task_target: str):
    return [
        row
        for row in collect_slice_frames(data, "right_hand_motion", task_target)
        if (canonical(row["seq_name"]) in labels or row["seq_name"] in labels) and row["curr_frame"].get("left") is not None
    ]


def build_override_map(train_rows, fixed_mode: str, cfg):
    grouped = defaultdict(list)
    for row in train_rows:
        grouped[subtype_key(row)].append(row)
    override = {}
    stats = {}
    for key, items in grouped.items():
        if len(items) < cfg["min_count"]:
            continue
        fixed = sum(float(row[f"left_{fixed_mode}_joint_score"]) for row in items) / len(items)
        best_mode = fixed_mode
        best_delta = 0.0
        for mode in LEFT_REPAIR_MODES:
            if mode == fixed_mode:
                continue
            score = sum(float(row[f"left_{mode}_joint_score"]) for row in items) / len(items)
            delta = score - fixed
            if delta > best_delta:
                best_delta = delta
                best_mode = mode
        stats[str(key)] = {"num_frames": len(items), "best_mode": best_mode, "best_delta": best_delta}
        if best_mode != fixed_mode and best_delta >= cfg["min_delta"]:
            override[key] = best_mode
    return override, stats


def evaluate_rows(test_rows, fixed_mode: str, override_map):
    out_rows = []
    pred_counts = Counter()
    for row in test_rows:
        key = subtype_key(row)
        pred_mode = override_map.get(key, fixed_mode)
        pred_counts[pred_mode] += 1
        rec = dict(row)
        oracle_mode = max(LEFT_REPAIR_MODES, key=lambda mode: float(row[f"left_{mode}_joint_score"]))
        for name, mode in [
            ("fixed_task_best", fixed_mode),
            ("sparse_override", pred_mode),
            ("oracle", oracle_mode),
        ]:
            prefix = f"left_{mode}"
            for field in ("right_grouped_match", "left_preserve", "joint_score"):
                rec[f"{name}_{field}"] = rec[f"{prefix}_{field}"]
        rec["pred_mode"] = pred_mode
        rec["oracle_mode"] = oracle_mode
        out_rows.append(rec)
    return out_rows, dict(pred_counts)


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


def choose_cfg(train_rows, fixed_mode: str):
    best = None
    for cfg in GRID:
        override_map, _ = build_override_map(train_rows, fixed_mode, cfg)
        eval_rows, pred_counts = evaluate_rows(train_rows, fixed_mode, override_map)
        score = summarize(eval_rows, "sparse_override")["joint_score_overall"]
        candidate = {
            "cfg": cfg,
            "train_joint_score": score,
            "num_overrides": len(override_map),
            "pred_mode_counts": pred_counts,
            "override_map": override_map,
        }
        if best is None or candidate["train_joint_score"] > best["train_joint_score"]:
            best = candidate
    return best


def run_task(train_data, test_data, labels, pair_bank, task_target: str):
    train_frames = build_frames(train_data, labels, task_target)
    test_frames = build_frames(test_data, labels, task_target)
    pair_model, pair_stats = train_pairguided_model(train_frames, pair_bank, task_target)
    train_rows = collect_rows(train_frames, pair_bank, task_target, pair_model)
    test_rows = collect_rows(test_frames, pair_bank, task_target, pair_model)

    fixed_mode = TASK_BEST[task_target]
    best = choose_cfg(train_rows, fixed_mode)
    override_map = best.pop("override_map")
    override_stats = build_override_map(train_rows, fixed_mode, best["cfg"])[1]
    eval_rows, pred_counts = evaluate_rows(test_rows, fixed_mode, override_map)

    summary = {name: summarize(eval_rows, name) for name in ("fixed_task_best", "sparse_override", "oracle")}
    paired = {
        "sparse_override_vs_fixed_task_best": paired_stats(eval_rows, "fixed_task_best", "sparse_override"),
        "oracle_vs_fixed_task_best": paired_stats(eval_rows, "fixed_task_best", "oracle"),
        "oracle_vs_sparse_override": paired_stats(eval_rows, "sparse_override", "oracle"),
    }
    return {
        "training_stats": {
            "pair_model": pair_stats,
            "sparse_override": best,
            "override_stats": override_stats,
        },
        "summary": summary,
        "paired": paired,
        "pred_mode_counts": pred_counts,
        "rows": eval_rows,
    }


def main():
    train_data = load_json(GEN / "temporal_hl_val.json")
    test_data = load_json(GEN / "temporal_hl_test.json")
    labels = overlap_labels(train_data, test_data)
    semantic_vocab = build_semantic_frame_vocab(train_data, labels)
    pair_bank = build_pair_bank(train_data, labels, semantic_vocab)

    task_results = {}
    for _, task_target in TASKS:
        task_results[f"right_hand_motion->{task_target}"] = run_task(train_data, test_data, labels, pair_bank, task_target)

    payload = {
        "artifacts": {"train_json": str(GEN / "temporal_hl_val.json"), "test_json": str(GEN / "temporal_hl_test.json")},
        "focus": {
            "goal": "sparse subtype override over fixed feasible-left policy",
            "tasks": [f"right_hand_motion->{target}" for _, target in TASKS],
            "grid": GRID,
        },
        "task_results": task_results,
    }

    out_json = GEN / "interaction_realized_feasible_left_sparse_override.json"
    out_md = SUM / "interaction_realized_feasible_left_sparse_override.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction Realized Feasible Left Sparse Override",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Override the fixed task-best policy only on a sparse set of train-supported subtypes with strong positive alternate-mode gain.",
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
        for method in ("fixed_task_best", "sparse_override", "oracle"):
            stats = result["summary"][method]
            lines.append(
                f"| {method} | {fmt(stats['right_grouped_match_overall'])} | {fmt(stats['left_preserve_overall'])} | {fmt(stats['joint_score_overall'])} |"
            )
        lines.extend(
            [
                "",
                f"Selected cfg: `{result['training_stats']['sparse_override']['cfg']}`, train joint `{fmt(result['training_stats']['sparse_override']['train_joint_score'])}`, overrides `{result['training_stats']['sparse_override']['num_overrides']}`",
                f"Pred mode counts: `{result['pred_mode_counts']}`",
                "",
                "| comparison | delta | wins | losses | ties |",
                "| --- | ---: | ---: | ---: | ---: |",
                f"| sparse_override vs fixed_task_best | {fmt(result['paired']['sparse_override_vs_fixed_task_best']['delta'])} | {result['paired']['sparse_override_vs_fixed_task_best']['wins']} | {result['paired']['sparse_override_vs_fixed_task_best']['losses']} | {result['paired']['sparse_override_vs_fixed_task_best']['ties']} |",
                f"| oracle vs fixed_task_best | {fmt(result['paired']['oracle_vs_fixed_task_best']['delta'])} | {result['paired']['oracle_vs_fixed_task_best']['wins']} | {result['paired']['oracle_vs_fixed_task_best']['losses']} | {result['paired']['oracle_vs_fixed_task_best']['ties']} |",
                f"| oracle vs sparse_override | {fmt(result['paired']['oracle_vs_sparse_override']['delta'])} | {result['paired']['oracle_vs_sparse_override']['wins']} | {result['paired']['oracle_vs_sparse_override']['losses']} | {result['paired']['oracle_vs_sparse_override']['ties']} |",
                "",
            ]
        )

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
