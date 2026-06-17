#!/usr/bin/env python3
"""Subtype gate for feasible-subset left repair.

This is an experiment memo, not paper text.

Learn a cheap selector over left-repair modes on the feasible two-hand subset
using only subtype-level averages on train:

- subtype key: `(other_hand_motion, interaction_motion_value)`
- candidate modes: `none`, `edge_transition_snap`, `finger_profile_snap`

Then evaluate on test and compare against:

- fixed `none`
- fixed task-best mode
- subtype gate
- per-frame oracle over candidate modes
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

from tools.build_interaction_realized_constraint_sweep import evaluate_edit, fmt
from tools.build_interaction_realized_pairguided_editor import (
    TASKS,
    select_pairguided_left_pool,
    train_pairguided_model,
)
from tools.build_interaction_realized_right_support_sweep import (
    reorder_to_budget,
    right_candidates_mode,
)
from tools.build_pairguided_reranker_multislice import (
    collect_slice_frames,
    relaxed_left_family_candidates_with_meta,
)
from tools.build_transition_conditioned_symbolic_editor import (
    GEN,
    SUM,
    build_pair_bank,
    build_semantic_frame_vocab,
    canonical,
    load_json,
    overlap_labels,
)
from tools.build_weak_slice_topk_joint_search import evaluate_joint


LEFT_REPAIR_MODES = ("none", "edge_transition_snap", "finger_profile_snap")
TASK_BEST = {
    "closing": "edge_transition_snap",
    "opening": "finger_profile_snap",
}
RIGHT_POOL_MODE = "relax_both"
FAMILY_BUDGET = 2
DEPTH = 20


def pick_best_from_cache(prev_frame, curr_frame, prev_geom, target_pool, left_pool, depth: int, joint_cache):
    best = None
    for right_row in target_pool:
        right_id = id(right_row)
        for left_item in left_pool[:depth]:
            left_row = left_item["row"]
            key = (right_id, id(left_row))
            if key not in joint_cache:
                joint_cache[key] = evaluate_joint(prev_frame, curr_frame, prev_geom, right_row, left_row)
            res = joint_cache[key]
            score = (res["joint_score"], res["left_preserve"], res["right_grouped_match"])
            if best is None or score > best[0]:
                best = (score, right_row, left_row)
    return best


def collect_rows(frames, pair_bank, task_target: str, hgb_model):
    rows = []
    for entry in frames:
        if entry["curr_frame"].get("left") is None:
            continue
        prev_frame = entry["prev_frame"]
        curr_frame = entry["curr_frame"]
        prev_geom = entry["prev_geom"]
        curr_geom = entry["curr_geom"]
        curr_attrs = entry["curr_attrs"]
        current_left_group = entry["current_opp_group"]

        raw_left_pool = relaxed_left_family_candidates_with_meta(
            pair_bank,
            current_left_group,
            curr_attrs,
            prev_geom,
            curr_geom,
            "left",
            FAMILY_BUDGET,
        )
        target_pool = right_candidates_mode(pair_bank, curr_attrs, prev_geom, curr_geom, task_target, RIGHT_POOL_MODE)
        ranked_left_pool = select_pairguided_left_pool(
            hgb_model,
            pair_bank,
            curr_attrs,
            prev_geom,
            curr_geom,
            current_left_group,
            target_pool,
        )
        left_pool = reorder_to_budget(ranked_left_pool, raw_left_pool)
        choice = pick_best_from_cache(
            prev_frame,
            curr_frame,
            prev_geom,
            target_pool,
            left_pool,
            DEPTH,
            {},
        ) if target_pool and left_pool else None
        right_row = None if choice is None else choice[1]
        left_row = None if choice is None else choice[2]

        rec = {
            "seq_name": entry["seq_name"],
            "frame_idx": curr_frame["frame_idx"],
            "other_hand_motion": str(curr_frame["left"].get("hand_motion", "unknown")),
            "interaction_motion_value": str(curr_frame.get("interaction_motion", "unknown")),
        }
        for mode in LEFT_REPAIR_MODES:
            prefix = f"left_{mode}"
            eval_rec = evaluate_edit(
                prev_frame,
                curr_frame,
                prev_geom,
                right_row=right_row,
                left_row=left_row,
                repair_mode=mode,
            )
            rec.update({f"{prefix}_{k}": v for k, v in eval_rec.items()})
        rows.append(rec)
    return rows


def train_subtype_gate(train_rows):
    by = defaultdict(lambda: defaultdict(list))
    for r in train_rows:
        key = (r["other_hand_motion"], r["interaction_motion_value"])
        for mode in LEFT_REPAIR_MODES:
            by[key][mode].append(float(r[f"left_{mode}_joint_score"]))
    gate = {}
    for key, mode_map in by.items():
        means = {mode: sum(vals) / len(vals) for mode, vals in mode_map.items()}
        gate[key] = max(means, key=means.get)
    return gate


def summarize_method(rows, prefix: str):
    n = len(rows)
    return {
        "num_frames": n,
        "right_grouped_match_overall": sum(row[f"{prefix}_right_grouped_match"] for row in rows) / n,
        "left_preserve_overall": sum(row[f"{prefix}_left_preserve"] for row in rows) / n,
        "joint_score_overall": sum(row[f"{prefix}_joint_score"] for row in rows) / n,
    }


def paired_stats(rows, a_prefix: str, b_prefix: str):
    import numpy as np
    a = np.asarray([float(r[f"{a_prefix}_joint_score"]) for r in rows], dtype=np.float32)
    b = np.asarray([float(r[f"{b_prefix}_joint_score"]) for r in rows], dtype=np.float32)
    diff = b - a
    return {
        "delta": float(diff.mean()),
        "wins": int((diff > 0).sum()),
        "losses": int((diff < 0).sum()),
        "ties": int((diff == 0).sum()),
    }


def run_task(train_frames, test_frames, pair_bank, task_target: str):
    hgb_model, hgb_stats = train_pairguided_model(train_frames, pair_bank, task_target)
    train_rows = collect_rows(train_frames, pair_bank, task_target, hgb_model)
    test_rows = collect_rows(test_frames, pair_bank, task_target, hgb_model)
    gate = train_subtype_gate(train_rows)

    task_best_mode = TASK_BEST[task_target]
    for r in test_rows:
        key = (r["other_hand_motion"], r["interaction_motion_value"])
        gate_mode = gate.get(key, task_best_mode)
        for name, source in [
            ("fixed_none", "left_none"),
            ("fixed_task_best", f"left_{task_best_mode}"),
            ("gate", f"left_{gate_mode}"),
        ]:
            for field in ("right_grouped_match", "left_preserve", "joint_score"):
                r[f"{name}_{field}"] = r[f"{source}_{field}"]
        # oracle over realistic candidate modes
        vals = {mode: r[f"left_{mode}_joint_score"] for mode in LEFT_REPAIR_MODES}
        best_mode = max(vals, key=vals.get)
        for field in ("right_grouped_match", "left_preserve", "joint_score"):
            r[f"oracle_{field}"] = r[f"left_{best_mode}_{field}"]
        r["gate_mode"] = gate_mode
        r["oracle_mode"] = best_mode

    summary = {
        name: summarize_method(test_rows, name)
        for name in ("fixed_none", "fixed_task_best", "gate", "oracle")
    }
    paired = {
        "gate_vs_fixed_task_best": paired_stats(test_rows, "fixed_task_best", "gate"),
        "oracle_vs_fixed_task_best": paired_stats(test_rows, "fixed_task_best", "oracle"),
        "gate_vs_fixed_none": paired_stats(test_rows, "fixed_none", "gate"),
    }
    return {
        "training_stats": {"hgb": hgb_stats},
        "gate_map": {f"{k[0]}|{k[1]}": v for k, v in gate.items()},
        "summary": summary,
        "paired": paired,
        "rows": test_rows,
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

    selected_tasks = [task for task in TASKS if args.task_target is None or task[1] == args.task_target]
    task_results = {}
    for task_field, task_target in selected_tasks:
        train_frames = [
            row for row in collect_slice_frames(train_data, task_field, task_target)
            if (canonical(row["seq_name"]) in labels or row["seq_name"] in labels) and row["curr_frame"].get("left") is not None
        ]
        test_frames = [
            row for row in collect_slice_frames(test_data, task_field, task_target)
            if (canonical(row["seq_name"]) in labels or row["seq_name"] in labels) and row["curr_frame"].get("left") is not None
        ]
        task_results[f"{task_field}->{task_target}"] = run_task(train_frames, test_frames, pair_bank, task_target)

    payload = {
        "artifacts": {"train_json": str(GEN / "temporal_hl_val.json"), "test_json": str(GEN / "temporal_hl_test.json")},
        "focus": {
            "goal": "cheap subtype gate over feasible-subset left repair",
            "tasks": [f"{field}->{target}" for field, target in selected_tasks],
            "candidate_modes": list(LEFT_REPAIR_MODES),
            "task_best": TASK_BEST,
            "task_filter": args.task_target,
        },
        "task_results": task_results,
    }

    suffix = "" if args.task_target is None else f"_{args.task_target}"
    out_json = GEN / f"interaction_realized_feasible_left_repair_gate{suffix}.json"
    out_md = SUM / f"interaction_realized_feasible_left_repair_gate{suffix}.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction Realized Feasible Left Repair Gate",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Cheap subtype gate over feasible-subset left repair.",
        "",
    ]
    for task_name, result in task_results.items():
        lines.extend([
            f"## {task_name}",
            "",
            "| method | right grouped | left preserve | joint overall |",
            "| --- | ---: | ---: | ---: |",
        ])
        for method in ("fixed_none", "fixed_task_best", "gate", "oracle"):
            s = result["summary"][method]
            lines.append(
                f"| {method} | {fmt(s['right_grouped_match_overall'])} | {fmt(s['left_preserve_overall'])} | {fmt(s['joint_score_overall'])} |"
            )
        lines.extend([
            "",
            "| comparison | delta | wins | losses | ties |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for key, stats in result["paired"].items():
            lines.append(
                f"| {key} | {fmt(stats['delta'])} | {stats['wins']} | {stats['losses']} | {stats['ties']} |"
            )
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
