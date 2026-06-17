#!/usr/bin/env python3
"""Targeted left-repair follow-up on feasible hard right-hand slices.

This is an experiment memo, not paper text.

Focus only on feasible two-hand frames where the opposite hand exists.
Keep the corrected strongest right-side setup fixed:

- right donor pool: `relax_both`
- left preserve pool: budget 2 + pair-guided reranking
- no extra right-target donor-side repair

Then vary only the left-side preserve repair strength.
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


LEFT_REPAIR_MODES = (
    "none",
    "edge_transition_snap",
    "finger_profile_snap",
    "full_curr_oracle",
)
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


def summarize_method(rows, prefix: str):
    n = len(rows)
    return {
        "num_frames": n,
        "right_grouped_match_overall": sum(row[f"{prefix}_right_grouped_match"] for row in rows) / n,
        "left_preserve_overall": sum(row[f"{prefix}_left_preserve"] for row in rows) / n,
        "joint_score_overall": sum(row[f"{prefix}_joint_score"] for row in rows) / n,
    }


def summarize_by_sequence(rows, prefixes):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["seq_name"]].append(row)
    out = []
    for seq_name, items in sorted(grouped.items()):
        rec = {"seq_name": seq_name, "num_frames": len(items)}
        for prefix in prefixes:
            rec[f"{prefix}_joint_score_overall"] = summarize_method(items, prefix)["joint_score_overall"]
        out.append(rec)
    return out


def run_task(train_frames, test_frames, pair_bank, task_target: str):
    hgb_model, hgb_stats = train_pairguided_model(train_frames, pair_bank, task_target)
    rows = []
    for entry in test_frames:
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

        rec = {
            "seq_name": entry["seq_name"],
            "frame_idx": curr_frame["frame_idx"],
            "other_hand_motion": str(curr_frame["left"].get("hand_motion", "unknown")),
            "interaction_motion_value": str(curr_frame.get("interaction_motion", "unknown")),
        }
        right_row = None if choice is None else choice[1]
        left_row = None if choice is None else choice[2]
        for mode in LEFT_REPAIR_MODES:
            prefix = f"hgb_relax_both_left_{mode}"
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

    prefixes = [f"hgb_relax_both_left_{mode}" for mode in LEFT_REPAIR_MODES]
    return {
        "training_stats": {"hgb": hgb_stats},
        "summary": {prefix: summarize_method(rows, prefix) for prefix in prefixes},
        "sequence_summary": summarize_by_sequence(rows, prefixes),
        "rows": rows,
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
            "goal": "targeted left-repair follow-up on the feasible two-hand subset",
            "tasks": [f"{field}->{target}" for field, target in selected_tasks],
            "right_pool_mode": RIGHT_POOL_MODE,
            "family_budget": FAMILY_BUDGET,
            "depth": DEPTH,
            "left_repair_modes": list(LEFT_REPAIR_MODES),
            "task_filter": args.task_target,
        },
        "task_results": task_results,
    }

    suffix = "" if args.task_target is None else f"_{args.task_target}"
    out_json = GEN / f"interaction_realized_feasible_left_repair_followup{suffix}.json"
    out_md = SUM / f"interaction_realized_feasible_left_repair_followup{suffix}.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction Realized Feasible Left Repair Follow-up",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Feasible-two-hand subset only: keep corrected `relax_both` support fixed and vary only the left-side preserve repair mode.",
        "",
    ]
    for task_name, result in task_results.items():
        lines.extend([
            f"## {task_name}",
            "",
            "| method | right grouped | left preserve | joint overall |",
            "| --- | ---: | ---: | ---: |",
        ])
        for method in [f"hgb_relax_both_left_{m}" for m in LEFT_REPAIR_MODES]:
            s = result["summary"][method]
            lines.append(
                f"| {method} | {fmt(s['right_grouped_match_overall'])} | {fmt(s['left_preserve_overall'])} | {fmt(s['joint_score_overall'])} |"
            )
        lines.extend([
            "",
            "### Sequence Summary",
            "",
            "| sequence | none | edge-transition | finger-profile | full-curr-oracle |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for row in result["sequence_summary"]:
            lines.append(
                f"| {row['seq_name']} | {fmt(row['hgb_relax_both_left_none_joint_score_overall'])} | "
                f"{fmt(row['hgb_relax_both_left_edge_transition_snap_joint_score_overall'])} | "
                f"{fmt(row['hgb_relax_both_left_finger_profile_snap_joint_score_overall'])} | "
                f"{fmt(row['hgb_relax_both_left_full_curr_oracle_joint_score_overall'])} |"
            )
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
