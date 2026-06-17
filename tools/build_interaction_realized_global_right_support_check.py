#!/usr/bin/env python3
"""Global right-support check on full hard right-hand interaction slices.

This is an experiment memo, not paper text.

The goal is to verify whether the right-hand donor pool should stay strict on
the full right-hand hard slices once we already use:

- pair-guided left-pool reranking
- preserve support budget = 2
- constrained preserve-hand repair

This intentionally avoids the unfinished gate path and only answers the
smaller, higher-confidence question:

- on the full right-hand hard slices, does `relax_both` beat `strict` under the
  current strongest realization protocol?
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

from tools.build_interaction_realized_constraint_sweep import evaluate_edit, fmt
from tools.build_interaction_realized_pairguided_editor import (
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


TASKS = (
    ("right_hand_motion", "closing"),
    ("right_hand_motion", "opening"),
)
RIGHT_POOL_MODES = ("strict", "relax_both")
FAMILY_BUDGET = 2
DEPTH = 20
REPAIR_MODE = "finger_profile_snap"


def summarize_method(rows, prefix: str):
    avail_key = f"{prefix}_available"
    n = len(rows)
    avail = sum(row[avail_key] for row in rows)
    out = {"num_frames": n, "available_rate": avail / n}
    for key in ("right_grouped_match", "left_preserve", "joint_score"):
        out[f"{key}_overall"] = sum(row[f"{prefix}_{key}"] for row in rows) / n
        out[f"{key}_on_available"] = sum(row[f"{prefix}_{key}"] for row in rows if row[avail_key]) / max(avail, 1)
    return out


def summarize_by_sequence(rows, prefixes):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["seq_name"]].append(row)
    out = []
    for seq_name, items in sorted(grouped.items()):
        rec = {"seq_name": seq_name, "num_frames": len(items)}
        for prefix in prefixes:
            summary = summarize_method(items, prefix)
            rec[f"{prefix}_joint_score_overall"] = summary["joint_score_overall"]
            rec[f"{prefix}_right_grouped_match_overall"] = summary["right_grouped_match_overall"]
            rec[f"{prefix}_left_preserve_overall"] = summary["left_preserve_overall"]
        out.append(rec)
    return out


def summarize_by_subtype(rows, prefixes):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["other_hand_motion"], row["interaction_motion_value"])].append(row)
    out = []
    for (other_hand_motion, interaction_motion_value), items in sorted(grouped.items()):
        rec = {
            "other_hand_motion": other_hand_motion,
            "interaction_motion_value": interaction_motion_value,
            "num_frames": len(items),
        }
        for prefix in prefixes:
            summary = summarize_method(items, prefix)
            rec[f"{prefix}_joint_score_overall"] = summary["joint_score_overall"]
        out.append(rec)
    return out


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


def run_task(train_frames, test_frames, pair_bank, task_target: str):
    hgb_model, hgb_stats = train_pairguided_model(train_frames, pair_bank, task_target)
    rows = []
    for entry in test_frames:
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
        rec = {
            "seq_name": entry["seq_name"],
            "frame_idx": curr_frame["frame_idx"],
            "other_hand_motion": "none" if curr_frame.get("left") is None else str(curr_frame["left"].get("hand_motion", "unknown")),
            "interaction_motion_value": str(curr_frame.get("interaction_motion", "unknown")),
        }

        joint_cache = {}
        for right_mode in RIGHT_POOL_MODES:
            target_pool = right_candidates_mode(pair_bank, curr_attrs, prev_geom, curr_geom, task_target, right_mode)
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
            prefix = f"hgb_{right_mode}"
            choice = pick_best_from_cache(
                prev_frame,
                curr_frame,
                prev_geom,
                target_pool,
                left_pool,
                DEPTH,
                joint_cache,
            ) if target_pool and left_pool else None
            eval_rec = evaluate_edit(
                prev_frame,
                curr_frame,
                prev_geom,
                None if choice is None else choice[1],
                None if choice is None else choice[2],
                repair_mode=REPAIR_MODE,
            )
            rec.update({f"{prefix}_{k}": v for k, v in eval_rec.items()})
        rows.append(rec)

    prefixes = [f"hgb_{mode}" for mode in RIGHT_POOL_MODES]
    return {
        "training_stats": {"hgb": hgb_stats},
        "summary": {prefix: summarize_method(rows, prefix) for prefix in prefixes},
        "sequence_summary": summarize_by_sequence(rows, prefixes),
        "subtype_summary": summarize_by_subtype(rows, prefixes),
        "rows": rows,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-target",
        choices=[target for _, target in TASKS],
        default=None,
        help="Run only one right-hand hard slice target.",
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
            if canonical(row["seq_name"]) in labels or row["seq_name"] in labels
        ]
        test_frames = [
            row for row in collect_slice_frames(test_data, task_field, task_target)
            if canonical(row["seq_name"]) in labels or row["seq_name"] in labels
        ]
        task_results[f"{task_field}->{task_target}"] = run_task(train_frames, test_frames, pair_bank, task_target)

    payload = {
        "artifacts": {"train_json": str(GEN / "temporal_hl_val.json"), "test_json": str(GEN / "temporal_hl_test.json")},
        "focus": {
            "tasks": [f"{field}->{target}" for field, target in selected_tasks],
            "slice": "interaction only",
            "goal": "check whether relaxed right donor support beats strict support on the full right-hand hard slices",
            "right_pool_modes": list(RIGHT_POOL_MODES),
            "family_budget": FAMILY_BUDGET,
            "depth": DEPTH,
            "left_repair": REPAIR_MODE,
            "task_filter": args.task_target,
        },
        "task_results": task_results,
    }

    suffix = "" if args.task_target is None else f"_{args.task_target}"
    out_json = GEN / f"interaction_realized_global_right_support_check{suffix}.json"
    out_md = SUM / f"interaction_realized_global_right_support_check{suffix}.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction Realized Global Right Support Check",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Compare `strict` and `relax_both` right-hand donor support on the full hard right-hand interaction slices.",
        "",
    ]
    for task_name, result in task_results.items():
        lines.extend([f"## {task_name}", "", "| method | avail | right grouped | left preserve | joint overall |", "| --- | ---: | ---: | ---: | ---: |"])
        for method in ("hgb_strict", "hgb_relax_both"):
            stats = result["summary"][method]
            lines.append(
                f"| {method} | {fmt(stats['available_rate'])} | {fmt(stats['right_grouped_match_overall'])} | "
                f"{fmt(stats['left_preserve_overall'])} | {fmt(stats['joint_score_overall'])} |"
            )
        lines.extend(["", "### Sequence Summary", "", "| sequence | hgb strict | hgb relax-both |", "| --- | ---: | ---: |"])
        for row in result["sequence_summary"]:
            lines.append(
                f"| {row['seq_name']} | {fmt(row['hgb_strict_joint_score_overall'])} | {fmt(row['hgb_relax_both_joint_score_overall'])} |"
            )
        lines.extend(["", "### Subtype Summary", "", "| other hand | interaction | n | hgb strict | hgb relax-both |", "| --- | --- | ---: | ---: | ---: |"])
        for row in result["subtype_summary"]:
            lines.append(
                f"| {row['other_hand_motion']} | {row['interaction_motion_value']} | {row['num_frames']} | "
                f"{fmt(row['hgb_strict_joint_score_overall'])} | {fmt(row['hgb_relax_both_joint_score_overall'])} |"
            )
        lines.append("")
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
