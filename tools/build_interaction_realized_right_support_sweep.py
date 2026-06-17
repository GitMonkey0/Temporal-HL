#!/usr/bin/env python3
"""Sweep right-target donor support relaxations on hard interaction slices.

This is an experiment memo, not paper text.

The current strongest editor still leaves specific hard sequence families weak,
especially ROM09. This sweep tests whether the right-hand donor pool is overly
constrained.
"""

from __future__ import annotations

import json
from collections import defaultdict

from tools.build_interaction_realized_constraint_sweep import evaluate_edit, fmt
from tools.build_interaction_realized_pairguided_editor import (
    TASKS,
    choose_best_split,
    select_pairguided_left_pool,
    train_pairguided_model,
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
from tools.build_weak_slice_split_donor_prototype import right_relaxed_candidates


RIGHT_POOL_MODES = ("strict", "relax_both")
FAMILY_BUDGET = 2
DEPTH = 20
TARGET_SEQS = {"ROM02_Interaction_2_Hand", "ROM09_Interaction_Fingers_Touching"}


def right_candidates_mode(pair_bank, curr_attrs, prev_geom, curr_geom, task_target: str, mode: str):
    from tools.build_transition_conditioned_symbolic_editor import pair_distance, pair_realizes_target
    out = []
    for row in pair_bank:
        attrs = row["curr_attrs"]
        if attrs["right_hand_motion"] != task_target:
            continue
        if attrs["hand_type"] != curr_attrs["hand_type"]:
            continue
        if mode in ("strict", "relax_left_motion"):
            if attrs["interaction_motion"] != curr_attrs["interaction_motion"]:
                continue
        if mode in ("strict", "relax_interaction"):
            if attrs["left_hand_motion"] != curr_attrs["left_hand_motion"]:
                continue
        if not pair_realizes_target(row, "right_hand_motion", task_target):
            continue
        out.append(row)
    out.sort(key=lambda row: (pair_distance(prev_geom, curr_geom, row, "right_hand_motion"), row["seq_name"], row["curr_frame_idx"]))
    return out


def reorder_to_budget(ranked_pool, raw_pool):
    raw_ids = {(item["row"]["seq_name"], item["row"]["curr_frame_idx"]) for item in raw_pool}
    out = [item for item in ranked_pool if (item["row"]["seq_name"], item["row"]["curr_frame_idx"]) in raw_ids]
    seen = {(item["row"]["seq_name"], item["row"]["curr_frame_idx"]) for item in out}
    for item in raw_pool:
        key = (item["row"]["seq_name"], item["row"]["curr_frame_idx"])
        if key not in seen:
            out.append(item)
    return out


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
        out.append(rec)
    return out


def run_task(train_frames, test_frames, pair_bank, task_target: str):
    hgb_model, hgb_stats = train_pairguided_model(train_frames, pair_bank, task_target)
    rows = []
    method_specs = [("hgb", right_mode) for right_mode in RIGHT_POOL_MODES]

    for entry in test_frames:
        prev_frame = entry["prev_frame"]
        curr_frame = entry["curr_frame"]
        prev_geom = entry["prev_geom"]
        curr_geom = entry["curr_geom"]
        curr_attrs = entry["curr_attrs"]
        current_left_group = entry["current_opp_group"]

        raw_left_pool = relaxed_left_family_candidates_with_meta(
            pair_bank, current_left_group, curr_attrs, prev_geom, curr_geom, "left", FAMILY_BUDGET
        )
        rec = {
            "seq_name": entry["seq_name"],
            "frame_idx": curr_frame["frame_idx"],
            "other_hand_motion": "none" if curr_frame.get("left") is None else str(curr_frame["left"].get("hand_motion", "unknown")),
            "interaction_motion_value": str(curr_frame.get("interaction_motion", "unknown")),
        }

        for selector_name, right_mode in method_specs:
            target_pool = right_candidates_mode(pair_bank, curr_attrs, prev_geom, curr_geom, task_target, right_mode)
            ranked = select_pairguided_left_pool(hgb_model, pair_bank, curr_attrs, prev_geom, curr_geom, current_left_group, target_pool)
            left_pool = reorder_to_budget(ranked, raw_left_pool)

            prefix = f"{selector_name}_{right_mode}"
            choice = choose_best_split(prev_frame, curr_frame, prev_geom, target_pool, left_pool, DEPTH) if target_pool and left_pool else None
            eval_rec = evaluate_edit(
                prev_frame, curr_frame, prev_geom,
                None if choice is None else choice[1],
                None if choice is None else choice[2],
                repair_mode="finger_profile_snap",
            )
            rec.update({f"{prefix}_{k}": v for k, v in eval_rec.items()})
        rows.append(rec)

    prefixes = [f"{selector}_{right_mode}" for selector, right_mode in method_specs]
    return {
        "training_stats": {"hgb": hgb_stats},
        "summary": {prefix: summarize_method(rows, prefix) for prefix in prefixes},
        "sequence_summary": summarize_by_sequence(rows, prefixes),
        "rows": rows,
    }


def main():
    train_data = load_json(GEN / "temporal_hl_val.json")
    test_data = load_json(GEN / "temporal_hl_test.json")
    labels = overlap_labels(train_data, test_data)
    semantic_vocab = build_semantic_frame_vocab(train_data, labels)
    pair_bank = build_pair_bank(train_data, labels, semantic_vocab)

    task_results = {}
    for task_field, task_target in TASKS:
        train_frames = [row for row in collect_slice_frames(train_data, task_field, task_target) if (canonical(row["seq_name"]) in labels or row["seq_name"] in labels) and row["seq_name"] in TARGET_SEQS]
        test_frames = [row for row in collect_slice_frames(test_data, task_field, task_target) if (canonical(row["seq_name"]) in labels or row["seq_name"] in labels) and row["seq_name"] in TARGET_SEQS]
        task_results[f"{task_field}->{task_target}"] = run_task(train_frames, test_frames, pair_bank, task_target)

    payload = {
        "artifacts": {"train_json": str(GEN / "temporal_hl_val.json"), "test_json": str(GEN / "temporal_hl_test.json")},
        "focus": {
            "tasks": [f"{field}->{target}" for field, target in TASKS],
            "slice": "interaction only",
            "goal": "test whether weak hard-sequence families need a more relaxed right-hand donor pool",
            "right_pool_modes": list(RIGHT_POOL_MODES),
            "family_budget": FAMILY_BUDGET,
            "depth": DEPTH,
            "left_repair": "finger_profile_snap",
        },
        "task_results": task_results,
    }

    out_json = GEN / "interaction_realized_right_support_sweep.json"
    out_md = SUM / "interaction_realized_right_support_sweep.md"
    out_json.write_text(json.dumps(payload, indent=2))

    key_methods = [
        "hgb_strict",
        "hgb_relax_both",
    ]
    lines = [
        "# Interaction Realized Right Support Sweep",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Left side fixed to budget2 + finger-profile repair; vary right-hand donor support rules on weak sequence families only.",
        "",
    ]
    for task_name, result in task_results.items():
        lines.extend([f"## {task_name}", "", "| method | avail | right grouped | left preserve | joint overall |", "| --- | ---: | ---: | ---: | ---: |"])
        for method in key_methods:
            stats = result["summary"][method]
            lines.append(f"| {method} | {fmt(stats['available_rate'])} | {fmt(stats['right_grouped_match_overall'])} | {fmt(stats['left_preserve_overall'])} | {fmt(stats['joint_score_overall'])} |")
        lines.extend(["", "### Sequence Summary", "", "| sequence | hgb strict | hgb relax-both |", "| --- | ---: | ---: |"])
        for row in result["sequence_summary"]:
            lines.append(
                f"| {row['seq_name']} | {fmt(row['hgb_strict_joint_score_overall'])} | {fmt(row['hgb_relax_both_joint_score_overall'])} |"
            )
        lines.append("")
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
