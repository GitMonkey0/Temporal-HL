#!/usr/bin/env python3
"""Support-scaling study for the interaction-aware realized editor.

Goal:

- test whether the current low absolute interaction-aware joint score is mainly
  a candidate-support bottleneck

We vary:

- preserve-donor family distance budget
- preserve-donor search depth
- base ordering vs pair-guided ordering

on the two hard right-hand interaction tasks.
"""

from __future__ import annotations

import json
from pathlib import Path

from tools.build_interaction_realized_pairguided_editor import (
    DEPTHS,
    GEN,
    SUM,
    TASKS,
    build_pair_bank,
    build_semantic_frame_vocab,
    candidate_pool_for_task,
    canonical,
    choose_best_split,
    collect_slice_frames,
    evaluate_selected_edit,
    fmt,
    load_json,
    overlap_labels,
    pick_best_symbolic_pair,
    relaxed_left_family_candidates_with_meta,
    run_task,  # not used directly but keeps import surface explicit
    select_pairguided_left_pool,
    summarize_by_subtype,
    summarize_method,
    train_pairguided_model,
)


ROOT = Path("/opt/tiger/hand")
FAMILY_BUDGETS = (1, 2)
DEPTHS = (10, 20)


def main():
    train_data = load_json(GEN / "temporal_hl_val.json")
    test_data = load_json(GEN / "temporal_hl_test.json")
    labels = overlap_labels(train_data, test_data)
    semantic_vocab = build_semantic_frame_vocab(train_data, labels)
    pair_bank = build_pair_bank(train_data, labels, semantic_vocab)

    task_results = {}
    for task_field, task_target in TASKS:
        train_frames = [
            row for row in collect_slice_frames(train_data, task_field, task_target)
            if canonical(row["seq_name"]) in labels or row["seq_name"] in labels
        ]
        test_frames = [
            row for row in collect_slice_frames(test_data, task_field, task_target)
            if canonical(row["seq_name"]) in labels or row["seq_name"] in labels
        ]
        model, train_stats = train_pairguided_model(train_frames, pair_bank, task_target)

        rows = []
        for entry in test_frames:
            prev_frame = entry["prev_frame"]
            curr_frame = entry["curr_frame"]
            prev_geom = entry["prev_geom"]
            curr_geom = entry["curr_geom"]
            curr_attrs = entry["curr_attrs"]
            current_left_group = entry["current_opp_group"]

            rec = {
                "seq_name": entry["seq_name"],
                "frame_idx": curr_frame["frame_idx"],
                "other_hand_motion": "none" if curr_frame.get("left") is None else str(curr_frame["left"].get("hand_motion", "unknown")),
                "interaction_motion_value": str(curr_frame.get("interaction_motion", "unknown")),
            }

            single = pick_best_symbolic_pair(pair_bank, curr_attrs, prev_geom, curr_geom, "right_hand_motion", task_target)
            single_eval = evaluate_selected_edit(prev_frame, curr_frame, prev_geom, single, None)
            rec.update({f"single_{k}": v for k, v in single_eval.items()})

            target_pool = candidate_pool_for_task(pair_bank, "right_hand_motion", task_target, curr_attrs, prev_geom, curr_geom)

            for family_budget in FAMILY_BUDGETS:
                base_left_pool = relaxed_left_family_candidates_with_meta(
                    pair_bank,
                    current_left_group,
                    curr_attrs,
                    prev_geom,
                    curr_geom,
                    "left",
                    family_budget,
                )
                pg_left_pool = select_pairguided_left_pool(
                    model,
                    pair_bank,
                    curr_attrs,
                    prev_geom,
                    curr_geom,
                    current_left_group,
                    target_pool,
                )
                if family_budget != 1:
                    # Rebuild pair-guided ordering under the larger support budget.
                    raw_pool = relaxed_left_family_candidates_with_meta(
                        pair_bank,
                        current_left_group,
                        curr_attrs,
                        prev_geom,
                        curr_geom,
                        "left",
                        family_budget,
                    )
                    # pair-guided ordering scores items independently of budgeted slicing,
                    # so keep only the reordered items that also belong to the expanded pool.
                    raw_ids = {(item["row"]["seq_name"], item["row"]["curr_frame_idx"]) for item in raw_pool}
                    pg_left_pool = [
                        item for item in pg_left_pool
                        if (item["row"]["seq_name"], item["row"]["curr_frame_idx"]) in raw_ids
                    ]
                    seen = {(item["row"]["seq_name"], item["row"]["curr_frame_idx"]) for item in pg_left_pool}
                    for item in raw_pool:
                        key = (item["row"]["seq_name"], item["row"]["curr_frame_idx"])
                        if key not in seen:
                            pg_left_pool.append(item)

                for depth in DEPTHS:
                    prefix = f"budget{family_budget}_base_top{depth}"
                    choice = choose_best_split(prev_frame, curr_frame, prev_geom, target_pool, base_left_pool, depth) if target_pool and base_left_pool else None
                    result = evaluate_selected_edit(
                        prev_frame,
                        curr_frame,
                        prev_geom,
                        None if choice is None else choice[1],
                        None if choice is None else choice[2],
                    )
                    rec.update({f"{prefix}_{k}": v for k, v in result.items()})

                    prefix = f"budget{family_budget}_pairguided_top{depth}"
                    choice = choose_best_split(prev_frame, curr_frame, prev_geom, target_pool, pg_left_pool, depth) if target_pool and pg_left_pool else None
                    result = evaluate_selected_edit(
                        prev_frame,
                        curr_frame,
                        prev_geom,
                        None if choice is None else choice[1],
                        None if choice is None else choice[2],
                    )
                    rec.update({f"{prefix}_{k}": v for k, v in result.items()})

            rows.append(rec)

        prefixes = ["single"]
        for family_budget in FAMILY_BUDGETS:
            for depth in DEPTHS:
                prefixes.append(f"budget{family_budget}_base_top{depth}")
                prefixes.append(f"budget{family_budget}_pairguided_top{depth}")

        task_results[f"{task_field}->{task_target}"] = {
            "training_stats": train_stats,
            "summary": {prefix: summarize_method(rows, prefix) for prefix in prefixes},
            "subtype_summary": summarize_by_subtype(rows, prefixes),
            "rows": rows,
        }

    payload = {
        "artifacts": {
            "train_json": str(GEN / "temporal_hl_val.json"),
            "test_json": str(GEN / "temporal_hl_test.json"),
        },
        "focus": {
            "tasks": [f"{field}->{target}" for field, target in TASKS],
            "slice": "interaction only",
            "goal": "test whether expanding preserve support and depth improves absolute realized interaction editing",
            "family_budgets": list(FAMILY_BUDGETS),
            "depths": list(DEPTHS),
        },
        "task_results": task_results,
    }

    out_json = GEN / "interaction_realized_support_scaling.json"
    out_md = SUM / "interaction_realized_support_scaling.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction Realized Support Scaling",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Focus: hard right-hand interaction slices.",
        "",
        "Vary:",
        "",
        "- preserve-donor family budget",
        "- search depth",
        "- base vs pair-guided ordering",
        "",
    ]

    for task_name, result in task_results.items():
        lines.extend(
            [
                f"## {task_name}",
                "",
                "| method | avail | right grouped | left preserve | joint on avail | joint overall |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for method, stats in result["summary"].items():
            lines.append(
                f"| {method} | {fmt(stats['available_rate'])} | {fmt(stats['right_grouped_match_on_available'])} | "
                f"{fmt(stats['left_preserve_on_available'])} | {fmt(stats['joint_score_on_available'])} | {fmt(stats['joint_score_overall'])} |"
            )
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
