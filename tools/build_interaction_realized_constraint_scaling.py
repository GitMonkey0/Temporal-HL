#!/usr/bin/env python3
"""Scale the new constrained-realization editor on hard right-hand slices.

This is an experiment memo, not paper text.

The goal is to harden the new realization-stage gain by testing whether the
best repair mechanism remains useful across:

- selector: base / hgb / mlp
- preserve support budget: 1 / 2
- search depth: 5 / 10 / 20

The core question is whether constrained preserve-hand repair is only a local
top-10 effect or a stable mechanism layer.
"""

from __future__ import annotations

import json
from collections import defaultdict

from tools.build_interaction_realized_constraint_sweep import (
    evaluate_edit,
    fmt,
    mlp_ranked_left_pool,
)
from tools.build_interaction_realized_mechanism_sweep import train_mlp_model
from tools.build_interaction_realized_pairguided_editor import (
    TASKS,
    choose_best_split,
    select_pairguided_left_pool,
    train_pairguided_model,
)
from tools.build_pairguided_reranker_multislice import (
    candidate_pool_for_task,
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
    pick_best_symbolic_pair,
)


REPAIR_MODES = ("none", "finger_profile_snap")
SELECTORS = ("base", "hgb", "mlp")
FAMILY_BUDGETS = (1, 2)
DEPTHS = (5, 10, 20)


def summarize_method(rows, prefix: str):
    avail_key = f"{prefix}_available"
    n = len(rows)
    avail = sum(row[avail_key] for row in rows)
    out = {
        "num_frames": n,
        "available_rate": avail / n,
    }
    for key in (
        "right_grouped_match",
        "left_preserve",
        "joint_score",
        "right_state_agreement",
        "right_transition_agreement",
        "left_state_agreement",
        "left_transition_agreement",
    ):
        out[f"{key}_overall"] = sum(row[f"{prefix}_{key}"] for row in rows) / n
        out[f"{key}_on_available"] = sum(row[f"{prefix}_{key}"] for row in rows if row[avail_key]) / max(avail, 1)
    return out


def summarize_by_subtype(rows, prefixes):
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
        for prefix in prefixes:
            summary = summarize_method(items, prefix)
            rec[f"{prefix}_joint_score_overall"] = summary["joint_score_overall"]
        out.append(rec)
    return out


def reorder_to_budget(ranked_pool, raw_pool):
    raw_ids = {(item["row"]["seq_name"], item["row"]["curr_frame_idx"]) for item in raw_pool}
    out = [
        item for item in ranked_pool
        if (item["row"]["seq_name"], item["row"]["curr_frame_idx"]) in raw_ids
    ]
    seen = {(item["row"]["seq_name"], item["row"]["curr_frame_idx"]) for item in out}
    for item in raw_pool:
        key = (item["row"]["seq_name"], item["row"]["curr_frame_idx"])
        if key not in seen:
            out.append(item)
    return out


def run_task(train_frames, test_frames, pair_bank, task_target: str):
    hgb_model, hgb_stats = train_pairguided_model(train_frames, pair_bank, task_target)
    mlp_model, mlp_stats = train_mlp_model(train_frames, pair_bank, seed=0)

    rows = []
    method_specs = [
        (selector, budget, depth, repair)
        for selector in SELECTORS
        for budget in FAMILY_BUDGETS
        for depth in DEPTHS
        for repair in REPAIR_MODES
    ]

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
        single_eval = evaluate_edit(prev_frame, curr_frame, prev_geom, single, None, repair_mode="none")
        rec.update({f"single_{k}": v for k, v in single_eval.items()})

        target_pool = candidate_pool_for_task(pair_bank, "right_hand_motion", task_target, curr_attrs, prev_geom, curr_geom)
        raw_budget_pools = {}
        for budget in FAMILY_BUDGETS:
            raw_budget_pools[budget] = relaxed_left_family_candidates_with_meta(
                pair_bank,
                current_left_group,
                curr_attrs,
                prev_geom,
                curr_geom,
                "left",
                budget,
            )
        hgb_ranked_budget1 = select_pairguided_left_pool(
            hgb_model,
            pair_bank,
            curr_attrs,
            prev_geom,
            curr_geom,
            current_left_group,
            target_pool,
        )
        mlp_ranked_budget1 = mlp_ranked_left_pool(
            mlp_model,
            pair_bank,
            curr_attrs,
            prev_geom,
            curr_geom,
            current_left_group,
            target_pool,
        )

        selector_budget_pool = {}
        for budget in FAMILY_BUDGETS:
            selector_budget_pool[("base", budget)] = raw_budget_pools[budget]
            selector_budget_pool[("hgb", budget)] = reorder_to_budget(hgb_ranked_budget1, raw_budget_pools[budget])
            selector_budget_pool[("mlp", budget)] = reorder_to_budget(mlp_ranked_budget1, raw_budget_pools[budget])

        for selector, budget, depth, repair in method_specs:
            prefix = f"{selector}_budget{budget}_{repair}_top{depth}"
            left_pool = selector_budget_pool[(selector, budget)]
            choice = choose_best_split(prev_frame, curr_frame, prev_geom, target_pool, left_pool, depth) if target_pool and left_pool else None
            eval_rec = evaluate_edit(
                prev_frame,
                curr_frame,
                prev_geom,
                None if choice is None else choice[1],
                None if choice is None else choice[2],
                repair_mode=repair,
            )
            rec.update({f"{prefix}_{k}": v for k, v in eval_rec.items()})
        rows.append(rec)

    prefixes = ["single"] + [
        f"{selector}_budget{budget}_{repair}_top{depth}"
        for selector, budget, depth, repair in method_specs
    ]
    return {
        "training_stats": {
            "hgb": hgb_stats,
            "mlp": mlp_stats,
        },
        "summary": {prefix: summarize_method(rows, prefix) for prefix in prefixes},
        "subtype_summary": summarize_by_subtype(rows, prefixes),
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
        "artifacts": {
            "train_json": str(GEN / "temporal_hl_val.json"),
            "test_json": str(GEN / "temporal_hl_test.json"),
        },
        "focus": {
            "tasks": [f"{field}->{target}" for field, target in TASKS],
            "slice": "interaction only",
            "goal": "test whether the new preserve-hand repair gain is stable across selector, support budget, and depth",
            "selectors": list(SELECTORS),
            "family_budgets": list(FAMILY_BUDGETS),
            "depths": list(DEPTHS),
            "repair_modes": list(REPAIR_MODES),
        },
        "task_results": task_results,
    }

    out_json = GEN / "interaction_realized_constraint_scaling.json"
    out_md = SUM / "interaction_realized_constraint_scaling.md"
    out_json.write_text(json.dumps(payload, indent=2))

    key_methods = [
        "single",
        "hgb_budget1_none_top10",
        "hgb_budget1_finger_profile_snap_top10",
        "hgb_budget2_finger_profile_snap_top10",
        "hgb_budget2_finger_profile_snap_top20",
        "mlp_budget1_none_top10",
        "mlp_budget1_finger_profile_snap_top10",
        "mlp_budget2_finger_profile_snap_top20",
    ]

    lines = [
        "# Interaction Realized Constraint Scaling",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Focus: hard right-hand interaction slices.",
        "",
        "We scale the new constrained-realization gain across selector, support budget, and depth.",
        "",
    ]

    for task_name, result in task_results.items():
        lines.extend(
            [
                f"## {task_name}",
                "",
                "| method | avail | right grouped | left preserve | joint overall |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for method in key_methods:
            stats = result["summary"][method]
            lines.append(
                f"| {method} | {fmt(stats['available_rate'])} | {fmt(stats['right_grouped_match_overall'])} | "
                f"{fmt(stats['left_preserve_overall'])} | {fmt(stats['joint_score_overall'])} |"
            )
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
