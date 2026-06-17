#!/usr/bin/env python3
"""Template-conditioned feasible left-repair selector.

This is an experiment memo, not paper text.

After framewise local selectors failed, test whether a coarser temporal /
template context helps:

- choose a single left-repair mode from feasible `val`
- condition on:
  - sequence template only
  - subtype only
  - sequence template + subtype
- evaluate on feasible `test`
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


def fmt(x: float) -> str:
    return f"{x:.4f}"


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


def row_key_seq(row):
    return row["seq_name"]


def row_key_subtype(row):
    return (row["other_hand_motion"], row["interaction_motion_value"])


def row_key_seq_subtype(row):
    return (row["seq_name"], row["other_hand_motion"], row["interaction_motion_value"])


def best_mode_for_items(items):
    means = {
        mode: sum(float(item[f"left_{mode}_joint_score"]) for item in items) / len(items)
        for mode in LEFT_REPAIR_MODES
    }
    return max(means, key=means.get), means


def fit_rule(rows, key_fn):
    grouped = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(row)
    mapping = {}
    stats = {}
    for key, items in grouped.items():
        mode, means = best_mode_for_items(items)
        mapping[key] = mode
        stats[str(key)] = {
            "num_frames": len(items),
            "means": means,
            "best_mode": mode,
        }
    return mapping, stats


def apply_rule(rows, mapping, key_fn, fallback: str):
    chosen = []
    counts = Counter()
    for row in rows:
        mode = mapping.get(key_fn(row), fallback)
        chosen.append(mode)
        counts[mode] += 1
    return chosen, dict(counts)


def build_frames(data, labels, task_target: str):
    return [
        row
        for row in collect_slice_frames(data, "right_hand_motion", task_target)
        if (canonical(row["seq_name"]) in labels or row["seq_name"] in labels) and row["curr_frame"].get("left") is not None
    ]


def enrich_rows(raw_rows, chosen_modes_by_method, fixed_mode: str):
    rows = []
    for idx, raw in enumerate(raw_rows):
        row = dict(raw)
        oracle_mode, _ = best_mode_for_items([raw])
        for name, mode in [
            ("fixed_none", "none"),
            ("fixed_task_best", fixed_mode),
            ("seq_gate", chosen_modes_by_method["seq_gate"][idx]),
            ("subtype_gate", chosen_modes_by_method["subtype_gate"][idx]),
            ("seq_subtype_gate", chosen_modes_by_method["seq_subtype_gate"][idx]),
            ("oracle", oracle_mode),
        ]:
            prefix = f"left_{mode}"
            for field in ("right_grouped_match", "left_preserve", "joint_score"):
                row[f"{name}_{field}"] = row[f"{prefix}_{field}"]
        row["seq_gate_mode"] = chosen_modes_by_method["seq_gate"][idx]
        row["subtype_gate_mode"] = chosen_modes_by_method["subtype_gate"][idx]
        row["seq_subtype_gate_mode"] = chosen_modes_by_method["seq_subtype_gate"][idx]
        row["oracle_mode"] = oracle_mode
        rows.append(row)
    return rows


def run_task(train_data, test_data, labels, pair_bank, task_target: str):
    train_frames = build_frames(train_data, labels, task_target)
    test_frames = build_frames(test_data, labels, task_target)
    pair_model, pair_stats = train_pairguided_model(train_frames, pair_bank, task_target)
    train_rows = collect_rows(train_frames, pair_bank, task_target, pair_model)
    test_rows = collect_rows(test_frames, pair_bank, task_target, pair_model)

    seq_map, seq_fit_stats = fit_rule(train_rows, row_key_seq)
    subtype_map, subtype_fit_stats = fit_rule(train_rows, row_key_subtype)
    seq_subtype_map, seq_subtype_fit_stats = fit_rule(train_rows, row_key_seq_subtype)

    fixed_mode = TASK_BEST[task_target]
    chosen_modes_by_method = {}
    chosen_modes_by_method["seq_gate"], seq_counts = apply_rule(test_rows, seq_map, row_key_seq, fixed_mode)
    chosen_modes_by_method["subtype_gate"], subtype_counts = apply_rule(test_rows, subtype_map, row_key_subtype, fixed_mode)
    chosen_modes_by_method["seq_subtype_gate"], seq_subtype_counts = apply_rule(
        test_rows, seq_subtype_map, row_key_seq_subtype, fixed_mode
    )

    rows = enrich_rows(test_rows, chosen_modes_by_method, fixed_mode)
    summary = {
        name: summarize(rows, name)
        for name in ("fixed_none", "fixed_task_best", "seq_gate", "subtype_gate", "seq_subtype_gate", "oracle")
    }
    paired = {
        "seq_gate_vs_fixed_task_best": paired_stats(rows, "fixed_task_best", "seq_gate"),
        "subtype_gate_vs_fixed_task_best": paired_stats(rows, "fixed_task_best", "subtype_gate"),
        "seq_subtype_gate_vs_fixed_task_best": paired_stats(rows, "fixed_task_best", "seq_subtype_gate"),
    }
    return {
        "training_stats": {
            "pair_model": pair_stats,
            "seq_gate": {"mapping": {str(k): v for k, v in seq_map.items()}, "fit_stats": seq_fit_stats},
            "subtype_gate": {"mapping": {str(k): v for k, v in subtype_map.items()}, "fit_stats": subtype_fit_stats},
            "seq_subtype_gate": {
                "mapping": {str(k): v for k, v in seq_subtype_map.items()},
                "fit_stats": seq_subtype_fit_stats,
            },
        },
        "test_pred_mode_counts": {
            "seq_gate": seq_counts,
            "subtype_gate": subtype_counts,
            "seq_subtype_gate": seq_subtype_counts,
        },
        "summary": summary,
        "paired": paired,
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

    selected_targets = [target for _, target in TASKS if args.task_target is None or args.task_target == target]
    selected_targets = sorted(set(selected_targets))
    task_results = {}
    for task_target in selected_targets:
        task_results[f"right_hand_motion->{task_target}"] = run_task(train_data, test_data, labels, pair_bank, task_target)

    payload = {
        "artifacts": {"train_json": str(GEN / "temporal_hl_val.json"), "test_json": str(GEN / "temporal_hl_test.json")},
        "focus": {
            "goal": "template-conditioned feasible left-repair selector",
            "tasks": [f"right_hand_motion->{target}" for target in selected_targets],
            "repair_modes": list(LEFT_REPAIR_MODES),
        },
        "task_results": task_results,
    }

    suffix = "" if args.task_target is None else f"_{args.task_target}"
    out_json = GEN / f"interaction_realized_feasible_left_template_gate{suffix}.json"
    out_md = SUM / f"interaction_realized_feasible_left_template_gate{suffix}.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction Realized Feasible Left Template Gate",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Template-conditioned feasible left-repair selection using sequence template, subtype, and sequence-template-plus-subtype rules fit on feasible `val` and evaluated on feasible `test`.",
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
        for method in ("fixed_none", "fixed_task_best", "seq_gate", "subtype_gate", "seq_subtype_gate", "oracle"):
            stats = result["summary"][method]
            lines.append(
                f"| {method} | {fmt(stats['right_grouped_match_overall'])} | {fmt(stats['left_preserve_overall'])} | {fmt(stats['joint_score_overall'])} |"
            )
        lines.extend(
            [
                "",
                "| comparison | delta | wins | losses | ties |",
                "| --- | ---: | ---: | ---: | ---: |",
                f"| seq_gate vs fixed_task_best | {fmt(result['paired']['seq_gate_vs_fixed_task_best']['delta'])} | {result['paired']['seq_gate_vs_fixed_task_best']['wins']} | {result['paired']['seq_gate_vs_fixed_task_best']['losses']} | {result['paired']['seq_gate_vs_fixed_task_best']['ties']} |",
                f"| subtype_gate vs fixed_task_best | {fmt(result['paired']['subtype_gate_vs_fixed_task_best']['delta'])} | {result['paired']['subtype_gate_vs_fixed_task_best']['wins']} | {result['paired']['subtype_gate_vs_fixed_task_best']['losses']} | {result['paired']['subtype_gate_vs_fixed_task_best']['ties']} |",
                f"| seq_subtype_gate vs fixed_task_best | {fmt(result['paired']['seq_subtype_gate_vs_fixed_task_best']['delta'])} | {result['paired']['seq_subtype_gate_vs_fixed_task_best']['wins']} | {result['paired']['seq_subtype_gate_vs_fixed_task_best']['losses']} | {result['paired']['seq_subtype_gate_vs_fixed_task_best']['ties']} |",
                "",
                "### Test Mode Counts",
                "",
                f"- `seq_gate`: {result['test_pred_mode_counts']['seq_gate']}",
                f"- `subtype_gate`: {result['test_pred_mode_counts']['subtype_gate']}",
                f"- `seq_subtype_gate`: {result['test_pred_mode_counts']['seq_subtype_gate']}",
                "",
            ]
        )

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
