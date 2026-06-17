#!/usr/bin/env python3
"""Feasibility-aware audit for corrected hard-slice interaction editing.

This is an experiment memo, not paper text.

The current strict joint metric requires:

- right-hand grouped target realization
- opposite-hand grouped preservation

However, a large portion of the corrected hard right-hand slices have
`curr_frame.left is None`, which makes opposite-hand preservation structurally
undefined rather than merely unsuccessful.

This audit does not replace the main metric. It separates:

- full reported joint score
- feasible-two-hand subset where the opposite hand exists
- feasibility-aware score that counts only right-target realization when the
  opposite hand is absent
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


ROOT = Path("/opt/tiger/hand")
GEN = ROOT / "experiments/generated"
SUM = GEN / "summary_tables"
SOURCE = GEN / "interaction_realized_global_right_support_bundle.json"
METHODS = ("hgb_strict", "hgb_relax_both")


def load_json(path: Path):
    return json.loads(path.read_text())


def fmt(x: float) -> str:
    return f"{x:.4f}"


def feasible_metrics(rows, prefix: str):
    n = len(rows)
    feasible = [r for r in rows if r["other_hand_motion"] != "none"]
    infeasible = [r for r in rows if r["other_hand_motion"] == "none"]

    def avg(items, field):
        return sum(float(r[f"{prefix}_{field}"]) for r in items) / max(len(items), 1)

    feasibility_aware = []
    for r in rows:
        right = int(r[f"{prefix}_right_grouped_match"])
        left_exists = r["other_hand_motion"] != "none"
        left = int(r[f"{prefix}_left_preserve"])
        feasibility_aware.append(right * left if left_exists else right)

    return {
        "num_frames": n,
        "feasible_frames": len(feasible),
        "feasible_rate": len(feasible) / n,
        "reported_joint_overall": avg(rows, "joint_score"),
        "right_grouped_overall": avg(rows, "right_grouped_match"),
        "left_preserve_overall": avg(rows, "left_preserve"),
        "feasible_joint_overall": avg(feasible, "joint_score"),
        "feasible_right_grouped_overall": avg(feasible, "right_grouped_match"),
        "feasible_left_preserve_overall": avg(feasible, "left_preserve"),
        "infeasible_right_grouped_overall": avg(infeasible, "right_grouped_match"),
        "feasibility_aware_overall": sum(feasibility_aware) / n,
    }


def summarize_by_sequence(rows, prefix: str):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["seq_name"]].append(row)
    out = []
    for seq_name, items in sorted(grouped.items()):
        stats = feasible_metrics(items, prefix)
        out.append({"seq_name": seq_name, **stats})
    return out


def summarize_by_subtype(rows, prefix: str):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["other_hand_motion"], row["interaction_motion_value"])].append(row)
    out = []
    for (other_hand_motion, interaction_motion_value), items in sorted(grouped.items()):
        stats = feasible_metrics(items, prefix)
        out.append(
            {
                "other_hand_motion": other_hand_motion,
                "interaction_motion_value": interaction_motion_value,
                **stats,
            }
        )
    return out


def main():
    source = load_json(SOURCE)
    task_results = {}
    for task_name, result in source["task_results"].items():
        rows = result["rows"]
        task_results[task_name] = {
            "summary": {method: feasible_metrics(rows, method) for method in METHODS},
            "sequence_summary": {method: summarize_by_sequence(rows, method) for method in METHODS},
            "subtype_summary": {method: summarize_by_subtype(rows, method) for method in METHODS},
        }

    payload = {
        "artifacts": {"source": str(SOURCE)},
        "focus": {
            "goal": "separate structural infeasibility from true editing failure on corrected hard right-hand slices",
            "methods": list(METHODS),
            "feasibility_rule": "opposite-hand preservation is only required when the current frame contains a left hand",
        },
        "task_results": task_results,
    }

    out_json = GEN / "interaction_realized_feasibility_audit.json"
    out_md = SUM / "interaction_realized_feasibility_audit.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction Realized Feasibility Audit",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Separate the full joint metric from the feasible two-hand subset and a feasibility-aware score.",
        "",
    ]
    for task_name, result in task_results.items():
        lines.extend([
            f"## {task_name}",
            "",
            "| method | feasible rate | reported joint | feasible joint | infeasible right | feasibility-aware |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for method in METHODS:
            stats = result["summary"][method]
            lines.append(
                f"| {method} | {fmt(stats['feasible_rate'])} | {fmt(stats['reported_joint_overall'])} | "
                f"{fmt(stats['feasible_joint_overall'])} | {fmt(stats['infeasible_right_grouped_overall'])} | "
                f"{fmt(stats['feasibility_aware_overall'])} |"
            )
        lines.extend([
            "",
            "### Sequence Summary (`hgb_relax_both`)",
            "",
            "| sequence | feasible rate | reported joint | feasible joint | infeasible right | feasibility-aware |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in result["sequence_summary"]["hgb_relax_both"]:
            lines.append(
                f"| {row['seq_name']} | {fmt(row['feasible_rate'])} | {fmt(row['reported_joint_overall'])} | "
                f"{fmt(row['feasible_joint_overall'])} | {fmt(row['infeasible_right_grouped_overall'])} | "
                f"{fmt(row['feasibility_aware_overall'])} |"
            )
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
