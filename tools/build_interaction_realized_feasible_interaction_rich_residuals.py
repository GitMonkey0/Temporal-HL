#!/usr/bin/env python3
"""Audit hard-slice residuals on feasible interaction-rich subtypes.

This is an experiment memo, not paper text.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/opt/tiger/hand")
GEN = ROOT / "experiments/generated"
SUM = ROOT / "experiments/generated/summary_tables"
SOURCE = GEN / "interaction_realized_constraint_scaling.json"

BEST_METHOD = "hgb_budget2_finger_profile_snap_top20"
BASELINE_METHOD = "hgb_budget2_none_top20"


def fmt(x):
    return f"{x:.4f}"


def weighted_metric(rows, method: str, metric: str) -> float:
    total = sum(row["num_frames"] for row in rows)
    if total == 0:
        return 0.0
    key = f"{method}_{metric}"
    return sum(row["num_frames"] * row[key] for row in rows) / total


def slice_rows(rows, *, include_none: bool | None):
    out = []
    for row in rows:
        is_none = row["other_hand_motion"] == "none"
        if include_none is None:
            out.append(row)
        elif include_none and is_none:
            out.append(row)
        elif (not include_none) and (not is_none):
            out.append(row)
    return out


def build_summary(rows):
    return {
        "num_frames": sum(row["num_frames"] for row in rows),
        "baseline_joint": weighted_metric(rows, BASELINE_METHOD, "joint_score_overall"),
        "best_joint": weighted_metric(rows, BEST_METHOD, "joint_score_overall"),
    }


def main():
    payload = json.loads(SOURCE.read_text())
    task_results = {}
    for task_name, task_payload in payload["task_results"].items():
        subtype_rows = task_payload["subtype_summary"]
        all_rows = slice_rows(subtype_rows, include_none=None)
        none_rows = slice_rows(subtype_rows, include_none=True)
        rich_rows = slice_rows(subtype_rows, include_none=False)

        all_summary = build_summary(all_rows)
        none_summary = build_summary(none_rows)
        rich_summary = build_summary(rich_rows)
        task_results[task_name] = {
            "best_method": BEST_METHOD,
            "baseline_method": BASELINE_METHOD,
            "all_subtypes": all_summary,
            "none_mass_only": none_summary,
            "feasible_interaction_rich_only": rich_summary,
            "deltas": {
                "best_minus_baseline_all": all_summary["best_joint"] - all_summary["baseline_joint"],
                "best_minus_baseline_rich": rich_summary["best_joint"] - rich_summary["baseline_joint"],
                "rich_minus_all": rich_summary["best_joint"] - all_summary["best_joint"],
                "rich_to_all_ratio": 0.0 if all_summary["best_joint"] == 0 else rich_summary["best_joint"] / all_summary["best_joint"],
            },
        }

    out_payload = {
        "focus": {
            "goal": "separate feasible interaction-rich residuals from structurally infeasible none-mass on hard right-hand slices",
            "source": str(SOURCE),
            "best_method": BEST_METHOD,
            "baseline_method": BASELINE_METHOD,
        },
        "task_results": task_results,
    }

    out_json = GEN / "interaction_realized_feasible_interaction_rich_residuals.json"
    out_md = SUM / "interaction_realized_feasible_interaction_rich_residuals.md"
    out_json.write_text(json.dumps(out_payload, indent=2))

    lines = [
        "# Interaction Realized Feasible Interaction-Rich Residuals",
        "",
        "This is an experiment memo, not paper text.",
        "",
        f"Source: `{SOURCE}`",
        f"Best method: `{BEST_METHOD}`; baseline: `{BASELINE_METHOD}`",
    ]
    for task_name, task_payload in task_results.items():
        lines.extend(
            [
                "",
                f"## {task_name}",
                "",
                "| slice | frames | baseline joint | best joint | delta |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for label, key in (
            ("all subtypes", "all_subtypes"),
            ("none mass only", "none_mass_only"),
            ("feasible interaction-rich only", "feasible_interaction_rich_only"),
        ):
            rec = task_payload[key]
            delta = rec["best_joint"] - rec["baseline_joint"]
            lines.append(
                f"| {label} | {rec['num_frames']} | {fmt(rec['baseline_joint'])} | {fmt(rec['best_joint'])} | {fmt(delta)} |"
            )
        lines.extend(
            [
                "",
                f"Feasible interaction-rich / all ratio: `{fmt(task_payload['deltas']['rich_to_all_ratio'])}`",
                f"Feasible interaction-rich uplift over all-subtype best: `{fmt(task_payload['deltas']['rich_minus_all'])}`",
            ]
        )

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
