#!/usr/bin/env python3
"""Sequence-level concentration report for the current hard-slice frontier.

This is an experiment memo, not paper text.

It checks whether the current strongest interaction-aware realized gains are
spread across sequences or dominated by one recurring sequence family.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("/opt/tiger/hand")
GEN = ROOT / "experiments/generated"
SUM = GEN / "summary_tables"

COMPARISONS = [
    ("hgb_budget1_none_top10", "hgb_budget1_finger_profile_snap_top10"),
    ("hgb_budget1_finger_profile_snap_top10", "hgb_budget2_finger_profile_snap_top10"),
    ("hgb_budget1_none_top10", "hgb_budget2_finger_profile_snap_top10"),
]


def load_json(path: Path):
    return json.loads(path.read_text())


def fmt(x: float) -> str:
    return f"{x:.4f}"


def mean_field(rows, prefix: str, field: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([float(row[f"{prefix}_{field}"]) for row in rows]))


def summarize_sequence_blocks(rows, a_name: str, b_name: str):
    by_seq = defaultdict(list)
    for row in rows:
        by_seq[row["seq_name"]].append(row)

    total_delta = mean_field(rows, b_name, "joint_score") - mean_field(rows, a_name, "joint_score")
    seq_rows = []
    for seq_name, items in sorted(by_seq.items()):
        a_mean = mean_field(items, a_name, "joint_score")
        b_mean = mean_field(items, b_name, "joint_score")
        delta = b_mean - a_mean
        seq_rows.append(
            {
                "seq_name": seq_name,
                "num_frames": len(items),
                "mean_a": a_mean,
                "mean_b": b_mean,
                "delta": delta,
                "weighted_delta_contribution": delta * len(items) / max(len(rows), 1),
            }
        )

    ranked = sorted(seq_rows, key=lambda row: abs(row["weighted_delta_contribution"]), reverse=True)
    cumulative = 0.0
    concentration = []
    for idx, row in enumerate(ranked, start=1):
        cumulative += row["weighted_delta_contribution"]
        share = 0.0 if total_delta == 0 else cumulative / total_delta
        concentration.append(
            {
                "rank": idx,
                "seq_name": row["seq_name"],
                "cumulative_delta": cumulative,
                "cumulative_share_of_total": share,
            }
        )

    leave_one_out = []
    seq_names = sorted(by_seq)
    for seq_name in seq_names:
        kept = [row for row in rows if row["seq_name"] != seq_name]
        loo_delta = mean_field(kept, b_name, "joint_score") - mean_field(kept, a_name, "joint_score")
        leave_one_out.append(
            {
                "removed_seq": seq_name,
                "remaining_frames": len(kept),
                "delta_without_seq": loo_delta,
            }
        )

    return {
        "overall_delta": total_delta,
        "num_sequences": len(by_seq),
        "sequence_rows": seq_rows,
        "concentration_curve": concentration,
        "leave_one_out": leave_one_out,
    }


def subtype_by_sequence(rows, a_name: str, b_name: str):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["seq_name"], row["other_hand_motion"], row["interaction_motion_value"])].append(row)
    out = []
    for key, items in sorted(grouped.items()):
        out.append(
            {
                "seq_name": key[0],
                "other_hand_motion": key[1],
                "interaction_motion_value": key[2],
                "num_frames": len(items),
                "delta": mean_field(items, b_name, "joint_score") - mean_field(items, a_name, "joint_score"),
            }
        )
    return out


def main():
    payload_in = load_json(GEN / "interaction_realized_constraint_scaling.json")

    task_results = {}
    for task_name, result in payload_in["task_results"].items():
        rows = result["rows"]
        comps = {}
        for a_name, b_name in COMPARISONS:
            comp_key = f"{a_name}__to__{b_name}"
            comps[comp_key] = {
                "sequence_joint_score": summarize_sequence_blocks(rows, a_name, b_name),
                "subtype_by_sequence_joint_score": subtype_by_sequence(rows, a_name, b_name),
            }
        task_results[task_name] = comps

    payload = {
        "artifacts": {
            "source": str(GEN / "interaction_realized_constraint_scaling.json"),
        },
        "focus": {
            "goal": "test whether current hard-slice gains are sequence-distributed rather than dominated by a single sequence",
            "comparisons": [f"{a} -> {b}" for a, b in COMPARISONS],
        },
        "task_results": task_results,
    }

    out_json = GEN / "interaction_realized_sequence_concentration.json"
    out_md = SUM / "interaction_realized_sequence_concentration.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction Realized Sequence Concentration",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Sequence-level concentration and leave-one-out analysis for the current hard-slice frontier.",
        "",
    ]
    for task_name, comps in task_results.items():
        lines.extend([f"## {task_name}", ""])
        for comp_name, comp in comps.items():
            seq_summary = comp["sequence_joint_score"]
            lines.extend(
                [
                    f"### {comp_name.replace('__to__', ' -> ')}",
                    "",
                    f"Overall delta: `{fmt(seq_summary['overall_delta'])}` across `{seq_summary['num_sequences']}` sequences.",
                    "",
                    "| sequence | frames | mean A | mean B | delta | weighted contribution |",
                    "| --- | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for row in sorted(seq_summary["sequence_rows"], key=lambda r: r["delta"], reverse=True):
                lines.append(
                    f"| {row['seq_name']} | {row['num_frames']} | {fmt(row['mean_a'])} | {fmt(row['mean_b'])} | "
                    f"{fmt(row['delta'])} | {fmt(row['weighted_delta_contribution'])} |"
                )
            lines.extend(
                [
                    "",
                    "| removed sequence | remaining frames | delta without sequence |",
                    "| --- | ---: | ---: |",
                ]
            )
            for row in seq_summary["leave_one_out"]:
                lines.append(
                    f"| {row['removed_seq']} | {row['remaining_frames']} | {fmt(row['delta_without_seq'])} |"
                )
            lines.append("")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
