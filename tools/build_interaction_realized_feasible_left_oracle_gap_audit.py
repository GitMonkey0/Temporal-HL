#!/usr/bin/env python3
"""Decompose feasible-left oracle headroom by sequence, subtype, and mode conflict.

This is an experiment memo, not paper text.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/opt/tiger/hand")
GEN = ROOT / "experiments/generated"
SUM = GEN / "summary_tables"
TASK_FILES = {
    "right_hand_motion->closing": GEN / "interaction_realized_feasible_left_repair_model_gate_closing.json",
    "right_hand_motion->opening": GEN / "interaction_realized_feasible_left_repair_model_gate_opening.json",
}


def fmt(x: float) -> str:
    return f"{x:.4f}"


def summarize_rows(rows):
    n = len(rows)
    fixed = sum(float(r["fixed_task_best_joint_score"]) for r in rows) / max(n, 1)
    oracle = sum(float(r["oracle_joint_score"]) for r in rows) / max(n, 1)
    return {
        "num_frames": n,
        "fixed_task_best": fixed,
        "oracle": oracle,
        "delta": oracle - fixed,
    }


def weighted_by_group(rows, key_fn):
    grouped = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(row)
    out = []
    for key, items in sorted(grouped.items(), key=lambda kv: (len(kv[1]), str(kv[0])), reverse=True):
        stats = summarize_rows(items)
        rec = {"group": key, **stats}
        out.append(rec)
    return out


def main():
    task_results = {}
    for task_name, path in TASK_FILES.items():
        obj = json.loads(path.read_text())
        rows = obj["task_results"][task_name]["rows"]
        total = summarize_rows(rows)

        match_rows = [r for r in rows if r["oracle_mode"] == ("edge_transition_snap" if "closing" in task_name else "finger_profile_snap")]
        conflict_rows = [r for r in rows if r["oracle_mode"] != ("edge_transition_snap" if "closing" in task_name else "finger_profile_snap")]
        none_oracle = [r for r in rows if r["oracle_mode"] == "none"]
        edge_oracle = [r for r in rows if r["oracle_mode"] == "edge_transition_snap"]
        finger_oracle = [r for r in rows if r["oracle_mode"] == "finger_profile_snap"]

        task_results[task_name] = {
            "overall": total,
            "mode_counts": dict(Counter(r["oracle_mode"] for r in rows)),
            "task_best_match": summarize_rows(match_rows),
            "task_best_conflict": summarize_rows(conflict_rows),
            "oracle_none": summarize_rows(none_oracle),
            "oracle_edge": summarize_rows(edge_oracle),
            "oracle_finger": summarize_rows(finger_oracle),
            "by_sequence": weighted_by_group(rows, lambda r: r["seq_name"]),
            "by_subtype": weighted_by_group(rows, lambda r: (r["other_hand_motion"], r["interaction_motion_value"])),
            "conflict_by_sequence": weighted_by_group(conflict_rows, lambda r: r["seq_name"]),
            "conflict_by_subtype": weighted_by_group(conflict_rows, lambda r: (r["other_hand_motion"], r["interaction_motion_value"])),
        }

    payload = {
        "focus": {
            "goal": "decompose feasible-left oracle headroom by sequence, subtype, and mode conflict",
            "sources": {k: str(v) for k, v in TASK_FILES.items()},
        },
        "task_results": task_results,
    }

    out_json = GEN / "interaction_realized_feasible_left_oracle_gap_audit.json"
    out_md = SUM / "interaction_realized_feasible_left_oracle_gap_audit.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction Realized Feasible Left Oracle Gap Audit",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Decompose the remaining feasible-left headroom against the fixed task-best policy.",
        "",
    ]
    for task_name, result in task_results.items():
        lines.extend(
            [
                f"## {task_name}",
                "",
                "| slice | frames | fixed task-best | oracle | delta |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for label, rec in (
            ("overall", result["overall"]),
            ("task-best matches oracle", result["task_best_match"]),
            ("task-best conflicts with oracle", result["task_best_conflict"]),
            ("oracle mode = none", result["oracle_none"]),
            ("oracle mode = edge_transition_snap", result["oracle_edge"]),
            ("oracle mode = finger_profile_snap", result["oracle_finger"]),
        ):
            lines.append(
                f"| {label} | {rec['num_frames']} | {fmt(rec['fixed_task_best'])} | {fmt(rec['oracle'])} | {fmt(rec['delta'])} |"
            )
        lines.extend(
            [
                "",
                f"Oracle mode counts: `{result['mode_counts']}`",
                "",
                "### Top Conflict Sequences",
                "",
                "| sequence | frames | fixed task-best | oracle | delta |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for rec in result["conflict_by_sequence"][:5]:
            lines.append(
                f"| {rec['group']} | {rec['num_frames']} | {fmt(rec['fixed_task_best'])} | {fmt(rec['oracle'])} | {fmt(rec['delta'])} |"
            )
        lines.extend(
            [
                "",
                "### Top Conflict Subtypes",
                "",
                "| subtype | frames | fixed task-best | oracle | delta |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for rec in result["conflict_by_subtype"][:8]:
            subtype = rec["group"]
            lines.append(
                f"| {subtype} | {rec['num_frames']} | {fmt(rec['fixed_task_best'])} | {fmt(rec['oracle'])} | {fmt(rec['delta'])} |"
            )
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
