#!/usr/bin/env python3
"""Oracle granularity audit for feasible left repair.

This is an experiment memo, not paper text.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


ROOT = Path("/opt/tiger/hand")
GEN = ROOT / "experiments/generated"
SUM = GEN / "summary_tables"
TASK_FILES = {
    "right_hand_motion->closing": GEN / "interaction_realized_feasible_left_repair_model_gate_closing.json",
    "right_hand_motion->opening": GEN / "interaction_realized_feasible_left_repair_model_gate_opening.json",
}
LEFT_REPAIR_MODES = ("none", "edge_transition_snap", "finger_profile_snap")
TASK_BEST = {
    "right_hand_motion->closing": "edge_transition_snap",
    "right_hand_motion->opening": "finger_profile_snap",
}
RUN_GAP = 12


def fmt(x: float) -> str:
    return f"{x:.4f}"


def group_runs(rows):
    rows = sorted(rows, key=lambda r: (r["seq_name"], int(r["frame_idx"])))
    runs = []
    cur = []
    prev = None
    for row in rows:
        contiguous = False
        if prev is not None:
            contiguous = row["seq_name"] == prev["seq_name"] and int(row["frame_idx"]) - int(prev["frame_idx"]) <= RUN_GAP
        if prev is None or contiguous:
            cur.append(row)
        else:
            runs.append(cur)
            cur = [row]
        prev = row
    if cur:
        runs.append(cur)
    return runs


def mean_joint(rows, mode_selector):
    return sum(float(row[f"left_{mode_selector(row)}_joint_score"]) for row in rows) / len(rows)


def summarize_task(rows, fixed_mode):
    frame_oracle = mean_joint(rows, lambda row: max(LEFT_REPAIR_MODES, key=lambda mode: float(row[f"left_{mode}_joint_score"])))
    fixed = mean_joint(rows, lambda row: fixed_mode)

    runs = group_runs(rows)
    run_oracle_total = 0.0
    for run in runs:
        best_mode = max(LEFT_REPAIR_MODES, key=lambda mode: sum(float(row[f"left_{mode}_joint_score"]) for row in run) / len(run))
        run_oracle_total += sum(float(row[f"left_{best_mode}_joint_score"]) for row in run)
    run_oracle = run_oracle_total / len(rows)

    by_sequence = defaultdict(list)
    by_subtype = defaultdict(list)
    for row in rows:
        by_sequence[row["seq_name"]].append(row)
        by_subtype[(row["other_hand_motion"], row["interaction_motion_value"])].append(row)

    seq_oracle_total = 0.0
    for items in by_sequence.values():
        best_mode = max(LEFT_REPAIR_MODES, key=lambda mode: sum(float(row[f"left_{mode}_joint_score"]) for row in items) / len(items))
        seq_oracle_total += sum(float(row[f"left_{best_mode}_joint_score"]) for row in items)
    seq_oracle = seq_oracle_total / len(rows)

    subtype_oracle_total = 0.0
    for items in by_subtype.values():
        best_mode = max(LEFT_REPAIR_MODES, key=lambda mode: sum(float(row[f"left_{mode}_joint_score"]) for row in items) / len(items))
        subtype_oracle_total += sum(float(row[f"left_{best_mode}_joint_score"]) for row in items)
    subtype_oracle = subtype_oracle_total / len(rows)

    return {
        "num_frames": len(rows),
        "num_runs": len(runs),
        "num_sequences": len(by_sequence),
        "num_subtypes": len(by_subtype),
        "fixed_task_best": fixed,
        "subtype_oracle": subtype_oracle,
        "run_oracle": run_oracle,
        "sequence_oracle": seq_oracle,
        "frame_oracle": frame_oracle,
        "delta_subtype_vs_fixed": subtype_oracle - fixed,
        "delta_run_vs_fixed": run_oracle - fixed,
        "delta_seq_vs_fixed": seq_oracle - fixed,
        "delta_frame_vs_fixed": frame_oracle - fixed,
        "delta_frame_vs_run": frame_oracle - run_oracle,
        "delta_run_vs_seq": run_oracle - seq_oracle,
        "delta_subtype_vs_seq": subtype_oracle - seq_oracle,
    }


def main():
    task_results = {}
    for task_name, path in TASK_FILES.items():
        obj = json.loads(path.read_text())
        rows = obj["task_results"][task_name]["rows"]
        fixed_mode = TASK_BEST[task_name]
        task_results[task_name] = summarize_task(rows, fixed_mode)

    payload = {
        "focus": {
            "goal": "measure the granularity of remaining feasible-left oracle headroom",
            "run_gap": RUN_GAP,
            "sources": {k: str(v) for k, v in TASK_FILES.items()},
        },
        "task_results": task_results,
    }

    out_json = GEN / "interaction_realized_feasible_left_oracle_granularity.json"
    out_md = SUM / "interaction_realized_feasible_left_oracle_granularity.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction Realized Feasible Left Oracle Granularity",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Compare upper bounds when the mode can change at different granularities: subtype, run, sequence, or frame.",
        "",
    ]
    for task_name, result in task_results.items():
        lines.extend(
            [
                f"## {task_name}",
                "",
                "| policy | joint overall | delta vs fixed task-best |",
                "| --- | ---: | ---: |",
                f"| fixed_task_best | {fmt(result['fixed_task_best'])} | {fmt(0.0)} |",
                f"| subtype_oracle | {fmt(result['subtype_oracle'])} | {fmt(result['delta_subtype_vs_fixed'])} |",
                f"| run_oracle | {fmt(result['run_oracle'])} | {fmt(result['delta_run_vs_fixed'])} |",
                f"| sequence_oracle | {fmt(result['sequence_oracle'])} | {fmt(result['delta_seq_vs_fixed'])} |",
                f"| frame_oracle | {fmt(result['frame_oracle'])} | {fmt(result['delta_frame_vs_fixed'])} |",
                "",
                f"Frames `{result['num_frames']}`, runs `{result['num_runs']}`, sequences `{result['num_sequences']}`, subtypes `{result['num_subtypes']}`",
                f"Frame-vs-run gap: `{fmt(result['delta_frame_vs_run'])}`; run-vs-sequence gap: `{fmt(result['delta_run_vs_seq'])}`; subtype-vs-sequence gap: `{fmt(result['delta_subtype_vs_seq'])}`",
                "",
            ]
        )

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
