#!/usr/bin/env python3
"""Soft preserve-objective policy for feasible left repair.

This is an experiment memo, not paper text.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/opt/tiger/hand")
GEN = ROOT / "experiments/generated"
SUM = GEN / "summary_tables"
TASK_FILES = {
    "right_hand_motion->closing": GEN / "interaction_realized_feasible_left_repair_model_gate_closing.json",
    "right_hand_motion->opening": GEN / "interaction_realized_feasible_left_repair_model_gate_opening.json",
}
LEFT_REPAIR_MODES = ("none", "edge_transition_snap", "finger_profile_snap")
STATE_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
TASK_BEST = {
    "right_hand_motion->closing": "edge_transition_snap",
    "right_hand_motion->opening": "finger_profile_snap",
}


def fmt(x: float) -> str:
    return f"{x:.4f}"


def score_mode(row, mode: str, state_weight: float) -> float:
    return float(row[f"left_{mode}_left_transition_agreement"]) + state_weight * float(row[f"left_{mode}_left_state_agreement"])


def choose_mode(row, state_weight: float) -> str:
    return max(LEFT_REPAIR_MODES, key=lambda mode: score_mode(row, mode, state_weight))


def summarize(rows, fixed_mode: str, state_weight: float):
    soft_modes = [choose_mode(row, state_weight) for row in rows]
    oracle_modes = [max(LEFT_REPAIR_MODES, key=lambda mode: float(row[f"left_{mode}_joint_score"])) for row in rows]

    def mean(field_fn):
        return sum(field_fn(idx, row) for idx, row in enumerate(rows)) / len(rows)

    soft_joint = mean(lambda idx, row: float(row[f"left_{soft_modes[idx]}_joint_score"]))
    fixed_joint = mean(lambda idx, row: float(row[f"left_{fixed_mode}_joint_score"]))
    oracle_joint = mean(lambda idx, row: float(row[f"left_{oracle_modes[idx]}_joint_score"]))
    soft_left_preserve = mean(lambda idx, row: float(row[f"left_{soft_modes[idx]}_left_preserve"]))
    fixed_left_preserve = mean(lambda idx, row: float(row[f"left_{fixed_mode}_left_preserve"]))
    oracle_left_preserve = mean(lambda idx, row: float(row[f"left_{oracle_modes[idx]}_left_preserve"]))
    soft_left_state = mean(lambda idx, row: float(row[f"left_{soft_modes[idx]}_left_state_agreement"]))
    fixed_left_state = mean(lambda idx, row: float(row[f"left_{fixed_mode}_left_state_agreement"]))
    oracle_left_state = mean(lambda idx, row: float(row[f"left_{oracle_modes[idx]}_left_state_agreement"]))
    soft_left_transition = mean(lambda idx, row: float(row[f"left_{soft_modes[idx]}_left_transition_agreement"]))
    fixed_left_transition = mean(lambda idx, row: float(row[f"left_{fixed_mode}_left_transition_agreement"]))
    oracle_left_transition = mean(lambda idx, row: float(row[f"left_{oracle_modes[idx]}_left_transition_agreement"]))
    wins = sum(
        float(row[f"left_{soft_modes[idx]}_joint_score"]) > float(row[f"left_{fixed_mode}_joint_score"])
        for idx, row in enumerate(rows)
    )
    losses = sum(
        float(row[f"left_{soft_modes[idx]}_joint_score"]) < float(row[f"left_{fixed_mode}_joint_score"])
        for idx, row in enumerate(rows)
    )
    return {
        "state_weight": state_weight,
        "soft_joint": soft_joint,
        "fixed_joint": fixed_joint,
        "oracle_joint": oracle_joint,
        "soft_left_preserve": soft_left_preserve,
        "fixed_left_preserve": fixed_left_preserve,
        "oracle_left_preserve": oracle_left_preserve,
        "soft_left_state": soft_left_state,
        "fixed_left_state": fixed_left_state,
        "oracle_left_state": oracle_left_state,
        "soft_left_transition": soft_left_transition,
        "fixed_left_transition": fixed_left_transition,
        "oracle_left_transition": oracle_left_transition,
        "delta_vs_fixed": soft_joint - fixed_joint,
        "gap_to_oracle": oracle_joint - soft_joint,
        "wins": int(wins),
        "losses": int(losses),
        "ties": int(len(rows) - wins - losses),
        "pred_mode_counts": {mode: soft_modes.count(mode) for mode in LEFT_REPAIR_MODES},
    }


def main():
    task_results = {}
    for task_name, path in TASK_FILES.items():
        obj = json.loads(path.read_text())
        task = obj["task_results"][task_name]
        rows = task["rows"]
        fixed_mode = TASK_BEST[task_name]
        candidates = [summarize(rows, fixed_mode, w) for w in STATE_WEIGHTS]
        best = max(candidates, key=lambda rec: rec["soft_joint"])
        task_results[task_name] = {
            "fixed_mode": fixed_mode,
            "candidates": candidates,
            "best": best,
        }

    payload = {
        "focus": {
            "goal": "test a softer preserve-side objective for feasible left repair",
            "state_weights": list(STATE_WEIGHTS),
            "sources": {k: str(v) for k, v in TASK_FILES.items()},
        },
        "task_results": task_results,
    }

    out_json = GEN / "interaction_realized_feasible_left_soft_objective.json"
    out_md = SUM / "interaction_realized_feasible_left_soft_objective.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction Realized Feasible Left Soft Objective",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Choose the left repair mode by a soft preserve objective: `left_transition_agreement + w * left_state_agreement`.",
        "",
    ]
    for task_name, result in task_results.items():
        lines.extend(
            [
                f"## {task_name}",
                "",
                "| state weight | soft joint | fixed joint | oracle joint | delta vs fixed | gap to oracle |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for rec in result["candidates"]:
            lines.append(
                f"| {fmt(rec['state_weight'])} | {fmt(rec['soft_joint'])} | {fmt(rec['fixed_joint'])} | {fmt(rec['oracle_joint'])} | {fmt(rec['delta_vs_fixed'])} | {fmt(rec['gap_to_oracle'])} |"
            )
        best = result["best"]
        lines.extend(
            [
                "",
                f"Best state weight: `{best['state_weight']}`",
                f"Pred mode counts: `{best['pred_mode_counts']}`",
                f"Left state: soft `{fmt(best['soft_left_state'])}` vs fixed `{fmt(best['fixed_left_state'])}` vs oracle `{fmt(best['oracle_left_state'])}`",
                f"Left transition: soft `{fmt(best['soft_left_transition'])}` vs fixed `{fmt(best['fixed_left_transition'])}` vs oracle `{fmt(best['oracle_left_transition'])}`",
                f"Joint wins/losses/ties vs fixed: `{best['wins']}/{best['losses']}/{best['ties']}`",
                "",
            ]
        )

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
