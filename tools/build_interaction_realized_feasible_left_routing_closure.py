#!/usr/bin/env python3
"""Aggregate lightweight routing results on feasible left repair.

This is an experiment memo, not paper text.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/opt/tiger/hand")
GEN = ROOT / "experiments/generated"
SUM = GEN / "summary_tables"


def load(name: str):
    return json.loads((GEN / name).read_text())


def fmt(x: float) -> str:
    return f"{x:.4f}"


def collect_task(task: str, task_target: str):
    fixed_best = load("interaction_realized_feasible_left_repair_bundle.json")["task_results"][task]

    repair_gate = load(f"interaction_realized_feasible_left_repair_gate_{task_target}.json")["task_results"][task]
    model_gate = load(f"interaction_realized_feasible_left_repair_model_gate_{task_target}.json")["task_results"][task]
    template_gate = load(f"interaction_realized_feasible_left_template_gate_{task_target}.json")["task_results"][task]
    temporal_window = load(f"interaction_realized_feasible_left_temporal_window_knn_{task_target}.json")["task_results"][task]
    gain_regressor = load("interaction_realized_feasible_left_gain_regressor.json")["task_results"][task]
    sparse_override = load("interaction_realized_feasible_left_sparse_override.json")["task_results"][task]

    out = {
        "fixed_task_best": fixed_best["summary"][fixed_best["best_mode"]]["joint_score_overall"],
        "oracle_framewise": None,
        "cheap_subtype_gate": repair_gate["summary"]["gate"]["joint_score_overall"],
        "learned_gate": model_gate["summary"]["gate"]["joint_score_overall"],
        "gain_regressor": gain_regressor["summary"]["gain_regressor"]["joint_score_overall"],
        "sparse_override": sparse_override["summary"]["sparse_override"]["joint_score_overall"],
        "template_seq_gate": template_gate["summary"]["seq_gate"]["joint_score_overall"],
        "template_subtype_gate": template_gate["summary"]["subtype_gate"]["joint_score_overall"],
        "template_seq_subtype_gate": template_gate["summary"]["seq_subtype_gate"]["joint_score_overall"],
        "temporal_window_knn": temporal_window["summary"]["temporal_window_knn"]["joint_score_overall"],
    }

    if task_target == "closing":
        apply_gate = load("interaction_realized_feasible_left_apply_gate.json")
        closing_temporal_routing = load("interaction_realized_feasible_left_closing_temporal_routing.json")
        closing_chunk_knn = load("interaction_realized_feasible_left_closing_chunk_knn.json")
        dense_knn = load("interaction_realized_feasible_left_dense_knn_closing.json")["task_results"][task]
        out.update(
            {
                "apply_gate": apply_gate["summary"]["gate"]["joint_score_overall"],
                "dense_knn": dense_knn["summary"]["dense_knn"]["joint_score_overall"],
                "closing_temporal_route_edge_vs_none": closing_temporal_routing["summary"]["edge_vs_none_route"]["joint_score_overall"],
                "closing_temporal_route_edge_vs_finger": closing_temporal_routing["summary"]["edge_vs_finger_route"]["joint_score_overall"],
                "chunk_knn": closing_chunk_knn["summary"]["chunk_knn"]["joint_score_overall"],
                "oracle_framewise": model_gate["summary"]["oracle"]["joint_score_overall"],
            }
        )
    else:
        dense_knn = load("interaction_realized_feasible_left_dense_knn_opening.json")["task_results"][task]
        opening_chunk_knn = load("interaction_realized_feasible_left_opening_chunk_knn.json")
        out.update(
            {
                "dense_knn": dense_knn["summary"]["dense_knn"]["joint_score_overall"],
                "chunk_knn": opening_chunk_knn["summary"]["chunk_knn"]["joint_score_overall"],
                "oracle_framewise": model_gate["summary"]["oracle"]["joint_score_overall"],
            }
        )

    return out


def delta_map(metrics: dict[str, float]):
    fixed = metrics["fixed_task_best"]
    return {name: value - fixed for name, value in metrics.items() if name != "fixed_task_best"}


def main():
    task_names = {
        "right_hand_motion->closing": "closing",
        "right_hand_motion->opening": "opening",
    }
    task_results = {}
    for task, task_target in task_names.items():
        metrics = collect_task(task, task_target)
        task_results[task] = {
            "metrics": metrics,
            "delta_vs_fixed_task_best": delta_map(metrics),
        }

    out_json = GEN / "interaction_realized_feasible_left_routing_closure.json"
    out_md = SUM / "interaction_realized_feasible_left_routing_closure.md"
    out_json.write_text(json.dumps({"task_results": task_results}, indent=2))

    lines = [
        "# Interaction Realized Feasible Left Routing Closure",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Aggregate the lightweight routing attempts on the feasible left-repair problem.",
        "",
    ]
    for task, result in task_results.items():
        lines.extend(
            [
                f"## {task}",
                "",
                "| method | joint overall | delta vs fixed task best |",
                "| --- | ---: | ---: |",
            ]
        )
        metrics = result["metrics"]
        deltas = result["delta_vs_fixed_task_best"]
        ordered = ["fixed_task_best"] + [name for name in metrics.keys() if name != "fixed_task_best"]
        for name in ordered:
            delta = 0.0 if name == "fixed_task_best" else deltas[name]
            lines.append(f"| {name} | {fmt(metrics[name])} | {fmt(delta)} |")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
