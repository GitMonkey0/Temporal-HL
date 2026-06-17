#!/usr/bin/env python3
"""Merge feasible left-repair follow-up runs and add paired significance.

This is an experiment memo, not paper text.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path("/opt/tiger/hand")
GEN = ROOT / "experiments/generated"
SUM = GEN / "summary_tables"
BOOTSTRAP_SAMPLES = 20000
PERMUTATION_SAMPLES = 20000
SEED = 0
TASK_SUFFIXES = {
    "right_hand_motion->closing": "closing",
    "right_hand_motion->opening": "opening",
}
BEST_MODE = {
    "right_hand_motion->closing": "hgb_relax_both_left_edge_transition_snap",
    "right_hand_motion->opening": "hgb_relax_both_left_finger_profile_snap",
}
BASE_MODE = "hgb_relax_both_left_none"


def load_json(path: Path):
    return json.loads(path.read_text())


def fmt(x: float) -> str:
    return f"{x:.4f}"


def collect_method_array(rows, prefix: str, field: str):
    return np.asarray([float(row[f"{prefix}_{field}"]) for row in rows], dtype=np.float32)


def paired_stats(a: np.ndarray, b: np.ndarray, rng: np.random.Generator):
    diff = b - a
    n = len(diff)
    observed = float(diff.mean())
    wins = int((diff > 0).sum())
    losses = int((diff < 0).sum())
    ties = int((diff == 0).sum())

    boot = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float32)
    for idx in range(BOOTSTRAP_SAMPLES):
        sample = diff[rng.integers(0, n, size=n)]
        boot[idx] = float(sample.mean())
    ci_lo, ci_hi = np.quantile(boot, [0.025, 0.975]).tolist()

    signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=(PERMUTATION_SAMPLES, n))
    perm = (signs * diff[None, :]).mean(axis=1)
    p_value = float((np.abs(perm) >= abs(observed)).mean())

    return {
        "num_frames": int(n),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "delta": observed,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "bootstrap_ci_low": float(ci_lo),
        "bootstrap_ci_high": float(ci_hi),
        "permutation_pvalue": p_value,
    }


def main():
    rng = np.random.default_rng(SEED)
    task_results = {}
    sources = {}
    for task_name, suffix in TASK_SUFFIXES.items():
        path = GEN / f"interaction_realized_feasible_left_repair_followup_{suffix}.json"
        obj = load_json(path)
        sources[task_name] = str(path)
        result = obj["task_results"][task_name]
        rows = result["rows"]
        best_mode = BEST_MODE[task_name]
        result["paired_significance"] = {
            "best_vs_none_joint_score": paired_stats(
                collect_method_array(rows, BASE_MODE, "joint_score"),
                collect_method_array(rows, best_mode, "joint_score"),
                rng,
            ),
            "best_vs_none_left_preserve": paired_stats(
                collect_method_array(rows, BASE_MODE, "left_preserve"),
                collect_method_array(rows, best_mode, "left_preserve"),
                rng,
            ),
        }
        result["best_mode"] = best_mode
        task_results[task_name] = result

    payload = {
        "artifacts": sources,
        "focus": {
            "goal": "merged feasible left-repair follow-up on the corrected hard right-hand slices",
            "base_mode": BASE_MODE,
            "best_mode_by_task": BEST_MODE,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "permutation_samples": PERMUTATION_SAMPLES,
            "seed": SEED,
        },
        "task_results": task_results,
    }

    out_json = GEN / "interaction_realized_feasible_left_repair_bundle.json"
    out_md = SUM / "interaction_realized_feasible_left_repair_bundle.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction Realized Feasible Left Repair Bundle",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Merged feasible-two-hand left-repair follow-up with paired significance for the best mode on each hard right-hand task.",
        "",
    ]
    for task_name, result in task_results.items():
        lines.extend([
            f"## {task_name}",
            "",
            "| method | right grouped | left preserve | joint overall |",
            "| --- | ---: | ---: | ---: |",
        ])
        for method, stats in result["summary"].items():
            lines.append(
                f"| {method} | {fmt(stats['right_grouped_match_overall'])} | {fmt(stats['left_preserve_overall'])} | {fmt(stats['joint_score_overall'])} |"
            )
        joint = result["paired_significance"]["best_vs_none_joint_score"]
        left = result["paired_significance"]["best_vs_none_left_preserve"]
        lines.extend([
            "",
            f"Best mode: `{result['best_mode']}`",
            "",
            "| field | delta | CI low | CI high | p-value | wins | losses | ties |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| joint_score | {fmt(joint['delta'])} | {fmt(joint['bootstrap_ci_low'])} | {fmt(joint['bootstrap_ci_high'])} | {joint['permutation_pvalue']:.6f} | {joint['wins']} | {joint['losses']} | {joint['ties']} |",
            f"| left_preserve | {fmt(left['delta'])} | {fmt(left['bootstrap_ci_low'])} | {fmt(left['bootstrap_ci_high'])} | {left['permutation_pvalue']:.6f} | {left['wins']} | {left['losses']} | {left['ties']} |",
            "",
        ])

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
