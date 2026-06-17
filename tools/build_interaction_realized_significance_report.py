#!/usr/bin/env python3
"""Paired significance report for the current hard-slice interaction frontier.

This is an experiment memo, not paper text.

It quantifies whether the new strongest interaction-aware realized editor
improves over prior baselines under paired analysis on the exact same frames.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("/opt/tiger/hand")
GEN = ROOT / "experiments/generated"
SUM = GEN / "summary_tables"
BOOTSTRAP_SAMPLES = 20000
PERMUTATION_SAMPLES = 20000
SEED = 0

COMPARISONS = [
    ("single", "hgb_budget1_none_top10"),
    ("hgb_budget1_none_top10", "hgb_budget1_finger_profile_snap_top10"),
    ("hgb_budget1_finger_profile_snap_top10", "hgb_budget2_finger_profile_snap_top10"),
    ("hgb_budget1_none_top10", "hgb_budget2_finger_profile_snap_top10"),
    ("mlp_budget1_none_top10", "mlp_budget2_finger_profile_snap_top10"),
]


def load_json(path: Path):
    return json.loads(path.read_text())


def fmt(x: float) -> str:
    return f"{x:.4f}"


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
        "win_minus_loss": wins - losses,
        "bootstrap_ci_low": float(ci_lo),
        "bootstrap_ci_high": float(ci_hi),
        "permutation_pvalue": p_value,
    }


def collect_method_array(rows, prefix: str, field: str):
    return np.asarray([float(row[f"{prefix}_{field}"]) for row in rows], dtype=np.float32)


def subtype_stats(rows, a_name: str, b_name: str, field: str, rng: np.random.Generator):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["other_hand_motion"], row["interaction_motion_value"])].append(row)
    out = []
    for key, items in sorted(grouped.items()):
        a = collect_method_array(items, a_name, field)
        b = collect_method_array(items, b_name, field)
        stats = paired_stats(a, b, rng)
        out.append(
            {
                "other_hand_motion": key[0],
                "interaction_motion_value": key[1],
                **stats,
            }
        )
    return out


def main():
    payload_in = load_json(GEN / "interaction_realized_constraint_scaling.json")
    rng = np.random.default_rng(SEED)

    task_results = {}
    for task_name, result in payload_in["task_results"].items():
        rows = result["rows"]
        comps = {}
        for a_name, b_name in COMPARISONS:
            comps[f"{a_name}__to__{b_name}"] = {
                "joint_score": paired_stats(
                    collect_method_array(rows, a_name, "joint_score"),
                    collect_method_array(rows, b_name, "joint_score"),
                    rng,
                ),
                "left_preserve": paired_stats(
                    collect_method_array(rows, a_name, "left_preserve"),
                    collect_method_array(rows, b_name, "left_preserve"),
                    rng,
                ),
                "right_grouped_match": paired_stats(
                    collect_method_array(rows, a_name, "right_grouped_match"),
                    collect_method_array(rows, b_name, "right_grouped_match"),
                    rng,
                ),
                "subtype_joint_score": subtype_stats(rows, a_name, b_name, "joint_score", rng),
            }
        task_results[task_name] = comps

    payload = {
        "artifacts": {
            "source": str(GEN / "interaction_realized_constraint_scaling.json"),
        },
        "focus": {
            "goal": "paired significance for the current hard-slice interaction frontier",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "permutation_samples": PERMUTATION_SAMPLES,
            "seed": SEED,
            "comparisons": [f"{a} -> {b}" for a, b in COMPARISONS],
        },
        "task_results": task_results,
    }

    out_json = GEN / "interaction_realized_significance_report.json"
    out_md = SUM / "interaction_realized_significance_report.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction Realized Significance Report",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Paired statistics on the exact same hard-slice interaction frames.",
        "",
    ]
    for task_name, comps in task_results.items():
        lines.extend(
            [
                f"## {task_name}",
                "",
                "| comparison | field | mean A | mean B | delta | CI low | CI high | p-value | wins | losses | ties |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for comp_name, comp in comps.items():
            for field in ("joint_score", "left_preserve", "right_grouped_match"):
                stats = comp[field]
                lines.append(
                    f"| {comp_name.replace('__to__', ' -> ')} | {field} | {fmt(stats['mean_a'])} | {fmt(stats['mean_b'])} | "
                    f"{fmt(stats['delta'])} | {fmt(stats['bootstrap_ci_low'])} | {fmt(stats['bootstrap_ci_high'])} | "
                    f"{stats['permutation_pvalue']:.6f} | {stats['wins']} | {stats['losses']} | {stats['ties']} |"
                )
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
