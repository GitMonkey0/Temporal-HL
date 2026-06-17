#!/usr/bin/env python3
"""Merge corrected global right-support checks and compute paired significance.

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
    source_artifacts = {}
    for task_name, suffix in TASK_SUFFIXES.items():
        path = GEN / f"interaction_realized_global_right_support_check_{suffix}.json"
        obj = load_json(path)
        source_artifacts[task_name] = str(path)
        result = obj["task_results"][task_name]
        rows = result["rows"]
        result["paired_significance"] = {
            "joint_score": paired_stats(
                collect_method_array(rows, "hgb_strict", "joint_score"),
                collect_method_array(rows, "hgb_relax_both", "joint_score"),
                rng,
            ),
            "right_grouped_match": paired_stats(
                collect_method_array(rows, "hgb_strict", "right_grouped_match"),
                collect_method_array(rows, "hgb_relax_both", "right_grouped_match"),
                rng,
            ),
            "left_preserve": paired_stats(
                collect_method_array(rows, "hgb_strict", "left_preserve"),
                collect_method_array(rows, "hgb_relax_both", "left_preserve"),
                rng,
            ),
        }
        task_results[task_name] = result

    payload = {
        "artifacts": source_artifacts,
        "focus": {
            "goal": "corrected global right-support bundle on full hard right-hand slices",
            "comparison": "hgb_strict -> hgb_relax_both",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "permutation_samples": PERMUTATION_SAMPLES,
            "seed": SEED,
            "note": "This bundle supersedes earlier closing-side candidate-pool results that were polluted by a task-target bug.",
        },
        "task_results": task_results,
    }

    out_json = GEN / "interaction_realized_global_right_support_bundle.json"
    out_md = SUM / "interaction_realized_global_right_support_bundle.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction Realized Global Right Support Bundle",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Corrected full-slice comparison for `hgb_strict` vs `hgb_relax_both` on the hard right-hand interaction tasks.",
        "",
    ]
    for task_name, result in task_results.items():
        lines.extend([
            f"## {task_name}",
            "",
            "| method | avail | right grouped | left preserve | joint overall |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for method in ("hgb_strict", "hgb_relax_both"):
            stats = result["summary"][method]
            lines.append(
                f"| {method} | {fmt(stats['available_rate'])} | {fmt(stats['right_grouped_match_overall'])} | "
                f"{fmt(stats['left_preserve_overall'])} | {fmt(stats['joint_score_overall'])} |"
            )
        lines.extend([
            "",
            "| field | delta | CI low | CI high | p-value | wins | losses | ties |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for field in ("joint_score", "right_grouped_match", "left_preserve"):
            stats = result["paired_significance"][field]
            lines.append(
                f"| {field} | {fmt(stats['delta'])} | {fmt(stats['bootstrap_ci_low'])} | {fmt(stats['bootstrap_ci_high'])} | "
                f"{stats['permutation_pvalue']:.6f} | {stats['wins']} | {stats['losses']} | {stats['ties']} |"
            )
        lines.extend([
            "",
            "### Sequence Summary",
            "",
            "| sequence | hgb strict | hgb relax-both |",
            "| --- | ---: | ---: |",
        ])
        for row in result["sequence_summary"]:
            lines.append(
                f"| {row['seq_name']} | {fmt(row['hgb_strict_joint_score_overall'])} | {fmt(row['hgb_relax_both_joint_score_overall'])} |"
            )
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
