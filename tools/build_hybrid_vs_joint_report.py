#!/usr/bin/env python3
"""Compare hybrid symbolic mainline and joint-sequence baseline on shared slices."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from tools.build_symbolic_vs_joint_report import (
    compare_error_frontier,
    compare_sequences,
    compare_slice_tables,
    plain_summary,
    sequence_outcome_table,
    write_csv,
    write_json,
    write_md_table,
    fmt,
)


ROOT = Path("/opt/tiger/hand")
GEN = ROOT / "experiments/generated"


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def build_fraction_report(symbolic_path: Path, joint_path: Path):
    symbolic = load_json(symbolic_path)
    joint = load_json(joint_path)
    symbolic_seq = sequence_outcome_table(symbolic)
    joint_seq = sequence_outcome_table(joint)
    slice_rows = compare_slice_tables(symbolic_seq, joint_seq)
    seq_rows = compare_sequences(symbolic_seq, joint_seq)
    error_rows = compare_error_frontier(
        symbolic["pretrained_error_counts"],
        joint["pretrained_error_counts"],
    )
    return {
        "symbolic_analysis": str(symbolic_path),
        "joint_analysis": str(joint_path),
        "slice_comparison": slice_rows,
        "sequence_comparison_top_symbolic": [row for row in seq_rows if row["symbolic_minus_joint"] > 0][:15],
        "sequence_comparison_top_joint": sorted(
            [row for row in seq_rows if row["symbolic_minus_joint"] < 0],
            key=lambda row: (row["symbolic_minus_joint"], row["seq_name"]),
        )[:15],
        "error_frontier_joint_minus_symbolic": error_rows,
        "plain_summary": plain_summary(seq_rows, slice_rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=GEN / "hybrid_symbolic_vs_joint_report.json",
    )
    args = parser.parse_args()

    fraction_10 = build_fraction_report(
        GEN / "hybrid_symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_changeonly_analysis.json",
        GEN / "joint_sequence_pre186_96_analysis_v1.json",
    )
    fraction_05 = build_fraction_report(
        GEN / "hybrid_symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_fraction05_changeonly_analysis.json",
        GEN / "joint_sequence_pre186_96_fraction05_analysis_v1.json",
    )

    payload = {
        "fraction_0.5": fraction_05,
        "fraction_1.0": fraction_10,
    }
    write_json(args.output, payload)

    summary_dir = GEN / "summary_tables"
    slice_rows = []
    for fraction_key, rep in payload.items():
        for row in rep["slice_comparison"]:
            slice_rows.append(
                {
                    "fraction": fraction_key,
                    "slice": row["slice"],
                    "hybrid_symbolic_accuracy": fmt(float(row["symbolic_accuracy"])),
                    "joint_accuracy": fmt(float(row["joint_accuracy"])),
                    "hybrid_minus_joint": fmt(float(row["symbolic_minus_joint"])),
                }
            )
    write_csv(
        summary_dir / "hybrid_symbolic_vs_joint_slices.csv",
        slice_rows,
        ["fraction", "slice", "hybrid_symbolic_accuracy", "joint_accuracy", "hybrid_minus_joint"],
    )
    write_md_table(
        summary_dir / "hybrid_symbolic_vs_joint_slices.md",
        ["fraction", "slice", "hybrid_symbolic_accuracy", "joint_accuracy", "hybrid_minus_joint"],
        [
            [row["fraction"], row["slice"], row["hybrid_symbolic_accuracy"], row["joint_accuracy"], row["hybrid_minus_joint"]]
            for row in slice_rows
        ],
        "Hybrid Symbolic vs Joint Slices",
    )

    print(f"output: {args.output}")
    print(
        "fraction_0.5 hybrid symbolic better / joint better:",
        payload["fraction_0.5"]["plain_summary"]["symbolic_better_sequences"],
        payload["fraction_0.5"]["plain_summary"]["joint_better_sequences"],
    )
    print(
        "fraction_1.0 hybrid symbolic better / joint better:",
        payload["fraction_1.0"]["plain_summary"]["symbolic_better_sequences"],
        payload["fraction_1.0"]["plain_summary"]["joint_better_sequences"],
    )


if __name__ == "__main__":
    main()
