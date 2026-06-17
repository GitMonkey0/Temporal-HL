#!/usr/bin/env python3
"""Build a protocol matrix across representation, fraction, and protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def metric_triplet(path: Path, fraction: str):
    payload = load_json(path)
    summary = payload["summary"][fraction]
    if "pretrained_sequence_accuracy_mean" in summary:
        return {
            "scratch_seq": summary["scratch_sequence_accuracy_mean"],
            "pretrained_seq": summary["pretrained_sequence_accuracy_mean"],
            "scratch_win": summary["scratch_window_accuracy_mean"],
            "pretrained_win": summary["pretrained_window_accuracy_mean"],
        }
    return {
        "scratch_seq": summary["baseline_sequence_accuracy_mean"],
        "pretrained_seq": summary["teacher_sequence_accuracy_mean"],
        "scratch_win": summary["baseline_window_accuracy_mean"],
        "pretrained_win": summary["teacher_window_accuracy_mean"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/generated/protocol_matrix.json"),
    )
    args = parser.parse_args()

    entries = [
        {
            "representation": "symbolic",
            "fraction": "0.5",
            "protocol": "default",
            "path": Path("/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_boost_rom05_curve_seedfixed.json"),
        },
        {
            "representation": "symbolic",
            "fraction": "1.0",
            "protocol": "default",
            "path": Path("/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_boost_rom05_curve_seedfixed.json"),
        },
        {
            "representation": "symbolic",
            "fraction": "1.0",
            "protocol": "all_split_norm_168_84",
            "path": Path("/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_all_168_84_boost_rom05.json"),
        },
        {
            "representation": "symbolic",
            "fraction": "0.5",
            "protocol": "pretrain_only_norm_168_84",
            "path": Path("/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_curve.json"),
        },
        {
            "representation": "symbolic",
            "fraction": "1.0",
            "protocol": "pretrain_only_norm_168_84",
            "path": Path("/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_curve.json"),
        },
        {
            "representation": "joint_sequence",
            "fraction": "0.5",
            "protocol": "default",
            "path": Path("/opt/tiger/hand/experiments/generated/joint_sequence_student_default_fraction05_v1.json"),
        },
        {
            "representation": "joint_sequence",
            "fraction": "1.0",
            "protocol": "default",
            "path": Path("/opt/tiger/hand/experiments/generated/joint_sequence_student_default_v1.json"),
        },
        {
            "representation": "joint_sequence",
            "fraction": "0.5",
            "protocol": "all_split_norm_186_96",
            "path": Path("/opt/tiger/hand/experiments/generated/joint_sequence_student_all_186_96_fraction05_v1.json"),
        },
        {
            "representation": "joint_sequence",
            "fraction": "1.0",
            "protocol": "all_split_norm_186_96",
            "path": Path("/opt/tiger/hand/experiments/generated/joint_sequence_student_all_186_96_v1.json"),
        },
        {
            "representation": "joint_sequence",
            "fraction": "0.5",
            "protocol": "pretrain_only_norm_186_96",
            "path": Path("/opt/tiger/hand/experiments/generated/joint_sequence_student_pre186_96_fraction05_v1.json"),
        },
        {
            "representation": "joint_sequence",
            "fraction": "1.0",
            "protocol": "pretrain_only_norm_186_96",
            "path": Path("/opt/tiger/hand/experiments/generated/joint_sequence_student_pre186_96_v1.json"),
        },
    ]

    rows = []
    for entry in entries:
        metrics = metric_triplet(entry["path"], entry["fraction"])
        rows.append(
            {
                "representation": entry["representation"],
                "fraction": float(entry["fraction"]),
                "protocol": entry["protocol"],
                "path": str(entry["path"]),
                **metrics,
            }
        )

    payload = {"rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"output: {args.output}")
    for row in rows:
        print(
            f"{row['representation']} fraction={row['fraction']} {row['protocol']} "
            f"scratch_seq={row['scratch_seq']:.4f} "
            f"pretrained_seq={row['pretrained_seq']:.4f} "
            f"scratch_win={row['scratch_win']:.4f} "
            f"pretrained_win={row['pretrained_win']:.4f}"
        )


if __name__ == "__main__":
    main()
