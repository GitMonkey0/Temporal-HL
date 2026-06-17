#!/usr/bin/env python3
"""Build a single summary bundle from current strongest experiment artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def top_slice_rows(path: Path, limit: int = 10):
    payload = load_json(path)
    return payload["pretrained_diff"][:limit]


def error_counts(path: Path, key: str):
    payload = load_json(path)
    return payload[key][:10]


def pick_row(rows, representation: str, fraction: float, protocol: str):
    for row in rows:
        if (
            row["representation"] == representation
            and abs(float(row["fraction"]) - fraction) < 1e-9
            and row["protocol"] == protocol
        ):
            return row
    raise KeyError((representation, fraction, protocol))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/generated/experiment_summary_bundle.json"),
    )
    args = parser.parse_args()

    protocol_matrix_path = Path("/opt/tiger/hand/experiments/generated/protocol_matrix.json")
    symbolic_slice_1_path = Path("/opt/tiger/hand/experiments/generated/symbolic_slice_compare_old_vs_new.json")
    symbolic_slice_05_path = Path("/opt/tiger/hand/experiments/generated/symbolic_slice_compare_old_vs_new_fraction05.json")
    symbolic_old_analysis_path = Path("/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_boost_rom05_analysis_seedfixed.json")
    symbolic_new_analysis_path = Path("/opt/tiger/hand/experiments/generated/symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_analysis.json")
    joint_slice_1_path = Path("/opt/tiger/hand/experiments/generated/joint_sequence_slice_compare_v1.json")
    joint_slice_05_path = Path("/opt/tiger/hand/experiments/generated/joint_sequence_slice_compare_fraction05_v1.json")
    joint_old_analysis_path = Path("/opt/tiger/hand/experiments/generated/joint_sequence_default_analysis_v1.json")
    joint_new_analysis_path = Path("/opt/tiger/hand/experiments/generated/joint_sequence_pre186_96_analysis_v1.json")
    joint_old_analysis_05_path = Path("/opt/tiger/hand/experiments/generated/joint_sequence_default_fraction05_analysis_v1.json")
    joint_new_analysis_05_path = Path("/opt/tiger/hand/experiments/generated/joint_sequence_pre186_96_fraction05_analysis_v1.json")

    protocol_matrix = load_json(protocol_matrix_path)
    rows = protocol_matrix["rows"]

    payload = {
        "artifacts": {
            "protocol_matrix": str(protocol_matrix_path),
            "symbolic_slice_fraction_1": str(symbolic_slice_1_path),
            "symbolic_slice_fraction_05": str(symbolic_slice_05_path),
            "symbolic_old_analysis": str(symbolic_old_analysis_path),
            "symbolic_new_analysis": str(symbolic_new_analysis_path),
            "joint_slice_fraction_1": str(joint_slice_1_path),
            "joint_slice_fraction_05": str(joint_slice_05_path),
            "joint_old_analysis": str(joint_old_analysis_path),
            "joint_new_analysis": str(joint_new_analysis_path),
            "joint_old_analysis_fraction_05": str(joint_old_analysis_05_path),
            "joint_new_analysis_fraction_05": str(joint_new_analysis_05_path),
        },
        "current_mainlines": {
            "symbolic": {
                "protocol": "pretrain_only_norm_168_84",
                "fraction_1.0": pick_row(rows, "symbolic", 1.0, "pretrain_only_norm_168_84"),
                "fraction_0.5": pick_row(rows, "symbolic", 0.5, "pretrain_only_norm_168_84"),
            },
            "joint_sequence": {
                "protocol": "pretrain_only_norm_186_96",
                "fraction_1.0": pick_row(rows, "joint_sequence", 1.0, "pretrain_only_norm_186_96"),
                "fraction_0.5": pick_row(rows, "joint_sequence", 0.5, "pretrain_only_norm_186_96"),
            },
        },
        "protocol_matrix": rows,
        "key_protocol_comparisons": {
            "symbolic_fraction_1.0": {
                "default": pick_row(rows, "symbolic", 1.0, "default"),
                "all_split": pick_row(rows, "symbolic", 1.0, "all_split_norm_168_84"),
                "pretrain_only": pick_row(rows, "symbolic", 1.0, "pretrain_only_norm_168_84"),
            },
            "symbolic_fraction_0.5": {
                "default": pick_row(rows, "symbolic", 0.5, "default"),
                "pretrain_only": pick_row(rows, "symbolic", 0.5, "pretrain_only_norm_168_84"),
            },
            "joint_sequence_fraction_1.0": {
                "default": pick_row(rows, "joint_sequence", 1.0, "default"),
                "all_split": pick_row(rows, "joint_sequence", 1.0, "all_split_norm_186_96"),
                "pretrain_only": pick_row(rows, "joint_sequence", 1.0, "pretrain_only_norm_186_96"),
            },
            "joint_sequence_fraction_0.5": {
                "default": pick_row(rows, "joint_sequence", 0.5, "default"),
                "all_split": pick_row(rows, "joint_sequence", 0.5, "all_split_norm_186_96"),
                "pretrain_only": pick_row(rows, "joint_sequence", 0.5, "pretrain_only_norm_186_96"),
            },
        },
        "slice_deltas": {
            "fraction_1.0_top": top_slice_rows(symbolic_slice_1_path),
            "fraction_0.5_top": top_slice_rows(symbolic_slice_05_path),
            "joint_fraction_1.0_top": top_slice_rows(joint_slice_1_path),
            "joint_fraction_0.5_top": top_slice_rows(joint_slice_05_path),
        },
        "error_frontier": {
            "old_symbolic_mainline": error_counts(
                symbolic_old_analysis_path, "pretrained_error_counts"
            ),
            "new_symbolic_mainline": error_counts(
                symbolic_new_analysis_path, "pretrained_error_counts"
            ),
            "old_joint_sequence_mainline": error_counts(
                joint_old_analysis_path, "pretrained_error_counts"
            ),
            "new_joint_sequence_mainline": error_counts(
                joint_new_analysis_path, "pretrained_error_counts"
            ),
            "old_joint_sequence_mainline_fraction_0.5": error_counts(
                joint_old_analysis_05_path, "pretrained_error_counts"
            ),
            "new_joint_sequence_mainline_fraction_0.5": error_counts(
                joint_new_analysis_05_path, "pretrained_error_counts"
            ),
        },
        "plain_summary": {
            "symbolic_fraction_1.0_gain": (
                pick_row(rows, "symbolic", 1.0, "pretrain_only_norm_168_84")["pretrained_seq"]
                - pick_row(rows, "symbolic", 1.0, "default")["pretrained_seq"]
            ),
            "symbolic_fraction_0.5_gain": (
                pick_row(rows, "symbolic", 0.5, "pretrain_only_norm_168_84")["pretrained_seq"]
                - pick_row(rows, "symbolic", 0.5, "default")["pretrained_seq"]
            ),
            "joint_fraction_1.0_gain": (
                pick_row(rows, "joint_sequence", 1.0, "pretrain_only_norm_186_96")["pretrained_seq"]
                - pick_row(rows, "joint_sequence", 1.0, "default")["pretrained_seq"]
            ),
            "joint_fraction_0.5_gain": (
                pick_row(rows, "joint_sequence", 0.5, "pretrain_only_norm_186_96")["pretrained_seq"]
                - pick_row(rows, "joint_sequence", 0.5, "default")["pretrained_seq"]
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"output: {args.output}")
    print(
        "symbolic gains:",
        payload["plain_summary"]["symbolic_fraction_0.5_gain"],
        payload["plain_summary"]["symbolic_fraction_1.0_gain"],
    )
    print(
        "joint gains:",
        payload["plain_summary"]["joint_fraction_0.5_gain"],
        payload["plain_summary"]["joint_fraction_1.0_gain"],
    )


if __name__ == "__main__":
    main()
