#!/usr/bin/env python3
"""Build unified slice and sequence frontier across symbolic mainline and key controls."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path("/opt/tiger/hand")
GEN = ROOT / "experiments/generated"


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md_table(path: Path, headers: list[str], rows: list[list[object]], title: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(f"# {title}\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(x) for x in row) + " |\n")


def fmt(x: float) -> str:
    return f"{x:.4f}"


def slice_membership(seq_name: str) -> set[str]:
    groups = {"all"}
    if "Interaction" in seq_name:
        groups.add("interaction")
    if "No_Interaction" in seq_name:
        groups.add("no_interaction")
    if "LT_" in seq_name or "Lt_" in seq_name:
        groups.add("left")
    if "RT_" in seq_name or "Rt_" in seq_name:
        groups.add("right")
    if "Wrist_ROM" in seq_name:
        groups.add("wrist_rom")
    if "Occlusion" in seq_name:
        groups.add("occlusion")
    if "Finger_Occlusions" in seq_name:
        groups.add("finger_occlusion")
    if "No_Occlusion" in seq_name:
        groups.add("no_occlusion")
    if "Touching" in seq_name:
        groups.add("touching")
    return groups


def sequence_rows(payload: dict[str, object], mode: str = "pretrained"):
    key = f"{mode}_predictions"
    out = {}
    for seq_name, item in payload["sequence_consistency"].items():
        preds = item[key]
        correct = sum(1 for pred in preds if pred["correct"])
        out[seq_name] = {
            "seq_name": seq_name,
            "correct": correct,
            "total": len(preds),
            "accuracy": correct / max(len(preds), 1),
        }
    return out


def slice_rows(seq_rows_map: dict[str, dict[str, object]]):
    totals = defaultdict(int)
    correct = defaultdict(int)
    for seq_name, row in seq_rows_map.items():
        for group in slice_membership(seq_name):
            totals[group] += row["total"]
            correct[group] += row["correct"]
    out = {}
    for group in sorted(totals):
        out[group] = {
            "slice": group,
            "accuracy": correct[group] / max(totals[group], 1),
            "correct": correct[group],
            "total": totals[group],
        }
    return out


def build_fraction_report(paths: dict[str, Path]):
    payloads = {name: load_json(path) for name, path in paths.items()}
    seq_tables = {name: sequence_rows(payload) for name, payload in payloads.items()}
    slice_tables = {name: slice_rows(rows) for name, rows in seq_tables.items()}

    slices = sorted(set().union(*[set(table.keys()) for table in slice_tables.values()]))
    slice_frontier = []
    for group in slices:
        row = {"slice": group}
        best_name = None
        best_acc = -1.0
        for name, table in slice_tables.items():
            acc = table[group]["accuracy"]
            row[f"{name}_accuracy"] = acc
            if acc > best_acc:
                best_acc = acc
                best_name = name
        row["best_model"] = best_name
        row["best_accuracy"] = best_acc
        row["symbolic_minus_joint"] = row["symbolic_accuracy"] - row["joint_accuracy"]
        row["symbolic_minus_temporal_hl"] = row["symbolic_accuracy"] - row["temporal_hl_accuracy"]
        row["symbolic_minus_refined_phase"] = row["symbolic_accuracy"] - row["refined_phase_accuracy"]
        slice_frontier.append(row)
    slice_frontier.sort(key=lambda row: (-row["symbolic_minus_joint"], row["slice"]))

    seq_names = sorted(set().union(*[set(table.keys()) for table in seq_tables.values()]))
    sequence_frontier = []
    for seq_name in seq_names:
        row = {"seq_name": seq_name}
        best_name = None
        best_acc = -1.0
        for name, table in seq_tables.items():
            acc = table[seq_name]["accuracy"]
            row[f"{name}_accuracy"] = acc
            row[f"{name}_correct"] = table[seq_name]["correct"]
            if acc > best_acc:
                best_acc = acc
                best_name = name
        row["best_model"] = best_name
        row["best_accuracy"] = best_acc
        row["symbolic_minus_joint"] = row["symbolic_accuracy"] - row["joint_accuracy"]
        row["symbolic_minus_temporal_hl"] = row["symbolic_accuracy"] - row["temporal_hl_accuracy"]
        row["symbolic_minus_refined_phase"] = row["symbolic_accuracy"] - row["refined_phase_accuracy"]
        sequence_frontier.append(row)
    sequence_frontier.sort(key=lambda row: (-row["symbolic_minus_joint"], row["seq_name"]))

    symbolic_best_slices = sum(1 for row in slice_frontier if row["best_model"] == "symbolic")
    symbolic_best_sequences = sum(1 for row in sequence_frontier if row["best_model"] == "symbolic")
    return {
        "artifacts": {name: str(path) for name, path in paths.items()},
        "slice_frontier": slice_frontier,
        "sequence_frontier_top_symbolic": [row for row in sequence_frontier if row["best_model"] == "symbolic"][:15],
        "sequence_frontier_top_non_symbolic": [row for row in sequence_frontier if row["best_model"] != "symbolic"][:15],
        "plain_summary": {
            "symbolic_best_slices": symbolic_best_slices,
            "symbolic_best_sequences": symbolic_best_sequences,
            "all_slice_symbolic_acc": next(row["symbolic_accuracy"] for row in slice_frontier if row["slice"] == "all"),
            "all_slice_joint_acc": next(row["joint_accuracy"] for row in slice_frontier if row["slice"] == "all"),
            "all_slice_temporal_hl_acc": next(row["temporal_hl_accuracy"] for row in slice_frontier if row["slice"] == "all"),
            "all_slice_refined_phase_acc": next(row["refined_phase_accuracy"] for row in slice_frontier if row["slice"] == "all"),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=GEN / "symbolic_control_frontier.json",
    )
    args = parser.parse_args()

    fraction_05 = build_fraction_report(
        {
            "symbolic": GEN / "symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_fraction05_analysis.json",
            "joint": GEN / "joint_sequence_pre186_96_fraction05_analysis_v1.json",
            "temporal_hl": GEN / "temporal_hl_sequence_pre168_84_fraction05_analysis_v1.json",
            "refined_phase": GEN / "refined_phase_sequence_pre186_96_fraction05_analysis_v1.json",
        }
    )
    fraction_10 = build_fraction_report(
        {
            "symbolic": GEN / "symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_analysis.json",
            "joint": GEN / "joint_sequence_pre186_96_analysis_v1.json",
            "temporal_hl": GEN / "temporal_hl_sequence_pre168_84_analysis_v1.json",
            "refined_phase": GEN / "refined_phase_sequence_pre186_96_analysis_v1.json",
        }
    )

    payload = {
        "fraction_0.5": fraction_05,
        "fraction_1.0": fraction_10,
    }
    write_json(args.output, payload)

    summary_dir = GEN / "summary_tables"
    slice_rows_out = []
    for fraction_key, rep in payload.items():
        for row in rep["slice_frontier"]:
            slice_rows_out.append(
                {
                    "fraction": fraction_key,
                    "slice": row["slice"],
                    "symbolic_accuracy": fmt(float(row["symbolic_accuracy"])),
                    "joint_accuracy": fmt(float(row["joint_accuracy"])),
                    "temporal_hl_accuracy": fmt(float(row["temporal_hl_accuracy"])),
                    "refined_phase_accuracy": fmt(float(row["refined_phase_accuracy"])),
                    "best_model": row["best_model"],
                }
            )
    write_csv(
        summary_dir / "symbolic_control_frontier_slices.csv",
        slice_rows_out,
        ["fraction", "slice", "symbolic_accuracy", "joint_accuracy", "temporal_hl_accuracy", "refined_phase_accuracy", "best_model"],
    )
    write_md_table(
        summary_dir / "symbolic_control_frontier_slices.md",
        ["fraction", "slice", "symbolic_accuracy", "joint_accuracy", "temporal_hl_accuracy", "refined_phase_accuracy", "best_model"],
        [
            [
                row["fraction"],
                row["slice"],
                row["symbolic_accuracy"],
                row["joint_accuracy"],
                row["temporal_hl_accuracy"],
                row["refined_phase_accuracy"],
                row["best_model"],
            ]
            for row in slice_rows_out
        ],
        "Symbolic Control Frontier Slices",
    )

    sequence_rows_out = []
    for fraction_key, rep in payload.items():
        for label, rows in [
            ("symbolic_best", rep["sequence_frontier_top_symbolic"]),
            ("non_symbolic_best", rep["sequence_frontier_top_non_symbolic"]),
        ]:
            for row in rows:
                sequence_rows_out.append(
                    {
                        "fraction": fraction_key,
                        "bucket": label,
                        "seq_name": row["seq_name"],
                        "symbolic_accuracy": fmt(float(row["symbolic_accuracy"])),
                        "joint_accuracy": fmt(float(row["joint_accuracy"])),
                        "temporal_hl_accuracy": fmt(float(row["temporal_hl_accuracy"])),
                        "refined_phase_accuracy": fmt(float(row["refined_phase_accuracy"])),
                        "best_model": row["best_model"],
                    }
                )
    write_csv(
        summary_dir / "symbolic_control_frontier_sequences.csv",
        sequence_rows_out,
        ["fraction", "bucket", "seq_name", "symbolic_accuracy", "joint_accuracy", "temporal_hl_accuracy", "refined_phase_accuracy", "best_model"],
    )
    write_md_table(
        summary_dir / "symbolic_control_frontier_sequences.md",
        ["fraction", "bucket", "seq_name", "symbolic_accuracy", "joint_accuracy", "temporal_hl_accuracy", "refined_phase_accuracy", "best_model"],
        [
            [
                row["fraction"],
                row["bucket"],
                row["seq_name"],
                row["symbolic_accuracy"],
                row["joint_accuracy"],
                row["temporal_hl_accuracy"],
                row["refined_phase_accuracy"],
                row["best_model"],
            ]
            for row in sequence_rows_out
        ],
        "Symbolic Control Frontier Sequences",
    )

    print(f"output: {args.output}")
    print("fraction_0.5 symbolic_best_slices", payload["fraction_0.5"]["plain_summary"]["symbolic_best_slices"])
    print("fraction_1.0 symbolic_best_slices", payload["fraction_1.0"]["plain_summary"]["symbolic_best_slices"])


if __name__ == "__main__":
    main()
