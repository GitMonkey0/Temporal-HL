#!/usr/bin/env python3
"""Compare strongest symbolic and joint-sequence mainlines on shared slices and sequences."""

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


def sequence_outcome_table(payload: dict[str, object], mode: str = "pretrained"):
    key = f"{mode}_predictions"
    rows = {}
    for seq_name, item in payload["sequence_consistency"].items():
        preds = item[key]
        num_correct = sum(1 for pred in preds if pred["correct"])
        rows[seq_name] = {
            "seq_name": seq_name,
            "num_correct": num_correct,
            "num_total": len(preds),
            "accuracy": num_correct / max(len(preds), 1),
            "predictions": preds,
        }
    return rows


def slice_accuracy_table(seq_rows: dict[str, dict[str, object]]):
    totals = defaultdict(int)
    correct = defaultdict(int)
    for seq_name, row in seq_rows.items():
        for group in slice_membership(seq_name):
            totals[group] += row["num_total"]
            correct[group] += row["num_correct"]
    out = []
    for group in sorted(totals):
        out.append(
            {
                "slice": group,
                "correct": correct[group],
                "total": totals[group],
                "accuracy": correct[group] / max(totals[group], 1),
            }
        )
    return out


def compare_slice_tables(symbolic_rows: dict[str, dict[str, object]], joint_rows: dict[str, dict[str, object]]):
    sym = {row["slice"]: row for row in slice_accuracy_table(symbolic_rows)}
    jnt = {row["slice"]: row for row in slice_accuracy_table(joint_rows)}
    out = []
    for key in sorted(set(sym) | set(jnt)):
        sym_acc = sym.get(key, {}).get("accuracy", 0.0)
        jnt_acc = jnt.get(key, {}).get("accuracy", 0.0)
        out.append(
            {
                "slice": key,
                "symbolic_accuracy": sym_acc,
                "joint_accuracy": jnt_acc,
                "symbolic_minus_joint": sym_acc - jnt_acc,
                "symbolic_total": sym.get(key, {}).get("total", 0),
                "joint_total": jnt.get(key, {}).get("total", 0),
            }
        )
    out.sort(key=lambda row: (-row["symbolic_minus_joint"], row["slice"]))
    return out


def compare_sequences(symbolic_rows: dict[str, dict[str, object]], joint_rows: dict[str, dict[str, object]]):
    out = []
    for seq_name in sorted(set(symbolic_rows) | set(joint_rows)):
        sym = symbolic_rows[seq_name]
        jnt = joint_rows[seq_name]
        out.append(
            {
                "seq_name": seq_name,
                "symbolic_correct": sym["num_correct"],
                "joint_correct": jnt["num_correct"],
                "num_total": sym["num_total"],
                "symbolic_accuracy": sym["accuracy"],
                "joint_accuracy": jnt["accuracy"],
                "symbolic_minus_joint": sym["accuracy"] - jnt["accuracy"],
            }
        )
    out.sort(key=lambda row: (-row["symbolic_minus_joint"], row["seq_name"]))
    return out


def compare_error_frontier(symbolic_errors: list[dict[str, object]], joint_errors: list[dict[str, object]], limit: int = 20):
    sym = {(row["target"], row["prediction"]): row["count"] for row in symbolic_errors}
    jnt = {(row["target"], row["prediction"]): row["count"] for row in joint_errors}
    rows = []
    for key in sorted(set(sym) | set(jnt)):
        sym_count = sym.get(key, 0)
        jnt_count = jnt.get(key, 0)
        rows.append(
            {
                "target": key[0],
                "prediction": key[1],
                "symbolic_count": sym_count,
                "joint_count": jnt_count,
                "joint_minus_symbolic": jnt_count - sym_count,
            }
        )
    rows.sort(key=lambda row: (-row["joint_minus_symbolic"], row["target"], row["prediction"]))
    return rows[:limit]


def plain_summary(seq_rows: list[dict[str, object]], slice_rows: list[dict[str, object]]):
    symbolic_better = sum(1 for row in seq_rows if row["symbolic_minus_joint"] > 0)
    joint_better = sum(1 for row in seq_rows if row["symbolic_minus_joint"] < 0)
    tie = sum(1 for row in seq_rows if abs(row["symbolic_minus_joint"]) < 1e-12)
    top_slice = max(slice_rows, key=lambda row: row["symbolic_minus_joint"])
    worst_slice = min(slice_rows, key=lambda row: row["symbolic_minus_joint"])
    return {
        "symbolic_better_sequences": symbolic_better,
        "joint_better_sequences": joint_better,
        "tied_sequences": tie,
        "best_symbolic_slice": top_slice["slice"],
        "best_symbolic_slice_gap": top_slice["symbolic_minus_joint"],
        "worst_symbolic_slice": worst_slice["slice"],
        "worst_symbolic_slice_gap": worst_slice["symbolic_minus_joint"],
    }


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
        default=GEN / "symbolic_vs_joint_report.json",
    )
    args = parser.parse_args()

    fraction_10 = build_fraction_report(
        GEN / "symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_analysis.json",
        GEN / "joint_sequence_pre186_96_analysis_v1.json",
    )
    fraction_05 = build_fraction_report(
        GEN / "symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_fraction05_analysis.json",
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
                    "symbolic_accuracy": fmt(float(row["symbolic_accuracy"])),
                    "joint_accuracy": fmt(float(row["joint_accuracy"])),
                    "symbolic_minus_joint": fmt(float(row["symbolic_minus_joint"])),
                }
            )
    write_csv(
        summary_dir / "symbolic_vs_joint_slices.csv",
        slice_rows,
        ["fraction", "slice", "symbolic_accuracy", "joint_accuracy", "symbolic_minus_joint"],
    )
    write_md_table(
        summary_dir / "symbolic_vs_joint_slices.md",
        ["fraction", "slice", "symbolic_accuracy", "joint_accuracy", "symbolic_minus_joint"],
        [
            [row["fraction"], row["slice"], row["symbolic_accuracy"], row["joint_accuracy"], row["symbolic_minus_joint"]]
            for row in slice_rows
        ],
        "Symbolic vs Joint Slices",
    )

    seq_rows = []
    for fraction_key, rep in payload.items():
        for row in rep["sequence_comparison_top_symbolic"]:
            seq_rows.append(
                {
                    "fraction": fraction_key,
                    "advantage": "symbolic",
                    "seq_name": row["seq_name"],
                    "symbolic_correct": row["symbolic_correct"],
                    "joint_correct": row["joint_correct"],
                    "num_total": row["num_total"],
                    "symbolic_minus_joint": fmt(float(row["symbolic_minus_joint"])),
                }
            )
        for row in rep["sequence_comparison_top_joint"]:
            seq_rows.append(
                {
                    "fraction": fraction_key,
                    "advantage": "joint",
                    "seq_name": row["seq_name"],
                    "symbolic_correct": row["symbolic_correct"],
                    "joint_correct": row["joint_correct"],
                    "num_total": row["num_total"],
                    "symbolic_minus_joint": fmt(float(row["symbolic_minus_joint"])),
                }
            )
    write_csv(
        summary_dir / "symbolic_vs_joint_sequences.csv",
        seq_rows,
        ["fraction", "advantage", "seq_name", "symbolic_correct", "joint_correct", "num_total", "symbolic_minus_joint"],
    )
    write_md_table(
        summary_dir / "symbolic_vs_joint_sequences.md",
        ["fraction", "advantage", "seq_name", "symbolic_correct", "joint_correct", "num_total", "symbolic_minus_joint"],
        [
            [
                row["fraction"],
                row["advantage"],
                row["seq_name"],
                row["symbolic_correct"],
                row["joint_correct"],
                row["num_total"],
                row["symbolic_minus_joint"],
            ]
            for row in seq_rows
        ],
        "Symbolic vs Joint Sequences",
    )

    frontier_rows = []
    for fraction_key, rep in payload.items():
        for row in rep["error_frontier_joint_minus_symbolic"]:
            frontier_rows.append(
                {
                    "fraction": fraction_key,
                    "target": row["target"],
                    "prediction": row["prediction"],
                    "symbolic_count": row["symbolic_count"],
                    "joint_count": row["joint_count"],
                    "joint_minus_symbolic": row["joint_minus_symbolic"],
                }
            )
    write_csv(
        summary_dir / "symbolic_vs_joint_error_frontier.csv",
        frontier_rows,
        ["fraction", "target", "prediction", "symbolic_count", "joint_count", "joint_minus_symbolic"],
    )
    write_md_table(
        summary_dir / "symbolic_vs_joint_error_frontier.md",
        ["fraction", "target", "prediction", "symbolic_count", "joint_count", "joint_minus_symbolic"],
        [
            [
                row["fraction"],
                row["target"],
                row["prediction"],
                row["symbolic_count"],
                row["joint_count"],
                row["joint_minus_symbolic"],
            ]
            for row in frontier_rows
        ],
        "Symbolic vs Joint Error Frontier",
    )

    print(f"output: {args.output}")
    print(
        "fraction_0.5 symbolic better / joint better:",
        payload["fraction_0.5"]["plain_summary"]["symbolic_better_sequences"],
        payload["fraction_0.5"]["plain_summary"]["joint_better_sequences"],
    )
    print(
        "fraction_1.0 symbolic better / joint better:",
        payload["fraction_1.0"]["plain_summary"]["symbolic_better_sequences"],
        payload["fraction_1.0"]["plain_summary"]["joint_better_sequences"],
    )


if __name__ == "__main__":
    main()
