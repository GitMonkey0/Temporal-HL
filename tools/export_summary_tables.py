#!/usr/bin/env python3
"""Export CSV and Markdown tables from experiment summary bundle."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def to_md_table(rows: list[dict[str, object]], fieldnames: list[str]) -> str:
    header = "| " + " | ".join(fieldnames) + " |"
    sep = "| " + " | ".join(["---"] * len(fieldnames)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(k, "")) for k in fieldnames) + " |")
    return "\n".join([header, sep, *body]) + "\n"


def write_md(path: Path, rows: list[dict[str, object]], fieldnames: list[str], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"# {title}\n\n" + to_md_table(rows, fieldnames)
    with path.open("w") as f:
        f.write(content)


def protocol_rows(bundle):
    rows = []
    for row in bundle["protocol_matrix"]:
        rows.append(
            {
                "representation": row["representation"],
                "fraction": row["fraction"],
                "protocol": row["protocol"],
                "scratch_seq": f"{row['scratch_seq']:.4f}",
                "pretrained_seq": f"{row['pretrained_seq']:.4f}",
                "scratch_win": f"{row['scratch_win']:.4f}",
                "pretrained_win": f"{row['pretrained_win']:.4f}",
            }
        )
    return rows


def slice_rows(bundle):
    rows = []
    for fraction_key, items in bundle["slice_deltas"].items():
        fraction = "1.0" if "1.0" in fraction_key else "0.5"
        for item in items:
            rows.append(
                {
                    "fraction": fraction,
                    "slice": item["slice"],
                    "old_accuracy": f"{item['old_accuracy']:.4f}",
                    "new_accuracy": f"{item['new_accuracy']:.4f}",
                    "delta": f"{item['delta']:+.4f}",
                    "old_total": item["old_total"],
                    "new_total": item["new_total"],
                }
            )
    return rows


def error_rows(bundle):
    rows = []
    for source_key, items in bundle["error_frontier"].items():
        for item in items:
            rows.append(
                {
                    "source": source_key,
                    "target": item["target"],
                    "prediction": item["prediction"],
                    "count": item["count"],
                }
            )
    return rows


def gain_rows(bundle):
    summary = bundle["plain_summary"]
    return [
        {"representation": "symbolic", "fraction": "0.5", "gain": f"{summary['symbolic_fraction_0.5_gain']:+.4f}"},
        {"representation": "symbolic", "fraction": "1.0", "gain": f"{summary['symbolic_fraction_1.0_gain']:+.4f}"},
        {"representation": "joint_sequence", "fraction": "0.5", "gain": f"{summary['joint_fraction_0.5_gain']:+.4f}"},
        {"representation": "joint_sequence", "fraction": "1.0", "gain": f"{summary['joint_fraction_1.0_gain']:+.4f}"},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("/opt/tiger/hand/experiments/generated/experiment_summary_bundle.json"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("/opt/tiger/hand/experiments/generated/summary_tables"),
    )
    args = parser.parse_args()

    bundle = load_json(args.bundle)

    protocol = protocol_rows(bundle)
    protocol_fields = ["representation", "fraction", "protocol", "scratch_seq", "pretrained_seq", "scratch_win", "pretrained_win"]
    write_csv(args.outdir / "protocol_matrix.csv", protocol, protocol_fields)
    write_md(args.outdir / "protocol_matrix.md", protocol, protocol_fields, "Protocol Matrix")

    slices = slice_rows(bundle)
    slice_fields = ["fraction", "slice", "old_accuracy", "new_accuracy", "delta", "old_total", "new_total"]
    write_csv(args.outdir / "slice_deltas.csv", slices, slice_fields)
    write_md(args.outdir / "slice_deltas.md", slices, slice_fields, "Slice Deltas")

    errors = error_rows(bundle)
    error_fields = ["source", "target", "prediction", "count"]
    write_csv(args.outdir / "error_frontier.csv", errors, error_fields)
    write_md(args.outdir / "error_frontier.md", errors, error_fields, "Error Frontier")

    gains = gain_rows(bundle)
    gain_fields = ["representation", "fraction", "gain"]
    write_csv(args.outdir / "gain_summary.csv", gains, gain_fields)
    write_md(args.outdir / "gain_summary.md", gains, gain_fields, "Gain Summary")

    manifest = {
        "bundle": str(args.bundle),
        "generated": {
            "protocol_matrix_csv": str(args.outdir / "protocol_matrix.csv"),
            "protocol_matrix_md": str(args.outdir / "protocol_matrix.md"),
            "slice_deltas_csv": str(args.outdir / "slice_deltas.csv"),
            "slice_deltas_md": str(args.outdir / "slice_deltas.md"),
            "error_frontier_csv": str(args.outdir / "error_frontier.csv"),
            "error_frontier_md": str(args.outdir / "error_frontier.md"),
            "gain_summary_csv": str(args.outdir / "gain_summary.csv"),
            "gain_summary_md": str(args.outdir / "gain_summary.md"),
        },
    }
    with (args.outdir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"output_dir: {args.outdir}")
    print("generated:", ", ".join(sorted(manifest["generated"].keys())))


if __name__ == "__main__":
    main()
