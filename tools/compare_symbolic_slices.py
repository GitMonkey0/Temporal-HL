#!/usr/bin/env python3
"""Compare sequence-slice performance between two symbolic analysis files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


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


def aggregate_slices(payload: dict[str, object], mode: str) -> dict[str, dict[str, float]]:
    totals = defaultdict(int)
    correct = defaultdict(int)
    runs = payload["runs"]
    for run in runs:
        for row in run[f"{mode}_errors"]:
            pass
    seq_entries = payload["sequence_consistency"]
    key_name = f"{mode}_predictions"
    for seq_name, item in seq_entries.items():
        predictions = item[key_name]
        for pred in predictions:
            for group in slice_membership(seq_name):
                totals[group] += 1
                if pred["correct"]:
                    correct[group] += 1
    out = {}
    for group in sorted(totals):
        out[group] = {
            "total": totals[group],
            "correct": correct[group],
            "accuracy": correct[group] / max(totals[group], 1),
        }
    return out


def diff_table(old_slices, new_slices):
    rows = []
    for group in sorted(set(old_slices) | set(new_slices)):
        old_acc = old_slices.get(group, {}).get("accuracy", 0.0)
        new_acc = new_slices.get(group, {}).get("accuracy", 0.0)
        rows.append(
            {
                "slice": group,
                "old_accuracy": old_acc,
                "new_accuracy": new_acc,
                "delta": new_acc - old_acc,
                "old_total": old_slices.get(group, {}).get("total", 0),
                "new_total": new_slices.get(group, {}).get("total", 0),
            }
        )
    rows.sort(key=lambda x: (-x["delta"], x["slice"]))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-analysis", type=Path, required=True)
    parser.add_argument("--new-analysis", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/generated/symbolic_slice_compare.json"),
    )
    args = parser.parse_args()

    old_payload = load_json(args.old_analysis)
    new_payload = load_json(args.new_analysis)
    payload = {
        "old_analysis": str(args.old_analysis),
        "new_analysis": str(args.new_analysis),
        "pretrained_old": aggregate_slices(old_payload, "pretrained"),
        "pretrained_new": aggregate_slices(new_payload, "pretrained"),
        "scratch_old": aggregate_slices(old_payload, "scratch"),
        "scratch_new": aggregate_slices(new_payload, "scratch"),
    }
    payload["pretrained_diff"] = diff_table(
        payload["pretrained_old"], payload["pretrained_new"]
    )
    payload["scratch_diff"] = diff_table(payload["scratch_old"], payload["scratch_new"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"output: {args.output}")
    for row in payload["pretrained_diff"]:
        print(
            f"{row['slice']}: old={row['old_accuracy']:.4f} "
            f"new={row['new_accuracy']:.4f} delta={row['delta']:+.4f}"
        )


if __name__ == "__main__":
    main()
