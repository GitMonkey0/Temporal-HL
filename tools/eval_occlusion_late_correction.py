#!/usr/bin/env python3
"""Evaluate restricted late correction on the symbolic mainline using symbolic retrieval."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from tools.compare_symbolic_slices import slice_membership
from tools.eval_sequence_symbolic_retrieval import (
    canonical_label,
    dtw_similarity,
    iter_sequences,
    overlap_labels,
    sequence_event_sets,
)


FAMILY_LABELS = {
    "ROM03_LT_No_Occlusion",
    "ROM03_RT_No_Occlusion",
    "ROM04_LT_Occlusion",
    "ROM04_RT_Occlusion",
    "ROM05_LT_Wrist_ROM",
    "ROM05_RT_Wrist_ROM",
    "ROM07_Rt_Finger_Occlusions",
    "ROM08_Lt_Finger_Occlusions",
}


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def build_gallery(gallery_data: dict[str, object], family_labels: set[str]):
    allowed = overlap_labels(gallery_data, gallery_data)
    allowed = {label for label in allowed if label in family_labels}
    gallery = []
    for sequence, label in iter_sequences(gallery_data, allowed):
        gallery.append(
            {
                "seq_name": sequence["seq_name"],
                "label": label,
                "events": sequence_event_sets(
                    sequence,
                    mode="temporal",
                    include_persistence=False,
                    include_segment_duration=False,
                ),
            }
        )
    return gallery


def build_query_events(query_data: dict[str, object], family_labels: set[str]):
    allowed = overlap_labels(query_data, query_data)
    allowed = {label for label in allowed if label in family_labels}
    queries = {}
    for sequence, label in iter_sequences(query_data, allowed):
        queries[sequence["seq_name"]] = {
            "seq_name": sequence["seq_name"],
            "label": label,
            "events": sequence_event_sets(
                sequence,
                mode="temporal",
                include_persistence=False,
                include_segment_duration=False,
            ),
        }
    return queries


def retrieve_family_label(query_item, gallery):
    ranking = []
    for item in gallery:
        score = dtw_similarity(query_item["events"], item["events"])
        ranking.append((item["label"], item["seq_name"], score))
    ranking.sort(key=lambda x: x[2], reverse=True)
    return ranking


def aggregate_slice_accuracy(rows):
    totals = Counter()
    correct = Counter()
    for row in rows:
        for group in slice_membership(row["seq_name"]):
            totals[group] += 1
            if row["correct"]:
                correct[group] += 1
    out = []
    for group in sorted(totals):
        out.append(
            {
                "slice": group,
                "total": totals[group],
                "correct": correct[group],
                "accuracy": correct[group] / max(totals[group], 1),
            }
        )
    return out


def compare_slices(old_rows, new_rows):
    old = {row["slice"]: row for row in aggregate_slice_accuracy(old_rows)}
    new = {row["slice"]: row for row in aggregate_slice_accuracy(new_rows)}
    out = []
    for key in sorted(set(old) | set(new)):
        out.append(
            {
                "slice": key,
                "old_accuracy": old.get(key, {}).get("accuracy", 0.0),
                "new_accuracy": new.get(key, {}).get("accuracy", 0.0),
                "delta": new.get(key, {}).get("accuracy", 0.0) - old.get(key, {}).get("accuracy", 0.0),
            }
        )
    out.sort(key=lambda row: (-row["delta"], row["slice"]))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-json", type=Path, required=True)
    parser.add_argument("--gallery-json", type=Path, required=True)
    parser.add_argument("--query-json", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/generated/occlusion_late_correction.json"),
    )
    args = parser.parse_args()

    analysis = load_json(args.analysis_json)
    gallery_data = load_json(args.gallery_json)
    query_data = load_json(args.query_json)
    gallery = build_gallery(gallery_data, FAMILY_LABELS)
    query_events = build_query_events(query_data, FAMILY_LABELS)

    seed_old_rows = defaultdict(list)
    seed_new_rows = defaultdict(list)
    family_rows = []

    for seq_name, item in sorted(analysis["sequence_consistency"].items()):
        preds = item["pretrained_predictions"]
        for pred in preds:
            seed = pred["seed"]
            original_pred = canonical_label(pred["prediction"])
            target = canonical_label(seq_name)
            old_row = {
                "seq_name": seq_name,
                "target": target,
                "prediction": original_pred,
                "correct": original_pred == target,
                "seed": seed,
            }
            new_row = dict(old_row)
            applied = False
            retrieval_pred = None
            top3 = None
            if original_pred in FAMILY_LABELS and seq_name in query_events:
                ranking = retrieve_family_label(query_events[seq_name], gallery)
                retrieval_pred = ranking[0][0]
                top3 = [[label, seq_name, float(score)] for label, seq_name, score in ranking[:3]]
                new_row["prediction"] = retrieval_pred
                new_row["correct"] = retrieval_pred == target
                applied = True
            seed_old_rows[seed].append(old_row)
            seed_new_rows[seed].append(new_row)
            if original_pred in FAMILY_LABELS:
                family_rows.append(
                    {
                        "seq_name": seq_name,
                        "seed": seed,
                        "target": target,
                        "original_prediction": original_pred,
                        "corrected_prediction": new_row["prediction"],
                        "original_correct": old_row["correct"],
                        "corrected_correct": new_row["correct"],
                        "applied": applied,
                        "retrieval_top3": top3,
                    }
                )

    per_seed = []
    for seed in sorted(seed_old_rows):
        old_acc = sum(row["correct"] for row in seed_old_rows[seed]) / max(len(seed_old_rows[seed]), 1)
        new_acc = sum(row["correct"] for row in seed_new_rows[seed]) / max(len(seed_new_rows[seed]), 1)
        per_seed.append(
            {
                "seed": seed,
                "old_accuracy": old_acc,
                "new_accuracy": new_acc,
                "delta": new_acc - old_acc,
            }
        )

    old_family_acc = sum(row["original_correct"] for row in family_rows) / max(len(family_rows), 1)
    new_family_acc = sum(row["corrected_correct"] for row in family_rows) / max(len(family_rows), 1)
    improved = [row for row in family_rows if (not row["original_correct"]) and row["corrected_correct"]]
    harmed = [row for row in family_rows if row["original_correct"] and (not row["corrected_correct"])]

    payload = {
        "analysis_json": str(args.analysis_json),
        "gallery_json": str(args.gallery_json),
        "query_json": str(args.query_json),
        "family_labels": sorted(FAMILY_LABELS),
        "per_seed_accuracy": per_seed,
        "family_summary": {
            "num_family_predictions": len(family_rows),
            "old_family_accuracy": old_family_acc,
            "new_family_accuracy": new_family_acc,
            "family_delta": new_family_acc - old_family_acc,
            "num_improved": len(improved),
            "num_harmed": len(harmed),
        },
        "slice_delta_overall": compare_slices(
            [row for rows in seed_old_rows.values() for row in rows],
            [row for rows in seed_new_rows.values() for row in rows],
        ),
        "slice_delta_family_only": compare_slices([
            {
                "seq_name": row["seq_name"],
                "correct": row["original_correct"],
            }
            for row in family_rows
        ], [
            {
                "seq_name": row["seq_name"],
                "correct": row["corrected_correct"],
            }
            for row in family_rows
        ]),
        "improved_cases": improved,
        "harmed_cases": harmed,
        "family_rows": family_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"output: {args.output}")
    print("family old/new", old_family_acc, new_family_acc, "delta", new_family_acc - old_family_acc)
    print("improved", len(improved), "harmed", len(harmed))


if __name__ == "__main__":
    main()
