#!/usr/bin/env python3
"""Audit whether family retrieval disagreements provide positive and negative supervision."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tools.analyze_symbolic_pretrain import FAMILY_LABELS
from tools.eval_occlusion_late_correction import build_gallery, build_query_events, retrieve_family_label
from tools.eval_sequence_symbolic_retrieval import canonical_label, load_json


def load_family_rows(path: Path):
    data = load_json(path)
    rows = []
    if "details" in data:
        seed = data["seed"]
        for row in data["details"]["pretrained"]["sequence_rows"]:
            pred = canonical_label(row["prediction"])
            if pred not in FAMILY_LABELS:
                continue
            rows.append(
                {
                    "seed": seed,
                    "seq_name": row["seq_name"],
                    "target": canonical_label(row["target"]),
                    "original_prediction": pred,
                    "original_correct": bool(row["correct"]),
                }
            )
    else:
        for seq_name, item in data["sequence_consistency"].items():
            target = canonical_label(seq_name)
            for pred in item["pretrained_predictions"]:
                original = canonical_label(pred["prediction"])
                if original not in FAMILY_LABELS:
                    continue
                rows.append(
                    {
                        "seed": pred["seed"],
                        "seq_name": seq_name,
                        "target": target,
                        "original_prediction": original,
                        "original_correct": bool(pred["correct"]),
                    }
                )
    return rows


def audit_rows(rows, query_events, gallery):
    by_seq = {}
    for row in rows:
        if row["seq_name"] not in query_events:
            continue
        if row["seq_name"] not in by_seq:
            ranking = retrieve_family_label(query_events[row["seq_name"]], gallery)
            by_seq[row["seq_name"]] = canonical_label(ranking[0][0])
    counts = Counter()
    disagreements = []
    for row in rows:
        retrieval_prediction = by_seq.get(row["seq_name"])
        if retrieval_prediction is None:
            continue
        counts["family_rows"] += 1
        if row["original_correct"]:
            counts["original_correct"] += 1
        if retrieval_prediction != row["original_prediction"]:
            counts["disagreement_rows"] += 1
            choose_retrieval_positive = retrieval_prediction == row["target"]
            choose_retrieval_negative = retrieval_prediction != row["target"]
            counts["disagreement_positive"] += int(choose_retrieval_positive)
            counts["disagreement_negative"] += int(choose_retrieval_negative)
            disagreements.append(
                {
                    "seed": row["seed"],
                    "seq_name": row["seq_name"],
                    "target": row["target"],
                    "original_prediction": row["original_prediction"],
                    "retrieval_prediction": retrieval_prediction,
                    "original_correct": row["original_correct"],
                    "retrieval_is_correct": choose_retrieval_positive,
                }
            )
        else:
            counts["agreement_rows"] += 1
    return {
        "counts": dict(counts),
        "disagreement_rows": disagreements,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-json", type=Path, nargs="+", required=True)
    parser.add_argument("--gallery-json", type=Path, required=True)
    parser.add_argument("--query-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("experiments/generated/family_disagreement_audit.json"))
    args = parser.parse_args()

    gallery = build_gallery(load_json(args.gallery_json), set(FAMILY_LABELS))
    query_events = build_query_events(load_json(args.query_json), set(FAMILY_LABELS))

    audits = []
    for path in args.analysis_json:
        rows = load_family_rows(path)
        payload = audit_rows(rows, query_events, gallery)
        payload["analysis_json"] = str(path)
        audits.append(payload)

    summary = []
    for item in audits:
        c = item["counts"]
        summary.append(
            {
                "analysis_json": item["analysis_json"],
                "family_rows": c.get("family_rows", 0),
                "agreement_rows": c.get("agreement_rows", 0),
                "disagreement_rows": c.get("disagreement_rows", 0),
                "disagreement_positive": c.get("disagreement_positive", 0),
                "disagreement_negative": c.get("disagreement_negative", 0),
            }
        )

    payload = {"audits": audits, "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"output: {args.output}")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
