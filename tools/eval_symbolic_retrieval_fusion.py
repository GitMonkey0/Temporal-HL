#!/usr/bin/env python3
"""Evaluate a retrieval fusion gate on configurable family-correction candidates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tools.audit_family_disagreement_broad import build_query_events_all, family_best_from_row, passes_trigger
from tools.compare_symbolic_slices import slice_membership
from tools.eval_occlusion_late_correction import FAMILY_LABELS, build_gallery, retrieve_family_label
from tools.eval_sequence_symbolic_retrieval import canonical_label, load_json


FAMILY_SET = set(FAMILY_LABELS)


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
        old_acc = old.get(key, {}).get("accuracy", 0.0)
        new_acc = new.get(key, {}).get("accuracy", 0.0)
        out.append(
            {
                "slice": key,
                "old_accuracy": old_acc,
                "new_accuracy": new_acc,
                "delta": new_acc - old_acc,
            }
        )
    out.sort(key=lambda row: (-row["delta"], row["slice"]))
    return out


def parse_thresholds(raw: str):
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("No thresholds provided")
    return sorted(set(values))


def margin_from_top3(top3):
    if top3 is None or len(top3) < 2:
        return None
    return float(top3[0][1] - top3[1][1])


def retrieval_margin_from_top3(top3):
    if top3 is None or len(top3) < 2:
        return None
    return float(top3[0][2] - top3[1][2])


def load_detailed_rows(paths: list[Path]):
    rows = []
    for path in paths:
        data = load_json(path)
        seed = data["seed"]
        for row in data["details"]["pretrained"]["sequence_rows"]:
            family_best_label, family_best_score = family_best_from_row(row)
            record = {
                "source": str(path),
                "seed": seed,
                "seq_name": row["seq_name"],
                "target": canonical_label(row["target"]),
                "prediction": canonical_label(row["prediction"]),
                "correct": bool(row["correct"]),
                "mainline_top3": row["top3"],
                "mainline_margin": margin_from_top3(row["top3"]),
                "prediction_in_family": canonical_label(row["prediction"]) in FAMILY_SET,
                "top3_family_labels": [canonical_label(label) for label, _ in row["top3"] if canonical_label(label) in FAMILY_SET],
                "family_best_label": family_best_label,
                "family_best_score": family_best_score,
                "top1_score": float(row["top3"][0][1]),
                "family_gap_to_top1": None if family_best_score is None else float(row["top3"][0][1]) - family_best_score,
            }
            record["target_in_family"] = record["target"] in FAMILY_SET
            rows.append(record)
    return rows


def attach_retrieval(base_rows, query_events, gallery):
    enriched = []
    cache = {}
    for row in base_rows:
        new_row = dict(row)
        retrieval_top3 = None
        retrieval_prediction = None
        retrieval_margin = None
        if row["seq_name"] in query_events:
            if row["seq_name"] not in cache:
                ranking = retrieve_family_label(query_events[row["seq_name"]], gallery)
                cache[row["seq_name"]] = [[label, seq_name, float(score)] for label, seq_name, score in ranking[:3]]
            retrieval_top3 = cache[row["seq_name"]]
            retrieval_prediction = canonical_label(retrieval_top3[0][0])
            retrieval_margin = retrieval_margin_from_top3(retrieval_top3)
        new_row["retrieval_top3"] = retrieval_top3
        new_row["retrieval_prediction"] = retrieval_prediction
        new_row["retrieval_margin"] = retrieval_margin
        enriched.append(new_row)
    return enriched


def parse_scope(raw: str):
    if raw not in {"all", "target_family_only"}:
        raise ValueError(f"Unknown scope: {raw}")
    return raw


def row_in_scope(row, scope):
    if scope == "target_family_only":
        return row["target_in_family"]
    return row["target_in_family"] or row["prediction_in_family"] or bool(row["top3_family_labels"]) or row["family_gap_to_top1"] is not None


def select_candidate_rows(base_rows, trigger, scope, margin_max, family_gap_max):
    out = []
    for row in base_rows:
        if not row_in_scope(row, scope):
            continue
        if not passes_trigger(row, trigger, margin_max, family_gap_max):
            continue
        out.append(row)
    return out


def evaluate_combo(base_rows, mainline_max_margin, retrieval_min_margin, trigger, scope, candidate_margin_max, candidate_family_gap_max):
    old_rows = []
    new_rows = []
    family_rows = []
    improved = 0
    harmed = 0
    applied = 0
    changed = 0

    candidate_rows = select_candidate_rows(base_rows, trigger, scope, candidate_margin_max, candidate_family_gap_max)

    for row in candidate_rows:
        old_rows.append({"seq_name": row["seq_name"], "correct": row["correct"]})
        new_prediction = row["prediction"]
        new_correct = row["correct"]
        retrieval_top3 = row.get("retrieval_top3")
        retrieval_prediction = row.get("retrieval_prediction")
        retrieval_margin = row.get("retrieval_margin")
        applied_here = False
        if retrieval_prediction is not None:
            if (
                retrieval_prediction != row["prediction"]
                and row["mainline_margin"] is not None
                and row["mainline_margin"] <= mainline_max_margin
                and retrieval_margin is not None
                and retrieval_margin >= retrieval_min_margin
            ):
                new_prediction = retrieval_prediction
                new_correct = new_prediction == row["target"]
                applied_here = True
                applied += 1
                changed += 1

        new_rows.append({"seq_name": row["seq_name"], "correct": new_correct})
        family_rows.append(
            {
                "seq_name": row["seq_name"],
                "seed": row["seed"],
                "target": row["target"],
                "original_prediction": row["prediction"],
                "new_prediction": new_prediction,
                "original_correct": row["correct"],
                "new_correct": new_correct,
                "mainline_margin": row["mainline_margin"],
                "retrieval_margin": retrieval_margin,
                "retrieval_prediction": retrieval_prediction,
                "retrieval_top3": retrieval_top3,
                "applied": applied_here,
            }
        )
        if (not row["correct"]) and new_correct:
            improved += 1
        if row["correct"] and (not new_correct):
            harmed += 1

    overall_old = sum(row["correct"] for row in old_rows) / max(len(old_rows), 1)
    overall_new = sum(row["correct"] for row in new_rows) / max(len(new_rows), 1)
    family_old = sum(row["original_correct"] for row in family_rows) / max(len(family_rows), 1)
    family_new = sum(row["new_correct"] for row in family_rows) / max(len(family_rows), 1)

    return {
        "candidate_trigger": trigger,
        "candidate_scope": scope,
        "candidate_margin_max": candidate_margin_max,
        "candidate_family_gap_max": candidate_family_gap_max,
        "mainline_max_margin": mainline_max_margin,
        "retrieval_min_margin": retrieval_min_margin,
        "overall_accuracy": {"old": overall_old, "new": overall_new, "delta": overall_new - overall_old},
        "candidate_accuracy": {"old": family_old, "new": family_new, "delta": family_new - family_old},
        "counts": {
            "num_candidate_rows": len(old_rows),
            "num_applied": applied,
            "num_changed_prediction": changed,
            "num_improved": improved,
            "num_harmed": harmed,
        },
        "slice_delta_overall": compare_slices(old_rows, new_rows),
        "slice_delta_candidate_only": compare_slices(
            [{"seq_name": row["seq_name"], "correct": row["original_correct"]} for row in family_rows],
            [{"seq_name": row["seq_name"], "correct": row["new_correct"]} for row in family_rows],
        ),
        "applied_cases": [row for row in family_rows if row["applied"]],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detailed-analysis", type=Path, nargs="+", required=True)
    parser.add_argument("--gallery-json", type=Path, required=True)
    parser.add_argument("--query-json", type=Path, required=True)
    parser.add_argument("--mainline-thresholds", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.5,2.0")
    parser.add_argument("--retrieval-thresholds", type=str, default="0.0,0.04,0.045,0.0475,0.05,0.0525,0.055,0.0575,0.06")
    parser.add_argument("--candidate-trigger", type=str, default="pred_in_family")
    parser.add_argument("--candidate-scope", type=str, default="target_family_only")
    parser.add_argument("--candidate-margin-max", type=float, default=0.0)
    parser.add_argument("--candidate-family-gap-max", type=float, default=0.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/generated/symbolic_retrieval_fusion.json"),
    )
    args = parser.parse_args()

    base_rows = load_detailed_rows(args.detailed_analysis)
    gallery_data = load_json(args.gallery_json)
    query_data = load_json(args.query_json)
    gallery = build_gallery(gallery_data, FAMILY_LABELS)
    query_events = build_query_events_all(query_data)
    base_rows = attach_retrieval(base_rows, query_events, gallery)
    candidate_scope = parse_scope(args.candidate_scope)

    mainline_thresholds = parse_thresholds(args.mainline_thresholds)
    retrieval_thresholds = parse_thresholds(args.retrieval_thresholds)

    sweeps = []
    for mainline_thr in mainline_thresholds:
        for retrieval_thr in retrieval_thresholds:
            sweeps.append(
                evaluate_combo(
                    base_rows,
                    mainline_thr,
                    retrieval_thr,
                    args.candidate_trigger,
                    candidate_scope,
                    args.candidate_margin_max,
                    args.candidate_family_gap_max,
                )
            )

    sweeps.sort(
        key=lambda row: (
            -row["overall_accuracy"]["delta"],
            row["counts"]["num_harmed"],
            -row["counts"]["num_improved"],
            row["counts"]["num_applied"],
            row["mainline_max_margin"],
            -row["retrieval_min_margin"],
        )
    )
    zero_harm = [row for row in sweeps if row["counts"]["num_harmed"] == 0]

    payload = {
        "detailed_analysis": [str(path) for path in args.detailed_analysis],
        "gallery_json": str(args.gallery_json),
        "query_json": str(args.query_json),
        "candidate_trigger": args.candidate_trigger,
        "candidate_scope": candidate_scope,
        "candidate_margin_max": args.candidate_margin_max,
        "candidate_family_gap_max": args.candidate_family_gap_max,
        "mainline_thresholds": mainline_thresholds,
        "retrieval_thresholds": retrieval_thresholds,
        "base_overall_accuracy": sum(row["correct"] for row in base_rows) / max(len(base_rows), 1),
        "base_candidate_rows": len(select_candidate_rows(base_rows, args.candidate_trigger, candidate_scope, args.candidate_margin_max, args.candidate_family_gap_max)),
        "sweeps": sweeps,
        "best_zero_harm": zero_harm[:10],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"output: {args.output}")
    for row in zero_harm[:5]:
        print(
            "zero_harm",
            "m<=",
            row["mainline_max_margin"],
            "r>=",
            row["retrieval_min_margin"],
            "overall",
            row["overall_accuracy"]["old"],
            "->",
            row["overall_accuracy"]["new"],
            "delta",
            row["overall_accuracy"]["delta"],
            "applied",
            row["counts"]["num_applied"],
            "improved",
            row["counts"]["num_improved"],
        )


if __name__ == "__main__":
    main()
