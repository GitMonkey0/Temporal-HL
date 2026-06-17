#!/usr/bin/env python3
"""Audit broader family-correction candidate cohorts beyond top1-in-family."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tools.analyze_symbolic_pretrain import FAMILY_LABELS
from tools.eval_occlusion_late_correction import build_gallery, retrieve_family_label
from tools.eval_sequence_symbolic_retrieval import canonical_label, load_json
from tools.eval_sequence_symbolic_retrieval import iter_sequences, overlap_labels, sequence_event_sets


FAMILY_SET = set(FAMILY_LABELS)


def top3_to_map(top3):
    return {canonical_label(label): float(score) for label, score in top3}


def family_best_from_row(row):
    if "family_scores" in row and row["family_scores"]:
        items = [
            (canonical_label(label), float(score))
            for label, score in row["family_scores"].items()
            if canonical_label(label) in FAMILY_SET
        ]
        if items:
            return max(items, key=lambda x: x[1])
    top3_family = [
        (canonical_label(label), float(score))
        for label, score in row["top3"]
        if canonical_label(label) in FAMILY_SET
    ]
    if top3_family:
        return max(top3_family, key=lambda x: x[1])
    return None, None


def load_rows(path: Path):
    data = load_json(path)
    seed = data["seed"]
    rows = []
    for row in data["details"]["pretrained"]["sequence_rows"]:
        pred = canonical_label(row["prediction"])
        target = canonical_label(row["target"])
        top3 = [(canonical_label(label), float(score)) for label, score in row["top3"]]
        top1_score = float(top3[0][1])
        top2_score = float(top3[1][1]) if len(top3) > 1 else top1_score
        family_best_label, family_best_score = family_best_from_row(row)
        rows.append(
            {
                "seed": seed,
                "source": str(path),
                "seq_name": row["seq_name"],
                "target": target,
                "original_prediction": pred,
                "original_correct": bool(row["correct"]),
                "top3": top3,
                "top3_map": top3_to_map(top3),
                "top1_score": top1_score,
                "top2_score": top2_score,
                "mainline_margin": top1_score - top2_score,
                "top3_family_labels": [label for label, _ in top3 if label in FAMILY_SET],
                "family_best_label": family_best_label,
                "family_best_score": family_best_score,
                "family_gap_to_top1": None if family_best_score is None else top1_score - family_best_score,
                "target_in_family": target in FAMILY_SET,
                "prediction_in_family": pred in FAMILY_SET,
            }
        )
    return rows


def build_query_events_all(query_data: dict[str, object]):
    allowed = overlap_labels(query_data, query_data)
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


def build_retrieval_cache(rows, query_events, gallery):
    cache = {}
    for row in rows:
        if row["seq_name"] in query_events and row["seq_name"] not in cache:
            ranking = retrieve_family_label(query_events[row["seq_name"]], gallery)
            cache[row["seq_name"]] = [
                [canonical_label(label), seq_name, float(score)]
                for label, seq_name, score in ranking[:3]
            ]
    return cache


def passes_trigger(row, trigger, margin_max, family_gap_max):
    if trigger == "pred_in_family":
        return row["prediction_in_family"]
    if trigger == "top3_family":
        return bool(row["top3_family_labels"])
    if trigger == "uncertain_margin":
        return row["mainline_margin"] <= margin_max
    if trigger == "family_gap":
        return row["family_gap_to_top1"] is not None and row["family_gap_to_top1"] <= family_gap_max
    if trigger == "pred_or_top3":
        return row["prediction_in_family"] or bool(row["top3_family_labels"])
    if trigger == "top3_or_margin":
        return bool(row["top3_family_labels"]) or row["mainline_margin"] <= margin_max
    if trigger == "pred_or_gap":
        return row["prediction_in_family"] or (
            row["family_gap_to_top1"] is not None and row["family_gap_to_top1"] <= family_gap_max
        )
    if trigger == "top3_or_gap":
        return bool(row["top3_family_labels"]) or (
            row["family_gap_to_top1"] is not None and row["family_gap_to_top1"] <= family_gap_max
        )
    if trigger == "pred_or_top3_or_gap":
        return (
            row["prediction_in_family"]
            or bool(row["top3_family_labels"])
            or (row["family_gap_to_top1"] is not None and row["family_gap_to_top1"] <= family_gap_max)
        )
    raise ValueError(f"Unknown trigger: {trigger}")


def audit_trigger(rows, retrieval_cache, trigger, margin_max, family_gap_max, scope):
    counts = Counter()
    disagreements = []
    candidate_rows = []
    for row in rows:
        if scope == "target_family_only" and not row["target_in_family"]:
            continue
        if scope == "all" and not (row["target_in_family"] or row["prediction_in_family"] or row["top3_family_labels"]):
            # Keep all likely family-confusion regions when not filtering by target.
            if row["family_gap_to_top1"] is None:
                continue
        if not passes_trigger(row, trigger, margin_max, family_gap_max):
            continue
        retrieval_top3 = retrieval_cache.get(row["seq_name"])
        if not retrieval_top3:
            continue
        retrieval_prediction = canonical_label(retrieval_top3[0][0])
        candidate = {
            "seed": row["seed"],
            "source": row["source"],
            "seq_name": row["seq_name"],
            "target": row["target"],
            "original_prediction": row["original_prediction"],
            "retrieval_prediction": retrieval_prediction,
            "original_correct": row["original_correct"],
            "prediction_in_family": row["prediction_in_family"],
            "top3_family_labels": row["top3_family_labels"],
            "mainline_margin": row["mainline_margin"],
            "family_best_label": row["family_best_label"],
            "family_best_score": row["family_best_score"],
            "family_gap_to_top1": row["family_gap_to_top1"],
            "retrieval_top3": retrieval_top3,
        }
        candidate_rows.append(candidate)
        counts["candidate_rows"] += 1
        counts["candidate_original_correct"] += int(row["original_correct"])
        if retrieval_prediction == row["target"]:
            counts["retrieval_correct"] += 1
        if retrieval_prediction != row["original_prediction"]:
            counts["disagreement_rows"] += 1
            positive = retrieval_prediction == row["target"]
            negative = retrieval_prediction != row["target"]
            counts["disagreement_positive"] += int(positive)
            counts["disagreement_negative"] += int(negative)
            disagreements.append(
                {
                    **candidate,
                    "retrieval_is_correct": positive,
                }
            )
        else:
            counts["agreement_rows"] += 1

    out = {
        "trigger": trigger,
        "scope": scope,
        "margin_max": margin_max,
        "family_gap_max": family_gap_max,
        "counts": dict(counts),
        "disagreement_rows": disagreements,
        "candidate_examples": candidate_rows[:20],
    }
    if counts["candidate_rows"]:
        out["rates"] = {
            "candidate_original_accuracy": counts["candidate_original_correct"] / counts["candidate_rows"],
            "candidate_retrieval_accuracy": counts["retrieval_correct"] / counts["candidate_rows"],
            "disagreement_rate": counts["disagreement_rows"] / counts["candidate_rows"],
        }
    else:
        out["rates"] = {
            "candidate_original_accuracy": 0.0,
            "candidate_retrieval_accuracy": 0.0,
            "disagreement_rate": 0.0,
        }
    return out


def parse_floats(raw: str):
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("Expected at least one float")
    return values


def parse_strings(raw: str):
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(item)
    if not values:
        raise ValueError("Expected at least one trigger")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-json", type=Path, nargs="+", required=True)
    parser.add_argument("--gallery-json", type=Path, required=True)
    parser.add_argument("--query-json", type=Path, required=True)
    parser.add_argument(
        "--triggers",
        type=str,
        default="pred_in_family,top3_family,uncertain_margin,family_gap,pred_or_top3,top3_or_margin,pred_or_gap,top3_or_gap,pred_or_top3_or_gap",
    )
    parser.add_argument("--margin-thresholds", type=str, default="0.25,0.5,0.75,1.0,1.5")
    parser.add_argument("--family-gap-thresholds", type=str, default="0.25,0.5,0.75,1.0,1.5,2.0")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/generated/family_disagreement_audit_broad.json"),
    )
    parser.add_argument("--scope", type=str, default="all", choices=["all", "target_family_only"])
    args = parser.parse_args()

    rows = []
    for path in args.analysis_json:
        rows.extend(load_rows(path))
    gallery = build_gallery(load_json(args.gallery_json), FAMILY_SET)
    query_events = build_query_events_all(load_json(args.query_json))
    retrieval_cache = build_retrieval_cache(rows, query_events, gallery)

    triggers = parse_strings(args.triggers)
    margin_thresholds = parse_floats(args.margin_thresholds)
    family_gap_thresholds = parse_floats(args.family_gap_thresholds)

    audits = []
    for trigger in triggers:
        if trigger in {"uncertain_margin", "top3_or_margin"}:
            for margin_max in margin_thresholds:
                audits.append(audit_trigger(rows, retrieval_cache, trigger, margin_max, family_gap_max=0.0, scope=args.scope))
        elif trigger in {"family_gap", "pred_or_gap", "top3_or_gap", "pred_or_top3_or_gap"}:
            for family_gap_max in family_gap_thresholds:
                audits.append(audit_trigger(rows, retrieval_cache, trigger, margin_max=0.0, family_gap_max=family_gap_max, scope=args.scope))
        else:
            audits.append(audit_trigger(rows, retrieval_cache, trigger, margin_max=0.0, family_gap_max=0.0, scope=args.scope))

    summary = []
    for item in audits:
        c = item["counts"]
        summary.append(
            {
                "trigger": item["trigger"],
                "scope": item["scope"],
                "margin_max": item["margin_max"],
                "family_gap_max": item["family_gap_max"],
                "candidate_rows": c.get("candidate_rows", 0),
                "agreement_rows": c.get("agreement_rows", 0),
                "disagreement_rows": c.get("disagreement_rows", 0),
                "disagreement_positive": c.get("disagreement_positive", 0),
                "disagreement_negative": c.get("disagreement_negative", 0),
                "candidate_original_accuracy": item["rates"]["candidate_original_accuracy"],
                "candidate_retrieval_accuracy": item["rates"]["candidate_retrieval_accuracy"],
            }
        )

    summary.sort(
        key=lambda row: (
            -row["disagreement_negative"],
            -row["disagreement_positive"],
            -row["candidate_rows"],
            row["trigger"],
            row["margin_max"],
            row["family_gap_max"],
        )
    )

    payload = {
        "analysis_json": [str(path) for path in args.analysis_json],
        "gallery_json": str(args.gallery_json),
        "query_json": str(args.query_json),
        "scope": args.scope,
        "num_rows_loaded": len(rows),
        "summary": summary,
        "audits": audits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"output: {args.output}")
    for row in summary[:15]:
        print(row)


if __name__ == "__main__":
    main()
