#!/usr/bin/env python3
"""Evaluate a simple label-pair correction rule on configurable candidate cohorts."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
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


def parse_scope(raw: str):
    if raw not in {"all", "target_family_only"}:
        raise ValueError(f"Unknown scope: {raw}")
    return raw


def row_in_scope(row, scope):
    if scope == "target_family_only":
        return row["target_in_family"]
    return row["target_in_family"] or row["prediction_in_family"] or bool(row["top3_family_labels"]) or row["family_gap_to_top1"] is not None


def load_detailed_rows(paths):
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
                "prediction_in_family": canonical_label(row["prediction"]) in FAMILY_SET,
                "top3_family_labels": [canonical_label(label) for label, _ in row["top3"] if canonical_label(label) in FAMILY_SET],
                "family_best_label": family_best_label,
                "family_best_score": family_best_score,
                "top1_score": float(row["top3"][0][1]),
                "mainline_margin": float(row["top3"][0][1] - row["top3"][1][1]) if len(row["top3"]) > 1 else 0.0,
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
        retrieval_prediction = None
        retrieval_top3 = None
        if row["seq_name"] in query_events:
            if row["seq_name"] not in cache:
                ranking = retrieve_family_label(query_events[row["seq_name"]], gallery)
                cache[row["seq_name"]] = [[canonical_label(label), seq_name, float(score)] for label, seq_name, score in ranking]
            retrieval_top3 = cache[row["seq_name"]][:3]
            retrieval_prediction = canonical_label(retrieval_top3[0][0])
        new_row["retrieval_prediction"] = retrieval_prediction
        new_row["retrieval_top3"] = retrieval_top3
        enriched.append(new_row)
    return enriched


def select_candidate_rows(base_rows, trigger, scope, margin_max, family_gap_max):
    out = []
    for row in base_rows:
        if not row_in_scope(row, scope):
            continue
        if not passes_trigger(row, trigger, margin_max, family_gap_max):
            continue
        if row.get("retrieval_prediction") is None:
            continue
        out.append(row)
    return out


def build_pair_rule(train_rows):
    by_pair = defaultdict(list)
    for row in train_rows:
        pair = (row["prediction"], row["retrieval_prediction"])
        would_fix = (row["retrieval_prediction"] != row["prediction"]) and (row["retrieval_prediction"] == row["target"])
        would_harm = row["correct"] and (row["retrieval_prediction"] != row["target"])
        by_pair[pair].append({"would_fix": would_fix, "would_harm": would_harm})
    rule = {}
    for pair, items in by_pair.items():
        fixes = sum(int(x["would_fix"]) for x in items)
        harms = sum(int(x["would_harm"]) for x in items)
        if fixes > harms and fixes > 0:
            rule[pair] = True
        else:
            rule[pair] = False
    return rule


def run_loso(candidate_rows):
    seeds = sorted({row["seed"] for row in candidate_rows})
    outcomes = []
    for test_seed in seeds:
        train_rows = [row for row in candidate_rows if row["seed"] != test_seed]
        test_rows = [row for row in candidate_rows if row["seed"] == test_seed]
        rule = build_pair_rule(train_rows)
        for row in test_rows:
            pair = (row["prediction"], row["retrieval_prediction"])
            apply_gate = bool(rule.get(pair, False)) and row["retrieval_prediction"] != row["prediction"]
            new_prediction = row["retrieval_prediction"] if apply_gate else row["prediction"]
            new_correct = new_prediction == row["target"]
            would_fix = (row["retrieval_prediction"] != row["prediction"]) and (row["retrieval_prediction"] == row["target"])
            would_harm = row["correct"] and (row["retrieval_prediction"] != row["target"])
            outcomes.append(
                {
                    "seed": row["seed"],
                    "seq_name": row["seq_name"],
                    "target": row["target"],
                    "original_prediction": row["prediction"],
                    "retrieval_prediction": row["retrieval_prediction"],
                    "new_prediction": new_prediction,
                    "original_correct": row["correct"],
                    "new_correct": new_correct,
                    "apply_gate": apply_gate,
                    "pair": list(pair),
                    "would_fix": would_fix,
                    "would_harm": would_harm,
                }
            )
    return outcomes


def summarize(outcomes):
    totals = Counter()
    correct = Counter()
    improved = harmed = applied = 0
    for row in outcomes:
        totals["candidate"] += 1
        correct["old"] += int(row["original_correct"])
        correct["new"] += int(row["new_correct"])
        applied += int(row["apply_gate"])
        improved += int((not row["original_correct"]) and row["new_correct"])
        harmed += int(row["original_correct"] and (not row["new_correct"]))
    old_rows = [{"seq_name": row["seq_name"], "correct": row["original_correct"]} for row in outcomes]
    new_rows = [{"seq_name": row["seq_name"], "correct": row["new_correct"]} for row in outcomes]
    return {
        "candidate_accuracy": {
            "old": correct["old"] / max(totals["candidate"], 1),
            "new": correct["new"] / max(totals["candidate"], 1),
            "delta": correct["new"] / max(totals["candidate"], 1) - correct["old"] / max(totals["candidate"], 1),
        },
        "counts": {
            "num_candidate_rows": totals["candidate"],
            "num_applied": applied,
            "num_improved": improved,
            "num_harmed": harmed,
        },
        "slice_delta_candidate_only": compare_slices(old_rows, new_rows),
        "applied_cases": [row for row in outcomes if row["apply_gate"]],
        "all_candidate_outcomes": outcomes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detailed-analysis", type=Path, nargs="+", required=True)
    parser.add_argument("--gallery-json", type=Path, required=True)
    parser.add_argument("--query-json", type=Path, required=True)
    parser.add_argument("--candidate-trigger", type=str, required=True)
    parser.add_argument("--candidate-scope", type=str, default="all")
    parser.add_argument("--candidate-margin-max", type=float, default=0.0)
    parser.add_argument("--candidate-family-gap-max", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=Path("experiments/generated/symbolic_pair_rule.json"))
    args = parser.parse_args()

    base_rows = load_detailed_rows(args.detailed_analysis)
    gallery = build_gallery(load_json(args.gallery_json), FAMILY_LABELS)
    query_events = build_query_events_all(load_json(args.query_json))
    base_rows = attach_retrieval(base_rows, query_events, gallery)
    candidate_scope = parse_scope(args.candidate_scope)
    candidate_rows = select_candidate_rows(
        base_rows,
        args.candidate_trigger,
        candidate_scope,
        args.candidate_margin_max,
        args.candidate_family_gap_max,
    )
    outcomes = run_loso(candidate_rows)
    summary = summarize(outcomes)
    payload = {
        "detailed_analysis": [str(path) for path in args.detailed_analysis],
        "gallery_json": str(args.gallery_json),
        "query_json": str(args.query_json),
        "candidate_trigger": args.candidate_trigger,
        "candidate_scope": candidate_scope,
        "candidate_margin_max": args.candidate_margin_max,
        "candidate_family_gap_max": args.candidate_family_gap_max,
        **summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"output: {args.output}")
    print(
        "candidate",
        payload["candidate_accuracy"]["old"],
        "->",
        payload["candidate_accuracy"]["new"],
        "delta",
        payload["candidate_accuracy"]["delta"],
        "counts",
        payload["counts"],
    )


if __name__ == "__main__":
    main()
