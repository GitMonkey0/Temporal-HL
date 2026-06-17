#!/usr/bin/env python3
"""Construct harder correction benchmark subsets from retrieval evidence."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from tools.eval_occlusion_late_correction import FAMILY_LABELS, build_gallery
from tools.eval_symbolic_learned_gate import (
    build_family_dataset,
    load_detailed_rows,
    parse_scope,
)
from tools.eval_sequence_symbolic_retrieval import load_json
from tools.audit_family_disagreement_broad import build_query_events_all


def pair_status_map(pair_rule_payload):
    rows = pair_rule_payload["all_candidate_outcomes"]
    status = {}
    for row in rows:
        pair = tuple(row["pair"])
        train = [x for x in rows if x["seed"] != row["seed"] and tuple(x["pair"]) == pair]
        fixes = sum(int(x["would_fix"]) for x in train)
        harms = sum(int(x["would_harm"]) for x in train)
        if fixes > harms and fixes > 0:
            tag = "seen_fix"
        elif harms >= fixes and harms > 0:
            tag = "seen_harm"
        else:
            tag = "unseen"
        status[(row["seed"], row["seq_name"])] = {
            "status": tag,
            "pair": list(pair),
            "train_fix": fixes,
            "train_harm": harms,
        }
    return status


def enrich_rows(dataset_rows, pair_status):
    enriched = []
    for row in dataset_rows:
        key = (row["seed"], row["seq_name"])
        pair_info = pair_status.get(
            key,
            {
                "status": "missing",
                "pair": [row["original_prediction"], row["retrieval_prediction"]],
                "train_fix": 0,
                "train_harm": 0,
            },
        )
        ranking = row.get("retrieval_ranking", [])
        labels = [label for label, _, _ in ranking]
        scores = [float(score) for _, _, score in ranking]
        unique_labels = len(set(labels))
        label_counts = Counter(labels)
        top1 = row["retrieval_prediction"]
        top1_votes = label_counts.get(top1, 0)
        second_votes = sorted(label_counts.values(), reverse=True)[1] if len(label_counts) > 1 else 0
        top1_vote_margin = top1_votes - second_votes
        row_enriched = {
            **row,
            "pair_status": pair_info["status"],
            "pair": pair_info["pair"],
            "pair_train_fix": pair_info["train_fix"],
            "pair_train_harm": pair_info["train_harm"],
            "ranking_unique_labels": unique_labels,
            "ranking_top1_vote_count": top1_votes,
            "ranking_top1_vote_margin": top1_vote_margin,
            "ranking_score_range": max(scores) - min(scores) if scores else 0.0,
            "ranking_top3_score_gap": scores[0] - scores[2] if len(scores) >= 3 else (scores[0] - scores[-1] if len(scores) > 1 else 0.0),
            "ranking_top2_score_gap": scores[0] - scores[1] if len(scores) >= 2 else 0.0,
            "changed": row["retrieval_prediction"] != row["original_prediction"],
        }
        enriched.append(row_enriched)
    return enriched


def select_subset(rows, subset):
    if subset == "all":
        return rows
    if subset == "changed":
        return [r for r in rows if r["changed"]]
    if subset == "pair_unseen_changed":
        return [r for r in rows if r["changed"] and r["pair_status"] == "unseen"]
    if subset == "pair_unseen_positive_only":
        return [r for r in rows if r["changed"] and r["pair_status"] == "unseen" and r["would_fix"]]
    if subset == "conflict_changed":
        return [
            r for r in rows
            if r["changed"]
            and (
                r["ranking_unique_labels"] >= 3
                or r["ranking_top1_vote_margin"] <= 1
                or r["ranking_top2_score_gap"] <= 0.05
            )
        ]
    if subset == "pair_hard_conflict":
        return [
            r for r in rows
            if r["changed"]
            and r["pair_status"] == "unseen"
            and (
                r["ranking_unique_labels"] >= 3
                or r["ranking_top1_vote_margin"] <= 1
                or r["ranking_top2_score_gap"] <= 0.05
            )
        ]
    if subset.startswith("gap_le_"):
        thr = float(subset.split("gap_le_", 1)[1])
        return [r for r in rows if r["changed"] and r["ranking_top2_score_gap"] <= thr]
    if subset.startswith("pair_unseen_gap_le_"):
        thr = float(subset.split("pair_unseen_gap_le_", 1)[1])
        return [
            r for r in rows
            if r["changed"] and r["pair_status"] == "unseen" and r["ranking_top2_score_gap"] <= thr
        ]
    raise ValueError(f"Unknown subset: {subset}")


def summarize(rows):
    c = Counter()
    for r in rows:
        c["rows"] += 1
        c[f"pair_status::{r['pair_status']}"] += 1
        c["changed"] += int(r["changed"])
        c["would_fix"] += int(r["would_fix"])
        c["would_harm"] += int(r["would_harm"])
        c["original_correct"] += int(r["original_correct"])
        c["top1_vote_margin_le1"] += int(r["ranking_top1_vote_margin"] <= 1)
        c["top2_gap_le005"] += int(r["ranking_top2_score_gap"] <= 0.05)
        c["unique_labels_ge3"] += int(r["ranking_unique_labels"] >= 3)
    return dict(c)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detailed-analysis", type=Path, nargs="+", required=True)
    parser.add_argument("--gallery-json", type=Path, required=True)
    parser.add_argument("--query-json", type=Path, required=True)
    parser.add_argument("--pair-rule-json", type=Path, required=True)
    parser.add_argument("--candidate-trigger", type=str, required=True)
    parser.add_argument("--candidate-scope", type=str, default="all")
    parser.add_argument("--candidate-margin-max", type=float, default=0.0)
    parser.add_argument("--candidate-family-gap-max", type=float, default=0.0)
    parser.add_argument(
        "--subsets",
        type=str,
        default="all,changed,pair_unseen_changed,conflict_changed,pair_hard_conflict,gap_le_0.01,gap_le_0.02,gap_le_0.04,gap_le_0.06,pair_unseen_gap_le_0.02,pair_unseen_gap_le_0.06",
    )
    parser.add_argument("--output", type=Path, default=Path("experiments/generated/correction_benchmark_candidates.json"))
    args = parser.parse_args()

    base_rows = load_detailed_rows(args.detailed_analysis)
    gallery = build_gallery(load_json(args.gallery_json), FAMILY_LABELS)
    query_events = build_query_events_all(load_json(args.query_json))
    scope = parse_scope(args.candidate_scope)
    dataset = build_family_dataset(
        base_rows,
        query_events,
        gallery,
        args.candidate_trigger,
        scope,
        args.candidate_margin_max,
        args.candidate_family_gap_max,
    )
    pair_status = pair_status_map(load_json(args.pair_rule_json))
    enriched = enrich_rows(dataset, pair_status)

    subset_names = [x.strip() for x in args.subsets.split(",") if x.strip()]
    subsets = []
    for subset_name in subset_names:
        rows = select_subset(enriched, subset_name)
        subsets.append(
            {
                "subset": subset_name,
                "summary": summarize(rows),
                "rows": rows,
            }
        )

    payload = {
        "detailed_analysis": [str(x) for x in args.detailed_analysis],
        "pair_rule_json": str(args.pair_rule_json),
        "candidate_trigger": args.candidate_trigger,
        "candidate_scope": scope,
        "candidate_margin_max": args.candidate_margin_max,
        "candidate_family_gap_max": args.candidate_family_gap_max,
        "subsets": subsets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"output: {args.output}")
    for subset in subsets:
        print(subset["subset"], subset["summary"])


if __name__ == "__main__":
    main()
