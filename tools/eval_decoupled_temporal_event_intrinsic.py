#!/usr/bin/env python3
"""Evaluate decoupled temporal-event channel variants on the intrinsic benchmark."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tools.eval_sequence_symbolic_retrieval import (
    canonical_label,
    dtw_similarity,
    iter_sequences,
    load_json,
    overlap_labels,
    sequence_event_sets,
)


def keep_token(token: str, channel_set: set[str]) -> bool:
    if token.startswith("hand_type="):
        return True
    if token.startswith("event::duration::"):
        return "duration" in channel_set
    if token.startswith("interaction="):
        return "interaction" in channel_set
    if ":state:" in token:
        return "state" in channel_set
    if ":trans:" in token:
        return "transition" in channel_set
    if ":motion:" in token:
        return "motion" in channel_set
    if ":state_persist:" in token or ":activity_persist:" in token or token.startswith("interaction_persist:") or token.startswith("interaction_activity_persist:"):
        return "persistence" in channel_set
    if ":state_segdur:" in token or ":activity_segdur:" in token or token.startswith("interaction_segdur:") or token.startswith("interaction_activity_segdur:"):
        return "segment_duration" in channel_set
    return False


def build_variant_rep(sequence, channel_set: set[str]):
    include_persistence = "persistence" in channel_set
    include_segment_duration = "segment_duration" in channel_set
    base = sequence_event_sets(
        sequence,
        "temporal",
        include_persistence=include_persistence,
        include_segment_duration=include_segment_duration,
    )
    return [{tok for tok in event if keep_token(tok, channel_set)} for event in base]


def build_items(dataset, allowed, channel_set):
    out = []
    for sequence, label in iter_sequences(dataset, allowed):
        out.append(
            {
                "seq_name": sequence["seq_name"],
                "label": label,
                "rep": build_variant_rep(sequence, channel_set),
            }
        )
    return out


def evaluate(gallery, queries):
    rows = []
    summary = Counter()
    positive_margins = []
    positive_ranks = []
    for query in queries:
        ranking = []
        for item in gallery:
            score = float(dtw_similarity(query["rep"], item["rep"]))
            ranking.append((item["label"], item["seq_name"], score))
        ranking.sort(key=lambda x: x[2], reverse=True)
        best_label, best_seq, best_score = ranking[0]
        correct_items = [x for x in ranking if canonical_label(x[0]) == canonical_label(query["label"])]
        wrong_items = [x for x in ranking if canonical_label(x[0]) != canonical_label(query["label"])]
        best_correct = correct_items[0]
        best_wrong = wrong_items[0] if wrong_items else None
        positive_margin = best_correct[2] - (best_wrong[2] if best_wrong else best_correct[2])
        correct_rank = next(i for i, x in enumerate(ranking, start=1) if canonical_label(x[0]) == canonical_label(query["label"]))
        positive_margins.append(positive_margin)
        positive_ranks.append(correct_rank)
        summary["queries"] += 1
        summary["top1_correct"] += int(canonical_label(best_label) == canonical_label(query["label"]))
        rows.append(
            {
                "seq_name": query["seq_name"],
                "label": query["label"],
                "top1_label": best_label,
                "top1_seq": best_seq,
                "top1_score": best_score,
                "top1_correct": canonical_label(best_label) == canonical_label(query["label"]),
                "best_wrong_label": None if best_wrong is None else best_wrong[0],
                "best_wrong_seq": None if best_wrong is None else best_wrong[1],
                "best_wrong_score": None if best_wrong is None else best_wrong[2],
                "positive_margin": positive_margin,
                "correct_rank": correct_rank,
            }
        )
    return {
        "summary": {
            "num_queries": summary["queries"],
            "top1_accuracy": summary["top1_correct"] / max(summary["queries"], 1),
            "mean_positive_margin": sum(positive_margins) / max(len(positive_margins), 1),
            "mean_correct_rank": sum(positive_ranks) / max(len(positive_ranks), 1),
        },
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gallery-json", type=Path, required=True)
    parser.add_argument("--query-json", type=Path, required=True)
    parser.add_argument(
        "--variants",
        type=str,
        default=(
            "full::state+transition+motion+interaction+duration,"
            "state_transition::state+transition,"
            "state_motion::state+motion,"
            "transition_motion::transition+motion,"
            "transition_only::transition,"
            "motion_only::motion,"
            "duration_only::duration,"
            "interaction_only::interaction,"
            "temporal_nostate::transition+motion+interaction+duration"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("experiments/generated/decoupled_temporal_event_intrinsic.json"))
    args = parser.parse_args()

    gallery_data = load_json(args.gallery_json)
    query_data = load_json(args.query_json)
    allowed = overlap_labels(gallery_data, query_data)

    results = []
    for item in args.variants.split(","):
        name, channels = item.split("::", 1)
        channel_set = set(channels.split("+"))
        gallery = build_items(gallery_data, allowed, channel_set)
        queries = build_items(query_data, allowed, channel_set)
        payload = evaluate(gallery, queries)
        payload["name"] = name
        payload["channels"] = sorted(channel_set)
        results.append(payload)

    results.sort(
        key=lambda x: (
            -x["summary"]["top1_accuracy"],
            -x["summary"]["mean_positive_margin"],
            x["summary"]["mean_correct_rank"],
            x["name"],
        )
    )
    out = {
        "gallery_json": str(args.gallery_json),
        "query_json": str(args.query_json),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(out, f, indent=2)

    print(f"output: {args.output}")
    for item in results:
        s = item["summary"]
        print(
            item["name"],
            item["channels"],
            "top1",
            s["top1_accuracy"],
            "mean_margin",
            s["mean_positive_margin"],
            "rank",
            s["mean_correct_rank"],
        )


if __name__ == "__main__":
    main()
