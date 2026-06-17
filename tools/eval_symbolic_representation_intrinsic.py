#!/usr/bin/env python3
"""Intrinsic discrimination metrics for symbolic sequence representations."""

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
    normalized_similarity,
    overlap_labels,
    sequence_event_sets,
    sequence_tokens,
)


def build_gallery(dataset, allowed, mode, token_mode, include_persistence=False, include_segment_duration=False):
    gallery = []
    for sequence, label in iter_sequences(dataset, allowed):
        if token_mode == "frame_tokens":
            rep = sequence_tokens(
                sequence,
                mode,
                use_rle=False,
                include_persistence=include_persistence,
            )
        elif token_mode == "rle_tokens":
            rep = sequence_tokens(
                sequence,
                mode,
                use_rle=True,
                include_persistence=include_persistence,
            )
        elif token_mode == "event_dtw":
            rep = sequence_event_sets(
                sequence,
                mode,
                include_persistence=include_persistence,
                include_segment_duration=include_segment_duration,
            )
        else:
            raise ValueError(f"Unknown token_mode: {token_mode}")
        gallery.append(
            {
                "seq_name": sequence["seq_name"],
                "label": label,
                "rep": rep,
            }
        )
    return gallery


def build_queries(dataset, allowed, mode, token_mode, include_persistence=False, include_segment_duration=False):
    queries = []
    for sequence, label in iter_sequences(dataset, allowed):
        if token_mode == "frame_tokens":
            rep = sequence_tokens(
                sequence,
                mode,
                use_rle=False,
                include_persistence=include_persistence,
            )
        elif token_mode == "rle_tokens":
            rep = sequence_tokens(
                sequence,
                mode,
                use_rle=True,
                include_persistence=include_persistence,
            )
        elif token_mode == "event_dtw":
            rep = sequence_event_sets(
                sequence,
                mode,
                include_persistence=include_persistence,
                include_segment_duration=include_segment_duration,
            )
        else:
            raise ValueError(f"Unknown token_mode: {token_mode}")
        queries.append(
            {
                "seq_name": sequence["seq_name"],
                "label": label,
                "rep": rep,
            }
        )
    return queries


def score_rep(query_rep, gallery_rep, token_mode):
    if token_mode in {"frame_tokens", "rle_tokens"}:
        return normalized_similarity(query_rep, gallery_rep)
    if token_mode == "event_dtw":
        return dtw_similarity(query_rep, gallery_rep)
    raise ValueError(token_mode)


def evaluate(gallery, queries, token_mode):
    rows = []
    summary = Counter()
    positive_margins = []
    positive_ranks = []
    pos_scores = []
    neg_scores = []
    for query in queries:
        ranking = []
        for item in gallery:
            score = score_rep(query["rep"], item["rep"], token_mode)
            ranking.append((item["label"], item["seq_name"], float(score)))
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
        pos_scores.append(best_correct[2])
        if best_wrong is not None:
            neg_scores.append(best_wrong[2])
        summary["queries"] += 1
        summary["top1_correct"] += int(canonical_label(best_label) == canonical_label(query["label"]))
        summary["margin_positive"] += int(positive_margin > 0)
        rows.append(
            {
                "seq_name": query["seq_name"],
                "label": query["label"],
                "top1_label": best_label,
                "top1_seq": best_seq,
                "top1_score": best_score,
                "top1_correct": canonical_label(best_label) == canonical_label(query["label"]),
                "best_correct_label": best_correct[0],
                "best_correct_seq": best_correct[1],
                "best_correct_score": best_correct[2],
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
            "margin_positive_rate": summary["margin_positive"] / max(summary["queries"], 1),
            "mean_positive_margin": sum(positive_margins) / max(len(positive_margins), 1),
            "mean_correct_rank": sum(positive_ranks) / max(len(positive_ranks), 1),
            "mean_best_positive_score": sum(pos_scores) / max(len(pos_scores), 1),
            "mean_best_negative_score": sum(neg_scores) / max(len(neg_scores), 1),
        },
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gallery-json", type=Path, required=True)
    parser.add_argument("--query-json", type=Path, required=True)
    parser.add_argument(
        "--configs",
        type=str,
        default="state_frame::state::frame_tokens,temporal_frame::temporal::frame_tokens,state_rle::state::rle_tokens,temporal_rle::temporal::rle_tokens,temporal_event::temporal::event_dtw",
    )
    parser.add_argument("--output", type=Path, default=Path("experiments/generated/symbolic_representation_intrinsic.json"))
    args = parser.parse_args()

    gallery_data = load_json(args.gallery_json)
    query_data = load_json(args.query_json)
    allowed = overlap_labels(gallery_data, query_data)
    configs = []
    for item in args.configs.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split("::")
        if len(parts) < 3:
            raise ValueError(f"Bad config: {item}")
        name, mode, token_mode = parts[:3]
        flags = set(parts[3:])
        configs.append(
            (
                name,
                mode,
                token_mode,
                "persist" in flags,
                "segdur" in flags,
            )
        )

    results = []
    for name, mode, token_mode, include_persistence, include_segment_duration in configs:
        gallery = build_gallery(
            gallery_data,
            allowed,
            mode,
            token_mode,
            include_persistence=include_persistence,
            include_segment_duration=include_segment_duration,
        )
        queries = build_queries(
            query_data,
            allowed,
            mode,
            token_mode,
            include_persistence=include_persistence,
            include_segment_duration=include_segment_duration,
        )
        payload = evaluate(gallery, queries, token_mode)
        payload["name"] = name
        payload["mode"] = mode
        payload["token_mode"] = token_mode
        payload["include_persistence"] = include_persistence
        payload["include_segment_duration"] = include_segment_duration
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
            "top1",
            s["top1_accuracy"],
            "margin+",
            s["margin_positive_rate"],
            "mean_margin",
            s["mean_positive_margin"],
            "rank",
            s["mean_correct_rank"],
        )


if __name__ == "__main__":
    main()
