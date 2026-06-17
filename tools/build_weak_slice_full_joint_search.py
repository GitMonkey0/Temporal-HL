#!/usr/bin/env python3
"""Full candidate-search diagnosis on the weak interaction slice.

This script tests whether the remaining bottleneck is caused by top-k truncation
or by the candidate sets themselves.
"""

from __future__ import annotations

import json
from collections import defaultdict

from tools.build_transition_conditioned_symbolic_editor import (
    GEN,
    SUM,
    build_pair_bank,
    build_semantic_frame_vocab,
    canonical,
    frame_geom,
    load_json,
    overlap_labels,
)
from tools.build_local_edit_audit import contiguous_runs, eligible_value, frame_attrs
from tools.build_weak_slice_joint_editor_prototype import (
    TASK_FIELD,
    TASK_TARGET,
    current_grouped_signature,
    fmt,
)
from tools.build_weak_slice_relaxed_support_topk import left_family_candidates
from tools.build_weak_slice_split_donor_prototype import (
    left_preserve_candidates,
    right_relaxed_candidates,
)
from tools.build_weak_slice_topk_joint_search import (
    TOPK_LEFT,
    TOPK_RIGHT,
    evaluate_joint,
)


def pick_best(prev_frame, curr_frame, prev_geom, right_pool, left_pool, right_k=None, left_k=None):
    rp = right_pool if right_k is None else right_pool[:right_k]
    lp = left_pool if left_k is None else left_pool[:left_k]
    best = None
    for r in rp:
        for l in lp:
            res = evaluate_joint(prev_frame, curr_frame, prev_geom, r, l)
            key = (
                res["joint_score"],
                res["left_preserve"],
                res["right_grouped_match"],
            )
            if best is None or key > best[0]:
                best = (key, res)
    return None if best is None else best[1]


def summarize(rows, prefix):
    n = len(rows)
    avail_key = f"{prefix}_available"
    joint_key = f"{prefix}_joint_score"
    right_key = f"{prefix}_right_grouped_match"
    left_key = f"{prefix}_left_preserve"
    avail = sum(row[avail_key] for row in rows)
    return {
        "num_frames": n,
        "available_rate": avail / n,
        "joint_score_on_available": sum(row[joint_key] for row in rows if row[avail_key]) / max(avail, 1),
        "right_grouped_match_on_available": sum(row[right_key] for row in rows if row[avail_key]) / max(avail, 1),
        "left_preserve_on_available": sum(row[left_key] for row in rows if row[avail_key]) / max(avail, 1),
        "joint_hit_rate_overall": sum(row[joint_key] for row in rows) / n,
    }


def summarize_subtypes(rows, prefix):
    avail_key = f"{prefix}_available"
    joint_key = f"{prefix}_joint_score"
    by = defaultdict(list)
    for row in rows:
        by[(row["other_hand_motion"], row["interaction_motion_value"])].append(row)
    out = []
    for key, items in sorted(by.items()):
        other_hand_motion, interaction_motion_value = key
        avail = sum(row[avail_key] for row in items)
        out.append(
            {
                "other_hand_motion": other_hand_motion,
                "interaction_motion_value": interaction_motion_value,
                "num_frames": len(items),
                "available_rate": avail / len(items),
                "joint_score_on_available": sum(row[joint_key] for row in items if row[avail_key]) / max(avail, 1),
                "joint_hit_rate_overall": sum(row[joint_key] for row in items) / len(items),
            }
        )
    return out


def main():
    train_data = load_json(GEN / "temporal_hl_val.json")
    test_data = load_json(GEN / "temporal_hl_test.json")
    labels = overlap_labels(train_data, test_data)
    semantic_vocab = build_semantic_frame_vocab(train_data, labels)
    pair_bank = build_pair_bank(train_data, labels, semantic_vocab)

    rows = []
    for sequence in test_data["sequences"]:
        if canonical(sequence["seq_name"]) not in labels and sequence["seq_name"] not in labels:
            continue
        frames = sequence["frames"]
        for start, end, value in contiguous_runs(frames, TASK_FIELD):
            if end - start < 3 or not eligible_value(TASK_FIELD, value, TASK_TARGET):
                continue
            for idx in range(max(start, 1), end):
                prev_frame = frames[idx - 1]
                curr_frame = frames[idx]
                if curr_frame.get("hand_type") != "interacting":
                    continue
                curr_attrs = frame_attrs(curr_frame)
                prev_geom = frame_geom(prev_frame)
                curr_geom = frame_geom(curr_frame)
                current_left_group = current_grouped_signature(curr_frame.get("left"))

                right_pool = right_relaxed_candidates(pair_bank, curr_attrs, prev_geom, curr_geom)
                exact_left_pool = left_preserve_candidates(pair_bank, current_left_group, curr_attrs, prev_geom)
                relaxed_left_pool = left_family_candidates(pair_bank, current_left_group, curr_attrs, prev_geom, max_profile_distance=1)

                rec = {
                    "seq_name": sequence["seq_name"],
                    "frame_idx": curr_frame["frame_idx"],
                    "other_hand_motion": "none" if curr_frame.get("left") is None else str(curr_frame["left"].get("hand_motion", "unknown")),
                    "interaction_motion_value": str(curr_frame.get("interaction_motion", "unknown")),
                    "right_pool_size": len(right_pool),
                    "exact_left_pool_size": len(exact_left_pool),
                    "relaxed_left_pool_size": len(relaxed_left_pool),
                }

                methods = {
                    "exact_topk": (right_pool, exact_left_pool, TOPK_RIGHT, TOPK_LEFT),
                    "exact_full": (right_pool, exact_left_pool, None, None),
                    "relaxed_topk": (right_pool, relaxed_left_pool, TOPK_RIGHT, TOPK_LEFT),
                    "relaxed_full": (right_pool, relaxed_left_pool, None, None),
                }
                for name, (rp, lp, rk, lk) in methods.items():
                    res = pick_best(prev_frame, curr_frame, prev_geom, rp, lp, rk, lk) if rp and lp else None
                    rec[f"{name}_available"] = int(res is not None)
                    rec[f"{name}_joint_score"] = 0 if res is None else res["joint_score"]
                    rec[f"{name}_right_grouped_match"] = 0 if res is None else res["right_grouped_match"]
                    rec[f"{name}_left_preserve"] = 0 if res is None else res["left_preserve"]

                rows.append(rec)

    prefixes = ["exact_topk", "exact_full", "relaxed_topk", "relaxed_full"]
    summary = {prefix: summarize(rows, prefix) for prefix in prefixes}
    ranked = sorted(
        ({"method": prefix, **summary[prefix]} for prefix in prefixes),
        key=lambda row: (
            row["joint_hit_rate_overall"],
            row["joint_score_on_available"],
            row["available_rate"],
        ),
        reverse=True,
    )
    subtype_focus = {prefix: summarize_subtypes(rows, prefix) for prefix in prefixes}

    payload = {
        "artifacts": {
            "train_json": str(GEN / "temporal_hl_val.json"),
            "test_json": str(GEN / "temporal_hl_test.json"),
        },
        "focus": {
            "task": f"{TASK_FIELD}->{TASK_TARGET}",
            "slice": "interaction only",
            "diagnosis": "top-k truncation vs full search",
            "topk": {"right": TOPK_RIGHT, "left": TOPK_LEFT},
        },
        "summary": summary,
        "ranked_methods": ranked,
        "subtype_summary": subtype_focus,
        "rows": rows,
    }

    out_json = GEN / "weak_slice_full_joint_search.json"
    out_md = SUM / "weak_slice_full_joint_search.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Weak Slice Full Joint Search Diagnosis",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Focus: `right_hand_motion -> opening` on interaction frames only.",
        "",
        "Diagnose whether top-k truncation is the main bottleneck by comparing",
        "top-k joint search against full candidate search for exact and relaxed",
        "left-support pools.",
        "",
        "## Method Ranking",
        "",
        "| method | avail | joint on avail | right match | left preserve | joint overall |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ranked:
        lines.append(
            f"| {row['method']} | {fmt(row['available_rate'])} | "
            f"{fmt(row['joint_score_on_available'])} | "
            f"{fmt(row['right_grouped_match_on_available'])} | "
            f"{fmt(row['left_preserve_on_available'])} | "
            f"{fmt(row['joint_hit_rate_overall'])} |"
        )

    for prefix in prefixes:
        lines.extend(
            [
                "",
                f"## By Subtype: {prefix}",
                "",
                "| other hand motion | interaction motion | frames | avail | joint on avail | joint overall |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in subtype_focus[prefix]:
            lines.append(
                f"| {row['other_hand_motion']} | {row['interaction_motion_value']} | "
                f"{row['num_frames']} | {fmt(row['available_rate'])} | "
                f"{fmt(row['joint_score_on_available'])} | "
                f"{fmt(row['joint_hit_rate_overall'])} |"
            )

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
