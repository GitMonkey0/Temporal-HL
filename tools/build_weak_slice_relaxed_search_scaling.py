#!/usr/bin/env python3
"""Search-depth scaling for the relaxed-support weak-slice editor."""

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
from tools.build_weak_slice_split_donor_prototype import right_relaxed_candidates
from tools.build_weak_slice_topk_joint_search import evaluate_joint


LEFT_DEPTHS = [1, 3, 5, 10, 20, 50, None]


def pick_best(prev_frame, curr_frame, prev_geom, right_pool, left_pool, left_depth):
    lp = left_pool if left_depth is None else left_pool[:left_depth]
    best = None
    for r in right_pool:
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
    avail_key = f"{prefix}_available"
    joint_key = f"{prefix}_joint_score"
    right_key = f"{prefix}_right_grouped_match"
    left_key = f"{prefix}_left_preserve"
    n = len(rows)
    avail = sum(row[avail_key] for row in rows)
    return {
        "num_frames": n,
        "available_rate": avail / n,
        "joint_score_on_available": sum(row[joint_key] for row in rows if row[avail_key]) / max(avail, 1),
        "right_grouped_match_on_available": sum(row[right_key] for row in rows if row[avail_key]) / max(avail, 1),
        "left_preserve_on_available": sum(row[left_key] for row in rows if row[avail_key]) / max(avail, 1),
        "joint_hit_rate_overall": sum(row[joint_key] for row in rows) / n,
    }


def summarize_by_subtype(rows, prefix):
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


def depth_name(depth):
    return "full" if depth is None else f"left_top_{depth}"


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
                relaxed_left_pool = left_family_candidates(pair_bank, current_left_group, curr_attrs, prev_geom, max_profile_distance=1)

                rec = {
                    "seq_name": sequence["seq_name"],
                    "frame_idx": curr_frame["frame_idx"],
                    "other_hand_motion": "none" if curr_frame.get("left") is None else str(curr_frame["left"].get("hand_motion", "unknown")),
                    "interaction_motion_value": str(curr_frame.get("interaction_motion", "unknown")),
                    "right_pool_size": len(right_pool),
                    "relaxed_left_pool_size": len(relaxed_left_pool),
                }

                for depth in LEFT_DEPTHS:
                    prefix = depth_name(depth)
                    res = pick_best(prev_frame, curr_frame, prev_geom, right_pool, relaxed_left_pool, depth) if right_pool and relaxed_left_pool else None
                    rec[f"{prefix}_available"] = int(res is not None)
                    rec[f"{prefix}_joint_score"] = 0 if res is None else res["joint_score"]
                    rec[f"{prefix}_right_grouped_match"] = 0 if res is None else res["right_grouped_match"]
                    rec[f"{prefix}_left_preserve"] = 0 if res is None else res["left_preserve"]

                rows.append(rec)

    prefixes = [depth_name(depth) for depth in LEFT_DEPTHS]
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
    focus_prefixes = ["left_top_5", "left_top_10", "left_top_20", "full"]
    subtype_summary = {prefix: summarize_by_subtype(rows, prefix) for prefix in focus_prefixes}

    payload = {
        "artifacts": {
            "train_json": str(GEN / "temporal_hl_val.json"),
            "test_json": str(GEN / "temporal_hl_test.json"),
        },
        "focus": {
            "task": f"{TASK_FIELD}->{TASK_TARGET}",
            "slice": "interaction only",
            "diagnosis": "relaxed support search-depth scaling",
            "left_depths": ["full" if d is None else d for d in LEFT_DEPTHS],
            "right_search": "all candidates",
        },
        "summary": summary,
        "ranked_methods": ranked,
        "subtype_summary": subtype_summary,
        "rows": rows,
    }

    out_json = GEN / "weak_slice_relaxed_search_scaling.json"
    out_md = SUM / "weak_slice_relaxed_search_scaling.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Weak Slice Relaxed Search Scaling",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Focus: `right_hand_motion -> opening` on interaction frames only.",
        "",
        "Keep the full right-donor candidate pool and vary only the relaxed",
        "left-donor search depth.",
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

    for prefix in focus_prefixes:
        lines.extend(
            [
                "",
                f"## By Subtype: {prefix}",
                "",
                "| other hand motion | interaction motion | frames | avail | joint on avail | joint overall |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in subtype_summary[prefix]:
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
