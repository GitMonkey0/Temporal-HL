#!/usr/bin/env python3
"""Coordination-aware reranking for relaxed left-support search.

Goal:

- keep the successful relaxed family support
- avoid full/deep left search when possible
- test whether better left-candidate ordering can make small search budgets
  approach the `left_top_20` weak-slice baseline
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
    grouped_motif_signature,
    load_json,
    overlap_labels,
)
from tools.build_geometry_locality_audit import hand_delta
from tools.build_local_edit_audit import contiguous_runs, eligible_value, frame_attrs
from tools.build_weak_slice_joint_editor_prototype import (
    TASK_FIELD,
    TASK_TARGET,
    current_grouped_signature,
    fmt,
)
from tools.build_weak_slice_topk_joint_search import evaluate_joint
from tools.build_weak_slice_split_donor_prototype import right_relaxed_candidates


LEVEL = {"still": 0, "local": 1, "active": 2}
DEPTHS = (5, 10)
REFERENCE_METHODS = {
    "base_top5": 5,
    "base_top10": 10,
    "base_top20": 20,
}
RERANK_METHODS = (
    "curr_left",
    "curr_left_cross",
    "pair_cross",
)


def parse_grouped(group: str):
    parts = group.split("|")
    if len(parts) != 6:
        return None, []
    return parts[0], parts[1:]


def left_family_candidates_with_meta(pair_bank, current_left_group, curr_attrs, prev_geom, curr_geom, max_profile_distance: int = 1):
    target_motion, target_profile = parse_grouped(current_left_group)
    out = []
    for row in pair_bank:
        attrs = row["curr_attrs"]
        if attrs["hand_type"] != curr_attrs["hand_type"]:
            continue
        if attrs["interaction_motion"] != curr_attrs["interaction_motion"]:
            continue
        donor_left = row["curr_frame"].get("left")
        if donor_left is None:
            continue
        donor_group = grouped_motif_signature(
            str(donor_left.get("hand_motion", "unknown")),
            list(donor_left.get("transition_labels", [])),
        )
        donor_motion, donor_profile = parse_grouped(donor_group)
        if donor_motion != target_motion:
            continue
        if len(donor_profile) != len(target_profile):
            continue
        family_dist = sum(abs(LEVEL[a] - LEVEL[b]) for a, b in zip(donor_profile, target_profile))
        if family_dist > max_profile_distance:
            continue
        donor_curr_geom = row["curr_geom"]
        curr_left_delta = hand_delta(curr_geom["left"], donor_curr_geom["left"])
        prev_left_delta = hand_delta(prev_geom["left"], donor_curr_geom["left"])
        curr_cross = curr_geom.get("cross_hand_distance")
        donor_cross = donor_curr_geom.get("cross_hand_distance")
        cross_delta_to_current = 0.0
        if curr_cross is not None and donor_cross is not None:
            cross_delta_to_current = abs(float(curr_cross) - float(donor_cross))
        out.append(
            {
                "row": row,
                "family_dist": family_dist,
                "curr_left_delta": curr_left_delta,
                "prev_left_delta": prev_left_delta,
                "cross_delta_to_current": cross_delta_to_current,
                "donor_cross_distance": donor_cross,
            }
        )
    out.sort(
        key=lambda item: (
            item["family_dist"],
            item["row"]["seq_name"],
            item["row"]["curr_frame_idx"],
        )
    )
    return out


def sort_candidates(meta_rows, method, right_row=None):
    if method == "base":
        ordered = list(meta_rows)
    elif method == "curr_left":
        ordered = sorted(
            meta_rows,
            key=lambda item: (
                item["family_dist"],
                item["curr_left_delta"],
                item["prev_left_delta"],
                item["cross_delta_to_current"],
                item["row"]["seq_name"],
                item["row"]["curr_frame_idx"],
            ),
        )
    elif method == "curr_left_cross":
        ordered = sorted(
            meta_rows,
            key=lambda item: (
                item["family_dist"],
                item["curr_left_delta"] + 0.05 * item["cross_delta_to_current"],
                item["curr_left_delta"],
                item["cross_delta_to_current"],
                item["prev_left_delta"],
                item["row"]["seq_name"],
                item["row"]["curr_frame_idx"],
            ),
        )
    elif method == "pair_cross":
        right_cross = None if right_row is None else right_row["curr_geom"].get("cross_hand_distance")
        def pair_key(item):
            pair_cross_gap = 0.0
            donor_cross = item["donor_cross_distance"]
            if right_cross is not None and donor_cross is not None:
                pair_cross_gap = abs(float(right_cross) - float(donor_cross))
            return (
                item["family_dist"],
                item["curr_left_delta"] + 0.05 * item["cross_delta_to_current"] + 0.05 * pair_cross_gap,
                pair_cross_gap,
                item["cross_delta_to_current"],
                item["curr_left_delta"],
                item["row"]["seq_name"],
                item["row"]["curr_frame_idx"],
            )
        ordered = sorted(meta_rows, key=pair_key)
    else:
        raise ValueError(method)
    return ordered


def pick_best(prev_frame, curr_frame, prev_geom, right_pool, left_meta_rows, method, left_depth):
    best = None
    for right_row in right_pool:
        ordered_left = sort_candidates(left_meta_rows, method, right_row=right_row)
        for item in ordered_left[:left_depth]:
            res = evaluate_joint(prev_frame, curr_frame, prev_geom, right_row, item["row"])
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
                left_meta_rows = left_family_candidates_with_meta(
                    pair_bank, current_left_group, curr_attrs, prev_geom, curr_geom, max_profile_distance=1
                )

                rec = {
                    "seq_name": sequence["seq_name"],
                    "frame_idx": curr_frame["frame_idx"],
                    "other_hand_motion": "none" if curr_frame.get("left") is None else str(curr_frame["left"].get("hand_motion", "unknown")),
                    "interaction_motion_value": str(curr_frame.get("interaction_motion", "unknown")),
                    "right_pool_size": len(right_pool),
                    "relaxed_left_pool_size": len(left_meta_rows),
                }

                for name, depth in REFERENCE_METHODS.items():
                    res = pick_best(prev_frame, curr_frame, prev_geom, right_pool, left_meta_rows, "base", depth) if right_pool and left_meta_rows else None
                    rec[f"{name}_available"] = int(res is not None)
                    rec[f"{name}_joint_score"] = 0 if res is None else res["joint_score"]
                    rec[f"{name}_right_grouped_match"] = 0 if res is None else res["right_grouped_match"]
                    rec[f"{name}_left_preserve"] = 0 if res is None else res["left_preserve"]

                for method in RERANK_METHODS:
                    for depth in DEPTHS:
                        prefix = f"{method}_top{depth}"
                        res = pick_best(prev_frame, curr_frame, prev_geom, right_pool, left_meta_rows, method, depth) if right_pool and left_meta_rows else None
                        rec[f"{prefix}_available"] = int(res is not None)
                        rec[f"{prefix}_joint_score"] = 0 if res is None else res["joint_score"]
                        rec[f"{prefix}_right_grouped_match"] = 0 if res is None else res["right_grouped_match"]
                        rec[f"{prefix}_left_preserve"] = 0 if res is None else res["left_preserve"]

                rows.append(rec)

    prefixes = list(REFERENCE_METHODS.keys()) + [f"{method}_top{depth}" for method in RERANK_METHODS for depth in DEPTHS]
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
    focus_prefixes = [
        "base_top5",
        "base_top10",
        "base_top20",
        "curr_left_top5",
        "curr_left_top10",
        "curr_left_cross_top5",
        "curr_left_cross_top10",
        "pair_cross_top5",
        "pair_cross_top10",
    ]
    subtype_summary = {prefix: summarize_subtypes(rows, prefix) for prefix in focus_prefixes}

    payload = {
        "artifacts": {
            "train_json": str(GEN / "temporal_hl_val.json"),
            "test_json": str(GEN / "temporal_hl_test.json"),
        },
        "focus": {
            "task": f"{TASK_FIELD}->{TASK_TARGET}",
            "slice": "interaction only",
            "goal": "coordination-aware rerank under small left-search budgets",
            "reference_methods": REFERENCE_METHODS,
            "rerank_methods": list(RERANK_METHODS),
            "depths": list(DEPTHS),
        },
        "summary": summary,
        "ranked_methods": ranked,
        "subtype_summary": subtype_summary,
        "rows": rows,
    }

    out_json = GEN / "weak_slice_coordination_rerank.json"
    out_md = SUM / "weak_slice_coordination_rerank.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Weak Slice Coordination-Aware Rerank",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Focus: `right_hand_motion -> opening` on interaction frames only.",
        "",
        "Goal: test whether better relaxed left-pool ordering can make small",
        "search budgets approach the `base_top20` weak-slice baseline.",
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
