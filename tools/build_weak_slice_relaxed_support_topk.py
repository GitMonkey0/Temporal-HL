#!/usr/bin/env python3
"""Coverage expansion via relaxed left-hand motif support plus top-k joint search.

This extends the successful top-k joint search by relaxing the left-hand donor
support from exact grouped-motif match to a family-level neighborhood.
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
from tools.build_local_edit_audit import contiguous_runs, eligible_value, frame_attrs
from tools.build_weak_slice_joint_editor_prototype import (
    TASK_FIELD,
    TASK_TARGET,
    current_grouped_signature,
    fmt,
)
from tools.build_weak_slice_split_donor_prototype import (
    left_preserve_candidates,
    right_relaxed_candidates,
)
from tools.build_weak_slice_topk_joint_search import (
    TOPK_LEFT,
    TOPK_RIGHT,
    evaluate_joint,
)


LEVEL = {"still": 0, "local": 1, "active": 2}


def parse_grouped(group: str):
    parts = group.split("|")
    if len(parts) != 6:
        return None, []
    return parts[0], parts[1:]


def left_family_candidates(pair_bank, current_left_group, curr_attrs, prev_geom, max_profile_distance: int = 1):
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
        donor_group = grouped_motif_signature(str(donor_left.get("hand_motion", "unknown")), list(donor_left.get("transition_labels", [])))
        donor_motion, donor_profile = parse_grouped(donor_group)
        if donor_motion != target_motion:
            continue
        if len(donor_profile) != len(target_profile):
            continue
        dist = sum(abs(LEVEL[a] - LEVEL[b]) for a, b in zip(donor_profile, target_profile))
        if dist > max_profile_distance:
            continue
        out.append((dist, row))
    out.sort(key=lambda item: (item[0], item[1]["seq_name"], item[1]["curr_frame_idx"]))
    return [row for _, row in out]


def summarize(rows):
    n = len(rows)
    def avg(field, avail):
        return sum(row[field] for row in rows if row[avail]) / max(sum(row[avail] for row in rows), 1)
    return {
        "num_frames": n,
        "exact_available_rate": sum(row["exact_available"] for row in rows) / n,
        "relaxed_available_rate": sum(row["relaxed_available"] for row in rows) / n,
        "exact_joint_score": avg("exact_joint_score", "exact_available"),
        "relaxed_joint_score": avg("relaxed_joint_score", "relaxed_available"),
        "exact_right_grouped_match": avg("exact_right_grouped_match", "exact_available"),
        "relaxed_right_grouped_match": avg("relaxed_right_grouped_match", "relaxed_available"),
        "exact_left_preserve": avg("exact_left_preserve", "exact_available"),
        "relaxed_left_preserve": avg("relaxed_left_preserve", "relaxed_available"),
    }


def summarize_by_subtype(rows):
    by = defaultdict(list)
    for row in rows:
        by[(row["other_hand_motion"], row["interaction_motion_value"])].append(row)
    out = []
    for key, items in sorted(by.items()):
        other_hand_motion, interaction_motion_value = key
        def avg(field, avail):
            return sum(row[field] for row in items if row[avail]) / max(sum(row[avail] for row in items), 1)
        out.append(
            {
                "other_hand_motion": other_hand_motion,
                "interaction_motion_value": interaction_motion_value,
                "num_frames": len(items),
                "exact_available_rate": sum(row["exact_available"] for row in items) / len(items),
                "relaxed_available_rate": sum(row["relaxed_available"] for row in items) / len(items),
                "exact_joint_score": avg("exact_joint_score", "exact_available"),
                "relaxed_joint_score": avg("relaxed_joint_score", "relaxed_available"),
            }
        )
    return out


def pick_topk_joint(prev_frame, curr_frame, prev_geom, right_pool, left_pool):
    best = None
    for r in right_pool[:TOPK_RIGHT]:
        for l in left_pool[:TOPK_LEFT]:
            res = evaluate_joint(prev_frame, curr_frame, prev_geom, r, l)
            key = (res["joint_score"], res["left_preserve"], res["right_grouped_match"])
            if best is None or key > best[0]:
                best = (key, res)
    return None if best is None else best[1]


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
                    "exact_available": int(bool(right_pool) and bool(exact_left_pool)),
                    "relaxed_available": int(bool(right_pool) and bool(relaxed_left_pool)),
                }

                exact = pick_topk_joint(prev_frame, curr_frame, prev_geom, right_pool, exact_left_pool) if right_pool and exact_left_pool else None
                relaxed = pick_topk_joint(prev_frame, curr_frame, prev_geom, right_pool, relaxed_left_pool) if right_pool and relaxed_left_pool else None

                if exact is not None:
                    rec["exact_joint_score"] = exact["joint_score"]
                    rec["exact_right_grouped_match"] = exact["right_grouped_match"]
                    rec["exact_left_preserve"] = exact["left_preserve"]
                else:
                    rec["exact_joint_score"] = 0
                    rec["exact_right_grouped_match"] = 0
                    rec["exact_left_preserve"] = 0

                if relaxed is not None:
                    rec["relaxed_joint_score"] = relaxed["joint_score"]
                    rec["relaxed_right_grouped_match"] = relaxed["right_grouped_match"]
                    rec["relaxed_left_preserve"] = relaxed["left_preserve"]
                else:
                    rec["relaxed_joint_score"] = 0
                    rec["relaxed_right_grouped_match"] = 0
                    rec["relaxed_left_preserve"] = 0

                rows.append(rec)

    payload = {
        "artifacts": {
            "train_json": str(GEN / "temporal_hl_val.json"),
            "test_json": str(GEN / "temporal_hl_test.json"),
        },
        "focus": {
            "task": f"{TASK_FIELD}->{TASK_TARGET}",
            "slice": "interaction only",
            "prototype": "relaxed left support + topk joint search",
            "left_family_distance": 1,
        },
        "summary": summarize(rows),
        "subtype_summary": summarize_by_subtype(rows),
        "rows": rows,
    }

    out_json = GEN / "weak_slice_relaxed_support_topk.json"
    out_md = SUM / "weak_slice_relaxed_support_topk.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Weak Slice Relaxed-Support Top-k Search",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Focus: `right_hand_motion -> opening` on interaction frames only.",
        "",
        "Compare exact left grouped-motif support against family-level relaxed",
        "left support, both using top-k joint search.",
        "",
        "## Overall",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key, value in payload["summary"].items():
        if isinstance(value, float):
            lines.append(f"| {key} | {fmt(value)} |")
        else:
            lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## By Subtype",
            "",
            "| other hand motion | interaction motion | frames | exact avail | relaxed avail | exact joint | relaxed joint |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["subtype_summary"]:
        lines.append(
            f"| {row['other_hand_motion']} | {row['interaction_motion_value']} | {row['num_frames']} | "
            f"{fmt(row['exact_available_rate'])} | {fmt(row['relaxed_available_rate'])} | {fmt(row['exact_joint_score'])} | {fmt(row['relaxed_joint_score'])} |"
        )
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
