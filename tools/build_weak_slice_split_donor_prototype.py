#!/usr/bin/env python3
"""Split-donor candidate-space expansion on the weak interaction slice.

Idea:

- choose a right-hand donor to realize the target right-hand motif
- choose a separate left-hand donor to preserve the current left-hand motif
- compose both deltas jointly

This tests whether the current weak slice is limited by single-donor support.
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
    compose_target_hand_transition,
    frame_geom,
    grouped_motif_signature,
    load_json,
    overlap_labels,
    pair_distance,
    pair_realizes_target,
)
from tools.build_conditional_hand_transplant_audit import target_hand_name
from tools.build_geometry_locality_audit import hand_delta
from tools.build_local_edit_audit import contiguous_runs, eligible_value, frame_attrs
from tools.build_weak_slice_joint_editor_prototype import (
    TASK_FIELD,
    TASK_TARGET,
    compose_joint_two_hands,
    current_grouped_signature,
    fmt,
    hand_motif_from_geom,
)


def right_relaxed_candidates(
    pair_bank,
    curr_attrs,
    test_prev_geom,
    test_curr_geom,
    task_field: str = TASK_FIELD,
    task_target: str = TASK_TARGET,
):
    out = []
    for row in pair_bank:
        attrs = row["curr_attrs"]
        if attrs[task_field] != task_target:
            continue
        if attrs["hand_type"] != curr_attrs["hand_type"]:
            continue
        if attrs["interaction_motion"] != curr_attrs["interaction_motion"]:
            continue
        if attrs["left_hand_motion"] != curr_attrs["left_hand_motion"]:
            continue
        if not pair_realizes_target(row, task_field, task_target):
            continue
        out.append(row)
    out.sort(key=lambda row: (pair_distance(test_prev_geom, test_curr_geom, row, task_field), row["seq_name"], row["curr_frame_idx"]))
    return out


def left_preserve_candidates(pair_bank, current_left_group, curr_attrs, test_prev_geom):
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
        donor_left_group = grouped_motif_signature(str(donor_left.get("hand_motion", "unknown")), list(donor_left.get("transition_labels", [])))
        if donor_left_group != current_left_group:
            continue
        out.append(row)
    out.sort(
        key=lambda row: (
            hand_delta(test_prev_geom["left"], row["curr_geom"]["left"]),
            row["seq_name"],
            row["curr_frame_idx"],
        )
    )
    return out


def compose_split(prev_geom, right_row, left_row):
    out = {}
    # right
    out["right"] = compose_target_hand_transition(prev_geom, right_row, TASK_FIELD)
    # left
    prev_left = prev_geom["left"]
    donor_prev = left_row["prev_geom"]["left"]
    donor_curr = left_row["curr_geom"]["left"]
    if prev_left is None or donor_prev is None or donor_curr is None:
        out["left"] = None
    else:
        import numpy as np

        delta_local = np.asarray(donor_curr["local_vectors"], dtype=np.float32) - np.asarray(donor_prev["local_vectors"], dtype=np.float32)
        delta_flex = np.asarray(donor_curr["flexion"], dtype=np.float32) - np.asarray(donor_prev["flexion"], dtype=np.float32)
        out["left"] = {
            "local_vectors": np.asarray(prev_left["local_vectors"], dtype=np.float32) + delta_local,
            "flexion": np.asarray(prev_left["flexion"], dtype=np.float32) + delta_flex,
        }
    return out


def summarize(rows):
    n = len(rows)
    return {
        "num_frames": n,
        "baseline_available_rate": sum(row["baseline_available"] for row in rows) / n,
        "split_available_rate": sum(row["split_available"] for row in rows) / n,
        "baseline_right_grouped_match": sum(row["baseline_right_grouped_match"] for row in rows if row["baseline_available"]) / max(sum(row["baseline_available"] for row in rows), 1),
        "split_right_grouped_match": sum(row["split_right_grouped_match"] for row in rows if row["split_available"]) / max(sum(row["split_available"] for row in rows), 1),
        "baseline_left_preserve": sum(row["baseline_left_preserve"] for row in rows if row["baseline_available"]) / max(sum(row["baseline_available"] for row in rows), 1),
        "split_left_preserve": sum(row["split_left_preserve"] for row in rows if row["split_available"]) / max(sum(row["split_available"] for row in rows), 1),
        "baseline_joint_score": sum(row["baseline_joint_score"] for row in rows if row["baseline_available"]) / max(sum(row["baseline_available"] for row in rows), 1),
        "split_joint_score": sum(row["split_joint_score"] for row in rows if row["split_available"]) / max(sum(row["split_available"] for row in rows), 1),
        "split_beats_baseline_rate": sum(row["split_joint_score"] > row["baseline_joint_score"] for row in rows if row["baseline_available"] and row["split_available"]) / max(sum(row["baseline_available"] and row["split_available"] for row in rows), 1),
    }


def summarize_by_subtype(rows):
    by = defaultdict(list)
    for row in rows:
        by[(row["other_hand_motion"], row["interaction_motion_value"])].append(row)
    out = []
    for key, items in sorted(by.items()):
        other_hand_motion, interaction_motion_value = key
        base_avail = sum(row["baseline_available"] for row in items)
        split_avail = sum(row["split_available"] for row in items)
        out.append(
            {
                "other_hand_motion": other_hand_motion,
                "interaction_motion_value": interaction_motion_value,
                "num_frames": len(items),
                "baseline_available_rate": base_avail / len(items),
                "split_available_rate": split_avail / len(items),
                "baseline_joint_score": sum(row["baseline_joint_score"] for row in items if row["baseline_available"]) / max(base_avail, 1),
                "split_joint_score": sum(row["split_joint_score"] for row in items if row["split_available"]) / max(split_avail, 1),
                "baseline_right_grouped_match": sum(row["baseline_right_grouped_match"] for row in items if row["baseline_available"]) / max(base_avail, 1),
                "split_right_grouped_match": sum(row["split_right_grouped_match"] for row in items if row["split_available"]) / max(split_avail, 1),
                "baseline_left_preserve": sum(row["baseline_left_preserve"] for row in items if row["baseline_available"]) / max(base_avail, 1),
                "split_left_preserve": sum(row["split_left_preserve"] for row in items if row["split_available"]) / max(split_avail, 1),
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
                test_prev_geom = frame_geom(prev_frame)
                test_curr_geom = frame_geom(curr_frame)
                current_left_group = current_grouped_signature(curr_frame.get("left"))

                baseline_right_pool = right_relaxed_candidates(pair_bank, curr_attrs, test_prev_geom, test_curr_geom)
                baseline = baseline_right_pool[0] if baseline_right_pool else None
                left_pool = left_preserve_candidates(pair_bank, current_left_group, curr_attrs, test_prev_geom)
                left_best = left_pool[0] if left_pool else None

                rec = {
                    "seq_name": sequence["seq_name"],
                    "frame_idx": curr_frame["frame_idx"],
                    "other_hand_motion": "none" if curr_frame.get("left") is None else str(curr_frame["left"].get("hand_motion", "unknown")),
                    "interaction_motion_value": str(curr_frame.get("interaction_motion", "unknown")),
                    "baseline_available": int(baseline is not None),
                    "split_available": int(baseline is not None and left_best is not None),
                }

                if baseline is not None:
                    baseline_joint = compose_joint_two_hands(test_prev_geom, baseline)
                    right_m = hand_motif_from_geom(prev_frame.get("right"), baseline["curr_frame"].get("right"), baseline_joint["right"])
                    left_m = hand_motif_from_geom(prev_frame.get("left"), curr_frame.get("left"), baseline_joint["left"])
                    donor_right_group = "unknown" if right_m is None else grouped_motif_signature(str(right_m["donor_hand_motion"]), list(right_m["donor_transition_labels"]))
                    edited_right_group = "unknown" if right_m is None else grouped_motif_signature(str(right_m["edited_hand_motion"]), list(right_m["edited_transition_labels"]))
                    edited_left_group = "unknown" if left_m is None else grouped_motif_signature(str(left_m["edited_hand_motion"]), list(left_m["edited_transition_labels"]))
                    rec["baseline_right_grouped_match"] = int(edited_right_group == donor_right_group)
                    rec["baseline_left_preserve"] = int(edited_left_group == current_left_group)
                    rec["baseline_joint_score"] = rec["baseline_right_grouped_match"] * rec["baseline_left_preserve"]
                else:
                    rec["baseline_right_grouped_match"] = 0
                    rec["baseline_left_preserve"] = 0
                    rec["baseline_joint_score"] = 0

                if baseline is not None and left_best is not None:
                    split_joint = compose_split(test_prev_geom, baseline, left_best)
                    right_m = hand_motif_from_geom(prev_frame.get("right"), baseline["curr_frame"].get("right"), split_joint["right"])
                    left_m = hand_motif_from_geom(prev_frame.get("left"), left_best["curr_frame"].get("left"), split_joint["left"])
                    donor_right_group = "unknown" if right_m is None else grouped_motif_signature(str(right_m["donor_hand_motion"]), list(right_m["donor_transition_labels"]))
                    edited_right_group = "unknown" if right_m is None else grouped_motif_signature(str(right_m["edited_hand_motion"]), list(right_m["edited_transition_labels"]))
                    edited_left_group = "unknown" if left_m is None else grouped_motif_signature(str(left_m["edited_hand_motion"]), list(left_m["edited_transition_labels"]))
                    rec["split_right_grouped_match"] = int(edited_right_group == donor_right_group)
                    rec["split_left_preserve"] = int(edited_left_group == current_left_group)
                    rec["split_joint_score"] = rec["split_right_grouped_match"] * rec["split_left_preserve"]
                else:
                    rec["split_right_grouped_match"] = 0
                    rec["split_left_preserve"] = 0
                    rec["split_joint_score"] = 0

                rows.append(rec)

    payload = {
        "artifacts": {
            "train_json": str(GEN / "temporal_hl_val.json"),
            "test_json": str(GEN / "temporal_hl_test.json"),
        },
        "focus": {
            "task": f"{TASK_FIELD}->{TASK_TARGET}",
            "slice": "interaction only",
            "prototype": "split-donor candidate expansion",
        },
        "summary": summarize(rows),
        "subtype_summary": summarize_by_subtype(rows),
        "rows": rows,
    }

    out_json = GEN / "weak_slice_split_donor_prototype.json"
    out_md = SUM / "weak_slice_split_donor_prototype.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Weak Slice Split-Donor Prototype",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Focus: `right_hand_motion -> opening` on interaction frames only.",
        "",
        "Baseline: single symbolic donor expanded only on right-hand support.",
        "Prototype: separate right-hand and left-hand donors, composed jointly.",
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
            "| other hand motion | interaction motion | frames | baseline avail | split avail | baseline right | split right | baseline left | split left | baseline joint | split joint |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["subtype_summary"]:
        lines.append(
            f"| {row['other_hand_motion']} | {row['interaction_motion_value']} | {row['num_frames']} | "
            f"{fmt(row['baseline_available_rate'])} | {fmt(row['split_available_rate'])} | {fmt(row['baseline_right_grouped_match'])} | {fmt(row['split_right_grouped_match'])} | "
            f"{fmt(row['baseline_left_preserve'])} | {fmt(row['split_left_preserve'])} | {fmt(row['baseline_joint_score'])} | {fmt(row['split_joint_score'])} |"
        )
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
