#!/usr/bin/env python3
"""Joint two-hand composition prototype on the remaining weak slice.

This script compares:

- target-hand-only symbolic composition
- joint two-hand symbolic composition using the same donor pair

Focus:

- task: right_hand_motion -> opening
- setting: interaction frames only
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.build_transition_conditioned_symbolic_editor import (
    GEN,
    SUM,
    EDGE_ORDER,
    FINGER_NAMES,
    MINOR_DEG,
    STAY_DEG,
    build_pair_bank,
    build_semantic_frame_vocab,
    canonical,
    compose_target_hand_transition,
    fmt,
    frame_geom,
    grouped_motif_signature,
    load_json,
    normalize,
    overlap_labels,
    pick_best_symbolic_pair,
    quantize_direction,
    summarize_flexion_transition,
    transition_label,
)
from tools.build_local_edit_audit import contiguous_runs, eligible_value, frame_attrs


TASK_FIELD = "right_hand_motion"
TASK_TARGET = "opening"


def hand_motif_from_geom(prev_hand_raw, curr_hand_raw, edited_hand_geom):
    if prev_hand_raw is None or curr_hand_raw is None or edited_hand_geom is None:
        return None

    prev_vectors = [normalize(list(map(float, vec))) for vec in prev_hand_raw.get("local_vectors", [])]
    prev_tokens = list(map(int, prev_hand_raw.get("token_ids", [])))
    edited_vectors = [normalize(list(map(float, vec))) for vec in np.asarray(edited_hand_geom["local_vectors"], dtype=np.float32).tolist()]
    edited_tokens = [quantize_direction(vec)[0] for vec in edited_vectors]
    edited_labels = [quantize_direction(vec)[1] for vec in edited_vectors]
    edited_transition_labels, _ = transition_label(prev_vectors, edited_vectors, prev_tokens, edited_tokens, STAY_DEG, MINOR_DEG)

    prev_scores_raw = prev_hand_raw.get("flexion_scores", {})
    prev_scores = {finger: float(prev_scores_raw.get(finger, 0.0)) for finger in FINGER_NAMES}
    curr_arr = np.asarray(edited_hand_geom["flexion"], dtype=np.float32)
    curr_scores = {finger: float(curr_arr[idx]) for idx, finger in enumerate(FINGER_NAMES)}
    edited_hand_motion = summarize_flexion_transition(prev_scores, curr_scores, 0.05)

    donor_transition_labels = list(curr_hand_raw.get("transition_labels", []))
    donor_token_labels = list(curr_hand_raw.get("token_labels", []))
    donor_hand_motion = str(curr_hand_raw.get("hand_motion", "unknown"))

    state_agreement = 0.0
    transition_agreement = 0.0
    if donor_token_labels:
        state_agreement = sum(int(a == b) for a, b in zip(edited_labels, donor_token_labels)) / max(len(donor_token_labels), 1)
    if donor_transition_labels:
        transition_agreement = sum(int(a == b) for a, b in zip(edited_transition_labels, donor_transition_labels)) / max(len(donor_transition_labels), 1)

    return {
        "edited_hand_motion": edited_hand_motion,
        "edited_transition_labels": edited_transition_labels,
        "edited_token_labels": edited_labels,
        "donor_hand_motion": donor_hand_motion,
        "donor_transition_labels": donor_transition_labels,
        "donor_token_labels": donor_token_labels,
        "state_agreement": state_agreement,
        "transition_agreement": transition_agreement,
    }


def compose_joint_two_hands(test_prev_geom, donor_row):
    out = {}
    for hand_name in ("right", "left"):
        prev_hand = test_prev_geom[hand_name]
        donor_prev = donor_row["prev_geom"][hand_name]
        donor_curr = donor_row["curr_geom"][hand_name]
        if prev_hand is None or donor_prev is None or donor_curr is None:
            out[hand_name] = None
            continue
        delta_local = np.asarray(donor_curr["local_vectors"], dtype=np.float32) - np.asarray(donor_prev["local_vectors"], dtype=np.float32)
        delta_flex = np.asarray(donor_curr["flexion"], dtype=np.float32) - np.asarray(donor_prev["flexion"], dtype=np.float32)
        out[hand_name] = {
            "local_vectors": np.asarray(prev_hand["local_vectors"], dtype=np.float32) + delta_local,
            "flexion": np.asarray(prev_hand["flexion"], dtype=np.float32) + delta_flex,
        }
    return out


def current_grouped_signature(curr_hand_raw):
    if curr_hand_raw is None:
        return "none"
    return grouped_motif_signature(str(curr_hand_raw.get("hand_motion", "unknown")), list(curr_hand_raw.get("transition_labels", [])))


def summarize(rows):
    n = len(rows)
    return {
        "num_frames": n,
        "target_only_right_grouped_match": sum(row["target_only_right_grouped_match"] for row in rows) / n,
        "joint_right_grouped_match": sum(row["joint_right_grouped_match"] for row in rows) / n,
        "target_only_left_preserve": sum(row["target_only_left_preserve"] for row in rows) / n,
        "joint_left_preserve": sum(row["joint_left_preserve"] for row in rows) / n,
        "target_only_joint_score": sum(row["target_only_joint_score"] for row in rows) / n,
        "joint_joint_score": sum(row["joint_joint_score"] for row in rows) / n,
        "joint_beats_target_rate": sum(row["joint_joint_score"] > row["target_only_joint_score"] for row in rows) / n,
        "joint_ge_target_rate": sum(row["joint_joint_score"] >= row["target_only_joint_score"] for row in rows) / n,
    }


def summarize_by_subtype(rows):
    by = defaultdict(list)
    for row in rows:
        by[(row["other_hand_motion"], row["interaction_motion_value"])].append(row)
    out = []
    for key, items in sorted(by.items()):
        other_hand_motion, interaction_motion_value = key
        out.append(
            {
                "other_hand_motion": other_hand_motion,
                "interaction_motion_value": interaction_motion_value,
                "num_frames": len(items),
                "target_only_right_grouped_match": sum(row["target_only_right_grouped_match"] for row in items) / len(items),
                "joint_right_grouped_match": sum(row["joint_right_grouped_match"] for row in items) / len(items),
                "target_only_left_preserve": sum(row["target_only_left_preserve"] for row in items) / len(items),
                "joint_left_preserve": sum(row["joint_left_preserve"] for row in items) / len(items),
                "target_only_joint_score": sum(row["target_only_joint_score"] for row in items) / len(items),
                "joint_joint_score": sum(row["joint_joint_score"] for row in items) / len(items),
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
                donor = pick_best_symbolic_pair(pair_bank, curr_attrs, test_prev_geom, test_curr_geom, TASK_FIELD, TASK_TARGET)
                if donor is None:
                    continue

                target_only_right = compose_target_hand_transition(test_prev_geom, donor, TASK_FIELD)
                target_only_left = test_prev_geom["left"]
                joint = compose_joint_two_hands(test_prev_geom, donor)

                right_target_only_motif = hand_motif_from_geom(prev_frame.get("right"), donor["curr_frame"].get("right"), target_only_right)
                right_joint_motif = hand_motif_from_geom(prev_frame.get("right"), donor["curr_frame"].get("right"), joint["right"])
                left_target_only_motif = hand_motif_from_geom(prev_frame.get("left"), curr_frame.get("left"), target_only_left)
                left_joint_motif = hand_motif_from_geom(prev_frame.get("left"), curr_frame.get("left"), joint["left"])

                donor_right_group = (
                    "unknown"
                    if right_target_only_motif is None
                    else grouped_motif_signature(str(right_target_only_motif["donor_hand_motion"]), list(right_target_only_motif["donor_transition_labels"]))
                )
                right_target_only_group = (
                    "unknown"
                    if right_target_only_motif is None
                    else grouped_motif_signature(str(right_target_only_motif["edited_hand_motion"]), list(right_target_only_motif["edited_transition_labels"]))
                )
                right_joint_group = (
                    "unknown"
                    if right_joint_motif is None
                    else grouped_motif_signature(str(right_joint_motif["edited_hand_motion"]), list(right_joint_motif["edited_transition_labels"]))
                )
                current_left_group = current_grouped_signature(curr_frame.get("left"))
                left_target_only_group = (
                    "unknown"
                    if left_target_only_motif is None
                    else grouped_motif_signature(str(left_target_only_motif["edited_hand_motion"]), list(left_target_only_motif["edited_transition_labels"]))
                )
                left_joint_group = (
                    "unknown"
                    if left_joint_motif is None
                    else grouped_motif_signature(str(left_joint_motif["edited_hand_motion"]), list(left_joint_motif["edited_transition_labels"]))
                )

                target_only_right_match = int(right_target_only_group == donor_right_group)
                joint_right_match = int(right_joint_group == donor_right_group)
                target_only_left_preserve = int(left_target_only_group == current_left_group)
                joint_left_preserve = int(left_joint_group == current_left_group)

                rows.append(
                    {
                        "seq_name": sequence["seq_name"],
                        "frame_idx": curr_frame["frame_idx"],
                        "other_hand_motion": "none" if curr_frame.get("left") is None else str(curr_frame["left"].get("hand_motion", "unknown")),
                        "interaction_motion_value": str(curr_frame.get("interaction_motion", "unknown")),
                        "target_only_right_grouped_match": target_only_right_match,
                        "joint_right_grouped_match": joint_right_match,
                        "target_only_left_preserve": target_only_left_preserve,
                        "joint_left_preserve": joint_left_preserve,
                        "target_only_joint_score": target_only_right_match * target_only_left_preserve,
                        "joint_joint_score": joint_right_match * joint_left_preserve,
                    }
                )

    payload = {
        "artifacts": {
            "train_json": str(GEN / "temporal_hl_val.json"),
            "test_json": str(GEN / "temporal_hl_test.json"),
        },
        "focus": {
            "task": f"{TASK_FIELD}->{TASK_TARGET}",
            "slice": "interaction only",
            "prototype": "joint two-hand composition",
        },
        "summary": summarize(rows),
        "subtype_summary": summarize_by_subtype(rows),
        "rows": rows,
    }

    out_json = GEN / "weak_slice_joint_editor_prototype.json"
    out_md = SUM / "weak_slice_joint_editor_prototype.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Weak Slice Joint Editor Prototype",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Focus: `right_hand_motion -> opening` on interaction frames only.",
        "",
        "Compare target-hand-only composition against joint two-hand composition",
        "using the same symbolic donor pair.",
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
            "| other hand motion | interaction motion | frames | target-only right | joint right | target-only left preserve | joint left preserve | target-only joint score | joint joint score |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["subtype_summary"]:
        lines.append(
            f"| {row['other_hand_motion']} | {row['interaction_motion_value']} | {row['num_frames']} | "
            f"{fmt(row['target_only_right_grouped_match'])} | {fmt(row['joint_right_grouped_match'])} | {fmt(row['target_only_left_preserve'])} | "
            f"{fmt(row['joint_left_preserve'])} | {fmt(row['target_only_joint_score'])} | {fmt(row['joint_joint_score'])} |"
        )
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
