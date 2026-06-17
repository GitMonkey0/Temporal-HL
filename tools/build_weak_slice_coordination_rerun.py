#!/usr/bin/env python3
"""Targeted rerun for the remaining weak interaction slice.

This script focuses only on:

- task: right_hand_motion -> opening
- setting: interacting frames

It compares:

- baseline symbolic donor-pair selection
- coordination-aware symbolic donor-pair selection that additionally preserves
  the other hand's transition activity profile
- opaque proxy baselines
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from tools.build_transition_conditioned_symbolic_editor import (
    ROOT,
    GEN,
    SUM,
    active_finger_count,
    best_proxy_attrs,
    build_pair_bank,
    build_semantic_frame_vocab,
    canonical,
    cluster_majority_attrs,
    compose_target_hand_transition,
    finger_activity_profile,
    fmt,
    grouped_motif_signature,
    load_json,
    overlap_labels,
    pair_distance,
    pair_realizes_target,
    pick_best_proxy_pair,
    pick_best_symbolic_pair,
    source_cluster_decoded,
    summarize_composed_motif,
)
from tools.build_conditional_hand_transplant_audit import preserved_fields
from tools.build_geometry_locality_audit import frame_geom
from tools.build_local_edit_audit import TRACKED_FIELDS, contiguous_runs, eligible_value, frame_attrs


TASK_FIELD = "right_hand_motion"
TASK_TARGET = "opening"


def other_hand_transition_profile(frame: dict[str, object]) -> tuple[str, ...] | None:
    left = frame.get("left")
    if left is None:
        return None
    return tuple(finger_activity_profile(list(left.get("transition_labels", []))))


def coordination_symbolic_candidates(pair_bank, curr_attrs, curr_profile, task_field, target_value):
    keep = preserved_fields(task_field)
    out = []
    for row in pair_bank:
        attrs = row["curr_attrs"]
        if attrs[task_field] != target_value:
            continue
        if any(attrs[field] != curr_attrs[field] for field in keep):
            continue
        donor_profile = other_hand_transition_profile(row["curr_frame"])
        if donor_profile != curr_profile:
            continue
        out.append(row)
    return out


def pick_best_coordination_symbolic_pair(pair_bank, curr_attrs, curr_profile, test_prev_geom, test_curr_geom, task_field, target_value):
    cands = [
        row
        for row in coordination_symbolic_candidates(pair_bank, curr_attrs, curr_profile, task_field, target_value)
        if pair_realizes_target(row, task_field, target_value)
    ]
    if not cands:
        return None
    cands.sort(key=lambda row: (pair_distance(test_prev_geom, test_curr_geom, row, task_field), row["seq_name"], row["curr_frame_idx"]))
    return cands[0]


def summarize(rows):
    out = {}
    n = len(rows)
    base_avail = sum(row["base_available"] for row in rows)
    coord_avail = sum(row["coord_available"] for row in rows)
    out["num_frames"] = n
    out["base_available_rate"] = base_avail / n
    out["coord_available_rate"] = coord_avail / n
    out["base_grouped_match"] = sum(row["base_grouped_match"] for row in rows if row["base_available"]) / max(base_avail, 1)
    out["coord_grouped_match"] = sum(row["coord_grouped_match"] for row in rows if row["coord_available"]) / max(coord_avail, 1)
    out["base_transition_agreement"] = sum(row["base_transition_agreement"] for row in rows if row["base_available"]) / max(base_avail, 1)
    out["coord_transition_agreement"] = sum(row["coord_transition_agreement"] for row in rows if row["coord_available"]) / max(coord_avail, 1)
    for source_name in ("semantic_frame", "continuous_frame"):
        out[f"{source_name}_proxy_grouped_match"] = sum(row[f"{source_name}_proxy_grouped_match"] for row in rows) / n
        out[f"{source_name}_proxy_transition_agreement"] = sum(row[f"{source_name}_proxy_transition_agreement"] for row in rows) / n
    both = [row for row in rows if row["base_available"] and row["coord_available"]]
    out["coord_beats_base_rate"] = sum(row["coord_grouped_match"] > row["base_grouped_match"] for row in both) / max(len(both), 1)
    out["coord_ge_base_rate"] = sum(row["coord_grouped_match"] >= row["base_grouped_match"] for row in both) / max(len(both), 1)
    return out


def summarize_by_subtype(rows):
    by = defaultdict(list)
    for row in rows:
        by[(row["other_hand_motion"], row["interaction_motion_value"], row["other_hand_profile"])].append(row)
    out = []
    for key, items in sorted(by.items()):
        other_hand_motion, interaction_motion_value, other_hand_profile = key
        n = len(items)
        base_avail = sum(row["base_available"] for row in items)
        coord_avail = sum(row["coord_available"] for row in items)
        out.append(
            {
                "other_hand_motion": other_hand_motion,
                "interaction_motion_value": interaction_motion_value,
                "other_hand_profile": other_hand_profile,
                "num_frames": n,
                "base_available_rate": base_avail / n,
                "coord_available_rate": coord_avail / n,
                "base_grouped_match": sum(row["base_grouped_match"] for row in items if row["base_available"]) / max(base_avail, 1),
                "coord_grouped_match": sum(row["coord_grouped_match"] for row in items if row["coord_available"]) / max(coord_avail, 1),
                "semantic_proxy_grouped_match": sum(row["semantic_frame_proxy_grouped_match"] for row in items) / n,
                "continuous_proxy_grouped_match": sum(row["continuous_frame_proxy_grouped_match"] for row in items) / n,
            }
        )
    return out


def main():
    train_data = load_json(GEN / "temporal_hl_val.json")
    test_data = load_json(GEN / "temporal_hl_test.json")
    labels = overlap_labels(train_data, test_data)
    semantic_vocab = build_semantic_frame_vocab(train_data, labels)
    pair_bank = build_pair_bank(train_data, labels, semantic_vocab)

    semantic_decoded = source_cluster_decoded("semantic_frame", pair_bank)
    continuous_decoded = source_cluster_decoded("continuous_frame", pair_bank)

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
                curr_profile = other_hand_transition_profile(curr_frame)
                test_prev_geom = frame_geom(prev_frame)
                test_curr_geom = frame_geom(curr_frame)

                base = pick_best_symbolic_pair(pair_bank, curr_attrs, test_prev_geom, test_curr_geom, TASK_FIELD, TASK_TARGET)
                coord = None if curr_profile is None else pick_best_coordination_symbolic_pair(
                    pair_bank, curr_attrs, curr_profile, test_prev_geom, test_curr_geom, TASK_FIELD, TASK_TARGET
                )

                rec = {
                    "seq_name": sequence["seq_name"],
                    "frame_idx": curr_frame["frame_idx"],
                    "other_hand_motion": "none" if curr_frame.get("left") is None else str(curr_frame["left"].get("hand_motion", "unknown")),
                    "interaction_motion_value": str(curr_frame.get("interaction_motion", "unknown")),
                    "other_hand_profile": "none" if curr_profile is None else "|".join(curr_profile),
                    "base_available": int(base is not None),
                    "coord_available": int(coord is not None),
                }

                if base is not None:
                    edited = compose_target_hand_transition(test_prev_geom, base, TASK_FIELD)
                    motif = None if edited is None else summarize_composed_motif(prev_frame, edited, base, TASK_FIELD)
                    donor_group = (
                        "unknown"
                        if motif is None
                        else grouped_motif_signature(str(motif["donor_hand_motion"]), list(motif["donor_transition_labels"]))
                    )
                    edited_group = (
                        "unknown"
                        if motif is None
                        else grouped_motif_signature(str(motif["edited_hand_motion"]), list(motif["edited_transition_labels"]))
                    )
                    rec["base_grouped_match"] = int(motif is not None and donor_group == edited_group)
                    rec["base_transition_agreement"] = 0.0 if motif is None else float(motif["transition_agreement"])
                else:
                    rec["base_grouped_match"] = 0
                    rec["base_transition_agreement"] = 0.0

                if coord is not None:
                    edited = compose_target_hand_transition(test_prev_geom, coord, TASK_FIELD)
                    motif = None if edited is None else summarize_composed_motif(prev_frame, edited, coord, TASK_FIELD)
                    donor_group = (
                        "unknown"
                        if motif is None
                        else grouped_motif_signature(str(motif["donor_hand_motion"]), list(motif["donor_transition_labels"]))
                    )
                    edited_group = (
                        "unknown"
                        if motif is None
                        else grouped_motif_signature(str(motif["edited_hand_motion"]), list(motif["edited_transition_labels"]))
                    )
                    rec["coord_grouped_match"] = int(motif is not None and donor_group == edited_group)
                    rec["coord_transition_agreement"] = 0.0 if motif is None else float(motif["transition_agreement"])
                else:
                    rec["coord_grouped_match"] = 0
                    rec["coord_transition_agreement"] = 0.0

                for source_name, decoded in (("semantic_frame", semantic_decoded), ("continuous_frame", continuous_decoded)):
                    proxy_attrs = best_proxy_attrs(curr_attrs, TASK_FIELD, TASK_TARGET, decoded)
                    prox = None if proxy_attrs is None else pick_best_proxy_pair(
                        pair_bank, source_name, decoded, proxy_attrs, test_prev_geom, test_curr_geom, TASK_FIELD, TASK_TARGET
                    )
                    if prox is None:
                        rec[f"{source_name}_proxy_grouped_match"] = 0
                        rec[f"{source_name}_proxy_transition_agreement"] = 0.0
                    else:
                        edited = compose_target_hand_transition(test_prev_geom, prox, TASK_FIELD)
                        motif = None if edited is None else summarize_composed_motif(prev_frame, edited, prox, TASK_FIELD)
                        donor_group = (
                            "unknown"
                            if motif is None
                            else grouped_motif_signature(str(motif["donor_hand_motion"]), list(motif["donor_transition_labels"]))
                        )
                        edited_group = (
                            "unknown"
                            if motif is None
                            else grouped_motif_signature(str(motif["edited_hand_motion"]), list(motif["edited_transition_labels"]))
                        )
                        rec[f"{source_name}_proxy_grouped_match"] = int(motif is not None and donor_group == edited_group)
                        rec[f"{source_name}_proxy_transition_agreement"] = 0.0 if motif is None else float(motif["transition_agreement"])

                rows.append(rec)

    payload = {
        "artifacts": {
            "train_json": str(GEN / "temporal_hl_val.json"),
            "test_json": str(GEN / "temporal_hl_test.json"),
        },
        "focus": {
            "task": f"{TASK_FIELD}->{TASK_TARGET}",
            "slice": "interaction only",
            "new_channel": "other-hand transition activity profile",
        },
        "summary": summarize(rows),
        "subtype_summary": summarize_by_subtype(rows),
        "rows": rows,
    }

    out_json = GEN / "weak_slice_coordination_rerun.json"
    out_md = SUM / "weak_slice_coordination_rerun.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Weak Slice Coordination Rerun",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Focus: `right_hand_motion -> opening` on interaction frames only.",
        "",
        "The coordination-aware variant adds one explicit temporal channel:",
        "`other-hand transition activity profile`.",
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
            "| other hand motion | interaction motion | other-hand profile | frames | base avail | coord avail | base grouped | coord grouped | semantic proxy grouped | continuous proxy grouped |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["subtype_summary"]:
        lines.append(
            f"| {row['other_hand_motion']} | {row['interaction_motion_value']} | {row['other_hand_profile']} | {row['num_frames']} | "
            f"{fmt(row['base_available_rate'])} | {fmt(row['coord_available_rate'])} | {fmt(row['base_grouped_match'])} | "
            f"{fmt(row['coord_grouped_match'])} | {fmt(row['semantic_proxy_grouped_match'])} | {fmt(row['continuous_proxy_grouped_match'])} |"
        )
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
