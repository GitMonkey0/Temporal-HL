#!/usr/bin/env python3
"""Mechanism rerun for the remaining weak interaction slice.

Focus:

- task: right_hand_motion -> opening
- setting: interaction frames only

This rerun keeps the same symbolic donor selector and tests whether the weak
slice is caused by the composition mechanism itself. We compare exact donor
delta transplant against several blended/scaled variants.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.build_transition_conditioned_symbolic_editor import (
    GEN,
    SUM,
    best_proxy_attrs,
    build_pair_bank,
    build_semantic_frame_vocab,
    canonical,
    compose_target_hand_transition,
    fmt,
    frame_geom,
    frame_vector,
    grouped_motif_signature,
    load_json,
    overlap_labels,
    pick_best_proxy_pair,
    pick_best_symbolic_pair,
    source_cluster_decoded,
    summarize_composed_motif,
)
from tools.build_local_edit_audit import contiguous_runs, eligible_value, frame_attrs


TASK_FIELD = "right_hand_motion"
TASK_TARGET = "opening"
ALPHAS = [0.25, 0.5, 0.75, 1.0]


def compose_scaled_transition(test_prev_geom, donor_row, alpha: float):
    prev_hand = test_prev_geom["right"]
    donor_prev = donor_row["prev_geom"]["right"]
    donor_curr = donor_row["curr_geom"]["right"]
    if prev_hand is None or donor_prev is None or donor_curr is None:
        return None
    delta_local = np.asarray(donor_curr["local_vectors"], dtype=np.float32) - np.asarray(donor_prev["local_vectors"], dtype=np.float32)
    delta_flex = np.asarray(donor_curr["flexion"], dtype=np.float32) - np.asarray(donor_prev["flexion"], dtype=np.float32)
    new_local = np.asarray(prev_hand["local_vectors"], dtype=np.float32) + alpha * delta_local
    new_flex = np.asarray(prev_hand["flexion"], dtype=np.float32) + alpha * delta_flex
    return {
        "local_vectors": new_local,
        "flexion": new_flex,
    }


def evaluate_variant(prev_frame, test_prev_geom, donor_row, alpha: float):
    edited = compose_scaled_transition(test_prev_geom, donor_row, alpha)
    motif = None if edited is None else summarize_composed_motif(prev_frame, edited, donor_row, TASK_FIELD)
    if motif is None:
        return {
            "grouped_match": 0,
            "state_agreement": 0.0,
            "transition_agreement": 0.0,
        }
    donor_group = grouped_motif_signature(str(motif["donor_hand_motion"]), list(motif["donor_transition_labels"]))
    edited_group = grouped_motif_signature(str(motif["edited_hand_motion"]), list(motif["edited_transition_labels"]))
    return {
        "grouped_match": int(donor_group == edited_group),
        "state_agreement": float(motif["state_agreement"]),
        "transition_agreement": float(motif["transition_agreement"]),
    }


def summarize(rows):
    out = {"num_frames": len(rows)}
    for alpha in ALPHAS:
        key = f"a{alpha}"
        out[f"{key}_grouped_match"] = sum(row[f"{key}_grouped_match"] for row in rows) / len(rows)
        out[f"{key}_state_agreement"] = sum(row[f"{key}_state_agreement"] for row in rows) / len(rows)
        out[f"{key}_transition_agreement"] = sum(row[f"{key}_transition_agreement"] for row in rows) / len(rows)
    out["best_alpha_by_grouped_match"] = max(ALPHAS, key=lambda a: out[f"a{a}_grouped_match"])
    return out


def summarize_by_subtype(rows):
    by = defaultdict(list)
    for row in rows:
        by[(row["other_hand_motion"], row["interaction_motion_value"])].append(row)
    out = []
    for key, items in sorted(by.items()):
        other_hand_motion, interaction_motion_value = key
        rec = {
            "other_hand_motion": other_hand_motion,
            "interaction_motion_value": interaction_motion_value,
            "num_frames": len(items),
        }
        for alpha in ALPHAS:
            akey = f"a{alpha}"
            rec[f"{akey}_grouped_match"] = sum(row[f"{akey}_grouped_match"] for row in items) / len(items)
            rec[f"{akey}_transition_agreement"] = sum(row[f"{akey}_transition_agreement"] for row in items) / len(items)
        out.append(rec)
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
                rec = {
                    "seq_name": sequence["seq_name"],
                    "frame_idx": curr_frame["frame_idx"],
                    "other_hand_motion": "none" if curr_frame.get("left") is None else str(curr_frame["left"].get("hand_motion", "unknown")),
                    "interaction_motion_value": str(curr_frame.get("interaction_motion", "unknown")),
                }
                for alpha in ALPHAS:
                    result = evaluate_variant(prev_frame, test_prev_geom, donor, alpha)
                    key = f"a{alpha}"
                    rec[f"{key}_grouped_match"] = result["grouped_match"]
                    rec[f"{key}_state_agreement"] = result["state_agreement"]
                    rec[f"{key}_transition_agreement"] = result["transition_agreement"]
                rows.append(rec)

    payload = {
        "artifacts": {
            "train_json": str(GEN / "temporal_hl_val.json"),
            "test_json": str(GEN / "temporal_hl_test.json"),
        },
        "focus": {
            "task": f"{TASK_FIELD}->{TASK_TARGET}",
            "slice": "interaction only",
            "mechanism": "scaled donor delta composition",
            "alphas": ALPHAS,
        },
        "summary": summarize(rows),
        "subtype_summary": summarize_by_subtype(rows),
        "rows": rows,
    }

    out_json = GEN / "weak_slice_mechanism_rerun.json"
    out_md = SUM / "weak_slice_mechanism_rerun.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Weak Slice Mechanism Rerun",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Focus: `right_hand_motion -> opening` on interaction frames only.",
        "",
        "This rerun keeps the same symbolic donor selector and only changes the",
        "target-hand composition mechanism by scaling donor deltas.",
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
            "| other hand motion | interaction motion | frames | a0.25 grouped | a0.5 grouped | a0.75 grouped | a1.0 grouped | a0.25 transition | a0.5 transition | a0.75 transition | a1.0 transition |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["subtype_summary"]:
        lines.append(
            f"| {row['other_hand_motion']} | {row['interaction_motion_value']} | {row['num_frames']} | "
            f"{fmt(row['a0.25_grouped_match'])} | {fmt(row['a0.5_grouped_match'])} | {fmt(row['a0.75_grouped_match'])} | {fmt(row['a1.0_grouped_match'])} | "
            f"{fmt(row['a0.25_transition_agreement'])} | {fmt(row['a0.5_transition_agreement'])} | {fmt(row['a0.75_transition_agreement'])} | {fmt(row['a1.0_transition_agreement'])} |"
        )
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
