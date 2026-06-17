#!/usr/bin/env python3
"""Soft coordination reranking on the remaining weak interaction slice.

This targeted rerun keeps the coarse symbolic conditioning from the baseline
selector, but reranks symbolic donor pairs with a soft penalty on the other
hand's temporal profile mismatch instead of requiring an exact match.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from tools.build_transition_conditioned_symbolic_editor import (
    GEN,
    SUM,
    best_proxy_attrs,
    build_pair_bank,
    build_semantic_frame_vocab,
    canonical,
    compose_target_hand_transition,
    finger_activity_profile,
    fmt,
    frame_geom,
    grouped_motif_signature,
    load_json,
    overlap_labels,
    pair_distance,
    pair_realizes_target,
    pick_best_proxy_pair,
    source_cluster_decoded,
    summarize_composed_motif,
    symbolic_pair_candidates,
)
from tools.build_local_edit_audit import contiguous_runs, eligible_value, frame_attrs


TASK_FIELD = "right_hand_motion"
TASK_TARGET = "opening"
LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.0]


def other_hand_profile(frame: dict[str, object]) -> tuple[str, ...] | None:
    left = frame.get("left")
    if left is None:
        return None
    return tuple(finger_activity_profile(list(left.get("transition_labels", []))))


def profile_distance(a: tuple[str, ...] | None, b: tuple[str, ...] | None) -> float:
    if a is None or b is None:
        return 5.0
    return float(sum(x != y for x, y in zip(a, b)))


def pick_soft_coordination_pair(pair_bank, curr_attrs, curr_profile, test_prev_geom, test_curr_geom, lam: float):
    cands = [
        row
        for row in symbolic_pair_candidates(pair_bank, curr_attrs, TASK_FIELD, TASK_TARGET)
        if pair_realizes_target(row, TASK_FIELD, TASK_TARGET)
    ]
    if not cands:
        return None

    def score(row):
        donor_profile = other_hand_profile(row["curr_frame"])
        return (
            pair_distance(test_prev_geom, test_curr_geom, row, TASK_FIELD)
            + lam * profile_distance(curr_profile, donor_profile),
            row["seq_name"],
            row["curr_frame_idx"],
        )

    cands.sort(key=score)
    return cands[0]


def evaluate_pair(prev_frame, test_prev_geom, donor_row):
    edited = compose_target_hand_transition(test_prev_geom, donor_row, TASK_FIELD)
    motif = None if edited is None else summarize_composed_motif(prev_frame, edited, donor_row, TASK_FIELD)
    if motif is None:
        return {"grouped_match": 0, "transition_agreement": 0.0, "state_agreement": 0.0}
    donor_group = grouped_motif_signature(str(motif["donor_hand_motion"]), list(motif["donor_transition_labels"]))
    edited_group = grouped_motif_signature(str(motif["edited_hand_motion"]), list(motif["edited_transition_labels"]))
    return {
        "grouped_match": int(donor_group == edited_group),
        "transition_agreement": float(motif["transition_agreement"]),
        "state_agreement": float(motif["state_agreement"]),
    }


def summarize(rows):
    out = {"num_frames": len(rows)}
    for lam in LAMBDAS:
        key = f"l{lam}"
        out[f"{key}_grouped_match"] = sum(row[f"{key}_grouped_match"] for row in rows) / len(rows)
        out[f"{key}_transition_agreement"] = sum(row[f"{key}_transition_agreement"] for row in rows) / len(rows)
        out[f"{key}_state_agreement"] = sum(row[f"{key}_state_agreement"] for row in rows) / len(rows)
    out["best_lambda_by_grouped_match"] = max(LAMBDAS, key=lambda lam: out[f"l{lam}_grouped_match"])
    for source_name in ("semantic_frame", "continuous_frame"):
        out[f"{source_name}_proxy_grouped_match"] = sum(row[f"{source_name}_proxy_grouped_match"] for row in rows) / len(rows)
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
        for lam in LAMBDAS:
            lkey = f"l{lam}"
            rec[f"{lkey}_grouped_match"] = sum(row[f"{lkey}_grouped_match"] for row in items) / len(items)
        out.append(rec)
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
                curr_profile = other_hand_profile(curr_frame)
                test_prev_geom = frame_geom(prev_frame)
                test_curr_geom = frame_geom(curr_frame)
                rec = {
                    "seq_name": sequence["seq_name"],
                    "frame_idx": curr_frame["frame_idx"],
                    "other_hand_motion": "none" if curr_frame.get("left") is None else str(curr_frame["left"].get("hand_motion", "unknown")),
                    "interaction_motion_value": str(curr_frame.get("interaction_motion", "unknown")),
                }

                for lam in LAMBDAS:
                    donor = pick_soft_coordination_pair(pair_bank, curr_attrs, curr_profile, test_prev_geom, test_curr_geom, lam)
                    result = {"grouped_match": 0, "transition_agreement": 0.0, "state_agreement": 0.0} if donor is None else evaluate_pair(prev_frame, test_prev_geom, donor)
                    key = f"l{lam}"
                    rec[f"{key}_grouped_match"] = result["grouped_match"]
                    rec[f"{key}_transition_agreement"] = result["transition_agreement"]
                    rec[f"{key}_state_agreement"] = result["state_agreement"]

                for source_name, decoded in (("semantic_frame", semantic_decoded), ("continuous_frame", continuous_decoded)):
                    proxy_attrs = best_proxy_attrs(curr_attrs, TASK_FIELD, TASK_TARGET, decoded)
                    prox = None if proxy_attrs is None else pick_best_proxy_pair(
                        pair_bank, source_name, decoded, proxy_attrs, test_prev_geom, test_curr_geom, TASK_FIELD, TASK_TARGET
                    )
                    if prox is None:
                        rec[f"{source_name}_proxy_grouped_match"] = 0
                    else:
                        result = evaluate_pair(prev_frame, test_prev_geom, prox)
                        rec[f"{source_name}_proxy_grouped_match"] = result["grouped_match"]

                rows.append(rec)

    payload = {
        "artifacts": {
            "train_json": str(GEN / "temporal_hl_val.json"),
            "test_json": str(GEN / "temporal_hl_test.json"),
        },
        "focus": {
            "task": f"{TASK_FIELD}->{TASK_TARGET}",
            "slice": "interaction only",
            "mechanism": "soft coordination reranking",
            "lambdas": LAMBDAS,
        },
        "summary": summarize(rows),
        "subtype_summary": summarize_by_subtype(rows),
        "rows": rows,
    }

    out_json = GEN / "weak_slice_soft_coordination_rerun.json"
    out_md = SUM / "weak_slice_soft_coordination_rerun.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Weak Slice Soft Coordination Rerun",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Focus: `right_hand_motion -> opening` on interaction frames only.",
        "",
        "This rerun keeps the same coarse symbolic conditioning and reranks donor",
        "pairs with a soft penalty on the other hand's transition activity profile",
        "mismatch.",
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
            "| other hand motion | interaction motion | frames | l0.0 grouped | l0.25 grouped | l0.5 grouped | l1.0 grouped | l2.0 grouped |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["subtype_summary"]:
        lines.append(
            f"| {row['other_hand_motion']} | {row['interaction_motion_value']} | {row['num_frames']} | "
            f"{fmt(row['l0.0_grouped_match'])} | {fmt(row['l0.25_grouped_match'])} | {fmt(row['l0.5_grouped_match'])} | "
            f"{fmt(row['l1.0_grouped_match'])} | {fmt(row['l2.0_grouped_match'])} |"
        )
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
