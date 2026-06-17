#!/usr/bin/env python3
"""Top-k x top-k joint search on the weak interaction slice.

This extends the split-donor prototype:

- collect top-k right-hand target donors
- collect top-k left-hand preserve donors
- evaluate all pairings with a joint score
- choose the best joint pair
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
    compose_joint_two_hands,
    current_grouped_signature,
    fmt,
    hand_motif_from_geom,
)
from tools.build_weak_slice_split_donor_prototype import (
    compose_split,
    left_preserve_candidates,
    right_relaxed_candidates,
)


TOPK_RIGHT = 5
TOPK_LEFT = 5


def evaluate_joint(prev_frame, curr_frame, prev_geom, right_row, left_row=None):
    if left_row is None:
        joint = compose_joint_two_hands(prev_geom, right_row)
        left_curr_ref = curr_frame.get("left")
    else:
        joint = compose_split(prev_geom, right_row, left_row)
        left_curr_ref = left_row["curr_frame"].get("left")

    right_m = hand_motif_from_geom(prev_frame.get("right"), right_row["curr_frame"].get("right"), joint["right"])
    left_m = hand_motif_from_geom(prev_frame.get("left"), left_curr_ref, joint["left"])

    donor_right_group = "unknown" if right_m is None else grouped_motif_signature(str(right_m["donor_hand_motion"]), list(right_m["donor_transition_labels"]))
    edited_right_group = "unknown" if right_m is None else grouped_motif_signature(str(right_m["edited_hand_motion"]), list(right_m["edited_transition_labels"]))
    current_left_group = current_grouped_signature(curr_frame.get("left"))
    edited_left_group = "unknown" if left_m is None else grouped_motif_signature(str(left_m["edited_hand_motion"]), list(left_m["edited_transition_labels"]))

    right_match = int(edited_right_group == donor_right_group)
    left_preserve = int(edited_left_group == current_left_group)
    return {
        "right_grouped_match": right_match,
        "left_preserve": left_preserve,
        "joint_score": right_match * left_preserve,
    }


def summarize(rows):
    n = len(rows)
    return {
        "num_frames": n,
        "baseline_available_rate": sum(row["baseline_available"] for row in rows) / n,
        "split_available_rate": sum(row["split_available"] for row in rows) / n,
        "topk_available_rate": sum(row["topk_available"] for row in rows) / n,
        "baseline_joint_score": sum(row["baseline_joint_score"] for row in rows if row["baseline_available"]) / max(sum(row["baseline_available"] for row in rows), 1),
        "split_joint_score": sum(row["split_joint_score"] for row in rows if row["split_available"]) / max(sum(row["split_available"] for row in rows), 1),
        "topk_joint_score": sum(row["topk_joint_score"] for row in rows if row["topk_available"]) / max(sum(row["topk_available"] for row in rows), 1),
        "baseline_right_grouped_match": sum(row["baseline_right_grouped_match"] for row in rows if row["baseline_available"]) / max(sum(row["baseline_available"] for row in rows), 1),
        "split_right_grouped_match": sum(row["split_right_grouped_match"] for row in rows if row["split_available"]) / max(sum(row["split_available"] for row in rows), 1),
        "topk_right_grouped_match": sum(row["topk_right_grouped_match"] for row in rows if row["topk_available"]) / max(sum(row["topk_available"] for row in rows), 1),
        "baseline_left_preserve": sum(row["baseline_left_preserve"] for row in rows if row["baseline_available"]) / max(sum(row["baseline_available"] for row in rows), 1),
        "split_left_preserve": sum(row["split_left_preserve"] for row in rows if row["split_available"]) / max(sum(row["split_available"] for row in rows), 1),
        "topk_left_preserve": sum(row["topk_left_preserve"] for row in rows if row["topk_available"]) / max(sum(row["topk_available"] for row in rows), 1),
    }


def summarize_by_subtype(rows):
    by = defaultdict(list)
    for row in rows:
        by[(row["other_hand_motion"], row["interaction_motion_value"])].append(row)
    out = []
    for key, items in sorted(by.items()):
        other_hand_motion, interaction_motion_value = key
        def avg(field, avail):
            denom = max(sum(row[avail] for row in items), 1)
            return sum(row[field] for row in items if row[avail]) / denom
        out.append(
            {
                "other_hand_motion": other_hand_motion,
                "interaction_motion_value": interaction_motion_value,
                "num_frames": len(items),
                "baseline_available_rate": sum(row["baseline_available"] for row in items) / len(items),
                "split_available_rate": sum(row["split_available"] for row in items) / len(items),
                "topk_available_rate": sum(row["topk_available"] for row in items) / len(items),
                "baseline_joint_score": avg("baseline_joint_score", "baseline_available"),
                "split_joint_score": avg("split_joint_score", "split_available"),
                "topk_joint_score": avg("topk_joint_score", "topk_available"),
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
                left_pool = left_preserve_candidates(pair_bank, current_left_group, curr_attrs, prev_geom)

                baseline = right_pool[0] if right_pool else None
                split_left = left_pool[0] if left_pool else None

                rec = {
                    "seq_name": sequence["seq_name"],
                    "frame_idx": curr_frame["frame_idx"],
                    "other_hand_motion": "none" if curr_frame.get("left") is None else str(curr_frame["left"].get("hand_motion", "unknown")),
                    "interaction_motion_value": str(curr_frame.get("interaction_motion", "unknown")),
                    "baseline_available": int(baseline is not None),
                    "split_available": int(baseline is not None and split_left is not None),
                    "topk_available": int(bool(right_pool) and bool(left_pool)),
                }

                if baseline is not None:
                    res = evaluate_joint(prev_frame, curr_frame, prev_geom, baseline, None)
                    rec["baseline_right_grouped_match"] = res["right_grouped_match"]
                    rec["baseline_left_preserve"] = res["left_preserve"]
                    rec["baseline_joint_score"] = res["joint_score"]
                else:
                    rec["baseline_right_grouped_match"] = 0
                    rec["baseline_left_preserve"] = 0
                    rec["baseline_joint_score"] = 0

                if baseline is not None and split_left is not None:
                    res = evaluate_joint(prev_frame, curr_frame, prev_geom, baseline, split_left)
                    rec["split_right_grouped_match"] = res["right_grouped_match"]
                    rec["split_left_preserve"] = res["left_preserve"]
                    rec["split_joint_score"] = res["joint_score"]
                else:
                    rec["split_right_grouped_match"] = 0
                    rec["split_left_preserve"] = 0
                    rec["split_joint_score"] = 0

                if right_pool and left_pool:
                    best = None
                    for r in right_pool[:TOPK_RIGHT]:
                        for l in left_pool[:TOPK_LEFT]:
                            res = evaluate_joint(prev_frame, curr_frame, prev_geom, r, l)
                            key = (
                                res["joint_score"],
                                res["left_preserve"],
                                res["right_grouped_match"],
                            )
                            if best is None or key > best[0]:
                                best = (key, res)
                    assert best is not None
                    res = best[1]
                    rec["topk_right_grouped_match"] = res["right_grouped_match"]
                    rec["topk_left_preserve"] = res["left_preserve"]
                    rec["topk_joint_score"] = res["joint_score"]
                else:
                    rec["topk_right_grouped_match"] = 0
                    rec["topk_left_preserve"] = 0
                    rec["topk_joint_score"] = 0

                rows.append(rec)

    payload = {
        "artifacts": {
            "train_json": str(GEN / "temporal_hl_val.json"),
            "test_json": str(GEN / "temporal_hl_test.json"),
        },
        "focus": {
            "task": f"{TASK_FIELD}->{TASK_TARGET}",
            "slice": "interaction only",
            "prototype": f"topk joint search ({TOPK_RIGHT}x{TOPK_LEFT})",
        },
        "summary": summarize(rows),
        "subtype_summary": summarize_by_subtype(rows),
        "rows": rows,
    }

    out_json = GEN / "weak_slice_topk_joint_search.json"
    out_md = SUM / "weak_slice_topk_joint_search.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Weak Slice Top-k Joint Search",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Focus: `right_hand_motion -> opening` on interaction frames only.",
        "",
        f"Top-k search over {TOPK_RIGHT} right donors x {TOPK_LEFT} left donors.",
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
            "| other hand motion | interaction motion | frames | baseline avail | split avail | topk avail | baseline joint | split joint | topk joint |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["subtype_summary"]:
        lines.append(
            f"| {row['other_hand_motion']} | {row['interaction_motion_value']} | {row['num_frames']} | "
            f"{fmt(row['baseline_available_rate'])} | {fmt(row['split_available_rate'])} | {fmt(row['topk_available_rate'])} | "
            f"{fmt(row['baseline_joint_score'])} | {fmt(row['split_joint_score'])} | {fmt(row['topk_joint_score'])} |"
        )
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
