#!/usr/bin/env python3
"""Adaptive support policies on the weak interaction slice.

This script evaluates whether multi-stage / gated left-support expansion can
improve coverage without paying the full quality cost of uniform relaxation.
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
from tools.build_weak_slice_relaxed_support_topk import (
    left_family_candidates,
    pick_topk_joint,
)
from tools.build_weak_slice_split_donor_prototype import (
    left_preserve_candidates,
    right_relaxed_candidates,
)


THRESHOLDS = (1, 2, 3, 5, 10)
MOTION_GATES = {
    "approach_or_separate": {"approach", "separate"},
    "approach_only": {"approach"},
    "separate_only": {"separate"},
}


def summarize_policy(rows, prefix: str):
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


def summarize_subtypes(rows, prefix: str):
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


def run_policy(rows, name, use_relaxed_fn):
    for row in rows:
        use_relaxed = bool(use_relaxed_fn(row))
        chosen = "relaxed" if use_relaxed and row["relaxed_available"] else "exact"
        if chosen == "exact" and not row["exact_available"] and row["relaxed_available"] and use_relaxed:
            chosen = "relaxed"
        if chosen == "exact":
            row[f"{name}_available"] = row["exact_available"]
            row[f"{name}_joint_score"] = row["exact_joint_score"]
            row[f"{name}_right_grouped_match"] = row["exact_right_grouped_match"]
            row[f"{name}_left_preserve"] = row["exact_left_preserve"]
            row[f"{name}_source"] = "exact"
        else:
            row[f"{name}_available"] = row["relaxed_available"]
            row[f"{name}_joint_score"] = row["relaxed_joint_score"]
            row[f"{name}_right_grouped_match"] = row["relaxed_right_grouped_match"]
            row[f"{name}_left_preserve"] = row["relaxed_left_preserve"]
            row[f"{name}_source"] = "relaxed"


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

                exact = pick_topk_joint(prev_frame, curr_frame, prev_geom, right_pool, exact_left_pool) if right_pool and exact_left_pool else None
                relaxed = pick_topk_joint(prev_frame, curr_frame, prev_geom, right_pool, relaxed_left_pool) if right_pool and relaxed_left_pool else None

                row = {
                    "seq_name": sequence["seq_name"],
                    "frame_idx": curr_frame["frame_idx"],
                    "other_hand_motion": "none" if curr_frame.get("left") is None else str(curr_frame["left"].get("hand_motion", "unknown")),
                    "interaction_motion_value": str(curr_frame.get("interaction_motion", "unknown")),
                    "right_pool_size": len(right_pool),
                    "exact_left_pool_size": len(exact_left_pool),
                    "relaxed_left_pool_size": len(relaxed_left_pool),
                    "exact_available": int(exact is not None),
                    "relaxed_available": int(relaxed is not None),
                    "exact_joint_score": 0 if exact is None else exact["joint_score"],
                    "exact_right_grouped_match": 0 if exact is None else exact["right_grouped_match"],
                    "exact_left_preserve": 0 if exact is None else exact["left_preserve"],
                    "relaxed_joint_score": 0 if relaxed is None else relaxed["joint_score"],
                    "relaxed_right_grouped_match": 0 if relaxed is None else relaxed["right_grouped_match"],
                    "relaxed_left_preserve": 0 if relaxed is None else relaxed["left_preserve"],
                }
                rows.append(row)

    policy_names = []

    run_policy(rows, "exact_policy", lambda row: False)
    policy_names.append("exact_policy")

    run_policy(rows, "uniform_relaxed_policy", lambda row: True)
    policy_names.append("uniform_relaxed_policy")

    run_policy(rows, "fallback_empty_policy", lambda row: row["exact_left_pool_size"] == 0)
    policy_names.append("fallback_empty_policy")

    for threshold in THRESHOLDS:
        name = f"threshold_lt_{threshold}_policy"
        run_policy(rows, name, lambda row, t=threshold: row["exact_left_pool_size"] < t)
        policy_names.append(name)

    for gate_name, gate_values in MOTION_GATES.items():
        name = f"{gate_name}_fallback_empty_policy"
        run_policy(
            rows,
            name,
            lambda row, g=gate_values: row["exact_left_pool_size"] == 0 and row["interaction_motion_value"] in g,
        )
        policy_names.append(name)

        for threshold in (2, 3, 5):
            name = f"{gate_name}_threshold_lt_{threshold}_policy"
            run_policy(
                rows,
                name,
                lambda row, g=gate_values, t=threshold: row["interaction_motion_value"] in g and row["exact_left_pool_size"] < t,
            )
            policy_names.append(name)

    policy_summary = {name: summarize_policy(rows, name) for name in policy_names}

    ranked = sorted(
        (
            {
                "policy": name,
                **policy_summary[name],
            }
            for name in policy_names
        ),
        key=lambda row: (
            row["joint_hit_rate_overall"],
            row["joint_score_on_available"],
            row["available_rate"],
        ),
        reverse=True,
    )

    focus_policies = [
        "exact_policy",
        "uniform_relaxed_policy",
        "fallback_empty_policy",
        "threshold_lt_2_policy",
        "threshold_lt_3_policy",
        "threshold_lt_5_policy",
        "approach_or_separate_threshold_lt_3_policy",
    ]
    subtype_focus = {
        name: summarize_subtypes(rows, name)
        for name in focus_policies
        if name in policy_summary
    }

    payload = {
        "artifacts": {
            "train_json": str(GEN / "temporal_hl_val.json"),
            "test_json": str(GEN / "temporal_hl_test.json"),
        },
        "focus": {
            "task": f"{TASK_FIELD}->{TASK_TARGET}",
            "slice": "interaction only",
            "prototype": "adaptive left support + topk joint search",
            "thresholds": list(THRESHOLDS),
            "motion_gates": {key: sorted(value) for key, value in MOTION_GATES.items()},
        },
        "policy_summary": policy_summary,
        "ranked_policies": ranked,
        "focus_policy_subtypes": subtype_focus,
        "rows": rows,
    }

    out_json = GEN / "weak_slice_adaptive_support_topk.json"
    out_md = SUM / "weak_slice_adaptive_support_topk.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Weak Slice Adaptive-Support Top-k Search",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Focus: `right_hand_motion -> opening` on interaction frames only.",
        "",
        "Adaptive policies compare exact left grouped-motif support against",
        "multi-stage fallbacks to family-level relaxed support.",
        "",
        "## Policy Ranking",
        "",
        "| policy | avail | joint on avail | right match | left preserve | joint overall |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ranked:
        lines.append(
            f"| {row['policy']} | {fmt(row['available_rate'])} | "
            f"{fmt(row['joint_score_on_available'])} | "
            f"{fmt(row['right_grouped_match_on_available'])} | "
            f"{fmt(row['left_preserve_on_available'])} | "
            f"{fmt(row['joint_hit_rate_overall'])} |"
        )

    lines.extend(
        [
            "",
            "## Focus Policies",
            "",
            "| policy | avail | joint on avail | joint overall |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name in focus_policies:
        if name not in policy_summary:
            continue
        row = policy_summary[name]
        lines.append(
            f"| {name} | {fmt(row['available_rate'])} | "
            f"{fmt(row['joint_score_on_available'])} | "
            f"{fmt(row['joint_hit_rate_overall'])} |"
        )

    for name in focus_policies:
        if name not in subtype_focus:
            continue
        lines.extend(
            [
                "",
                f"## By Subtype: {name}",
                "",
                "| other hand motion | interaction motion | frames | avail | joint on avail | joint overall |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in subtype_focus[name]:
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
