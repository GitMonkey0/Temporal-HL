#!/usr/bin/env python3
"""Realized interaction editor using split-donor and pair-guided selection.

This script targets the main remaining weakness:

- right-hand hand-motion edits on interaction frames

It upgrades the realized editor from target-hand-only donor composition to a
split-donor interaction-aware editor:

- choose a right-hand target donor that realizes the requested right-hand motion
- choose a left-hand preserve donor from a relaxed grouped family pool
- compare:
  - single-donor target-only composition
  - base split-donor search over top-k preserve donors
  - pair-guided split-donor reranking over the same candidate pool

The evaluator reports realized grouped-motif fidelity of the edited right hand
and grouped preservation of the opposite hand.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from tools.build_transition_conditioned_symbolic_editor import (
    GEN,
    SUM,
    build_pair_bank,
    build_semantic_frame_vocab,
    canonical,
    compose_target_hand_transition,
    fmt,
    grouped_motif_signature,
    load_json,
    overlap_labels,
    pick_best_symbolic_pair,
    summarize_composed_motif,
)
from tools.build_geometry_locality_audit import frame_geom
from tools.build_local_edit_audit import contiguous_runs, eligible_value, frame_attrs
from tools.build_pairguided_reranker_multislice import (
    INTERACTION_VALUES,
    MOTION_VALUES,
    build_examples,
    candidate_pool_for_task,
    collect_slice_frames,
    one_hot,
    opposite_hand_name,
    pair_feature_vector,
    relaxed_left_family_candidates_with_meta,
    target_right_features,
)
from tools.build_weak_slice_joint_editor_prototype import current_grouped_signature
from tools.build_weak_slice_topk_joint_search import evaluate_joint


ROOT = Path("/opt/tiger/hand")
TASKS = [
    ("right_hand_motion", "closing"),
    ("right_hand_motion", "opening"),
]
DEPTHS = (5, 10)


def compose_left_preserve_transition(prev_geom, donor_row):
    prev_left = prev_geom["left"]
    donor_prev = donor_row["prev_geom"]["left"]
    donor_curr = donor_row["curr_geom"]["left"]
    if prev_left is None or donor_prev is None or donor_curr is None:
        return None
    delta_local = np.asarray(donor_curr["local_vectors"], dtype=np.float32) - np.asarray(donor_prev["local_vectors"], dtype=np.float32)
    delta_flex = np.asarray(donor_curr["flexion"], dtype=np.float32) - np.asarray(donor_prev["flexion"], dtype=np.float32)
    return {
        "local_vectors": np.asarray(prev_left["local_vectors"], dtype=np.float32) + delta_local,
        "flexion": np.asarray(prev_left["flexion"], dtype=np.float32) + delta_flex,
    }


def compose_split(prev_geom, right_row, left_row):
    return {
        "right": compose_target_hand_transition(prev_geom, right_row, "right_hand_motion"),
        "left": compose_left_preserve_transition(prev_geom, left_row),
    }


def summarize_method(rows, prefix: str):
    avail_key = f"{prefix}_available"
    right_group_key = f"{prefix}_right_grouped_match"
    left_group_key = f"{prefix}_left_preserve"
    joint_key = f"{prefix}_joint_score"
    right_state_key = f"{prefix}_right_state_agreement"
    right_trans_key = f"{prefix}_right_transition_agreement"
    n = len(rows)
    avail = sum(row[avail_key] for row in rows)
    return {
        "num_frames": n,
        "available_rate": avail / n,
        "right_grouped_match_on_available": sum(row[right_group_key] for row in rows if row[avail_key]) / max(avail, 1),
        "left_preserve_on_available": sum(row[left_group_key] for row in rows if row[avail_key]) / max(avail, 1),
        "joint_score_on_available": sum(row[joint_key] for row in rows if row[avail_key]) / max(avail, 1),
        "right_state_agreement_on_available": sum(row[right_state_key] for row in rows if row[avail_key]) / max(avail, 1),
        "right_transition_agreement_on_available": sum(row[right_trans_key] for row in rows if row[avail_key]) / max(avail, 1),
        "joint_score_overall": sum(row[joint_key] for row in rows) / n,
    }


def summarize_by_subtype(rows, prefixes):
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
        for prefix in prefixes:
            summary = summarize_method(items, prefix)
            rec[f"{prefix}_available_rate"] = summary["available_rate"]
            rec[f"{prefix}_joint_score_on_available"] = summary["joint_score_on_available"]
            rec[f"{prefix}_joint_score_overall"] = summary["joint_score_overall"]
            rec[f"{prefix}_right_grouped_match_on_available"] = summary["right_grouped_match_on_available"]
            rec[f"{prefix}_left_preserve_on_available"] = summary["left_preserve_on_available"]
        out.append(rec)
    return out


def train_pairguided_model(train_frames, pair_bank, task_target: str):
    x_train, y_train, meta = build_examples(train_frames, pair_bank, "right_hand_motion", task_target)
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    pos_weight = 1.0 if pos == 0 else max(1.0, neg / max(pos, 1))
    sample_weight = np.where(y_train == 1, pos_weight, 1.0).astype(np.float32)

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=5,
        max_iter=250,
        min_samples_leaf=30,
        random_state=0,
    )
    model.fit(x_train, y_train, sample_weight=sample_weight)
    return model, {
        "num_examples": int(len(y_train)),
        "num_positive": pos,
        "num_negative": neg,
        "positive_weight": float(pos_weight),
        **{k: int(v) for k, v in meta.items()},
    }


def select_pairguided_left_pool(model, pair_bank, curr_attrs, prev_geom, curr_geom, current_left_group, target_pool):
    opp_pool = relaxed_left_family_candidates_with_meta(
        pair_bank,
        current_left_group,
        curr_attrs,
        prev_geom,
        curr_geom,
        opposite_hand_name("right_hand_motion"),
        1,
    )
    if not target_pool or not opp_pool:
        return []
    opp_pool_size = len(opp_pool)
    target_pool_size = len(target_pool)
    target_meta = {id(row): target_right_features(row, "right_hand_motion", prev_geom, curr_geom) for row in target_pool}
    scored_opp = []
    for opp_item in opp_pool:
        feats = np.asarray(
            [pair_feature_vector(opp_item, target_meta[id(target_row)], opp_pool_size, target_pool_size) for target_row in target_pool],
            dtype=np.float32,
        )
        score = float(model.predict_proba(feats)[:, 1].max())
        scored_opp.append((score, opp_item))
    return [item for _, item in sorted(scored_opp, key=lambda pair: pair[0], reverse=True)]


def choose_best_split(prev_frame, curr_frame, prev_geom, target_pool, left_pool, depth: int):
    best = None
    for right_row in target_pool:
        for left_item in left_pool[:depth]:
            res = evaluate_joint(prev_frame, curr_frame, prev_geom, right_row, left_item["row"])
            key = (res["joint_score"], res["left_preserve"], res["right_grouped_match"])
            if best is None or key > best[0]:
                best = (key, right_row, left_item["row"], res)
    return best


def evaluate_selected_edit(prev_frame, curr_frame, prev_geom, right_row=None, left_row=None):
    if right_row is None:
        return {
            "available": 0,
            "right_grouped_match": 0,
            "left_preserve": 0,
            "joint_score": 0,
            "right_state_agreement": 0.0,
            "right_transition_agreement": 0.0,
        }

    if left_row is None:
        edited_right = compose_target_hand_transition(prev_geom, right_row, "right_hand_motion")
        edited_left = prev_geom["left"]
    else:
        split = compose_split(prev_geom, right_row, left_row)
        edited_right = split["right"]
        edited_left = split["left"]

    right_motif = None if edited_right is None else summarize_composed_motif(prev_frame, edited_right, right_row, "right_hand_motion")
    donor_grouped = "unknown"
    edited_right_group = "unknown"
    if right_motif is not None:
        donor_grouped = grouped_motif_signature(str(right_motif["donor_hand_motion"]), list(right_motif["donor_transition_labels"]))
        edited_right_group = grouped_motif_signature(str(right_motif["edited_hand_motion"]), list(right_motif["edited_transition_labels"]))

    if left_row is None:
        left_ref = curr_frame.get("left")
    else:
        left_ref = left_row["curr_frame"].get("left")
    left_motif = None if edited_left is None else summarize_composed_motif(prev_frame, edited_left, {"curr_frame": {"left": left_ref}}, "left_hand_motion")
    current_left_group = current_grouped_signature(curr_frame.get("left"))
    edited_left_group = "unknown"
    if left_motif is not None:
        edited_left_group = grouped_motif_signature(str(left_motif["edited_hand_motion"]), list(left_motif["edited_transition_labels"]))

    right_match = int(right_motif is not None and edited_right_group == donor_grouped)
    left_preserve = int(left_motif is not None and edited_left_group == current_left_group)
    return {
        "available": 1,
        "right_grouped_match": right_match,
        "left_preserve": left_preserve,
        "joint_score": right_match * left_preserve,
        "right_state_agreement": 0.0 if right_motif is None else float(right_motif["state_agreement"]),
        "right_transition_agreement": 0.0 if right_motif is None else float(right_motif["transition_agreement"]),
    }


def run_task(train_frames, test_frames, pair_bank, task_target: str):
    model, train_stats = train_pairguided_model(train_frames, pair_bank, task_target)
    rows = []
    for entry in test_frames:
        prev_frame = entry["prev_frame"]
        curr_frame = entry["curr_frame"]
        prev_geom = entry["prev_geom"]
        curr_geom = entry["curr_geom"]
        curr_attrs = entry["curr_attrs"]
        current_left_group = entry["current_opp_group"]

        single = pick_best_symbolic_pair(pair_bank, curr_attrs, prev_geom, curr_geom, "right_hand_motion", task_target)
        target_pool = candidate_pool_for_task(pair_bank, "right_hand_motion", task_target, curr_attrs, prev_geom, curr_geom)
        base_left_pool = relaxed_left_family_candidates_with_meta(pair_bank, current_left_group, curr_attrs, prev_geom, curr_geom, "left", 1)
        pg_left_pool = select_pairguided_left_pool(model, pair_bank, curr_attrs, prev_geom, curr_geom, current_left_group, target_pool)

        rec = {
            "seq_name": entry["seq_name"],
            "frame_idx": curr_frame["frame_idx"],
            "other_hand_motion": "none" if curr_frame.get("left") is None else str(curr_frame["left"].get("hand_motion", "unknown")),
            "interaction_motion_value": str(curr_frame.get("interaction_motion", "unknown")),
        }

        single_eval = evaluate_selected_edit(prev_frame, curr_frame, prev_geom, single, None)
        rec.update({f"single_{k}": v for k, v in single_eval.items()})

        for depth in DEPTHS:
            base_choice = choose_best_split(prev_frame, curr_frame, prev_geom, target_pool, base_left_pool, depth) if target_pool and base_left_pool else None
            base_eval = evaluate_selected_edit(
                prev_frame,
                curr_frame,
                prev_geom,
                None if base_choice is None else base_choice[1],
                None if base_choice is None else base_choice[2],
            )
            rec.update({f"base_top{depth}_{k}": v for k, v in base_eval.items()})

            pg_choice = choose_best_split(prev_frame, curr_frame, prev_geom, target_pool, pg_left_pool, depth) if target_pool and pg_left_pool else None
            pg_eval = evaluate_selected_edit(
                prev_frame,
                curr_frame,
                prev_geom,
                None if pg_choice is None else pg_choice[1],
                None if pg_choice is None else pg_choice[2],
            )
            rec.update({f"pairguided_top{depth}_{k}": v for k, v in pg_eval.items()})

        rows.append(rec)

    prefixes = ["single"] + [f"base_top{d}" for d in DEPTHS] + [f"pairguided_top{d}" for d in DEPTHS]
    return {
        "training_stats": train_stats,
        "summary": {prefix: summarize_method(rows, prefix) for prefix in prefixes},
        "subtype_summary": summarize_by_subtype(rows, prefixes),
        "rows": rows,
    }


def main():
    train_data = load_json(GEN / "temporal_hl_val.json")
    test_data = load_json(GEN / "temporal_hl_test.json")
    labels = overlap_labels(train_data, test_data)
    semantic_vocab = build_semantic_frame_vocab(train_data, labels)
    pair_bank = build_pair_bank(train_data, labels, semantic_vocab)

    task_results = {}
    for task_field, task_target in TASKS:
        train_frames = [
            row for row in collect_slice_frames(train_data, task_field, task_target)
            if canonical(row["seq_name"]) in labels or row["seq_name"] in labels
        ]
        test_frames = [
            row for row in collect_slice_frames(test_data, task_field, task_target)
            if canonical(row["seq_name"]) in labels or row["seq_name"] in labels
        ]
        task_results[f"{task_field}->{task_target}"] = run_task(train_frames, test_frames, pair_bank, task_target)

    payload = {
        "artifacts": {
            "train_json": str(GEN / "temporal_hl_val.json"),
            "test_json": str(GEN / "temporal_hl_test.json"),
        },
        "focus": {
            "tasks": [f"{field}->{target}" for field, target in TASKS],
            "slice": "interaction only",
            "goal": "test whether stronger split-donor selection improves realized interaction editing on hard right-hand slices",
        },
        "task_results": task_results,
    }

    out_json = GEN / "interaction_realized_pairguided_editor.json"
    out_md = SUM / "interaction_realized_pairguided_editor.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction Realized Pair-Guided Editor",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Focus: right-hand hand-motion edits on interaction frames only.",
        "",
        "Methods compared:",
        "",
        "- `single`: target-hand-only symbolic donor composition",
        "- `base_topk`: split-donor joint search with unlearned preserve-donor ordering",
        "- `pairguided_topk`: split-donor joint search with learned pair-guided preserve ordering",
        "",
    ]

    for task_name, result in task_results.items():
        lines.extend(
            [
                f"## {task_name}",
                "",
                "### Training Stats",
                "",
                "| metric | value |",
                "| --- | ---: |",
            ]
        )
        for key, value in result["training_stats"].items():
            lines.append(f"| {key} | {value} |")
        lines.extend(
            [
                "",
                "### Method Summary",
                "",
                "| method | avail | right grouped | left preserve | joint on avail | joint overall | right state | right transition |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for method, stats in result["summary"].items():
            lines.append(
                f"| {method} | {fmt(stats['available_rate'])} | {fmt(stats['right_grouped_match_on_available'])} | "
                f"{fmt(stats['left_preserve_on_available'])} | {fmt(stats['joint_score_on_available'])} | {fmt(stats['joint_score_overall'])} | "
                f"{fmt(stats['right_state_agreement_on_available'])} | {fmt(stats['right_transition_agreement_on_available'])} |"
            )

        lines.extend(
            [
                "",
                "### By Subtype",
                "",
                "| other hand motion | interaction motion | frames | single overall | base top-5 overall | pair-guided top-5 overall | base top-10 overall | pair-guided top-10 overall |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in result["subtype_summary"]:
            lines.append(
                f"| {row['other_hand_motion']} | {row['interaction_motion_value']} | {row['num_frames']} | "
                f"{fmt(row['single_joint_score_overall'])} | {fmt(row['base_top5_joint_score_overall'])} | "
                f"{fmt(row['pairguided_top5_joint_score_overall'])} | {fmt(row['base_top10_joint_score_overall'])} | "
                f"{fmt(row['pairguided_top10_joint_score_overall'])} |"
            )
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
