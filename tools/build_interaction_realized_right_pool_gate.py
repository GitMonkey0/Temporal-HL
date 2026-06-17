#!/usr/bin/env python3
"""Learn when to relax the right-hand donor pool on hard interaction slices.

This is an experiment memo, not paper text.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from tools.build_interaction_realized_constraint_sweep import evaluate_edit, fmt
from tools.build_interaction_realized_pairguided_editor import (
    TASKS,
    choose_best_split,
    select_pairguided_left_pool,
    train_pairguided_model,
)
from tools.build_pairguided_reranker_multislice import (
    collect_slice_frames,
    relaxed_left_family_candidates_with_meta,
)
from tools.build_transition_conditioned_symbolic_editor import (
    GEN,
    SUM,
    build_pair_bank,
    build_semantic_frame_vocab,
    canonical,
    load_json,
    overlap_labels,
)
from tools.build_interaction_realized_right_support_sweep import right_candidates_mode, reorder_to_budget


FAMILY_BUDGET = 2
DEPTH = 10
RIGHT_POOL_CAP = 20

MOTION_VALUES = ["none", "start", "opening", "closing", "mixed", "steady", "unknown"]
INTERACTION_VALUES = ["approach", "separate", "steady", "unknown"]


def one_hot(value: str, vocab: list[str]) -> list[float]:
    return [1.0 if value == item else 0.0 for item in vocab]


def summarize_method(rows, prefix: str):
    avail_key = f"{prefix}_available"
    n = len(rows)
    avail = sum(row[avail_key] for row in rows)
    out = {"num_frames": n, "available_rate": avail / n}
    for key in ("right_grouped_match", "left_preserve", "joint_score"):
        out[f"{key}_overall"] = sum(row[f"{prefix}_{key}"] for row in rows) / n
        out[f"{key}_on_available"] = sum(row[f"{prefix}_{key}"] for row in rows if row[avail_key]) / max(avail, 1)
    return out


def summarize_by_sequence(rows, prefixes):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["seq_name"]].append(row)
    out = []
    for seq_name, items in sorted(grouped.items()):
        rec = {"seq_name": seq_name, "num_frames": len(items)}
        for prefix in prefixes:
            summary = summarize_method(items, prefix)
            rec[f"{prefix}_joint_score_overall"] = summary["joint_score_overall"]
        out.append(rec)
    return out


def top_distance(pool, prev_geom, curr_geom):
    if not pool:
        return 1e6
    from tools.build_transition_conditioned_symbolic_editor import pair_distance
    return float(pair_distance(prev_geom, curr_geom, pool[0], "right_hand_motion"))


def feature_vector(entry, strict_pool, relax_pool):
    curr_attrs = entry["curr_attrs"]
    feats = [
        float(len(strict_pool)),
        float(len(relax_pool)),
        top_distance(strict_pool, entry["prev_geom"], entry["curr_geom"]),
        top_distance(relax_pool, entry["prev_geom"], entry["curr_geom"]),
        float(max(0, len(relax_pool) - len(strict_pool))),
    ]
    feats.extend(one_hot(str(curr_attrs.get("left_hand_motion", "unknown")), MOTION_VALUES))
    feats.extend(one_hot(str(curr_attrs.get("interaction_motion", "unknown")), INTERACTION_VALUES))
    feats.extend(one_hot(str(curr_attrs.get("right_hand_motion", "unknown")), MOTION_VALUES))
    return np.asarray(feats, dtype=np.float32)


def build_example(entry, pair_bank, hgb_model, task_target: str):
    prev_frame = entry["prev_frame"]
    curr_frame = entry["curr_frame"]
    prev_geom = entry["prev_geom"]
    curr_geom = entry["curr_geom"]
    curr_attrs = entry["curr_attrs"]
    current_left_group = entry["current_opp_group"]

    strict_pool = right_candidates_mode(pair_bank, curr_attrs, prev_geom, curr_geom, task_target, "strict")[:RIGHT_POOL_CAP]
    relax_pool = right_candidates_mode(pair_bank, curr_attrs, prev_geom, curr_geom, task_target, "relax_both")[:RIGHT_POOL_CAP]

    raw_left_pool = relaxed_left_family_candidates_with_meta(
        pair_bank, current_left_group, curr_attrs, prev_geom, curr_geom, "left", FAMILY_BUDGET
    )
    strict_ranked = select_pairguided_left_pool(hgb_model, pair_bank, curr_attrs, prev_geom, curr_geom, current_left_group, strict_pool)
    strict_left_pool = reorder_to_budget(strict_ranked, raw_left_pool)
    relax_ranked = select_pairguided_left_pool(hgb_model, pair_bank, curr_attrs, prev_geom, curr_geom, current_left_group, relax_pool)
    relax_left_pool = reorder_to_budget(relax_ranked, raw_left_pool)

    strict_choice = choose_best_split(prev_frame, curr_frame, prev_geom, strict_pool, strict_left_pool, DEPTH) if strict_pool and strict_left_pool else None
    relax_choice = choose_best_split(prev_frame, curr_frame, prev_geom, relax_pool, relax_left_pool, DEPTH) if relax_pool and relax_left_pool else None

    strict_eval = evaluate_edit(prev_frame, curr_frame, prev_geom, None if strict_choice is None else strict_choice[1], None if strict_choice is None else strict_choice[2], repair_mode="finger_profile_snap")
    relax_eval = evaluate_edit(prev_frame, curr_frame, prev_geom, None if relax_choice is None else relax_choice[1], None if relax_choice is None else relax_choice[2], repair_mode="finger_profile_snap")
    feat = feature_vector(entry, strict_pool, relax_pool)
    label = int(
        (relax_eval["joint_score"], relax_eval["right_grouped_match"], relax_eval["left_preserve"])
        > (strict_eval["joint_score"], strict_eval["right_grouped_match"], strict_eval["left_preserve"])
    )
    return feat, label, strict_eval, relax_eval


def train_gate(train_frames, pair_bank, task_target: str):
    hgb_model, hgb_stats = train_pairguided_model(train_frames, pair_bank, task_target)
    xs, ys = [], []
    for entry in train_frames:
        feat, label, _, _ = build_example(entry, pair_bank, hgb_model, task_target)
        xs.append(feat)
        ys.append(label)
    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.int32)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    pos_weight = 1.0 if pos == 0 else max(1.0, neg / max(pos, 1))
    sample_weight = np.where(y == 1, pos_weight, 1.0).astype(np.float32)
    gate = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=4,
        max_iter=200,
        min_samples_leaf=20,
        random_state=0,
    )
    gate.fit(x, y, sample_weight=sample_weight)
    return gate, hgb_model, {
        "num_examples": int(len(y)),
        "num_positive": pos,
        "num_negative": neg,
        "positive_weight": float(pos_weight),
        **hgb_stats,
    }


def run_task(train_frames, test_frames, pair_bank, task_target: str):
    gate, hgb_model, train_stats = train_gate(train_frames, pair_bank, task_target)
    rows = []
    for entry in test_frames:
        feat, _, strict_eval, relax_eval = build_example(entry, pair_bank, hgb_model, task_target)
        prob = float(gate.predict_proba(feat.reshape(1, -1))[0, 1])
        use_relax = int(prob >= 0.5)
        chosen = relax_eval if use_relax else strict_eval
        oracle = relax_eval if (relax_eval["joint_score"], relax_eval["right_grouped_match"], relax_eval["left_preserve"]) > (strict_eval["joint_score"], strict_eval["right_grouped_match"], strict_eval["left_preserve"]) else strict_eval
        rec = {
            "seq_name": entry["seq_name"],
            "frame_idx": entry["curr_frame"]["frame_idx"],
            "other_hand_motion": "none" if entry["curr_frame"].get("left") is None else str(entry["curr_frame"]["left"].get("hand_motion", "unknown")),
            "interaction_motion_value": str(entry["curr_frame"].get("interaction_motion", "unknown")),
            "gate_relax_prob": prob,
            "gate_use_relax": use_relax,
        }
        for prefix, obj in [("strict", strict_eval), ("relax_both", relax_eval), ("gate", chosen), ("oracle_best", oracle)]:
            rec.update({f"{prefix}_{k}": v for k, v in obj.items()})
        rows.append(rec)

    prefixes = ["strict", "relax_both", "gate", "oracle_best"]
    return {
        "training_stats": train_stats,
        "summary": {prefix: summarize_method(rows, prefix) for prefix in prefixes},
        "sequence_summary": summarize_by_sequence(rows, prefixes),
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
        train_frames = [row for row in collect_slice_frames(train_data, task_field, task_target) if canonical(row["seq_name"]) in labels or row["seq_name"] in labels]
        test_frames = [row for row in collect_slice_frames(test_data, task_field, task_target) if canonical(row["seq_name"]) in labels or row["seq_name"] in labels]
        task_results[f"{task_field}->{task_target}"] = run_task(train_frames, test_frames, pair_bank, task_target)

    payload = {
        "artifacts": {"train_json": str(GEN / "temporal_hl_val.json"), "test_json": str(GEN / "temporal_hl_test.json")},
        "focus": {
            "tasks": [f"{field}->{target}" for field, target in TASKS],
            "slice": "interaction only",
            "goal": "learn when to use strict vs relaxed right donor support",
            "family_budget": FAMILY_BUDGET,
            "depth": DEPTH,
            "right_pool_cap": RIGHT_POOL_CAP,
        },
        "task_results": task_results,
    }

    out_json = GEN / "interaction_realized_right_pool_gate.json"
    out_md = SUM / "interaction_realized_right_pool_gate.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction Realized Right Pool Gate",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Compare strict, relax-both, learned gate, and oracle-best on full hard slices.",
        "",
    ]
    for task_name, result in task_results.items():
        lines.extend([f"## {task_name}", "", "| method | avail | right grouped | left preserve | joint overall |", "| --- | ---: | ---: | ---: | ---: |"])
        for method in ["strict", "relax_both", "gate", "oracle_best"]:
            stats = result["summary"][method]
            lines.append(f"| {method} | {fmt(stats['available_rate'])} | {fmt(stats['right_grouped_match_overall'])} | {fmt(stats['left_preserve_overall'])} | {fmt(stats['joint_score_overall'])} |")
        lines.extend(["", "### Sequence Summary", "", "| sequence | strict | relax-both | gate | oracle-best |", "| --- | ---: | ---: | ---: | ---: |"])
        for row in result["sequence_summary"]:
            lines.append(
                f"| {row['seq_name']} | {fmt(row['strict_joint_score_overall'])} | {fmt(row['relax_both_joint_score_overall'])} | "
                f"{fmt(row['gate_joint_score_overall'])} | {fmt(row['oracle_best_joint_score_overall'])} |"
            )
        lines.append("")
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
