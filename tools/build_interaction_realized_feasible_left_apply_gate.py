#!/usr/bin/env python3
"""Binary apply gate for the best feasible left repair on closing.

This is an experiment memo, not paper text.

Target only the remaining meaningful case:

- task: `right_hand_motion->closing`
- feasible two-hand subset only
- fixed best left repair: `edge_transition_snap`

Learn when to apply that repair vs keep `none`.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from tools.build_interaction_realized_constraint_sweep import evaluate_edit
from tools.build_interaction_realized_pairguided_editor import (
    select_pairguided_left_pool,
    train_pairguided_model,
)
from tools.build_interaction_realized_right_support_sweep import (
    reorder_to_budget,
    right_candidates_mode,
)
from tools.build_pairguided_reranker_multislice import collect_slice_frames
from tools.build_transition_conditioned_symbolic_editor import (
    GEN,
    SUM,
    build_pair_bank,
    build_semantic_frame_vocab,
    canonical,
    load_json,
    overlap_labels,
)
from tools.build_pairguided_reranker_multislice import relaxed_left_family_candidates_with_meta
from tools.build_weak_slice_topk_joint_search import evaluate_joint


TASK_FIELD = "right_hand_motion"
TASK_TARGET = "closing"
BEST_MODE = "edge_transition_snap"
BASE_PREFIX = "left_none"
BEST_PREFIX = f"left_{BEST_MODE}"
OTHER_HAND_VALUES = ["none", "start", "opening", "closing", "mixed", "steady", "unknown"]
INTERACTION_VALUES = ["approach", "separate", "steady", "unknown"]


def fmt(x: float) -> str:
    return f"{x:.4f}"


def one_hot(value: str, vocab: list[str]) -> list[float]:
    return [1.0 if value == item else 0.0 for item in vocab]


def pick_best_from_cache(prev_frame, curr_frame, prev_geom, target_pool, left_pool, depth: int, joint_cache):
    best = None
    for right_row in target_pool:
        right_id = id(right_row)
        for left_item in left_pool[:depth]:
            left_row = left_item["row"]
            key = (right_id, id(left_row))
            if key not in joint_cache:
                joint_cache[key] = evaluate_joint(prev_frame, curr_frame, prev_geom, right_row, left_row)
            res = joint_cache[key]
            score = (res["joint_score"], res["left_preserve"], res["right_grouped_match"])
            if best is None or score > best[0]:
                best = (score, right_row, left_row)
    return best


def collect_rows(frames, pair_bank, pair_model):
    rows = []
    for entry in frames:
        if entry["curr_frame"].get("left") is None:
            continue
        prev_frame = entry["prev_frame"]
        curr_frame = entry["curr_frame"]
        prev_geom = entry["prev_geom"]
        curr_geom = entry["curr_geom"]
        curr_attrs = entry["curr_attrs"]
        current_left_group = entry["current_opp_group"]

        raw_left_pool = relaxed_left_family_candidates_with_meta(
            pair_bank,
            current_left_group,
            curr_attrs,
            prev_geom,
            curr_geom,
            "left",
            2,
        )
        target_pool = right_candidates_mode(pair_bank, curr_attrs, prev_geom, curr_geom, TASK_TARGET, "relax_both")
        ranked_left_pool = select_pairguided_left_pool(
            pair_model,
            pair_bank,
            curr_attrs,
            prev_geom,
            curr_geom,
            current_left_group,
            target_pool,
        )
        left_pool = reorder_to_budget(ranked_left_pool, raw_left_pool)
        choice = pick_best_from_cache(
            prev_frame,
            curr_frame,
            prev_geom,
            target_pool,
            left_pool,
            20,
            {},
        ) if target_pool and left_pool else None
        right_row = None if choice is None else choice[1]
        left_row = None if choice is None else choice[2]

        rec = {
            "seq_name": entry["seq_name"],
            "frame_idx": curr_frame["frame_idx"],
            "other_hand_motion": str(curr_frame["left"].get("hand_motion", "unknown")),
            "interaction_motion_value": str(curr_frame.get("interaction_motion", "unknown")),
        }
        for mode in ("none", BEST_MODE):
            prefix = f"left_{mode}"
            eval_rec = evaluate_edit(
                prev_frame,
                curr_frame,
                prev_geom,
                right_row=right_row,
                left_row=left_row,
                repair_mode=mode,
            )
            rec.update({f"{prefix}_{k}": v for k, v in eval_rec.items()})
        rows.append(rec)
    return rows


def feature_vector(row):
    feats = [
        float(row[f"{BASE_PREFIX}_left_state_agreement"]),
        float(row[f"{BASE_PREFIX}_left_transition_agreement"]),
        float(row[f"{BASE_PREFIX}_right_state_agreement"]),
        float(row[f"{BASE_PREFIX}_right_transition_agreement"]),
    ]
    feats.extend(one_hot(row["other_hand_motion"], OTHER_HAND_VALUES))
    feats.extend(one_hot(row["interaction_motion_value"], INTERACTION_VALUES))
    return np.asarray(feats, dtype=np.float32)


def train_gate(train_rows):
    xs, ys = [], []
    for row in train_rows:
        improve = int(float(row[f"{BEST_PREFIX}_joint_score"]) > float(row[f"{BASE_PREFIX}_joint_score"]))
        xs.append(feature_vector(row))
        ys.append(improve)
    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.int32)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    pos_weight = 1.0 if pos == 0 else max(1.0, neg / max(pos, 1))
    sample_weight = np.where(y == 1, pos_weight, 1.0).astype(np.float32)
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=4,
        max_iter=200,
        min_samples_leaf=10,
        random_state=0,
    )
    model.fit(x, y, sample_weight=sample_weight)
    return model, {
        "num_examples": int(len(y)),
        "num_positive": pos,
        "num_negative": neg,
        "positive_weight": float(pos_weight),
    }


def summarize_method(rows, prefix: str):
    n = len(rows)
    return {
        "num_frames": n,
        "right_grouped_match_overall": sum(row[f"{prefix}_right_grouped_match"] for row in rows) / n,
        "left_preserve_overall": sum(row[f"{prefix}_left_preserve"] for row in rows) / n,
        "joint_score_overall": sum(row[f"{prefix}_joint_score"] for row in rows) / n,
    }


def paired_stats(rows, a_prefix: str, b_prefix: str):
    a = np.asarray([float(r[f"{a_prefix}_joint_score"]) for r in rows], dtype=np.float32)
    b = np.asarray([float(r[f"{b_prefix}_joint_score"]) for r in rows], dtype=np.float32)
    diff = b - a
    return {
        "delta": float(diff.mean()),
        "wins": int((diff > 0).sum()),
        "losses": int((diff < 0).sum()),
        "ties": int((diff == 0).sum()),
    }


def main():
    train_data = load_json(GEN / "temporal_hl_val.json")
    test_data = load_json(GEN / "temporal_hl_test.json")
    labels = overlap_labels(train_data, test_data)
    semantic_vocab = build_semantic_frame_vocab(train_data, labels)
    pair_bank = build_pair_bank(train_data, labels, semantic_vocab)

    train_frames = [
        row for row in collect_slice_frames(train_data, TASK_FIELD, TASK_TARGET)
        if (canonical(row["seq_name"]) in labels or row["seq_name"] in labels) and row["curr_frame"].get("left") is not None
    ]
    test_frames = [
        row for row in collect_slice_frames(test_data, TASK_FIELD, TASK_TARGET)
        if (canonical(row["seq_name"]) in labels or row["seq_name"] in labels) and row["curr_frame"].get("left") is not None
    ]

    pair_model, pair_stats = train_pairguided_model(train_frames, pair_bank, TASK_TARGET)
    train_rows = collect_rows(train_frames, pair_bank, pair_model)
    gate_model, gate_stats = train_gate(train_rows)
    test_rows = collect_rows(test_frames, pair_bank, pair_model)

    for row in test_rows:
        use_best = int(gate_model.predict(feature_vector(row).reshape(1, -1))[0])
        chosen = BEST_PREFIX if use_best else BASE_PREFIX
        for name, source in [
            ("fixed_none", BASE_PREFIX),
            ("fixed_best", BEST_PREFIX),
            ("gate", chosen),
        ]:
            for field in ("right_grouped_match", "left_preserve", "joint_score"):
                row[f"{name}_{field}"] = row[f"{source}_{field}"]
        best_val = max(float(row[f"{BASE_PREFIX}_joint_score"]), float(row[f"{BEST_PREFIX}_joint_score"]))
        oracle_source = BEST_PREFIX if float(row[f"{BEST_PREFIX}_joint_score"]) >= float(row[f"{BASE_PREFIX}_joint_score"]) else BASE_PREFIX
        for field in ("right_grouped_match", "left_preserve", "joint_score"):
            row[f"oracle_{field}"] = row[f"{oracle_source}_{field}"]
        row["gate_use_best"] = use_best
        row["oracle_source"] = oracle_source

    summary = {name: summarize_method(test_rows, name) for name in ("fixed_none", "fixed_best", "gate", "oracle")}
    paired = {
        "gate_vs_fixed_best": paired_stats(test_rows, "fixed_best", "gate"),
        "oracle_vs_fixed_best": paired_stats(test_rows, "fixed_best", "oracle"),
        "gate_vs_fixed_none": paired_stats(test_rows, "fixed_none", "gate"),
    }

    payload = {
        "artifacts": {"train_json": str(GEN / "temporal_hl_val.json"), "test_json": str(GEN / "temporal_hl_test.json")},
        "focus": {
            "goal": "binary apply gate for the best feasible left repair on closing",
            "task": f"{TASK_FIELD}->{TASK_TARGET}",
            "best_mode": BEST_MODE,
        },
        "training_stats": {
            "pair_model": pair_stats,
            "gate_model": gate_stats,
        },
        "summary": summary,
        "paired": paired,
        "rows": test_rows,
    }

    out_json = GEN / "interaction_realized_feasible_left_apply_gate.json"
    out_md = SUM / "interaction_realized_feasible_left_apply_gate.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction Realized Feasible Left Apply Gate",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Binary apply gate for the best feasible left repair on closing.",
        "",
        "| method | right grouped | left preserve | joint overall |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method in ("fixed_none", "fixed_best", "gate", "oracle"):
        s = summary[method]
        lines.append(
            f"| {method} | {fmt(s['right_grouped_match_overall'])} | {fmt(s['left_preserve_overall'])} | {fmt(s['joint_score_overall'])} |"
        )
    lines.extend([
        "",
        "| comparison | delta | wins | losses | ties |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for key, stats in paired.items():
        lines.append(
            f"| {key} | {fmt(stats['delta'])} | {stats['wins']} | {stats['losses']} | {stats['ties']} |"
        )
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
