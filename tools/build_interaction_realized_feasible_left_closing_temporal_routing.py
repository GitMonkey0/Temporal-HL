#!/usr/bin/env python3
"""Closing-specific temporal routing for feasible left repair.

This is an experiment memo, not paper text.

Target the remaining hard case directly:

- task fixed to `right_hand_motion->closing`
- base strong policy is fixed `edge_transition_snap`
- test two temporal routing families using one-hop context:
  - `edge_vs_none`: abstain from edge only when temporal evidence says so
  - `edge_vs_finger`: switch repair family when temporal evidence says so
"""

from __future__ import annotations

import json
from collections import Counter

from tools.build_interaction_realized_feasible_left_temporal_window_knn import (
    attach_context,
    build_frames,
    distance,
)
from tools.build_interaction_realized_feasible_left_repair_model_gate import collect_rows
from tools.build_interaction_realized_pairguided_editor import train_pairguided_model
from tools.build_transition_conditioned_symbolic_editor import (
    GEN,
    SUM,
    build_pair_bank,
    build_semantic_frame_vocab,
    load_json,
    overlap_labels,
)


CFG = {"k": 3, "subtype_weight": 0.25, "agreement_weight": 0.5, "epsilon": 0.05}
THRESHOLDS = [-1.0, -0.75, -0.5, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]


def fmt(x: float) -> str:
    return f"{x:.4f}"


def knn_delta(train_rows, row, left_mode: str, right_mode: str):
    neighbors = sorted(((distance(train_row, row, CFG), train_row) for train_row in train_rows), key=lambda x: x[0])[: CFG["k"]]
    weighted_delta = 0.0
    total_weight = 0.0
    for dist, train_row in neighbors:
        weight = 1.0 / (CFG["epsilon"] + dist)
        weighted_delta += weight * (
            float(train_row[f"left_{left_mode}_joint_score"]) - float(train_row[f"left_{right_mode}_joint_score"])
        )
        total_weight += weight
    return 0.0 if total_weight == 0.0 else weighted_delta / total_weight


def choose_threshold(train_rows, lhs: str, rhs: str, choose_lhs_when_delta_gt: bool):
    train_pred = []
    for idx, row in enumerate(train_rows):
        train_pred.append(knn_delta(train_rows[:idx] + train_rows[idx + 1 :], row, lhs, rhs))
    best = None
    for threshold in THRESHOLDS:
        if choose_lhs_when_delta_gt:
            modes = [lhs if delta > threshold else rhs for delta in train_pred]
        else:
            modes = [lhs if delta < threshold else rhs for delta in train_pred]
        score = sum(float(row[f"left_{mode}_joint_score"]) for row, mode in zip(train_rows, modes)) / len(train_rows)
        candidate = {
            "leave_one_out_joint_score": score,
            "threshold": threshold,
            "pred_mode_counts": dict(Counter(modes)),
        }
        if best is None or candidate["leave_one_out_joint_score"] > best["leave_one_out_joint_score"]:
            best = candidate
    return best


def summarize(rows, prefix: str):
    n = len(rows)
    return {
        "num_frames": n,
        "right_grouped_match_overall": sum(float(row[f"{prefix}_right_grouped_match"]) for row in rows) / n,
        "left_preserve_overall": sum(float(row[f"{prefix}_left_preserve"]) for row in rows) / n,
        "joint_score_overall": sum(float(row[f"{prefix}_joint_score"]) for row in rows) / n,
    }


def paired_stats(rows, a_prefix: str, b_prefix: str):
    a = [float(row[f"{a_prefix}_joint_score"]) for row in rows]
    b = [float(row[f"{b_prefix}_joint_score"]) for row in rows]
    diff = [y - x for x, y in zip(a, b)]
    return {
        "delta": sum(diff) / len(diff),
        "wins": int(sum(x > 0 for x in diff)),
        "losses": int(sum(x < 0 for x in diff)),
        "ties": int(sum(x == 0 for x in diff)),
    }


def main():
    train_data = load_json(GEN / "temporal_hl_val.json")
    test_data = load_json(GEN / "temporal_hl_test.json")
    labels = overlap_labels(train_data, test_data)
    semantic_vocab = build_semantic_frame_vocab(train_data, labels)
    pair_bank = build_pair_bank(train_data, labels, semantic_vocab)

    train_frames = build_frames(train_data, labels, "closing")
    test_frames = build_frames(test_data, labels, "closing")
    pair_model, pair_stats = train_pairguided_model(train_frames, pair_bank, "closing")
    train_rows = attach_context(collect_rows(train_frames, pair_bank, "closing", pair_model))
    test_rows = attach_context(collect_rows(test_frames, pair_bank, "closing", pair_model))

    edge_none = choose_threshold(train_rows, "none", "edge_transition_snap", choose_lhs_when_delta_gt=False)
    edge_finger = choose_threshold(train_rows, "finger_profile_snap", "edge_transition_snap", choose_lhs_when_delta_gt=True)

    output_rows = []
    edge_none_modes = []
    edge_finger_modes = []
    for row in test_rows:
        d_edge_none = knn_delta(train_rows, row, "none", "edge_transition_snap")
        mode_edge_none = "none" if d_edge_none < edge_none["threshold"] else "edge_transition_snap"
        d_edge_finger = knn_delta(train_rows, row, "finger_profile_snap", "edge_transition_snap")
        mode_edge_finger = "finger_profile_snap" if d_edge_finger > edge_finger["threshold"] else "edge_transition_snap"
        edge_none_modes.append(mode_edge_none)
        edge_finger_modes.append(mode_edge_finger)

        rec = dict(row)
        oracle_mode = max(("none", "edge_transition_snap", "finger_profile_snap"), key=lambda mode: float(rec[f"left_{mode}_joint_score"]))
        for name, mode in [
            ("fixed_edge", "edge_transition_snap"),
            ("edge_vs_none_route", mode_edge_none),
            ("edge_vs_finger_route", mode_edge_finger),
            ("oracle", oracle_mode),
        ]:
            prefix = f"left_{mode}"
            for field in ("right_grouped_match", "left_preserve", "joint_score"):
                rec[f"{name}_{field}"] = rec[f"{prefix}_{field}"]
        rec["edge_vs_none_mode"] = mode_edge_none
        rec["edge_vs_finger_mode"] = mode_edge_finger
        rec["oracle_mode"] = oracle_mode
        output_rows.append(rec)

    payload = {
        "focus": {
            "goal": "closing-specific temporal routing on the feasible subset",
            "task": "right_hand_motion->closing",
            "cfg": CFG,
            "routes": {
                "edge_vs_none": edge_none,
                "edge_vs_finger": edge_finger,
            },
        },
        "training_stats": {"pair_model": pair_stats},
        "summary": {
            name: summarize(output_rows, name)
            for name in ("fixed_edge", "edge_vs_none_route", "edge_vs_finger_route", "oracle")
        },
        "paired": {
            "edge_vs_none_route_vs_fixed_edge": paired_stats(output_rows, "fixed_edge", "edge_vs_none_route"),
            "edge_vs_finger_route_vs_fixed_edge": paired_stats(output_rows, "fixed_edge", "edge_vs_finger_route"),
            "oracle_vs_fixed_edge": paired_stats(output_rows, "fixed_edge", "oracle"),
        },
        "test_pred_mode_counts": {
            "edge_vs_none_route": dict(Counter(edge_none_modes)),
            "edge_vs_finger_route": dict(Counter(edge_finger_modes)),
        },
        "rows": output_rows,
    }

    out_json = GEN / "interaction_realized_feasible_left_closing_temporal_routing.json"
    out_md = SUM / "interaction_realized_feasible_left_closing_temporal_routing.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction Realized Feasible Left Closing Temporal Routing",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Closing-specific one-hop temporal routing from the fixed `edge_transition_snap` policy.",
        "",
        "| method | right grouped | left preserve | joint overall |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method in ("fixed_edge", "edge_vs_none_route", "edge_vs_finger_route", "oracle"):
        stats = payload["summary"][method]
        lines.append(
            f"| {method} | {fmt(stats['right_grouped_match_overall'])} | {fmt(stats['left_preserve_overall'])} | {fmt(stats['joint_score_overall'])} |"
        )
    lines.extend(
        [
            "",
            f"Config: `k={CFG['k']}`, subtype weight `{CFG['subtype_weight']}`, agreement weight `{CFG['agreement_weight']}`",
            "",
            f"Train-selected `edge_vs_none` threshold: `{edge_none['threshold']}`",
            f"Train-selected `edge_vs_finger` threshold: `{edge_finger['threshold']}`",
            "",
            "| comparison | delta | wins | losses | ties |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| edge_vs_none_route vs fixed_edge | {fmt(payload['paired']['edge_vs_none_route_vs_fixed_edge']['delta'])} | {payload['paired']['edge_vs_none_route_vs_fixed_edge']['wins']} | {payload['paired']['edge_vs_none_route_vs_fixed_edge']['losses']} | {payload['paired']['edge_vs_none_route_vs_fixed_edge']['ties']} |",
            f"| edge_vs_finger_route vs fixed_edge | {fmt(payload['paired']['edge_vs_finger_route_vs_fixed_edge']['delta'])} | {payload['paired']['edge_vs_finger_route_vs_fixed_edge']['wins']} | {payload['paired']['edge_vs_finger_route_vs_fixed_edge']['losses']} | {payload['paired']['edge_vs_finger_route_vs_fixed_edge']['ties']} |",
            f"| oracle vs fixed_edge | {fmt(payload['paired']['oracle_vs_fixed_edge']['delta'])} | {payload['paired']['oracle_vs_fixed_edge']['wins']} | {payload['paired']['oracle_vs_fixed_edge']['losses']} | {payload['paired']['oracle_vs_fixed_edge']['ties']} |",
            "",
            f"Test mode counts `edge_vs_none_route`: {payload['test_pred_mode_counts']['edge_vs_none_route']}",
            f"Test mode counts `edge_vs_finger_route`: {payload['test_pred_mode_counts']['edge_vs_finger_route']}",
            "",
        ]
    )

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
