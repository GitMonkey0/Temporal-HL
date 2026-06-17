#!/usr/bin/env python3
"""Opening-specific temporal routing for feasible left repair.

This is an experiment memo, not paper text.
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

    train_frames = build_frames(train_data, labels, "opening")
    test_frames = build_frames(test_data, labels, "opening")
    pair_model, pair_stats = train_pairguided_model(train_frames, pair_bank, "opening")
    train_rows = attach_context(collect_rows(train_frames, pair_bank, "opening", pair_model))
    test_rows = attach_context(collect_rows(test_frames, pair_bank, "opening", pair_model))

    finger_none = choose_threshold(train_rows, "none", "finger_profile_snap", choose_lhs_when_delta_gt=False)
    finger_edge = choose_threshold(train_rows, "edge_transition_snap", "finger_profile_snap", choose_lhs_when_delta_gt=True)

    output_rows = []
    finger_none_modes = []
    finger_edge_modes = []
    for row in test_rows:
        d_finger_none = knn_delta(train_rows, row, "none", "finger_profile_snap")
        mode_finger_none = "none" if d_finger_none < finger_none["threshold"] else "finger_profile_snap"
        d_finger_edge = knn_delta(train_rows, row, "edge_transition_snap", "finger_profile_snap")
        mode_finger_edge = "edge_transition_snap" if d_finger_edge > finger_edge["threshold"] else "finger_profile_snap"
        finger_none_modes.append(mode_finger_none)
        finger_edge_modes.append(mode_finger_edge)

        rec = dict(row)
        oracle_mode = max(("none", "edge_transition_snap", "finger_profile_snap"), key=lambda mode: float(rec[f"left_{mode}_joint_score"]))
        for name, mode in [
            ("fixed_finger", "finger_profile_snap"),
            ("finger_vs_none_route", mode_finger_none),
            ("finger_vs_edge_route", mode_finger_edge),
            ("oracle", oracle_mode),
        ]:
            prefix = f"left_{mode}"
            for field in ("right_grouped_match", "left_preserve", "joint_score"):
                rec[f"{name}_{field}"] = rec[f"{prefix}_{field}"]
        rec["finger_vs_none_mode"] = mode_finger_none
        rec["finger_vs_edge_mode"] = mode_finger_edge
        rec["oracle_mode"] = oracle_mode
        output_rows.append(rec)

    payload = {
        "focus": {
            "goal": "opening-specific temporal routing on the feasible subset",
            "task": "right_hand_motion->opening",
            "cfg": CFG,
            "routes": {
                "finger_vs_none": finger_none,
                "finger_vs_edge": finger_edge,
            },
        },
        "training_stats": {"pair_model": pair_stats},
        "summary": {
            name: summarize(output_rows, name)
            for name in ("fixed_finger", "finger_vs_none_route", "finger_vs_edge_route", "oracle")
        },
        "paired": {
            "finger_vs_none_route_vs_fixed_finger": paired_stats(output_rows, "fixed_finger", "finger_vs_none_route"),
            "finger_vs_edge_route_vs_fixed_finger": paired_stats(output_rows, "fixed_finger", "finger_vs_edge_route"),
            "oracle_vs_fixed_finger": paired_stats(output_rows, "fixed_finger", "oracle"),
        },
        "test_pred_mode_counts": {
            "finger_vs_none_route": dict(Counter(finger_none_modes)),
            "finger_vs_edge_route": dict(Counter(finger_edge_modes)),
        },
        "rows": output_rows,
    }

    out_json = GEN / "interaction_realized_feasible_left_opening_temporal_routing.json"
    out_md = SUM / "interaction_realized_feasible_left_opening_temporal_routing.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction Realized Feasible Left Opening Temporal Routing",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Opening-specific one-hop temporal routing from the fixed `finger_profile_snap` policy.",
        "",
        "| method | right grouped | left preserve | joint overall |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method in ("fixed_finger", "finger_vs_none_route", "finger_vs_edge_route", "oracle"):
        stats = payload["summary"][method]
        lines.append(
            f"| {method} | {fmt(stats['right_grouped_match_overall'])} | {fmt(stats['left_preserve_overall'])} | {fmt(stats['joint_score_overall'])} |"
        )
    lines.extend(
        [
            "",
            f"Config: `k={CFG['k']}`, subtype weight `{CFG['subtype_weight']}`, agreement weight `{CFG['agreement_weight']}`",
            "",
            f"Train-selected `finger_vs_none` threshold: `{finger_none['threshold']}`",
            f"Train-selected `finger_vs_edge` threshold: `{finger_edge['threshold']}`",
            "",
            "| comparison | delta | wins | losses | ties |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| finger_vs_none_route vs fixed_finger | {fmt(payload['paired']['finger_vs_none_route_vs_fixed_finger']['delta'])} | {payload['paired']['finger_vs_none_route_vs_fixed_finger']['wins']} | {payload['paired']['finger_vs_none_route_vs_fixed_finger']['losses']} | {payload['paired']['finger_vs_none_route_vs_fixed_finger']['ties']} |",
            f"| finger_vs_edge_route vs fixed_finger | {fmt(payload['paired']['finger_vs_edge_route_vs_fixed_finger']['delta'])} | {payload['paired']['finger_vs_edge_route_vs_fixed_finger']['wins']} | {payload['paired']['finger_vs_edge_route_vs_fixed_finger']['losses']} | {payload['paired']['finger_vs_edge_route_vs_fixed_finger']['ties']} |",
            f"| oracle vs fixed_finger | {fmt(payload['paired']['oracle_vs_fixed_finger']['delta'])} | {payload['paired']['oracle_vs_fixed_finger']['wins']} | {payload['paired']['oracle_vs_fixed_finger']['losses']} | {payload['paired']['oracle_vs_fixed_finger']['ties']} |",
            "",
            f"Test mode counts `finger_vs_none_route`: {payload['test_pred_mode_counts']['finger_vs_none_route']}",
            f"Test mode counts `finger_vs_edge_route`: {payload['test_pred_mode_counts']['finger_vs_edge_route']}",
            "",
        ]
    )

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
