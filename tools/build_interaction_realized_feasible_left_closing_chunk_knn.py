#!/usr/bin/env python3
"""Closing chunk-level KNN selector on the feasible subset.

This is an experiment memo, not paper text.

Test whether closing repair becomes more stable when the decision unit is a
 short temporal run instead of a single frame.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

from tools.build_interaction_realized_feasible_left_repair_model_gate import collect_rows
from tools.build_interaction_realized_pairguided_editor import train_pairguided_model
from tools.build_interaction_realized_feasible_left_temporal_window_knn import build_frames
from tools.build_transition_conditioned_symbolic_editor import (
    GEN,
    SUM,
    build_pair_bank,
    build_semantic_frame_vocab,
    load_json,
    overlap_labels,
)


MODES = ("none", "edge_transition_snap", "finger_profile_snap")
OTHER_VALUES = ["none", "start", "opening", "closing", "mixed", "steady", "unknown"]
INTERACTION_VALUES = ["approach", "separate", "steady", "unknown"]
GRID = []
for k in (1, 3, 5, 7):
    for subtype_weight in (0.25, 0.5, 1.0):
        for agreement_weight in (0.5, 1.0):
            for length_weight in (0.1, 0.25, 0.5):
                GRID.append(
                    {
                        "k": k,
                        "subtype_weight": subtype_weight,
                        "agreement_weight": agreement_weight,
                        "length_weight": length_weight,
                    }
                )


def fmt(x: float) -> str:
    return f"{x:.4f}"


def build_runs(rows):
    runs = []
    cur = []
    prev = None
    for row in rows:
        ok = False
        if prev is not None:
            ok = row["seq_name"] == prev["seq_name"] and int(row["frame_idx"]) - int(prev["frame_idx"]) <= 12
        if prev is None or ok:
            cur.append(row)
        else:
            runs.append(cur)
            cur = [row]
        prev = row
    if cur:
        runs.append(cur)
    return runs


def summarize_run(run):
    subtype_counter = Counter((row["other_hand_motion"], row["interaction_motion_value"]) for row in run)
    major_subtype = subtype_counter.most_common(1)[0][0]
    means = {mode: sum(float(row[f"left_{mode}_joint_score"]) for row in run) / len(run) for mode in MODES}
    return {
        "seq_name": run[0]["seq_name"],
        "start_frame_idx": int(run[0]["frame_idx"]),
        "end_frame_idx": int(run[-1]["frame_idx"]),
        "num_frames": len(run),
        "major_other_hand_motion": major_subtype[0],
        "major_interaction_motion_value": major_subtype[1],
        "subtype_counter": {str(k): int(v) for k, v in subtype_counter.items()},
        "left_state_mean": sum(float(row["left_none_left_state_agreement"]) for row in run) / len(run),
        "left_transition_mean": sum(float(row["left_none_left_transition_agreement"]) for row in run) / len(run),
        "right_state_mean": sum(float(row["left_none_right_state_agreement"]) for row in run) / len(run),
        "right_transition_mean": sum(float(row["left_none_right_transition_agreement"]) for row in run) / len(run),
        "means": means,
        "best_mode": max(means, key=means.get),
    }


def one_hot(counter, vocab, axis: int):
    out = []
    total = sum(counter.values())
    for value in vocab:
        count = 0
        for key, freq in counter.items():
            key = eval(key) if isinstance(key, str) and key.startswith("(") else key
            if key[axis] == value:
                count += freq
        out.append(count / total if total else 0.0)
    return out


def feature_vector(summary):
    subtype_counter = {}
    for key, value in summary["subtype_counter"].items():
        try:
            parsed = eval(key)
        except Exception:
            parsed = ("unknown", "unknown")
        subtype_counter[parsed] = value
    feats = [float(summary["num_frames"])]
    feats.extend(one_hot(subtype_counter, OTHER_VALUES, axis=0))
    feats.extend(one_hot(subtype_counter, INTERACTION_VALUES, axis=1))
    feats.extend(
        [
            float(summary["left_state_mean"]),
            float(summary["left_transition_mean"]),
            float(summary["right_state_mean"]),
            float(summary["right_transition_mean"]),
        ]
    )
    return feats


def distance(a, b, cfg):
    av = feature_vector(a)
    bv = feature_vector(b)
    total = cfg["length_weight"] * abs(av[0] - bv[0])
    offset = 1
    total += cfg["subtype_weight"] * sum(abs(x - y) for x, y in zip(av[offset : offset + len(OTHER_VALUES)], bv[offset : offset + len(OTHER_VALUES)]))
    offset += len(OTHER_VALUES)
    total += cfg["subtype_weight"] * sum(abs(x - y) for x, y in zip(av[offset : offset + len(INTERACTION_VALUES)], bv[offset : offset + len(INTERACTION_VALUES)]))
    offset += len(INTERACTION_VALUES)
    total += cfg["agreement_weight"] * sum(abs(x - y) for x, y in zip(av[offset:], bv[offset:]))
    return total


def predict_mode(train_summaries, query, cfg):
    neighbors = sorted(((distance(item, query, cfg), item) for item in train_summaries), key=lambda x: x[0])[: cfg["k"]]
    votes = Counter(item["best_mode"] for _, item in neighbors)
    score_sums = defaultdict(float)
    for _, item in neighbors:
        for mode in MODES:
            score_sums[mode] += item["means"][mode]
    return max(MODES, key=lambda mode: (votes[mode], score_sums[mode]))


def evaluate_runs(run_summaries, mode_choices):
    total = 0.0
    n = 0
    for summary, mode in zip(run_summaries, mode_choices):
        total += summary["means"][mode] * summary["num_frames"]
        n += summary["num_frames"]
    return total / n


def choose_cfg(train_summaries):
    best = None
    for cfg in GRID:
        modes = []
        for idx, summary in enumerate(train_summaries):
            modes.append(predict_mode(train_summaries[:idx] + train_summaries[idx + 1 :], summary, cfg))
        score = evaluate_runs(train_summaries, modes)
        candidate = {
            "cfg": cfg,
            "leave_one_out_joint_score": score,
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
    train_rows = collect_rows(train_frames, pair_bank, "closing", pair_model)
    test_rows = collect_rows(test_frames, pair_bank, "closing", pair_model)

    train_runs_raw = build_runs(train_rows)
    test_runs_raw = build_runs(test_rows)
    train_summaries = [summarize_run(run) for run in train_runs_raw]
    test_summaries = [summarize_run(run) for run in test_runs_raw]

    best = choose_cfg(train_summaries)
    pred_modes = [predict_mode(train_summaries, summary, best["cfg"]) for summary in test_summaries]

    output_rows = []
    for run_rows, pred_mode in zip(test_runs_raw, pred_modes):
        for row in run_rows:
            rec = dict(row)
            oracle_mode = max(MODES, key=lambda mode: float(rec[f"left_{mode}_joint_score"]))
            for name, mode in [
                ("fixed_edge", "edge_transition_snap"),
                ("chunk_knn", pred_mode),
                ("oracle", oracle_mode),
            ]:
                prefix = f"left_{mode}"
                for field in ("right_grouped_match", "left_preserve", "joint_score"):
                    rec[f"{name}_{field}"] = rec[f"{prefix}_{field}"]
            rec["chunk_knn_mode"] = pred_mode
            rec["oracle_mode"] = oracle_mode
            output_rows.append(rec)

    payload = {
        "focus": {
            "goal": "closing chunk-level KNN selector on the feasible subset",
            "task": "right_hand_motion->closing",
            "grid_size": len(GRID),
        },
        "training_stats": {
            "pair_model": pair_stats,
            "chunk_knn": best,
            "num_train_runs": len(train_summaries),
            "num_test_runs": len(test_summaries),
        },
        "summary": {name: summarize(output_rows, name) for name in ("fixed_edge", "chunk_knn", "oracle")},
        "paired": {
            "chunk_knn_vs_fixed_edge": paired_stats(output_rows, "fixed_edge", "chunk_knn"),
            "oracle_vs_fixed_edge": paired_stats(output_rows, "fixed_edge", "oracle"),
        },
        "test_pred_mode_counts": dict(Counter(pred_modes)),
        "rows": output_rows,
    }

    out_json = GEN / "interaction_realized_feasible_left_closing_chunk_knn.json"
    out_md = SUM / "interaction_realized_feasible_left_closing_chunk_knn.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction Realized Feasible Left Closing Chunk KNN",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Closing-only run-level KNN selector with a single repair decision per short temporal run.",
        "",
        "| method | right grouped | left preserve | joint overall |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method in ("fixed_edge", "chunk_knn", "oracle"):
        stats = payload["summary"][method]
        lines.append(
            f"| {method} | {fmt(stats['right_grouped_match_overall'])} | {fmt(stats['left_preserve_overall'])} | {fmt(stats['joint_score_overall'])} |"
        )
    lines.extend(
        [
            "",
            f"Train runs: `{payload['training_stats']['num_train_runs']}`, test runs: `{payload['training_stats']['num_test_runs']}`",
            f"Selected config: `{best['cfg']}`, leave-one-out joint `{fmt(best['leave_one_out_joint_score'])}`",
            "",
            "| comparison | delta | wins | losses | ties |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| chunk_knn vs fixed_edge | {fmt(payload['paired']['chunk_knn_vs_fixed_edge']['delta'])} | {payload['paired']['chunk_knn_vs_fixed_edge']['wins']} | {payload['paired']['chunk_knn_vs_fixed_edge']['losses']} | {payload['paired']['chunk_knn_vs_fixed_edge']['ties']} |",
            f"| oracle vs fixed_edge | {fmt(payload['paired']['oracle_vs_fixed_edge']['delta'])} | {payload['paired']['oracle_vs_fixed_edge']['wins']} | {payload['paired']['oracle_vs_fixed_edge']['losses']} | {payload['paired']['oracle_vs_fixed_edge']['ties']} |",
            "",
            f"Test run mode counts: {payload['test_pred_mode_counts']}",
            "",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
