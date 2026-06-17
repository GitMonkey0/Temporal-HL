#!/usr/bin/env python3
"""Opening chunk-level KNN selector on the feasible subset.

This is an experiment memo, not paper text.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

from tools.build_interaction_realized_feasible_left_closing_chunk_knn import (
    GRID,
    MODES,
    build_runs,
    choose_cfg,
    distance,
    evaluate_runs,
    feature_vector,
    fmt,
    paired_stats,
    predict_mode,
    summarize,
    summarize_run,
)
from tools.build_interaction_realized_feasible_left_repair_model_gate import collect_rows
from tools.build_interaction_realized_feasible_left_temporal_window_knn import build_frames
from tools.build_interaction_realized_pairguided_editor import train_pairguided_model
from tools.build_transition_conditioned_symbolic_editor import (
    GEN,
    SUM,
    build_pair_bank,
    build_semantic_frame_vocab,
    load_json,
    overlap_labels,
)


def main():
    train_data = load_json(GEN / "temporal_hl_val.json")
    test_data = load_json(GEN / "temporal_hl_test.json")
    labels = overlap_labels(train_data, test_data)
    semantic_vocab = build_semantic_frame_vocab(train_data, labels)
    pair_bank = build_pair_bank(train_data, labels, semantic_vocab)

    train_frames = build_frames(train_data, labels, "opening")
    test_frames = build_frames(test_data, labels, "opening")
    pair_model, pair_stats = train_pairguided_model(train_frames, pair_bank, "opening")
    train_rows = collect_rows(train_frames, pair_bank, "opening", pair_model)
    test_rows = collect_rows(test_frames, pair_bank, "opening", pair_model)

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
                ("fixed_finger", "finger_profile_snap"),
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
            "goal": "opening chunk-level KNN selector on the feasible subset",
            "task": "right_hand_motion->opening",
            "grid_size": len(GRID),
        },
        "training_stats": {
            "pair_model": pair_stats,
            "chunk_knn": best,
            "num_train_runs": len(train_summaries),
            "num_test_runs": len(test_summaries),
        },
        "summary": {name: summarize(output_rows, name) for name in ("fixed_finger", "chunk_knn", "oracle")},
        "paired": {
            "chunk_knn_vs_fixed_finger": paired_stats(output_rows, "fixed_finger", "chunk_knn"),
            "oracle_vs_fixed_finger": paired_stats(output_rows, "fixed_finger", "oracle"),
        },
        "test_pred_mode_counts": dict(Counter(pred_modes)),
        "rows": output_rows,
    }

    out_json = GEN / "interaction_realized_feasible_left_opening_chunk_knn.json"
    out_md = SUM / "interaction_realized_feasible_left_opening_chunk_knn.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction Realized Feasible Left Opening Chunk KNN",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Opening-only run-level KNN selector with a single repair decision per short temporal run.",
        "",
        "| method | right grouped | left preserve | joint overall |",
        "| --- | ---: | ---: | ---: |",
        f"| fixed_finger | {fmt(payload['summary']['fixed_finger']['right_grouped_match_overall'])} | {fmt(payload['summary']['fixed_finger']['left_preserve_overall'])} | {fmt(payload['summary']['fixed_finger']['joint_score_overall'])} |",
        f"| chunk_knn | {fmt(payload['summary']['chunk_knn']['right_grouped_match_overall'])} | {fmt(payload['summary']['chunk_knn']['left_preserve_overall'])} | {fmt(payload['summary']['chunk_knn']['joint_score_overall'])} |",
        f"| oracle | {fmt(payload['summary']['oracle']['right_grouped_match_overall'])} | {fmt(payload['summary']['oracle']['left_preserve_overall'])} | {fmt(payload['summary']['oracle']['joint_score_overall'])} |",
        "",
        f"Train runs: `{payload['training_stats']['num_train_runs']}`, test runs: `{payload['training_stats']['num_test_runs']}`",
        f"Selected config: `{payload['training_stats']['chunk_knn']['cfg']}`, leave-one-out joint `{fmt(payload['training_stats']['chunk_knn']['leave_one_out_joint_score'])}`",
        "",
        "| comparison | delta | wins | losses | ties |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| chunk_knn vs fixed_finger | {fmt(payload['paired']['chunk_knn_vs_fixed_finger']['delta'])} | {payload['paired']['chunk_knn_vs_fixed_finger']['wins']} | {payload['paired']['chunk_knn_vs_fixed_finger']['losses']} | {payload['paired']['chunk_knn_vs_fixed_finger']['ties']} |",
        f"| oracle vs fixed_finger | {fmt(payload['paired']['oracle_vs_fixed_finger']['delta'])} | {payload['paired']['oracle_vs_fixed_finger']['wins']} | {payload['paired']['oracle_vs_fixed_finger']['losses']} | {payload['paired']['oracle_vs_fixed_finger']['ties']} |",
        "",
        f"Test run mode counts: `{payload['test_pred_mode_counts']}`",
        "",
    ]
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
