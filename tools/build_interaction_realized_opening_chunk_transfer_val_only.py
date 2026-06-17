#!/usr/bin/env python3
"""Val-only chunk transfer test on broader corrected feasible opening.

This is an experiment memo, not paper text.
"""

from __future__ import annotations

import json

from tools.build_interaction_realized_closing_chunk_transfer import (
    all_sequence_labels,
    evaluate_support,
    load_json,
)
from tools.build_interaction_realized_feasible_left_repair_model_gate import collect_rows
from tools.build_interaction_realized_feasible_left_temporal_window_knn import (
    attach_context,
)
from tools.build_interaction_realized_pairguided_editor import train_pairguided_model
from tools.build_pairguided_reranker_multislice import collect_slice_frames
from tools.build_transition_conditioned_symbolic_editor import (
    GEN,
    SUM,
    build_pair_bank,
    build_semantic_frame_vocab,
)


TASK_TARGET = "opening"


def subset_to_opening_sequences(data):
    rows = collect_slice_frames(data, "right_hand_motion", TASK_TARGET)
    keep = {row["seq_name"] for row in rows}
    return {"sequences": [seq for seq in data["sequences"] if seq["seq_name"] in keep]}


def build_rows(data, pair_bank, pair_model):
    frames = [
        row
        for row in collect_slice_frames(data, "right_hand_motion", TASK_TARGET)
        if row["curr_frame"].get("left") is not None
    ]
    rows = attach_context(collect_rows(frames, pair_bank, TASK_TARGET, pair_model))
    return {
        "rows": rows,
        "num_frames": len(frames),
        "num_sequences": len(data["sequences"]),
        "pair_bank_size": len(pair_bank),
    }


def fmt(x):
    return f"{x:.4f}"


def main():
    val_data = load_json(GEN / "temporal_hl_val.json")
    test_data = load_json(GEN / "temporal_hl_test.json")

    val_subset = subset_to_opening_sequences(val_data)
    test_subset = subset_to_opening_sequences(test_data)

    labels = all_sequence_labels(val_subset)
    semantic_vocab = build_semantic_frame_vocab(val_subset, labels)
    pair_bank = build_pair_bank(val_subset, labels, semantic_vocab)

    val_frames = [
        row
        for row in collect_slice_frames(val_subset, "right_hand_motion", TASK_TARGET)
        if row["curr_frame"].get("left") is not None
    ]
    pair_model, pair_stats = train_pairguided_model(val_frames, pair_bank, TASK_TARGET)

    val_payload = build_rows(val_subset, pair_bank, pair_model)
    test_payload = build_rows(test_subset, pair_bank, pair_model)
    result = evaluate_support(val_payload["rows"], test_payload["rows"], 2)

    payload = {
        "focus": {
            "goal": "chunk-level transfer on broader corrected feasible opening",
            "task": "right_hand_motion->opening",
            "support": "val_only",
            "chunk_len": 2,
        },
        "training_stats": {
            "num_support_frames": val_payload["num_frames"],
            "num_support_sequences": val_payload["num_sequences"],
            "pair_bank_size": val_payload["pair_bank_size"],
            "pair_model": pair_stats,
        },
        "result": result,
    }

    out_json = GEN / "interaction_realized_opening_chunk_transfer_val_only.json"
    out_md = SUM / "interaction_realized_opening_chunk_transfer_val_only.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction Realized Opening Chunk Transfer Val Only",
        "",
        "This is an experiment memo, not paper text.",
        "",
        f"Support frames: `{val_payload['num_frames']}`; support sequences: `{val_payload['num_sequences']}`; pair bank size: `{val_payload['pair_bank_size']}`",
        f"Test frames: `{test_payload['num_frames']}`; test sequences: `{test_payload['num_sequences']}`",
        "",
        "| method | right grouped | left preserve | joint overall |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method in ("fixed_edge", "chunk_cls", "chunk_reg", "oracle_binary"):
        stats = result["summary"][method]
        lines.append(
            f"| {method} | {fmt(stats['right_grouped_match_overall'])} | {fmt(stats['left_preserve_overall'])} | {fmt(stats['joint_score_overall'])} |"
        )
    lines.extend(
        [
            "",
            f"Pred chunk modes: `{result['pred_mode_counts']}`",
            "",
            "| comparison | delta | wins | losses | ties |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| chunk_cls vs fixed_edge | {fmt(result['paired']['chunk_cls_vs_fixed_edge']['delta'])} | {result['paired']['chunk_cls_vs_fixed_edge']['wins']} | {result['paired']['chunk_cls_vs_fixed_edge']['losses']} | {result['paired']['chunk_cls_vs_fixed_edge']['ties']} |",
            f"| chunk_reg vs fixed_edge | {fmt(result['paired']['chunk_reg_vs_fixed_edge']['delta'])} | {result['paired']['chunk_reg_vs_fixed_edge']['wins']} | {result['paired']['chunk_reg_vs_fixed_edge']['losses']} | {result['paired']['chunk_reg_vs_fixed_edge']['ties']} |",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
