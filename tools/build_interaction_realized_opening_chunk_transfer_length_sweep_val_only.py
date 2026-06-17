#!/usr/bin/env python3
"""Length sweep for val-only chunk transfer on broader corrected feasible opening.

This is an experiment memo, not paper text.
"""

from __future__ import annotations

import json

from tools.build_interaction_realized_closing_chunk_transfer import (
    evaluate_support,
    load_json,
)
from tools.build_interaction_realized_opening_chunk_transfer_val_only import (
    TASK_TARGET,
    build_rows,
    subset_to_opening_sequences,
)
from tools.build_interaction_realized_pairguided_editor import train_pairguided_model
from tools.build_pairguided_reranker_multislice import collect_slice_frames
from tools.build_transition_conditioned_symbolic_editor import (
    GEN,
    SUM,
    build_pair_bank,
    build_semantic_frame_vocab,
)


CHUNK_LENGTHS = (2, 3, 4)


def fmt(x):
    return f"{x:.4f}"


def main():
    val_data = load_json(GEN / "temporal_hl_val.json")
    test_data = load_json(GEN / "temporal_hl_test.json")

    val_subset = subset_to_opening_sequences(val_data)
    test_subset = subset_to_opening_sequences(test_data)

    labels = {seq["seq_name"] for seq in val_subset["sequences"]}
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

    results = {}
    for chunk_len in CHUNK_LENGTHS:
        results[f"chunk_len_{chunk_len}"] = evaluate_support(
            val_payload["rows"],
            test_payload["rows"],
            chunk_len,
        )

    payload = {
        "focus": {
            "goal": "chunk-length robustness on broader corrected feasible opening",
            "task": "right_hand_motion->opening",
            "support": "val_only",
            "chunk_lengths": list(CHUNK_LENGTHS),
        },
        "training_stats": {
            "num_support_frames": val_payload["num_frames"],
            "num_support_sequences": val_payload["num_sequences"],
            "pair_bank_size": val_payload["pair_bank_size"],
            "pair_model": pair_stats,
        },
        "results": results,
    }

    out_json = GEN / "interaction_realized_opening_chunk_transfer_length_sweep_val_only.json"
    out_md = SUM / "interaction_realized_opening_chunk_transfer_length_sweep_val_only.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction Realized Opening Chunk Transfer Length Sweep Val Only",
        "",
        "This is an experiment memo, not paper text.",
        "",
        f"Support frames: `{val_payload['num_frames']}`; support sequences: `{val_payload['num_sequences']}`; pair bank size: `{val_payload['pair_bank_size']}`",
        f"Test frames: `{test_payload['num_frames']}`; test sequences: `{test_payload['num_sequences']}`",
        "",
        "| chunk len | fixed edge | chunk cls | chunk reg | oracle binary | delta vs fixed edge |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for chunk_name, result in results.items():
        fixed_edge = result["summary"]["fixed_edge"]["joint_score_overall"]
        chunk_cls = result["summary"]["chunk_cls"]["joint_score_overall"]
        chunk_reg = result["summary"]["chunk_reg"]["joint_score_overall"]
        oracle_binary = result["summary"]["oracle_binary"]["joint_score_overall"]
        delta = result["paired"]["chunk_cls_vs_fixed_edge"]["delta"]
        lines.append(
            f"| {chunk_name} | {fmt(fixed_edge)} | {fmt(chunk_cls)} | {fmt(chunk_reg)} | {fmt(oracle_binary)} | {fmt(delta)} |"
        )

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
