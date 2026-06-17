#!/usr/bin/env python3
"""Chunk-level transfer test on broader corrected feasible closing.

This is an experiment memo, not paper text.

Goal:
- test whether the learned chunk-level closing gain survives beyond the narrow
  fast-path regime

Protocol:
- evaluate on the broader corrected feasible closing slice
- compare support sources:
  - val_only
  - train_plus_val
- compare:
  - fixed edge
  - chunk-level classifier
  - chunk-level regressor
  - binary oracle
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

from tools.build_interaction_realized_closing_chunk_learned_scorer import (
    CFG,
    apply_chunk_predictions,
    fit_classifier,
    fit_regressors,
    fmt,
    predict_classifier,
    predict_regressor,
    summarize_chunks,
)
from tools.build_interaction_realized_feasible_left_repair_model_gate import collect_rows
from tools.build_interaction_realized_feasible_left_temporal_window_knn import (
    attach_context,
    build_frames,
)
from tools.build_interaction_realized_pairguided_editor import train_pairguided_model
from tools.build_transition_conditioned_symbolic_editor import (
    GEN,
    SUM,
    canonical,
    build_pair_bank,
    build_semantic_frame_vocab,
    load_json,
)
from tools.build_interaction_realized_closing_run_learned_scorer import summarize_rows, paired_stats
from tools.build_pairguided_reranker_multislice import collect_slice_frames


TASK_TARGET = "closing"
CHUNK_LENGTHS = (2,)
CACHE_DIR = Path("/opt/tiger/hand/experiments/generated/cache/closing_chunk_transfer")
FRAME_BATCH = 8


def all_sequence_labels(data):
    return {canonical(seq["seq_name"]) for seq in data["sequences"]} | {seq["seq_name"] for seq in data["sequences"]}


def subset_to_closing_sequences(data):
    rows = collect_slice_frames(data, "right_hand_motion", TASK_TARGET)
    keep = {row["seq_name"] for row in rows}
    return {"sequences": [seq for seq in data["sequences"] if seq["seq_name"] in keep]}


def merge_data(*datasets):
    out = {"sequences": []}
    for dataset in datasets:
        out["sequences"].extend(dataset["sequences"])
    return out


def rows_cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}.json"


def model_cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}.pkl"


def build_rows_with_bank(data, pair_bank, pair_model_stats, pair_model):
    frames = [
        row
        for row in collect_slice_frames(data, "right_hand_motion", TASK_TARGET)
        if row["curr_frame"].get("left") is not None
    ]
    rows = attach_context(collect_rows(frames, pair_bank, TASK_TARGET, pair_model))
    return {
        "rows": rows,
        "pair_model": pair_model_stats,
        "num_frames": len(frames),
        "num_sequences": len(data["sequences"]),
        "pair_bank_size": len(pair_bank),
    }


def build_rows_incremental(name: str, data, pair_bank, pair_model_stats, pair_model, max_new_units: int | None = None):
    path = rows_cache_path(name)
    tmp_path = path.with_suffix(".partial.json")
    if path.exists():
        return json.loads(path.read_text())
    frames = [
        row
        for row in collect_slice_frames(data, "right_hand_motion", TASK_TARGET)
        if row["curr_frame"].get("left") is not None
    ]

    completed = set()
    rows = []
    if tmp_path.exists():
        partial = json.loads(tmp_path.read_text())
        completed = set(partial.get("completed_sequences", []))
        rows = partial.get("rows", [])

    grouped_frames = defaultdict(list)
    for frame in frames:
        name_key = frame["seq_name"]
        grouped_frames[name_key].append(frame)
    seq_order = sorted(grouped_frames, key=lambda name: (len(grouped_frames[name]), name))

    grouped_rows = []
    new_units = 0
    for seq_name in seq_order:
        seq_frames = grouped_frames[seq_name]
        total_batches = (len(seq_frames) + FRAME_BATCH - 1) // FRAME_BATCH
        batch_ids = [f"{seq_name}::batch{idx}" for idx in range(total_batches)]
        if all(batch_id in completed for batch_id in batch_ids):
            continue
        for idx in range(total_batches):
            batch_id = f"{seq_name}::batch{idx}"
            if batch_id in completed:
                continue
            batch_frames = seq_frames[idx * FRAME_BATCH : (idx + 1) * FRAME_BATCH]
            grouped_rows.extend(collect_rows(batch_frames, pair_bank, TASK_TARGET, pair_model))
            completed.add(batch_id)
            new_units += 1
            tmp_payload = {
                "rows": rows + grouped_rows,
                "pair_model": pair_model_stats,
                "num_frames": len(frames),
                "num_sequences": len(data["sequences"]),
                "pair_bank_size": len(pair_bank),
                "completed_sequences": sorted(completed),
            }
            tmp_path.write_text(json.dumps(tmp_payload, indent=2))
            if max_new_units is not None and new_units >= max_new_units:
                return tmp_payload

    payload = {
        "rows": rows + grouped_rows,
        "pair_model": pair_model_stats,
        "num_frames": len(frames),
        "num_sequences": len(data["sequences"]),
        "pair_bank_size": len(pair_bank),
    }
    path.write_text(json.dumps(payload, indent=2))
    if tmp_path.exists():
        tmp_path.unlink()
    return payload


def load_or_build_rows(name: str, data, pair_bank, pair_model_stats, pair_model, max_new_units: int | None = None):
    path = rows_cache_path(name)
    if path.exists():
        return json.loads(path.read_text())
    payload = build_rows_incremental(name, data, pair_bank, pair_model_stats, pair_model, max_new_units=max_new_units)
    if max_new_units is None:
        path.write_text(json.dumps(payload, indent=2))
    return payload


def load_rows_cache(name: str):
    path = rows_cache_path(name)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_or_train_pair_model(name: str, frames, pair_bank):
    path = model_cache_path(name)
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)
    pair_model, pair_stats = train_pairguided_model(frames, pair_bank, TASK_TARGET)
    with path.open("wb") as f:
        pickle.dump((pair_model, pair_stats), f)
    return pair_model, pair_stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-only",
        choices=("test_subset", "support_val_only", "support_train_only", "support_train_plus_val", "all"),
        default=None,
        help="Only build and store row caches, then exit.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Assume required caches already exist and only build the final report.",
    )
    parser.add_argument(
        "--max-new-units",
        type=int,
        default=None,
        help="When building a cache, stop after writing this many new sequence-batches.",
    )
    return parser.parse_args()


def evaluate_support(train_rows, test_rows, chunk_len):
    train_chunks = summarize_chunks(train_rows, 12, chunk_len)
    test_chunks = summarize_chunks(test_rows, 12, chunk_len)

    cls_model = fit_classifier(train_chunks)
    reg_model = fit_regressors(train_chunks)
    cls_pred = [predict_classifier(cls_model, chunk) for chunk in test_chunks]
    reg_pred = [predict_regressor(reg_model, chunk)[0] for chunk in test_chunks]

    cls_rows = apply_chunk_predictions(test_chunks, cls_pred)
    reg_rows = apply_chunk_predictions(test_chunks, reg_pred)
    merged_rows = []
    for row_cls, row_reg in zip(cls_rows, reg_rows):
        rec = dict(row_cls)
        for field in ("right_grouped_match", "left_preserve", "joint_score"):
            rec[f"chunk_reg_{field}"] = row_reg[f"chunk_target_{field}"]
        merged_rows.append(rec)

    return {
        "num_support_chunks": len(train_chunks),
        "summary": {
            "fixed_edge": summarize_rows(merged_rows, "fixed_edge"),
            "chunk_cls": summarize_rows(merged_rows, "chunk_target"),
            "chunk_reg": summarize_rows(merged_rows, "chunk_reg"),
            "oracle_binary": summarize_rows(merged_rows, "oracle_binary"),
        },
        "paired": {
            "chunk_cls_vs_fixed_edge": paired_stats(merged_rows, "fixed_edge", "chunk_target"),
            "chunk_reg_vs_fixed_edge": paired_stats(merged_rows, "fixed_edge", "chunk_reg"),
            "chunk_cls_vs_chunk_reg": paired_stats(merged_rows, "chunk_reg", "chunk_target"),
        },
        "pred_mode_counts": {
            "chunk_cls": dict(Counter(cls_pred)),
            "chunk_reg": dict(Counter(reg_pred)),
        },
    }


def main():
    args = parse_args()
    train_data = load_json(GEN / "temporal_hl_train.json")
    val_data = load_json(GEN / "temporal_hl_val.json")
    test_data = load_json(GEN / "temporal_hl_test.json")

    train_subset = subset_to_closing_sequences(train_data)
    val_subset = subset_to_closing_sequences(val_data)
    test_subset = subset_to_closing_sequences(test_data)

    support_sources = {
        "val_only": val_subset,
        "train_plus_val": merge_data(train_subset, val_subset),
    }

    val_labels = all_sequence_labels(val_subset)
    val_semantic_vocab = build_semantic_frame_vocab(val_subset, val_labels)
    val_pair_bank = build_pair_bank(val_subset, val_labels, val_semantic_vocab)
    val_train_frames = [
        row
        for row in collect_slice_frames(val_subset, "right_hand_motion", TASK_TARGET)
        if row["curr_frame"].get("left") is not None
    ]
    val_pair_model, val_pair_stats = load_or_train_pair_model("val_pair_model", val_train_frames, val_pair_bank)

    train_labels = all_sequence_labels(train_subset)
    train_semantic_vocab = build_semantic_frame_vocab(train_subset, train_labels)
    train_pair_bank = build_pair_bank(train_subset, train_labels, train_semantic_vocab)

    if args.cache_only is not None:
        targets = (
            ("test_subset", test_subset, val_pair_bank, val_pair_stats, val_pair_model),
            ("support_val_only", support_sources["val_only"], val_pair_bank, val_pair_stats, val_pair_model),
            ("support_train_only", train_subset, train_pair_bank, val_pair_stats, val_pair_model),
            ("support_train_plus_val", support_sources["train_plus_val"], val_pair_bank, val_pair_stats, val_pair_model),
        )
        for name, data, pair_bank, pair_model_stats, pair_model in targets:
            if args.cache_only not in ("all", name):
                continue
            payload = load_or_build_rows(name, data, pair_bank, pair_model_stats, pair_model)
            print(f"cached {name}: frames={payload['num_frames']} sequences={payload['num_sequences']} bank={payload['pair_bank_size']}")
        return

    test_payload = load_or_build_rows("test_subset", test_subset, val_pair_bank, val_pair_stats, val_pair_model)

    task_results = {}
    for support_name, support_data in support_sources.items():
        if support_name == "train_plus_val":
            train_only_payload = load_rows_cache("support_train_only")
            val_only_payload = load_rows_cache("support_val_only")
            if train_only_payload is not None and val_only_payload is not None:
                train_payload = {
                    "rows": train_only_payload["rows"] + val_only_payload["rows"],
                    "pair_model": {
                        "train_only": train_only_payload["pair_model"],
                        "val_only": val_only_payload["pair_model"],
                    },
                    "num_frames": int(train_only_payload["num_frames"]) + int(val_only_payload["num_frames"]),
                    "num_sequences": int(train_only_payload["num_sequences"]) + int(val_only_payload["num_sequences"]),
                    "pair_bank_size": max(int(train_only_payload["pair_bank_size"]), int(val_only_payload["pair_bank_size"])),
                }
            else:
                train_payload = load_or_build_rows(f"support_{support_name}", support_data, val_pair_bank, val_pair_stats, val_pair_model)
        else:
            train_payload = load_or_build_rows(
                f"support_{support_name}",
                support_data,
                val_pair_bank,
                val_pair_stats,
                val_pair_model,
            )
        support_result = {
            "training_stats": {
                "pair_model": train_payload["pair_model"],
                "num_support_frames": train_payload["num_frames"],
                "num_support_sequences": train_payload["num_sequences"],
                "pair_bank_size": train_payload["pair_bank_size"],
            },
            "chunk_lengths": {},
        }
        for chunk_len in CHUNK_LENGTHS:
            support_result["chunk_lengths"][f"chunk_len_{chunk_len}"] = evaluate_support(
                train_payload["rows"], test_payload["rows"], chunk_len
            )
        task_results[support_name] = support_result

    payload = {
        "focus": {
            "goal": "chunk-level transfer on broader corrected feasible closing",
            "task": "right_hand_motion->closing",
            "support_sources": list(support_sources.keys()),
        },
        "test_stats": {
            "num_test_frames": test_payload["num_frames"],
            "num_test_sequences": test_payload["num_sequences"],
            "pair_bank_size": test_payload["pair_bank_size"],
        },
        "task_results": task_results,
    }

    out_json = GEN / "interaction_realized_closing_chunk_transfer.json"
    out_md = SUM / "interaction_realized_closing_chunk_transfer.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Interaction Realized Closing Chunk Transfer",
        "",
        "This is an experiment memo, not paper text.",
        "",
        f"Test frames: `{test_payload['num_frames']}`; test sequences: `{test_payload['num_sequences']}`; test pair bank size: `{test_payload['pair_bank_size']}`",
        "",
    ]
    for support_name, result in task_results.items():
        lines.extend(
            [
                f"## {support_name}",
                "",
                f"Support frames: `{result['training_stats']['num_support_frames']}`",
                f"Support sequences: `{result['training_stats']['num_support_sequences']}`",
                f"Pair bank size: `{result['training_stats']['pair_bank_size']}`",
                "",
            ]
        )
        for chunk_name, chunk_result in result["chunk_lengths"].items():
            lines.extend(
                [
                    f"### {chunk_name}",
                    "",
                    f"Support chunks: `{chunk_result['num_support_chunks']}`",
                    "",
                    "| method | right grouped | left preserve | joint overall |",
                    "| --- | ---: | ---: | ---: |",
                ]
            )
            for method in ("fixed_edge", "chunk_cls", "chunk_reg", "oracle_binary"):
                stats = chunk_result["summary"][method]
                lines.append(
                    f"| {method} | {fmt(stats['right_grouped_match_overall'])} | {fmt(stats['left_preserve_overall'])} | {fmt(stats['joint_score_overall'])} |"
                )
            lines.extend(
                [
                    "",
                    f"Pred chunk modes: `{chunk_result['pred_mode_counts']}`",
                    "",
                    "| comparison | delta | wins | losses | ties |",
                    "| --- | ---: | ---: | ---: | ---: |",
                    f"| chunk_cls vs fixed_edge | {fmt(chunk_result['paired']['chunk_cls_vs_fixed_edge']['delta'])} | {chunk_result['paired']['chunk_cls_vs_fixed_edge']['wins']} | {chunk_result['paired']['chunk_cls_vs_fixed_edge']['losses']} | {chunk_result['paired']['chunk_cls_vs_fixed_edge']['ties']} |",
                    f"| chunk_reg vs fixed_edge | {fmt(chunk_result['paired']['chunk_reg_vs_fixed_edge']['delta'])} | {chunk_result['paired']['chunk_reg_vs_fixed_edge']['wins']} | {chunk_result['paired']['chunk_reg_vs_fixed_edge']['losses']} | {chunk_result['paired']['chunk_reg_vs_fixed_edge']['ties']} |",
                    f"| chunk_cls vs chunk_reg | {fmt(chunk_result['paired']['chunk_cls_vs_chunk_reg']['delta'])} | {chunk_result['paired']['chunk_cls_vs_chunk_reg']['wins']} | {chunk_result['paired']['chunk_cls_vs_chunk_reg']['losses']} | {chunk_result['paired']['chunk_cls_vs_chunk_reg']['ties']} |",
                    "",
                ]
            )

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
