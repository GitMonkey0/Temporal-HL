#!/usr/bin/env python3
"""Parallel cache builder for broader closing chunk transfer support rows.

This is an experiment utility, not paper text.

Why this exists:
- the original incremental cache path is intentionally conservative and robust
- on the train-only broader closing support it can become unnecessarily slow
- this script recomputes the train-only support cache in parallel so that the
  stronger train-plus-val transfer report can be finalized faster
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

from tools.build_interaction_realized_closing_chunk_transfer import (
    CACHE_DIR,
    FRAME_BATCH,
    TASK_TARGET,
    all_sequence_labels,
    load_json,
    load_or_train_pair_model,
    subset_to_closing_sequences,
)
from tools.build_interaction_realized_feasible_left_repair_model_gate import collect_rows
from tools.build_pairguided_reranker_multislice import collect_slice_frames
from tools.build_transition_conditioned_symbolic_editor import (
    GEN,
    build_pair_bank,
    build_semantic_frame_vocab,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing support_train_only.json if it already exists.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=8,
        help="Flush partial progress after this many completed batches.",
    )
    return parser.parse_args()


_WORKER_PAIR_BANK = None
_WORKER_PAIR_MODEL = None


def _init_worker(pair_bank, pair_model):
    global _WORKER_PAIR_BANK, _WORKER_PAIR_MODEL
    _WORKER_PAIR_BANK = pair_bank
    _WORKER_PAIR_MODEL = pair_model


def _process_batch(batch_id, batch_frames):
    return batch_id, collect_rows(batch_frames, _WORKER_PAIR_BANK, TASK_TARGET, _WORKER_PAIR_MODEL)


def main():
    args = parse_args()
    out_path = CACHE_DIR / "support_train_only.json"
    partial_path = CACHE_DIR / "support_train_only.parallel.partial.json"
    legacy_partial_path = CACHE_DIR / "support_train_only.partial.json"
    if out_path.exists() and not args.overwrite:
        print(f"exists: {out_path}")
        return

    train_data = load_json(GEN / "temporal_hl_train.json")
    val_data = load_json(GEN / "temporal_hl_val.json")

    train_subset = subset_to_closing_sequences(train_data)
    val_subset = subset_to_closing_sequences(val_data)

    train_labels = all_sequence_labels(train_subset)
    train_semantic_vocab = build_semantic_frame_vocab(train_subset, train_labels)
    train_pair_bank = build_pair_bank(train_subset, train_labels, train_semantic_vocab)

    val_labels = all_sequence_labels(val_subset)
    val_semantic_vocab = build_semantic_frame_vocab(val_subset, val_labels)
    val_pair_bank = build_pair_bank(val_subset, val_labels, val_semantic_vocab)
    val_train_frames = [
        row
        for row in collect_slice_frames(val_subset, "right_hand_motion", TASK_TARGET)
        if row["curr_frame"].get("left") is not None
    ]
    val_pair_model, val_pair_stats = load_or_train_pair_model(
        "val_pair_model", val_train_frames, val_pair_bank
    )

    frames = [
        row
        for row in collect_slice_frames(train_subset, "right_hand_motion", TASK_TARGET)
        if row["curr_frame"].get("left") is not None
    ]
    grouped = defaultdict(list)
    for frame in frames:
        grouped[frame["seq_name"]].append(frame)

    batch_frames = []
    for seq_name in sorted(grouped, key=lambda name: (len(grouped[name]), name)):
        seq_frames = grouped[seq_name]
        total_batches = (len(seq_frames) + FRAME_BATCH - 1) // FRAME_BATCH
        for idx in range(total_batches):
            batch_frames.append(
                (
                    f"{seq_name}::batch{idx}",
                    seq_frames[idx * FRAME_BATCH : (idx + 1) * FRAME_BATCH],
                )
            )

    rows = []
    completed = set()
    if partial_path.exists() and not args.overwrite:
        partial = json.loads(partial_path.read_text())
        rows = partial.get("rows", [])
        completed = set(partial.get("completed_sequences", []))
    elif legacy_partial_path.exists() and not args.overwrite:
        partial = json.loads(legacy_partial_path.read_text())
        rows = partial.get("rows", [])
        completed = set(partial.get("completed_sequences", []))

    pending_batches = [(batch_id, frames) for batch_id, frames in batch_frames if batch_id not in completed]

    def flush_partial():
        payload = {
            "rows": rows,
            "pair_model": val_pair_stats,
            "num_frames": len(frames),
            "num_sequences": len(train_subset["sequences"]),
            "pair_bank_size": len(train_pair_bank),
            "completed_sequences": sorted(completed),
        }
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        partial_path.write_text(json.dumps(payload, indent=2))

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(train_pair_bank, val_pair_model),
    ) as ex:
        future_map = {
            ex.submit(_process_batch, batch_id, batch): batch_id
            for batch_id, batch in pending_batches
        }
        completed_since_flush = 0
        while future_map:
            done, _ = wait(future_map, return_when=FIRST_COMPLETED)
            for fut in done:
                batch_id, batch_rows = fut.result()
                rows.extend(batch_rows)
                completed.add(batch_id)
                completed_since_flush += 1
                del future_map[fut]
            if completed_since_flush >= args.flush_every:
                flush_partial()
                print(
                    f"partial {len(completed)}/{len(batch_frames)} batches rows={len(rows)}",
                    flush=True,
                )
                completed_since_flush = 0

    if pending_batches:
        flush_partial()

    payload = {
        "rows": rows,
        "pair_model": val_pair_stats,
        "num_frames": len(frames),
        "num_sequences": len(train_subset["sequences"]),
        "pair_bank_size": len(train_pair_bank),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    if partial_path.exists():
        partial_path.unlink()
    print(
        f"wrote {out_path} rows={len(rows)} frames={len(frames)} "
        f"sequences={len(train_subset['sequences'])} bank={len(train_pair_bank)}"
    )


if __name__ == "__main__":
    main()
