#!/usr/bin/env python3
"""Diagnose why strict hand-motion geometry matching is unstable."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from tools.build_learned_token_proxy_report import (
    build_semantic_frame_vocab,
    frame_token_set,
    frame_vector,
    overlap_labels,
)
from tools.build_local_edit_audit import (
    TRACKED_FIELDS,
    best_proxy_edit,
    cluster_majority_attrs,
    contiguous_runs,
    eligible_value,
    frame_attrs,
)


ROOT = Path("/opt/tiger/hand")
GEN = ROOT / "experiments/generated"
SUM = GEN / "summary_tables"
TASKS = [
    ("right_hand_motion", "opening"),
    ("right_hand_motion", "closing"),
    ("left_hand_motion", "opening"),
    ("left_hand_motion", "closing"),
]


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def canonical(seq_name: str) -> str:
    return seq_name.replace("ROM07_RT_", "ROM07_Rt_").replace("ROM08_LT_", "ROM08_Lt_")


def build_train_bank(train_data, labels, semantic_vocab):
    token_to_idx = {tok: idx for idx, tok in enumerate(semantic_vocab)}
    rows = []
    for sequence in train_data["sequences"]:
        if sequence["seq_name"] not in labels and canonical(sequence["seq_name"]) not in labels:
            continue
        for frame in sequence["frames"]:
            semantic_vec = np.zeros(len(semantic_vocab), dtype=np.float32)
            for tok in frame_token_set(frame, mode="temporal", include_persistence=True):
                idx = token_to_idx.get(tok)
                if idx is not None:
                    semantic_vec[idx] = 1.0
            rows.append(
                {
                    "seq_name": sequence["seq_name"],
                    "frame_idx": frame["frame_idx"],
                    "attrs": frame_attrs(frame),
                    "semantic_vec": semantic_vec,
                    "continuous_vec": np.asarray(frame_vector(frame), dtype=np.float32),
                }
            )
    return rows


def candidate_symbolic_frames(train_bank, original_attrs, task_field, target_value):
    out = []
    mismatch_counter = Counter()
    for row in train_bank:
        attrs = row["attrs"]
        if attrs[task_field] != target_value:
            continue
        mismatches = [
            field for field in TRACKED_FIELDS
            if field != task_field and attrs[field] != original_attrs[field]
        ]
        if not mismatches:
            out.append(row)
        else:
            mismatch_counter.update(mismatches)
    return out, mismatch_counter


def source_cluster_decoded(source_name, train_bank, semantic_vocab):
    if source_name == "semantic_frame":
        matrix = np.asarray([row["semantic_vec"] for row in train_bank], dtype=np.float32)
    else:
        matrix = np.asarray([row["continuous_vec"] for row in train_bank], dtype=np.float32)
    kmeans = KMeans(n_clusters=32, random_state=0, n_init=10)
    kmeans.fit(matrix)
    decoded = cluster_majority_attrs(list(map(int, kmeans.labels_)), [row["attrs"] for row in train_bank], 32)
    return decoded


def main():
    train_data = load_json(GEN / "temporal_hl_val.json")
    test_data = load_json(GEN / "temporal_hl_test.json")
    labels = overlap_labels(train_data, test_data)
    semantic_vocab = build_semantic_frame_vocab(train_data, labels)
    train_bank = build_train_bank(train_data, labels, semantic_vocab)
    semantic_decoded = source_cluster_decoded("semantic_frame", train_bank, semantic_vocab)
    continuous_decoded = source_cluster_decoded("continuous_frame", train_bank, semantic_vocab)

    rows = []
    for sequence in test_data["sequences"]:
        if sequence["seq_name"] not in labels and canonical(sequence["seq_name"]) not in labels:
            continue
        for task_field, task_target in TASKS:
            for start, end, value in contiguous_runs(sequence["frames"], task_field):
                if end - start < 3 or not eligible_value(task_field, value, task_target):
                    continue
                for idx in range(start, end):
                    attrs = frame_attrs(sequence["frames"][idx])
                    symbolic_cands, mismatch_counter = candidate_symbolic_frames(train_bank, attrs, task_field, task_target)
                    sem_proxy = best_proxy_edit(attrs, task_field, task_target, semantic_decoded)
                    cont_proxy = best_proxy_edit(attrs, task_field, task_target, continuous_decoded)
                    rows.append(
                        {
                            "seq_name": sequence["seq_name"],
                            "frame_idx": sequence["frames"][idx]["frame_idx"],
                            "task": f"{task_field}->{task_target}",
                            "num_strict_symbolic_candidates": len(symbolic_cands),
                            "semantic_proxy_available": sem_proxy is not None,
                            "continuous_proxy_available": cont_proxy is not None,
                            "top_mismatch_fields": mismatch_counter.most_common(4),
                        }
                    )

    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)

    summary = []
    for task, items in sorted(by_task.items()):
        summary.append(
            {
                "task": task,
                "num_frames": len(items),
                "frames_with_any_strict_symbolic_candidate": sum(row["num_strict_symbolic_candidates"] > 0 for row in items),
                "mean_num_strict_symbolic_candidates": sum(row["num_strict_symbolic_candidates"] for row in items) / max(len(items), 1),
                "semantic_proxy_available_rate": sum(row["semantic_proxy_available"] for row in items) / max(len(items), 1),
                "continuous_proxy_available_rate": sum(row["continuous_proxy_available"] for row in items) / max(len(items), 1),
                "top_blocking_fields": Counter(
                    field
                    for row in items
                    for field, _ in row["top_mismatch_fields"]
                ).most_common(5),
            }
        )

    payload = {
        "artifacts": {
            "train_json": str(GEN / "temporal_hl_val.json"),
            "test_json": str(GEN / "temporal_hl_test.json"),
        },
        "summary": summary,
    }

    out_json = GEN / "hand_motion_geometry_diagnostic.json"
    out_md = SUM / "hand_motion_geometry_diagnostic.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Hand-Motion Geometry Diagnostic",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "| task | frames | frames with strict symbolic cand | mean strict symbolic cands | semantic proxy avail | continuous proxy avail | top blocking fields |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary:
        lines.append(
            f"| {row['task']} | {row['num_frames']} | {row['frames_with_any_strict_symbolic_candidate']} | "
            f"{row['mean_num_strict_symbolic_candidates']:.2f} | {row['semantic_proxy_available_rate']:.4f} | "
            f"{row['continuous_proxy_available_rate']:.4f} | {row['top_blocking_fields']} |"
        )
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
