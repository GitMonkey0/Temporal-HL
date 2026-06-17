#!/usr/bin/env python3
"""Transition-conditioned donor-pair audit for hand-motion edits."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from tools.build_conditional_hand_transplant_audit import (
    TASKS,
    canonical,
    preserved_fields,
    target_hand_name,
)
from tools.build_geometry_locality_audit import frame_geom, hand_delta
from tools.build_learned_token_proxy_report import (
    build_semantic_frame_vocab,
    frame_token_set,
    frame_vector,
    overlap_labels,
)
from tools.build_local_edit_audit import TRACKED_FIELDS, cluster_majority_attrs, contiguous_runs, eligible_value, frame_attrs
from tools.build_temporal_hl import summarize_flexion_transition


ROOT = Path("/opt/tiger/hand")
GEN = ROOT / "experiments/generated"
SUM = GEN / "summary_tables"
FLEX_THRESHOLD = 0.05


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def fmt(x: float) -> str:
    return f"{x:.4f}"


def build_pair_bank(train_data, labels, semantic_vocab):
    token_to_idx = {tok: idx for idx, tok in enumerate(semantic_vocab)}
    bank = []
    for sequence in train_data["sequences"]:
        if canonical(sequence["seq_name"]) not in labels and sequence["seq_name"] not in labels:
            continue
        frames = sequence["frames"]
        for idx in range(1, len(frames)):
            prev_frame = frames[idx - 1]
            curr_frame = frames[idx]
            semantic_vec = np.zeros(len(semantic_vocab), dtype=np.float32)
            for tok in frame_token_set(curr_frame, mode="temporal", include_persistence=True):
                j = token_to_idx.get(tok)
                if j is not None:
                    semantic_vec[j] = 1.0
            bank.append(
                {
                    "seq_name": sequence["seq_name"],
                    "prev_frame_idx": prev_frame["frame_idx"],
                    "curr_frame_idx": curr_frame["frame_idx"],
                    "prev_attrs": frame_attrs(prev_frame),
                    "curr_attrs": frame_attrs(curr_frame),
                    "prev_geom": frame_geom(prev_frame),
                    "curr_geom": frame_geom(curr_frame),
                    "prev_right": prev_frame.get("right"),
                    "prev_left": prev_frame.get("left"),
                    "curr_right": curr_frame.get("right"),
                    "curr_left": curr_frame.get("left"),
                    "semantic_vec": semantic_vec,
                    "continuous_vec": np.asarray(frame_vector(curr_frame), dtype=np.float32),
                }
            )
    return bank


def source_cluster_decoded(source_name, pair_bank):
    if source_name == "semantic_frame":
        matrix = np.asarray([row["semantic_vec"] for row in pair_bank], dtype=np.float32)
    else:
        matrix = np.asarray([row["continuous_vec"] for row in pair_bank], dtype=np.float32)
    kmeans = KMeans(n_clusters=32, random_state=0, n_init=10)
    kmeans.fit(matrix)
    assignments = list(map(int, kmeans.labels_))
    decoded = cluster_majority_attrs(assignments, [row["curr_attrs"] for row in pair_bank], 32)
    for row, cid in zip(pair_bank, assignments):
        row[f"{source_name}_cluster"] = cid
    return decoded


def realized_motion(prev_hand: dict[str, object] | None, curr_hand: dict[str, object] | None) -> str:
    if prev_hand is None or curr_hand is None:
        return "unknown"
    return summarize_flexion_transition(
        prev_hand.get("flexion_scores", {}),
        curr_hand.get("flexion_scores", {}),
        FLEX_THRESHOLD,
    )


def pair_realizes_target(row: dict[str, object], task_field: str, target_value: str) -> bool:
    hand_name = target_hand_name(task_field)
    prev_hand = row[f"prev_{hand_name}"]
    curr_hand = row[f"curr_{hand_name}"]
    return realized_motion(prev_hand, curr_hand) == target_value


def pair_distance(test_prev_geom, test_curr_geom, donor_row, task_field: str) -> float:
    hand_name = target_hand_name(task_field)
    return (
        hand_delta(test_prev_geom[hand_name], donor_row["prev_geom"][hand_name])
        + hand_delta(test_curr_geom[hand_name], donor_row["curr_geom"][hand_name])
    )


def symbolic_pair_candidates(pair_bank, curr_attrs, task_field, target_value):
    keep = preserved_fields(task_field)
    out = []
    for row in pair_bank:
        attrs = row["curr_attrs"]
        if attrs[task_field] != target_value:
            continue
        if any(attrs[field] != curr_attrs[field] for field in keep):
            continue
        out.append(row)
    return out


def pick_best_symbolic_pair(pair_bank, curr_attrs, test_prev_geom, test_curr_geom, task_field, target_value):
    cands = [row for row in symbolic_pair_candidates(pair_bank, curr_attrs, task_field, target_value) if pair_realizes_target(row, task_field, target_value)]
    if not cands:
        return None
    cands.sort(key=lambda row: (pair_distance(test_prev_geom, test_curr_geom, row, task_field), row["seq_name"], row["curr_frame_idx"]))
    return cands[0]


def best_proxy_attrs(curr_attrs, task_field, target_value, cluster_decoded):
    keep = preserved_fields(task_field)
    candidates = []
    for cid, attrs in cluster_decoded.items():
        if attrs[task_field] != target_value:
            continue
        preserved_mismatch = sum(1 for field in keep if attrs[field] != curr_attrs[field])
        total_collateral = sum(1 for field in TRACKED_FIELDS if field != task_field and attrs[field] != curr_attrs[field])
        candidates.append((preserved_mismatch, total_collateral, cid, attrs))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    return candidates[0][3]


def pick_best_proxy_pair(pair_bank, source_name, cluster_decoded, proxy_attrs, test_prev_geom, test_curr_geom, task_field, target_value):
    matched_clusters = [cid for cid, attrs in cluster_decoded.items() if attrs == proxy_attrs]
    cands = [
        row for row in pair_bank
        if row[f"{source_name}_cluster"] in matched_clusters
        and row["curr_attrs"][task_field] == target_value
        and pair_realizes_target(row, task_field, target_value)
    ]
    if not cands:
        return None
    cands.sort(key=lambda row: (pair_distance(test_prev_geom, test_curr_geom, row, task_field), row["seq_name"], row["curr_frame_idx"]))
    return cands[0]


def summarize(rows):
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    out = []
    for task, items in sorted(by_task.items()):
        n = len(items)
        out.append(
            {
                "task": task,
                "num_frames": n,
                "symbolic_pair_available_rate": sum(row["symbolic_pair_available"] for row in items) / n,
                "proxy_pair_available_rate": sum(row["proxy_pair_available"] for row in items) / n,
                "symbolic_pair_distance": sum(row["symbolic_pair_distance"] for row in items if row["symbolic_pair_available"]) / max(sum(row["symbolic_pair_available"] for row in items), 1),
                "proxy_pair_distance": sum(row["proxy_pair_distance"] for row in items if row["proxy_pair_available"]) / max(sum(row["proxy_pair_available"] for row in items), 1),
                "symbolic_beats_proxy_rate": sum(row["symbolic_beats_proxy"] for row in items) / n,
                "proxy_preserved_clean_rate": sum(row["proxy_preserved_clean"] for row in items) / n,
                "proxy_total_semantic_collateral": sum(row["proxy_total_semantic_collateral"] for row in items) / n,
            }
        )
    return out


def main():
    train_data = load_json(GEN / "temporal_hl_val.json")
    test_data = load_json(GEN / "temporal_hl_test.json")
    labels = overlap_labels(train_data, test_data)
    semantic_vocab = build_semantic_frame_vocab(train_data, labels)
    pair_bank = build_pair_bank(train_data, labels, semantic_vocab)

    semantic_decoded = source_cluster_decoded("semantic_frame", pair_bank)
    continuous_decoded = source_cluster_decoded("continuous_frame", pair_bank)

    payload = {
        "artifacts": {
            "train_json": str(GEN / "temporal_hl_val.json"),
            "test_json": str(GEN / "temporal_hl_test.json"),
        },
        "flex_threshold": FLEX_THRESHOLD,
        "sources": [],
    }

    for source_name, decoded in [("semantic_frame", semantic_decoded), ("continuous_frame", continuous_decoded)]:
        rows = []
        for sequence in test_data["sequences"]:
            if canonical(sequence["seq_name"]) not in labels and sequence["seq_name"] not in labels:
                continue
            frames = sequence["frames"]
            for task_field, task_target in TASKS:
                for start, end, value in contiguous_runs(frames, task_field):
                    if end - start < 3 or not eligible_value(task_field, value, task_target):
                        continue
                    for idx in range(max(start, 1), end):
                        prev_frame = frames[idx - 1]
                        curr_frame = frames[idx]
                        curr_attrs = frame_attrs(curr_frame)
                        test_prev_geom = frame_geom(prev_frame)
                        test_curr_geom = frame_geom(curr_frame)
                        sym = pick_best_symbolic_pair(pair_bank, curr_attrs, test_prev_geom, test_curr_geom, task_field, task_target)
                        proxy_attrs = best_proxy_attrs(curr_attrs, task_field, task_target, decoded)
                        prox = None if proxy_attrs is None else pick_best_proxy_pair(pair_bank, source_name, decoded, proxy_attrs, test_prev_geom, test_curr_geom, task_field, task_target)
                        proxy_preserved_mismatch = 0 if proxy_attrs is None else sum(1 for field in preserved_fields(task_field) if proxy_attrs[field] != curr_attrs[field])
                        proxy_total_semantic_collateral = 0 if proxy_attrs is None else sum(
                            1 for field in TRACKED_FIELDS if field != task_field and proxy_attrs[field] != curr_attrs[field]
                        )
                        sym_avail = int(sym is not None)
                        prox_avail = int(prox is not None)
                        sym_dist = pair_distance(test_prev_geom, test_curr_geom, sym, task_field) if sym is not None else 0.0
                        prox_dist = pair_distance(test_prev_geom, test_curr_geom, prox, task_field) if prox is not None else 0.0
                        rows.append(
                            {
                                "task": f"{task_field}->{task_target}",
                                "symbolic_pair_available": sym_avail,
                                "proxy_pair_available": prox_avail,
                                "symbolic_pair_distance": sym_dist,
                                "proxy_pair_distance": prox_dist,
                                "symbolic_beats_proxy": int(sym is not None and prox is not None and sym_dist <= prox_dist),
                                "proxy_preserved_clean": int(proxy_preserved_mismatch == 0),
                                "proxy_total_semantic_collateral": proxy_total_semantic_collateral,
                            }
                        )
        payload["sources"].append({"source": source_name, "summary": summarize(rows), "rows": rows})

    out_json = GEN / "transition_conditioned_hand_motion_audit.json"
    out_md = SUM / "transition_conditioned_hand_motion_audit.md"
    out_json.write_text(json.dumps(payload, indent=2))
    lines = [
        "# Transition-Conditioned Hand-Motion Audit",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "This audit compares two-frame donor pairs that already realize the requested hand-motion label.",
    ]
    for source in payload["sources"]:
        lines.extend([
            "",
            f"## Source: {source['source']}",
            "",
            "| task | frames | symbolic pair avail | proxy pair avail | symbolic pair dist | proxy pair dist | symbolic beats proxy | proxy preserved clean | proxy semantic collateral |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in source["summary"]:
            lines.append(
                f"| {row['task']} | {row['num_frames']} | {fmt(row['symbolic_pair_available_rate'])} | {fmt(row['proxy_pair_available_rate'])} | "
                f"{fmt(row['symbolic_pair_distance'])} | {fmt(row['proxy_pair_distance'])} | {fmt(row['symbolic_beats_proxy_rate'])} | "
                f"{fmt(row['proxy_preserved_clean_rate'])} | {fmt(row['proxy_total_semantic_collateral'])} |"
            )
    out_md.write_text('\n'.join(lines) + '\n')
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
