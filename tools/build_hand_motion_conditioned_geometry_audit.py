#!/usr/bin/env python3
"""Conditioned geometry-locality audit for hand-motion edits."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from tools.build_geometry_locality_audit import (
    edit_target_and_collateral_delta,
    fmt,
    frame_geom,
    geometry_distance,
    hand_delta,
    interaction_delta,
)
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


def preserved_fields(task_field: str) -> list[str]:
    if task_field == "right_hand_motion":
        return [
            "hand_type",
            "interaction_motion",
            "left_hand_motion",
            "left_state_signature",
        ]
    if task_field == "left_hand_motion":
        return [
            "hand_type",
            "interaction_motion",
            "right_hand_motion",
            "right_state_signature",
        ]
    raise ValueError(task_field)


def build_train_frame_bank(train_data, labels, semantic_vocab):
    token_to_idx = {tok: idx for idx, tok in enumerate(semantic_vocab)}
    bank = []
    for sequence in train_data["sequences"]:
        if canonical(sequence["seq_name"]) not in labels and sequence["seq_name"] not in labels:
            continue
        for frame in sequence["frames"]:
            semantic_vec = np.zeros(len(semantic_vocab), dtype=np.float32)
            for tok in frame_token_set(frame, mode="temporal", include_persistence=True):
                idx = token_to_idx.get(tok)
                if idx is not None:
                    semantic_vec[idx] = 1.0
            bank.append(
                {
                    "seq_name": sequence["seq_name"],
                    "frame_idx": frame["frame_idx"],
                    "attrs": frame_attrs(frame),
                    "geom": frame_geom(frame),
                    "semantic_vec": semantic_vec,
                    "continuous_vec": np.asarray(frame_vector(frame), dtype=np.float32),
                }
            )
    return bank


def conditioned_symbolic_candidates(train_bank, original_attrs, task_field, target_value):
    keep = preserved_fields(task_field)
    out = []
    for row in train_bank:
        attrs = row["attrs"]
        if attrs[task_field] != target_value:
            continue
        if any(attrs[field] != original_attrs[field] for field in keep):
            continue
        out.append(row)
    return out


def pick_best_conditioned_symbolic(train_bank, original_attrs, original_geom, task_field, target_value):
    cands = conditioned_symbolic_candidates(train_bank, original_attrs, task_field, target_value)
    if not cands:
        return None
    cands.sort(key=lambda row: (geometry_distance(original_geom, row["geom"]), row["seq_name"], row["frame_idx"]))
    return cands[0]


def conditioned_proxy_attrs(original_attrs, task_field, target_value, cluster_decoded):
    keep = set(preserved_fields(task_field))
    candidates = []
    for cid, attrs in cluster_decoded.items():
        if attrs[task_field] != target_value:
            continue
        preserved_mismatch = sum(1 for field in keep if attrs[field] != original_attrs[field])
        total_collateral = sum(
            1
            for field in TRACKED_FIELDS
            if field != task_field and attrs[field] != original_attrs[field]
        )
        candidates.append((preserved_mismatch, total_collateral, cid, attrs))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    return candidates[0][3]


def source_cluster_decoded(source_name, train_frames):
    if source_name == "semantic_frame":
        matrix = np.asarray([row["semantic_vec"] for row in train_frames], dtype=np.float32)
    else:
        matrix = np.asarray([row["continuous_vec"] for row in train_frames], dtype=np.float32)
    kmeans = KMeans(n_clusters=32, random_state=0, n_init=10)
    kmeans.fit(matrix)
    assignments = list(map(int, kmeans.labels_))
    decoded = cluster_majority_attrs(assignments, [row["attrs"] for row in train_frames], 32)
    for row, cid in zip(train_frames, assignments):
        row[f"{source_name}_cluster"] = cid
    return decoded


def pick_best_conditioned_proxy(train_bank, source_name, cluster_decoded, original_attrs, original_geom, task_field, target_value):
    chosen = conditioned_proxy_attrs(original_attrs, task_field, target_value, cluster_decoded)
    if chosen is None:
        return None, None
    matched_clusters = [cid for cid, attrs in cluster_decoded.items() if attrs == chosen]
    cands = [row for row in train_bank if row[f"{source_name}_cluster"] in matched_clusters and row["attrs"][task_field] == target_value]
    keep = preserved_fields(task_field)
    strict_cands = [row for row in cands if all(row["attrs"][field] == original_attrs[field] for field in keep)]
    usable = strict_cands if strict_cands else cands
    if not usable:
        return chosen, None
    usable.sort(key=lambda row: (geometry_distance(original_geom, row["geom"]), row["seq_name"], row["frame_idx"]))
    return chosen, usable[0]


def summarize(rows):
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    out = []
    for task, items in sorted(by_task.items()):
        n = len(items)
        sym_target = sum(row["symbolic_target_delta"] for row in items) / max(n, 1)
        sym_coll = sum(row["symbolic_collateral_delta"] for row in items) / max(n, 1)
        prox_target = sum(row["proxy_target_delta"] for row in items) / max(n, 1)
        prox_coll = sum(row["proxy_collateral_delta"] for row in items) / max(n, 1)
        out.append(
            {
                "task": task,
                "num_frames": n,
                "symbolic_target_delta": sym_target,
                "symbolic_collateral_delta": sym_coll,
                "symbolic_locality_ratio": sym_target / max(sym_coll, 1e-6),
                "proxy_target_delta": prox_target,
                "proxy_collateral_delta": prox_coll,
                "proxy_locality_ratio": prox_target / max(prox_coll, 1e-6),
                "proxy_conditioned_clean_rate": sum(row["proxy_conditioned_clean"] for row in items) / max(n, 1),
                "proxy_target_success_rate": sum(row["proxy_target_success"] for row in items) / max(n, 1),
                "proxy_semantic_collateral_fields": sum(row["proxy_semantic_collateral_fields"] for row in items) / max(n, 1),
                "symbolic_candidate_rate": sum(row["symbolic_has_candidate"] for row in items) / max(n, 1),
            }
        )
    return out


def main():
    train_data = load_json(GEN / "temporal_hl_val.json")
    test_data = load_json(GEN / "temporal_hl_test.json")
    labels = overlap_labels(train_data, test_data)
    semantic_vocab = build_semantic_frame_vocab(train_data, labels)
    train_bank = build_train_frame_bank(train_data, labels, semantic_vocab)

    semantic_decoded = source_cluster_decoded("semantic_frame", train_bank)
    continuous_decoded = source_cluster_decoded("continuous_frame", train_bank)

    payload = {
        "artifacts": {
            "train_json": str(GEN / "temporal_hl_val.json"),
            "test_json": str(GEN / "temporal_hl_test.json"),
        },
        "conditioning_rule": {
            "right_hand_motion": preserved_fields("right_hand_motion"),
            "left_hand_motion": preserved_fields("left_hand_motion"),
        },
        "sources": [],
    }

    for source_name, decoded in [
        ("semantic_frame", semantic_decoded),
        ("continuous_frame", continuous_decoded),
    ]:
        rows = []
        for sequence in test_data["sequences"]:
            if canonical(sequence["seq_name"]) not in labels and sequence["seq_name"] not in labels:
                continue
            frames = sequence["frames"]
            for task_field, task_target in TASKS:
                for start, end, value in contiguous_runs(frames, task_field):
                    if end - start < 3 or not eligible_value(task_field, value, task_target):
                        continue
                    for idx in range(start, end):
                        frame = frames[idx]
                        attrs = frame_attrs(frame)
                        geom = frame_geom(frame)
                        sym = pick_best_conditioned_symbolic(train_bank, attrs, geom, task_field, task_target)
                        proxy_attrs, prox = pick_best_conditioned_proxy(train_bank, source_name, decoded, attrs, geom, task_field, task_target)
                        if sym is None or prox is None or proxy_attrs is None:
                            continue
                        sym_delta = edit_target_and_collateral_delta(geom, sym["geom"], task_field)
                        prox_delta = edit_target_and_collateral_delta(geom, prox["geom"], task_field)
                        keep = preserved_fields(task_field)
                        proxy_semantic_collateral = sum(
                            1
                            for field in TRACKED_FIELDS
                            if field != task_field and proxy_attrs[field] != attrs[field]
                        )
                        rows.append(
                            {
                                "task": f"{task_field}->{task_target}",
                                "seq_name": sequence["seq_name"],
                                "frame_idx": frame["frame_idx"],
                                "symbolic_target_delta": sym_delta["target_delta"],
                                "symbolic_collateral_delta": sym_delta["collateral_delta"],
                                "proxy_target_delta": prox_delta["target_delta"],
                                "proxy_collateral_delta": prox_delta["collateral_delta"],
                                "proxy_conditioned_clean": int(all(proxy_attrs[field] == attrs[field] for field in keep)),
                                "proxy_target_success": int(proxy_attrs[task_field] == task_target),
                                "proxy_semantic_collateral_fields": proxy_semantic_collateral,
                                "symbolic_has_candidate": 1,
                                "symbolic_source_seq": sym["seq_name"],
                                "symbolic_source_frame": sym["frame_idx"],
                                "proxy_source_seq": prox["seq_name"],
                                "proxy_source_frame": prox["frame_idx"],
                            }
                        )
        payload["sources"].append(
            {
                "source": source_name,
                "summary": summarize(rows),
                "rows": rows,
            }
        )

    out_json = GEN / "hand_motion_conditioned_geometry_audit.json"
    out_md = SUM / "hand_motion_conditioned_geometry_audit.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Hand-Motion Conditioned Geometry Audit",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Conditioning rule:",
        f"- `right_hand_motion`: preserve {preserved_fields('right_hand_motion')}",
        f"- `left_hand_motion`: preserve {preserved_fields('left_hand_motion')}",
    ]
    for source in payload["sources"]:
        lines.extend([
            "",
            f"## Source: {source['source']}",
            "",
            "| task | frames | symbolic target delta | symbolic collateral | symbolic locality | proxy target delta | proxy collateral | proxy locality | proxy conditioned clean | proxy semantic collateral | proxy target success |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in source["summary"]:
            lines.append(
                f"| {row['task']} | {row['num_frames']} | {fmt(row['symbolic_target_delta'])} | {fmt(row['symbolic_collateral_delta'])} | "
                f"{fmt(row['symbolic_locality_ratio'])} | {fmt(row['proxy_target_delta'])} | {fmt(row['proxy_collateral_delta'])} | "
                f"{fmt(row['proxy_locality_ratio'])} | {fmt(row['proxy_conditioned_clean_rate'])} | {fmt(row['proxy_semantic_collateral_fields'])} | {fmt(row['proxy_target_success_rate'])} |"
            )
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
