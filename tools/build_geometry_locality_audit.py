#!/usr/bin/env python3
"""Geometry-aware locality audit for symbolic edits vs opaque token proxies."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from tools.build_learned_token_proxy_report import (
    build_semantic_frame_vocab,
    collect_frame_matrix,
    collect_semantic_frame_matrix,
    frame_token_set,
    frame_vector,
    overlap_labels,
)
from tools.build_local_edit_audit import (
    TASKS,
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
GEOMETRY_TASKS = [
    ("interaction_motion", "approach"),
    ("interaction_motion", "separate"),
]


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def fmt(x: float) -> str:
    return f"{x:.4f}"


def canonical(seq_name: str) -> str:
    return seq_name.replace("ROM07_RT_", "ROM07_Rt_").replace("ROM08_LT_", "ROM08_Lt_")


def hand_geom(frame: dict[str, object], hand_name: str):
    rec = frame.get(hand_name)
    if rec is None:
        return None
    vecs = np.asarray(rec.get("local_vectors", []), dtype=np.float32)
    flex = rec.get("flexion_scores", {})
    flex_arr = np.asarray(
        [float(flex.get(k, 0.0)) for k in ("thumb", "index", "middle", "ring", "pinky")],
        dtype=np.float32,
    )
    return {
        "local_vectors": vecs,
        "flexion": flex_arr,
        "hand_motion": str(rec.get("hand_motion", "unknown")),
    }


def frame_geom(frame: dict[str, object]):
    return {
        "right": hand_geom(frame, "right"),
        "left": hand_geom(frame, "left"),
        "cross_hand_distance": None if frame.get("cross_hand_distance") is None else float(frame.get("cross_hand_distance")),
        "interaction_motion": str(frame.get("interaction_motion", "unknown")),
        "hand_type": str(frame.get("hand_type", "unknown")),
    }


def mean_local_delta(a, b):
    if a is None or b is None:
        return 0.0
    va = a["local_vectors"]
    vb = b["local_vectors"]
    if va.size == 0 or vb.size == 0:
        return 0.0
    return float(np.mean(np.linalg.norm(va - vb, axis=1)))


def mean_flex_delta(a, b):
    if a is None or b is None:
        return 0.0
    return float(np.mean(np.abs(a["flexion"] - b["flexion"])))


def hand_delta(a, b):
    return mean_local_delta(a, b) + mean_flex_delta(a, b)


def interaction_delta(a, b):
    da = a["cross_hand_distance"]
    db = b["cross_hand_distance"]
    if da is None or db is None:
        return 0.0
    return abs(da - db) / 100.0


def edit_target_and_collateral_delta(original_geom, edited_geom, task_field: str):
    if task_field == "right_hand_motion":
        return {
            "target_delta": hand_delta(original_geom["right"], edited_geom["right"]),
            "collateral_delta": hand_delta(original_geom["left"], edited_geom["left"]) + interaction_delta(original_geom, edited_geom),
        }
    if task_field == "left_hand_motion":
        return {
            "target_delta": hand_delta(original_geom["left"], edited_geom["left"]),
            "collateral_delta": hand_delta(original_geom["right"], edited_geom["right"]) + interaction_delta(original_geom, edited_geom),
        }
    return {
        "target_delta": interaction_delta(original_geom, edited_geom),
        "collateral_delta": hand_delta(original_geom["right"], edited_geom["right"]) + hand_delta(original_geom["left"], edited_geom["left"]),
    }


def source_cluster_decoded(source_name, train_frames, semantic_vocab):
    if source_name == "semantic_frame":
        matrix = np.asarray([row["semantic_vec"] for row in train_frames], dtype=np.float32)
    else:
        matrix = np.asarray([row["continuous_vec"] for row in train_frames], dtype=np.float32)
    kmeans = KMeans(n_clusters=32, random_state=0, n_init=10)
    kmeans.fit(matrix)
    assignments = list(map(int, kmeans.labels_))
    attrs = [row["attrs"] for row in train_frames]
    decoded = cluster_majority_attrs(assignments, attrs, 32)
    for row, cid in zip(train_frames, assignments):
        row[f"{source_name}_cluster"] = cid
    return decoded


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


def candidate_symbolic_frames(train_bank, original_attrs, task_field, target_value):
    out = []
    for row in train_bank:
        attrs = row["attrs"]
        if attrs[task_field] != target_value:
            continue
        ok = True
        for field in TRACKED_FIELDS:
            if field == task_field:
                continue
            if attrs[field] != original_attrs[field]:
                ok = False
                break
        if ok:
            out.append(row)
    return out


def geometry_distance(original_geom, candidate_geom):
    return (
        hand_delta(original_geom["right"], candidate_geom["right"])
        + hand_delta(original_geom["left"], candidate_geom["left"])
        + interaction_delta(original_geom, candidate_geom)
    )


def pick_best_symbolic_candidate(train_bank, original_attrs, original_geom, task_field, target_value):
    cands = candidate_symbolic_frames(train_bank, original_attrs, task_field, target_value)
    if not cands:
        return None
    cands.sort(key=lambda row: (geometry_distance(original_geom, row["geom"]), row["seq_name"], row["frame_idx"]))
    return cands[0]


def pick_best_proxy_candidate(train_bank, source_name, cluster_decoded, original_attrs, original_geom, task_field, target_value):
    best_attrs = best_proxy_edit(original_attrs, task_field, target_value, cluster_decoded)
    if best_attrs is None:
        return None
    # candidate frames from clusters whose decoded attrs equal the chosen proxy attrs signature
    matched_clusters = []
    for cid, attrs in cluster_decoded.items():
        if attrs == best_attrs:
            matched_clusters.append(cid)
    cands = [row for row in train_bank if row[f"{source_name}_cluster"] in matched_clusters and row["attrs"][task_field] == target_value]
    if not cands:
        return None
    cands.sort(key=lambda row: (geometry_distance(original_geom, row["geom"]), row["seq_name"], row["frame_idx"]))
    return cands[0]


def summarize(rows):
    by_task = defaultdict(list)
    for row in rows:
        by_task[f"{row['task_field']}->{row['task_target']}"].append(row)
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
            }
        )
    return out


def main():
    train_data = load_json(GEN / "temporal_hl_val.json")
    test_data = load_json(GEN / "temporal_hl_test.json")
    labels = overlap_labels(train_data, test_data)
    semantic_vocab = build_semantic_frame_vocab(train_data, labels)
    train_bank = build_train_frame_bank(train_data, labels, semantic_vocab)

    semantic_decoded = source_cluster_decoded("semantic_frame", train_bank, semantic_vocab)
    continuous_decoded = source_cluster_decoded("continuous_frame", train_bank, semantic_vocab)

    payload = {
        "artifacts": {
            "train_json": str(GEN / "temporal_hl_val.json"),
            "test_json": str(GEN / "temporal_hl_test.json"),
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
            for task_field, task_target in GEOMETRY_TASKS:
                for start, end, value in contiguous_runs(frames, task_field):
                    if end - start < 3 or not eligible_value(task_field, value, task_target):
                        continue
                    for idx in range(start, end):
                        frame = frames[idx]
                        attrs = frame_attrs(frame)
                        geom = frame_geom(frame)
                        sym = pick_best_symbolic_candidate(train_bank, attrs, geom, task_field, task_target)
                        prox = pick_best_proxy_candidate(train_bank, source_name, decoded, attrs, geom, task_field, task_target)
                        if sym is None or prox is None:
                            continue
                        sym_delta = edit_target_and_collateral_delta(geom, sym["geom"], task_field)
                        prox_delta = edit_target_and_collateral_delta(geom, prox["geom"], task_field)
                        rows.append(
                            {
                                "seq_name": sequence["seq_name"],
                                "frame_idx": frame["frame_idx"],
                                "task_field": task_field,
                                "task_target": task_target,
                                "symbolic_target_delta": sym_delta["target_delta"],
                                "symbolic_collateral_delta": sym_delta["collateral_delta"],
                                "proxy_target_delta": prox_delta["target_delta"],
                                "proxy_collateral_delta": prox_delta["collateral_delta"],
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

    out_json = GEN / "geometry_locality_audit.json"
    out_md = SUM / "geometry_locality_audit.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Geometry Locality Audit",
        "",
        "This is an experiment memo, not paper text.",
    ]
    for source in payload["sources"]:
        lines.extend([
            "",
            f"## Source: {source['source']}",
            "",
            "| task | frames | symbolic target delta | symbolic collateral | symbolic locality | proxy target delta | proxy collateral | proxy locality |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in source["summary"]:
            lines.append(
                f"| {row['task']} | {row['num_frames']} | {fmt(row['symbolic_target_delta'])} | {fmt(row['symbolic_collateral_delta'])} | {fmt(row['symbolic_locality_ratio'])} | {fmt(row['proxy_target_delta'])} | {fmt(row['proxy_collateral_delta'])} | {fmt(row['proxy_locality_ratio'])} |"
            )
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
GEOMETRY_TASKS = [
    ("interaction_motion", "approach"),
    ("interaction_motion", "separate"),
]
