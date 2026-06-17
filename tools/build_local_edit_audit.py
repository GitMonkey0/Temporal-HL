#!/usr/bin/env python3
"""Audit local editability of symbolic fields vs opaque token proxies."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.cluster import KMeans

from tools.build_learned_token_proxy_report import (
    build_semantic_frame_vocab,
    collect_frame_matrix,
    collect_semantic_frame_matrix,
    frame_vector,
    frame_token_set,
    overlap_labels,
)


ROOT = Path("/opt/tiger/hand")
GEN = ROOT / "experiments/generated"
SUM = GEN / "summary_tables"


TRACKED_FIELDS = [
    "hand_type",
    "interaction_motion",
    "right_hand_motion",
    "left_hand_motion",
    "right_state_signature",
    "left_state_signature",
]


TASKS = [
    ("right_hand_motion", "opening"),
    ("right_hand_motion", "closing"),
    ("left_hand_motion", "opening"),
    ("left_hand_motion", "closing"),
    ("interaction_motion", "approach"),
    ("interaction_motion", "separate"),
]


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def fmt(x: float) -> str:
    return f"{x:.4f}"


def pack(values):
    if values is None:
        return "none"
    return "|".join(str(v) for v in values)


def frame_attrs(frame: dict[str, object]) -> dict[str, str]:
    right = frame.get("right")
    left = frame.get("left")
    return {
        "hand_type": str(frame.get("hand_type", "unknown")),
        "interaction_motion": str(frame.get("interaction_motion", "unknown")),
        "right_hand_motion": "none" if right is None else str(right.get("hand_motion", "unknown")),
        "left_hand_motion": "none" if left is None else str(left.get("hand_motion", "unknown")),
        "right_state_signature": "none" if right is None else pack(right.get("token_labels")),
        "left_state_signature": "none" if left is None else pack(left.get("token_labels")),
    }


def contiguous_runs(frames, field: str):
    runs = []
    start = 0
    prev = frame_attrs(frames[0])[field]
    for idx in range(1, len(frames)):
        cur = frame_attrs(frames[idx])[field]
        if cur != prev:
            runs.append((start, idx, prev))
            start = idx
            prev = cur
    runs.append((start, len(frames), prev))
    return runs


def eligible_value(field: str, value: str, target: str) -> bool:
    if value == target:
        return False
    if value in {"unknown", "start"}:
        return False
    if field.endswith("hand_motion") and value == "none":
        return False
    if field == "interaction_motion" and value == "unknown":
        return False
    return True


def symbolic_edit(attrs: dict[str, str], field: str, target: str):
    out = dict(attrs)
    out[field] = target
    return out


def collateral_count(original: dict[str, str], edited: dict[str, str], target_field: str):
    return sum(
        1
        for key in TRACKED_FIELDS
        if key != target_field and edited[key] != original[key]
    )


def clean_edit(original: dict[str, str], edited: dict[str, str], target_field: str):
    return collateral_count(original, edited, target_field) == 0


def cluster_majority_attrs(assignments, attrs_list, n_clusters: int):
    by_cluster = defaultdict(list)
    for cluster_id, attrs in zip(assignments, attrs_list):
        by_cluster[int(cluster_id)].append(attrs)
    out = {}
    for cluster_id in range(n_clusters):
        rows = by_cluster[cluster_id]
        decoded = {}
        for field in TRACKED_FIELDS:
            counter = Counter(row[field] for row in rows)
            decoded[field] = counter.most_common(1)[0][0] if counter else "unknown"
        out[cluster_id] = decoded
    return out


def best_proxy_edit(original_attrs, target_field, target_value, cluster_decoded):
    candidates = [
        (cluster_id, attrs)
        for cluster_id, attrs in cluster_decoded.items()
        if attrs[target_field] == target_value
    ]
    if not candidates:
        return None
    scored = []
    for cluster_id, attrs in candidates:
        coll = collateral_count(original_attrs, attrs, target_field)
        scored.append((coll, cluster_id, attrs))
    scored.sort(key=lambda x: (x[0], x[1]))
    return scored[0][2]


def evaluate_source(test_data, labels, cluster_decoded):
    rows = []
    for sequence in test_data["sequences"]:
        label = sequence["seq_name"]
        if label not in labels and label.replace("ROM07_RT_", "ROM07_Rt_").replace("ROM08_LT_", "ROM08_Lt_") not in labels:
            continue
        frames = sequence["frames"]
        for field, target in TASKS:
            runs = contiguous_runs(frames, field)
            for start, end, value in runs:
                if end - start < 3:
                    continue
                if not eligible_value(field, value, target):
                    continue
                symbolic_clean = 0
                symbolic_coll = 0
                proxy_clean = 0
                proxy_coll = 0
                proxy_success = 0
                count = 0
                for idx in range(start, end):
                    attrs = frame_attrs(frames[idx])
                    symbolic_attrs = symbolic_edit(attrs, field, target)
                    symbolic_clean += int(clean_edit(attrs, symbolic_attrs, field))
                    symbolic_coll += collateral_count(attrs, symbolic_attrs, field)
                    proxy_attrs = best_proxy_edit(attrs, field, target, cluster_decoded)
                    if proxy_attrs is not None:
                        proxy_success += int(proxy_attrs[field] == target)
                        proxy_clean += int(clean_edit(attrs, proxy_attrs, field))
                        proxy_coll += collateral_count(attrs, proxy_attrs, field)
                    count += 1
                rows.append(
                    {
                        "seq_name": sequence["seq_name"],
                        "task_field": field,
                        "task_target": target,
                        "run_length": end - start,
                        "num_frames": count,
                        "symbolic_clean_edit_rate": symbolic_clean / max(count, 1),
                        "symbolic_mean_collateral_fields": symbolic_coll / max(count, 1),
                        "proxy_clean_edit_rate": proxy_clean / max(count, 1),
                        "proxy_mean_collateral_fields": proxy_coll / max(count, 1),
                        "proxy_target_success_rate": proxy_success / max(count, 1),
                    }
                )
    return rows


def summarize(rows):
    by_task = defaultdict(list)
    for row in rows:
        by_task[(row["task_field"], row["task_target"])].append(row)
    out = []
    for (field, target), items in sorted(by_task.items()):
        out.append(
            {
                "task_field": field,
                "task_target": target,
                "num_segments": len(items),
                "num_frames": sum(row["num_frames"] for row in items),
                "symbolic_clean_edit_rate": sum(row["symbolic_clean_edit_rate"] * row["num_frames"] for row in items)
                / max(sum(row["num_frames"] for row in items), 1),
                "symbolic_mean_collateral_fields": sum(row["symbolic_mean_collateral_fields"] * row["num_frames"] for row in items)
                / max(sum(row["num_frames"] for row in items), 1),
                "proxy_clean_edit_rate": sum(row["proxy_clean_edit_rate"] * row["num_frames"] for row in items)
                / max(sum(row["num_frames"] for row in items), 1),
                "proxy_mean_collateral_fields": sum(row["proxy_mean_collateral_fields"] * row["num_frames"] for row in items)
                / max(sum(row["num_frames"] for row in items), 1),
                "proxy_target_success_rate": sum(row["proxy_target_success_rate"] * row["num_frames"] for row in items)
                / max(sum(row["num_frames"] for row in items), 1),
            }
        )
    return out


def main():
    train_data = load_json(GEN / "temporal_hl_val.json")
    test_data = load_json(GEN / "temporal_hl_test.json")
    labels = overlap_labels(train_data, test_data)

    semantic_vocab = build_semantic_frame_vocab(train_data, labels)
    semantic_train_matrix = collect_semantic_frame_matrix(train_data, labels, semantic_vocab)
    continuous_train_matrix = collect_frame_matrix(train_data, labels)

    payload = {
        "artifacts": {
            "train_json": str(GEN / "temporal_hl_val.json"),
            "test_json": str(GEN / "temporal_hl_test.json"),
        },
        "sources": [],
    }

    for source_name, source_matrix, encode_frame in [
        ("semantic_frame", semantic_train_matrix, None),
        ("continuous_frame", continuous_train_matrix, None),
    ]:
        kmeans = KMeans(n_clusters=32, random_state=0, n_init=10)
        kmeans.fit(source_matrix)

        train_attrs = []
        assignments = []
        for sequence in train_data["sequences"]:
            label = sequence["seq_name"]
            canon = label
            if canon not in labels and canon.replace("ROM07_RT_", "ROM07_Rt_").replace("ROM08_LT_", "ROM08_Lt_") not in labels:
                continue
            for frame in sequence["frames"]:
                train_attrs.append(frame_attrs(frame))
                if source_name == "semantic_frame":
                    vec = np.zeros(len(semantic_vocab), dtype=np.float32)
                    token_to_idx = {tok: idx for idx, tok in enumerate(semantic_vocab)}
                    for tok in frame_token_set(frame, mode="temporal", include_persistence=True):
                        idx = token_to_idx.get(tok)
                        if idx is not None:
                            vec[idx] = 1.0
                    assignments.append(int(kmeans.predict(vec.reshape(1, -1))[0]))
                else:
                    vec = np.asarray(frame_vector(frame), dtype=np.float32)
                    assignments.append(int(kmeans.predict(vec.reshape(1, -1))[0]))
        cluster_decoded = cluster_majority_attrs(assignments, train_attrs, 32)
        raw_rows = evaluate_source(test_data, labels, cluster_decoded)
        payload["sources"].append(
            {
                "source": source_name,
                "num_clusters": 32,
                "task_summary": summarize(raw_rows),
                "raw_rows": raw_rows,
            }
        )

    out_json = GEN / "local_edit_audit.json"
    out_md = SUM / "local_edit_audit.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Local Edit Audit",
        "",
        "This is an experiment memo, not paper text.",
    ]
    for source in payload["sources"]:
        lines.extend([
            "",
            f"## Source: {source['source']}",
            "",
            "| task | segments | frames | symbolic clean | symbolic collateral | proxy clean | proxy collateral | proxy target success |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in source["task_summary"]:
            task = f"{row['task_field']}->{row['task_target']}"
            lines.append(
                f"| {task} | {row['num_segments']} | {row['num_frames']} | {fmt(row['symbolic_clean_edit_rate'])} | {fmt(row['symbolic_mean_collateral_fields'])} | {fmt(row['proxy_clean_edit_rate'])} | {fmt(row['proxy_mean_collateral_fields'])} | {fmt(row['proxy_target_success_rate'])} |"
            )
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    import numpy as np

    main()
