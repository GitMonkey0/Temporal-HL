#!/usr/bin/env python3
"""Build a frame-level casebook for local edit audit examples."""

from __future__ import annotations

import json
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
    clean_edit,
    cluster_majority_attrs,
    contiguous_runs,
    eligible_value,
    frame_attrs,
    symbolic_edit,
)


ROOT = Path("/opt/tiger/hand")
GEN = ROOT / "experiments/generated"
SUM = GEN / "summary_tables"


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def changed_fields(original: dict[str, str], edited: dict[str, str], target_field: str):
    return [
        key for key in TRACKED_FIELDS
        if key != target_field and edited[key] != original[key]
    ]


def collect_candidate_rows(test_data, labels, cluster_decoded):
    rows = []
    for sequence in test_data["sequences"]:
        seq_name = sequence["seq_name"]
        canon = seq_name.replace("ROM07_RT_", "ROM07_Rt_").replace("ROM08_LT_", "ROM08_Lt_")
        if seq_name not in labels and canon not in labels:
            continue
        frames = sequence["frames"]
        for field, target in TASKS:
            for start, end, value in contiguous_runs(frames, field):
                if end - start < 3 or not eligible_value(field, value, target):
                    continue
                coll_total = 0
                clean_total = 0
                success_total = 0
                frame_rows = []
                for idx in range(start, end):
                    attrs = frame_attrs(frames[idx])
                    sym = symbolic_edit(attrs, field, target)
                    prox = best_proxy_edit(attrs, field, target, cluster_decoded)
                    prox = prox if prox is not None else dict(attrs)
                    prox_coll = changed_fields(attrs, prox, field)
                    frame_rows.append(
                        {
                            "frame_idx": frames[idx]["frame_idx"],
                            "original": attrs,
                            "symbolic": sym,
                            "proxy": prox,
                            "proxy_changed_fields": prox_coll,
                            "symbolic_clean": clean_edit(attrs, sym, field),
                            "proxy_clean": clean_edit(attrs, prox, field),
                            "proxy_target_success": prox[field] == target,
                        }
                    )
                    coll_total += len(prox_coll)
                    clean_total += int(clean_edit(attrs, prox, field))
                    success_total += int(prox[field] == target)
                rows.append(
                    {
                        "seq_name": seq_name,
                        "task_field": field,
                        "task_target": target,
                        "start": start,
                        "end": end,
                        "run_length": end - start,
                        "proxy_mean_collateral_fields": coll_total / max(end - start, 1),
                        "proxy_clean_edit_rate": clean_total / max(end - start, 1),
                        "proxy_target_success_rate": success_total / max(end - start, 1),
                        "frame_rows": frame_rows,
                    }
                )
    return rows


def source_cluster_decoded(source_name, train_data, labels, semantic_vocab):
    if source_name == "semantic_frame":
        matrix = collect_semantic_frame_matrix(train_data, labels, semantic_vocab)
    else:
        matrix = collect_frame_matrix(train_data, labels)
    kmeans = KMeans(n_clusters=32, random_state=0, n_init=10)
    kmeans.fit(matrix)

    train_attrs = []
    assignments = []
    token_to_idx = {tok: idx for idx, tok in enumerate(semantic_vocab)}
    for sequence in train_data["sequences"]:
        seq_name = sequence["seq_name"]
        canon = seq_name.replace("ROM07_RT_", "ROM07_Rt_").replace("ROM08_LT_", "ROM08_Lt_")
        if seq_name not in labels and canon not in labels:
            continue
        for frame in sequence["frames"]:
            train_attrs.append(frame_attrs(frame))
            if source_name == "semantic_frame":
                vec = np.zeros(len(semantic_vocab), dtype=np.float32)
                for tok in frame_token_set(frame, mode="temporal", include_persistence=True):
                    idx = token_to_idx.get(tok)
                    if idx is not None:
                        vec[idx] = 1.0
            else:
                vec = np.asarray(frame_vector(frame), dtype=np.float32)
            assignments.append(int(kmeans.predict(vec.reshape(1, -1))[0]))
    return cluster_majority_attrs(assignments, train_attrs, 32)


def main():
    train_data = load_json(GEN / "temporal_hl_val.json")
    test_data = load_json(GEN / "temporal_hl_test.json")
    labels = overlap_labels(train_data, test_data)
    semantic_vocab = build_semantic_frame_vocab(train_data, labels)

    payload = {
        "artifacts": {
            "train_json": str(GEN / "temporal_hl_val.json"),
            "test_json": str(GEN / "temporal_hl_test.json"),
            "local_edit_audit": str(GEN / "local_edit_audit.json"),
        },
        "sources": [],
    }

    for source_name in ["semantic_frame", "continuous_frame"]:
        cluster_decoded = source_cluster_decoded(source_name, train_data, labels, semantic_vocab)
        rows = collect_candidate_rows(test_data, labels, cluster_decoded)
        by_task = defaultdict(list)
        for row in rows:
            by_task[(row["task_field"], row["task_target"])].append(row)
        selected = []
        for task in TASKS:
            task_rows = by_task[task]
            task_rows.sort(
                key=lambda row: (
                    -row["proxy_mean_collateral_fields"],
                    row["proxy_clean_edit_rate"],
                    -row["run_length"],
                    row["seq_name"],
                )
            )
            selected.extend(task_rows[:2])
        payload["sources"].append(
            {
                "source": source_name,
                "selected_cases": selected,
            }
        )

    out_json = GEN / "local_edit_casebook.json"
    out_md = SUM / "local_edit_casebook.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Local Edit Casebook",
        "",
        "This is an experiment memo, not paper text.",
    ]
    for source in payload["sources"]:
        lines.extend(["", f"## Source: {source['source']}"])
        for case in source["selected_cases"]:
            lines.extend([
                "",
                f"### {case['seq_name']} | {case['task_field']} -> {case['task_target']} | frames {case['start']}:{case['end']}",
                "",
                f"- run length: `{case['run_length']}`",
                f"- proxy clean edit rate: `{case['proxy_clean_edit_rate']:.4f}`",
                f"- proxy mean collateral fields: `{case['proxy_mean_collateral_fields']:.4f}`",
                f"- proxy target success rate: `{case['proxy_target_success_rate']:.4f}`",
                "",
                "| frame_idx | original target field | symbolic target field | proxy target field | proxy changed fields | original right motion | proxy right motion | original left motion | proxy left motion | original interaction | proxy interaction |",
                "| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ])
            for row in case["frame_rows"][:8]:
                orig = row["original"]
                sym = row["symbolic"]
                prox = row["proxy"]
                lines.append(
                    f"| {row['frame_idx']} | {orig[case['task_field']]} | {sym[case['task_field']]} | {prox[case['task_field']]} | {row['proxy_changed_fields']} | "
                    f"{orig['right_hand_motion']} | {prox['right_hand_motion']} | {orig['left_hand_motion']} | {prox['left_hand_motion']} | "
                    f"{orig['interaction_motion']} | {prox['interaction_motion']} |"
                )
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
