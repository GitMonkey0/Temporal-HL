#!/usr/bin/env python3
"""Counterfactual consistency audit for symbolic edits vs opaque proxies."""

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
)
from tools.build_learned_token_proxy_report import (
    build_semantic_frame_vocab,
    frame_token_set,
    frame_vector,
    overlap_labels,
)
from tools.build_local_edit_audit import (
    TASKS,
    TRACKED_FIELDS,
    cluster_majority_attrs,
    contiguous_runs,
    eligible_value,
    frame_attrs,
)


ROOT = Path("/opt/tiger/hand")
GEN = ROOT / "experiments/generated"
SUM = GEN / "summary_tables"


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def canonical(seq_name: str) -> str:
    return seq_name.replace("ROM07_RT_", "ROM07_Rt_").replace("ROM08_LT_", "ROM08_Lt_")


def preserved_fields(task_field: str) -> list[str]:
    if task_field == "right_hand_motion":
        return ["hand_type", "interaction_motion", "left_hand_motion", "left_state_signature"]
    if task_field == "left_hand_motion":
        return ["hand_type", "interaction_motion", "right_hand_motion", "right_state_signature"]
    if task_field == "interaction_motion":
        return [
            "hand_type",
            "right_hand_motion",
            "left_hand_motion",
            "right_state_signature",
            "left_state_signature",
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


def source_cluster_decoded(source_name, train_bank):
    if source_name == "semantic_frame":
        matrix = np.asarray([row["semantic_vec"] for row in train_bank], dtype=np.float32)
    else:
        matrix = np.asarray([row["continuous_vec"] for row in train_bank], dtype=np.float32)
    kmeans = KMeans(n_clusters=32, random_state=0, n_init=10)
    kmeans.fit(matrix)
    assignments = list(map(int, kmeans.labels_))
    decoded = cluster_majority_attrs(assignments, [row["attrs"] for row in train_bank], 32)
    for row, cid in zip(train_bank, assignments):
        row[f"{source_name}_cluster"] = cid
    return decoded


def matching_symbolic_candidates(train_bank, original_attrs, task_field, target_value):
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


def pick_best_symbolic(train_bank, original_attrs, original_geom, task_field, target_value):
    cands = matching_symbolic_candidates(train_bank, original_attrs, task_field, target_value)
    if not cands:
        return None
    cands.sort(key=lambda row: (geometry_distance(original_geom, row["geom"]), row["seq_name"], row["frame_idx"]))
    return cands[0]


def pick_best_proxy_attrs(original_attrs, task_field, target_value, cluster_decoded):
    keep = preserved_fields(task_field)
    candidates = []
    for cid, attrs in cluster_decoded.items():
        if attrs[task_field] != target_value:
            continue
        preserved_mismatch = sum(1 for field in keep if attrs[field] != original_attrs[field])
        total_collateral = sum(
            1 for field in TRACKED_FIELDS if field != task_field and attrs[field] != original_attrs[field]
        )
        candidates.append((preserved_mismatch, total_collateral, cid, attrs))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    return candidates[0][3]


def pick_best_proxy_frame(train_bank, source_name, cluster_decoded, proxy_attrs, original_geom, task_field, target_value):
    matched_clusters = [cid for cid, attrs in cluster_decoded.items() if attrs == proxy_attrs]
    cands = [row for row in train_bank if row[f"{source_name}_cluster"] in matched_clusters and row["attrs"][task_field] == target_value]
    if not cands:
        return None
    cands.sort(key=lambda row: (geometry_distance(original_geom, row["geom"]), row["seq_name"], row["frame_idx"]))
    return cands[0]


def target_share(target_delta: float, collateral_delta: float) -> float:
    return target_delta / max(target_delta + collateral_delta, 1e-6)


def preserved_share(num_preserved_fields: int, mismatches: int) -> float:
    return 1.0 - (mismatches / max(num_preserved_fields, 1))


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
                "symbolic_target_delta": sum(row["symbolic_target_delta"] for row in items) / n,
                "symbolic_context_drift": sum(row["symbolic_context_drift"] for row in items) / n,
                "symbolic_target_share": sum(row["symbolic_target_share"] for row in items) / n,
                "symbolic_counterfactual_score": sum(row["symbolic_counterfactual_score"] for row in items) / n,
                "proxy_target_delta": sum(row["proxy_target_delta"] for row in items) / n,
                "proxy_context_drift": sum(row["proxy_context_drift"] for row in items) / n,
                "proxy_target_share": sum(row["proxy_target_share"] for row in items) / n,
                "proxy_counterfactual_score": sum(row["proxy_counterfactual_score"] for row in items) / n,
                "proxy_preserved_clean_rate": sum(row["proxy_preserved_clean"] for row in items) / n,
                "proxy_preserved_mismatch_fields": sum(row["proxy_preserved_mismatch_fields"] for row in items) / n,
                "proxy_total_semantic_collateral": sum(row["proxy_total_semantic_collateral"] for row in items) / n,
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
        "conditioning": {field: preserved_fields(field) for field, _ in TASKS},
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
                    for idx in range(start, end):
                        frame = frames[idx]
                        attrs = frame_attrs(frame)
                        geom = frame_geom(frame)
                        sym = pick_best_symbolic(train_bank, attrs, geom, task_field, task_target)
                        if sym is None:
                            continue
                        proxy_attrs = pick_best_proxy_attrs(attrs, task_field, task_target, decoded)
                        if proxy_attrs is None:
                            continue
                        prox = pick_best_proxy_frame(train_bank, source_name, decoded, proxy_attrs, geom, task_field, task_target)
                        if prox is None:
                            continue

                        keep = preserved_fields(task_field)
                        sym_delta = edit_target_and_collateral_delta(geom, sym["geom"], task_field)
                        prox_delta = edit_target_and_collateral_delta(geom, prox["geom"], task_field)
                        proxy_preserved_mismatch = sum(1 for field in keep if proxy_attrs[field] != attrs[field])
                        proxy_total_semantic_collateral = sum(
                            1 for field in TRACKED_FIELDS if field != task_field and proxy_attrs[field] != attrs[field]
                        )
                        keep_count = len(keep)
                        sym_target_share = target_share(sym_delta["target_delta"], sym_delta["collateral_delta"])
                        prox_target_share = target_share(prox_delta["target_delta"], prox_delta["collateral_delta"])
                        rows.append(
                            {
                                "task": f"{task_field}->{task_target}",
                                "seq_name": sequence["seq_name"],
                                "frame_idx": frame["frame_idx"],
                                "symbolic_target_delta": sym_delta["target_delta"],
                                "symbolic_context_drift": sym_delta["collateral_delta"],
                                "symbolic_target_share": sym_target_share,
                                "symbolic_counterfactual_score": sym_target_share,
                                "proxy_target_delta": prox_delta["target_delta"],
                                "proxy_context_drift": prox_delta["collateral_delta"],
                                "proxy_target_share": prox_target_share,
                                "proxy_counterfactual_score": prox_target_share * preserved_share(keep_count, proxy_preserved_mismatch),
                                "proxy_preserved_clean": int(proxy_preserved_mismatch == 0),
                                "proxy_preserved_mismatch_fields": proxy_preserved_mismatch,
                                "proxy_total_semantic_collateral": proxy_total_semantic_collateral,
                            }
                        )
        payload["sources"].append({"source": source_name, "summary": summarize(rows), "rows": rows})

    out_json = GEN / "counterfactual_edit_audit.json"
    out_md = SUM / "counterfactual_edit_audit.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Counterfactual Edit Audit",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Target share is defined as `target_delta / (target_delta + context_drift)`.",
    ]
    for source in payload["sources"]:
        lines.extend(
            [
                "",
                f"## Source: {source['source']}",
                "",
                "| task | frames | symbolic target | symbolic context | symbolic share | symbolic cf score | proxy target | proxy context | proxy share | proxy cf score | proxy preserved clean | proxy preserved mismatch | proxy semantic collateral |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in source["summary"]:
            lines.append(
                f"| {row['task']} | {row['num_frames']} | {fmt(row['symbolic_target_delta'])} | {fmt(row['symbolic_context_drift'])} | {fmt(row['symbolic_target_share'])} | {fmt(row['symbolic_counterfactual_score'])} | "
                f"{fmt(row['proxy_target_delta'])} | {fmt(row['proxy_context_drift'])} | {fmt(row['proxy_target_share'])} | {fmt(row['proxy_counterfactual_score'])} | "
                f"{fmt(row['proxy_preserved_clean_rate'])} | {fmt(row['proxy_preserved_mismatch_fields'])} | {fmt(row['proxy_total_semantic_collateral'])} |"
            )
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
