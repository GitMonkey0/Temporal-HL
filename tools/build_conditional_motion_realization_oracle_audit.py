#!/usr/bin/env python3
"""Realization-aware conditional audit for hand-motion edits."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from tools.build_conditional_hand_transplant_audit import (
    TASKS,
    build_train_frame_bank,
    canonical,
    preserved_fields,
    target_hand_distance,
    target_hand_name,
)
from tools.build_geometry_locality_audit import frame_geom
from tools.build_learned_token_proxy_report import build_semantic_frame_vocab, overlap_labels
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


def realized_motion(prev_hand: dict[str, object] | None, donor_hand: dict[str, object] | None) -> str:
    if prev_hand is None or donor_hand is None:
        return "unknown"
    prev_scores = prev_hand.get("flexion_scores", {})
    donor_scores = donor_hand.get("flexion_scores", {})
    return summarize_flexion_transition(prev_scores, donor_scores, FLEX_THRESHOLD)


def donor_hand_record(row: dict[str, object], task_field: str) -> dict[str, object] | None:
    hand_name = target_hand_name(task_field)
    geom = row.get("geom", {})
    if isinstance(geom, dict):
        return geom.get(hand_name)  # type: ignore[return-value]
    return None


def symbolic_candidates(train_bank, original_attrs, task_field, target_value):
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


def best_symbolic_realizing(train_bank, original_attrs, original_geom, prev_hand, task_field, target_value):
    cands = symbolic_candidates(train_bank, original_attrs, task_field, target_value)
    hand_name = target_hand_name(task_field)
    realizing = [row for row in cands if realized_motion(prev_hand, donor_hand_record(row, task_field)) == target_value]
    usable = realizing if realizing else cands
    if not usable:
        return None, False
    usable.sort(key=lambda row: (target_hand_distance(original_geom, row["geom"], task_field), row["seq_name"], row["frame_idx"]))
    return usable[0], bool(realizing)


def best_proxy_attrs(original_attrs, task_field, target_value, cluster_decoded):
    keep = preserved_fields(task_field)
    candidates = []
    for cid, attrs in cluster_decoded.items():
        if attrs[task_field] != target_value:
            continue
        preserved_mismatch = sum(1 for field in keep if attrs[field] != original_attrs[field])
        total_collateral = sum(1 for field in TRACKED_FIELDS if field != task_field and attrs[field] != original_attrs[field])
        candidates.append((preserved_mismatch, total_collateral, cid, attrs))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    return candidates[0][3]


def best_proxy_realizing(train_bank, source_name, cluster_decoded, proxy_attrs, original_geom, prev_hand, task_field, target_value):
    hand_name = target_hand_name(task_field)
    matched_clusters = [cid for cid, attrs in cluster_decoded.items() if attrs == proxy_attrs]
    cands = [row for row in train_bank if row[f"{source_name}_cluster"] in matched_clusters and row["attrs"][task_field] == target_value]
    realizing = [row for row in cands if realized_motion(prev_hand, donor_hand_record(row, task_field)) == target_value]
    usable = realizing if realizing else cands
    if not usable:
        return None, False
    usable.sort(key=lambda row: (target_hand_distance(original_geom, row["geom"], task_field), row["seq_name"], row["frame_idx"]))
    return usable[0], bool(realizing)


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
                "symbolic_oracle_success": sum(row["symbolic_oracle_success"] for row in items) / n,
                "proxy_oracle_success": sum(row["proxy_oracle_success"] for row in items) / n,
                "symbolic_realizing_candidate_rate": sum(row["symbolic_realizing_candidate"] for row in items) / n,
                "proxy_realizing_candidate_rate": sum(row["proxy_realizing_candidate"] for row in items) / n,
                "symbolic_beats_proxy": sum(row["symbolic_beats_proxy"] for row in items) / n,
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
                hand_name = target_hand_name(task_field)
                for start, end, value in contiguous_runs(frames, task_field):
                    if end - start < 3 or not eligible_value(task_field, value, task_target):
                        continue
                    for idx in range(max(start, 1), end):
                        prev_frame = frames[idx - 1]
                        frame = frames[idx]
                        attrs = frame_attrs(frame)
                        prev_hand = prev_frame.get(hand_name)
                        geom = frame_geom(frame)
                        sym, sym_has = best_symbolic_realizing(train_bank, attrs, geom, prev_hand, task_field, task_target)
                        if sym is None:
                            continue
                        proxy_attrs = best_proxy_attrs(attrs, task_field, task_target, decoded)
                        if proxy_attrs is None:
                            continue
                        prox, prox_has = best_proxy_realizing(train_bank, source_name, decoded, proxy_attrs, geom, prev_hand, task_field, task_target)
                        if prox is None:
                            continue
                        proxy_preserved_mismatch = sum(1 for field in preserved_fields(task_field) if proxy_attrs[field] != attrs[field])
                        proxy_total_semantic_collateral = sum(
                            1 for field in TRACKED_FIELDS if field != task_field and proxy_attrs[field] != attrs[field]
                        )
                        sym_motion = realized_motion(prev_hand, donor_hand_record(sym, task_field))
                        prox_motion = realized_motion(prev_hand, donor_hand_record(prox, task_field))
                        rows.append(
                            {
                                "task": f"{task_field}->{task_target}",
                                "symbolic_oracle_success": int(sym_motion == task_target),
                                "proxy_oracle_success": int(prox_motion == task_target),
                                "symbolic_realizing_candidate": int(sym_has),
                                "proxy_realizing_candidate": int(prox_has),
                                "symbolic_beats_proxy": int((sym_motion == task_target) and (prox_motion != task_target)),
                                "proxy_preserved_clean": int(proxy_preserved_mismatch == 0),
                                "proxy_preserved_mismatch_fields": proxy_preserved_mismatch,
                                "proxy_total_semantic_collateral": proxy_total_semantic_collateral,
                            }
                        )
        payload["sources"].append({"source": source_name, "summary": summarize(rows), "rows": rows})

    out_json = GEN / "conditional_motion_realization_oracle_audit.json"
    out_md = SUM / "conditional_motion_realization_oracle_audit.md"
    out_json.write_text(json.dumps(payload, indent=2))
    lines = [
        "# Conditional Motion Realization Oracle Audit",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "This audit selects donors that realize the requested motion under the original previous frame whenever such candidates exist.",
    ]
    for source in payload["sources"]:
        lines.extend([
            "",
            f"## Source: {source['source']}",
            "",
            "| task | frames | symbolic oracle success | proxy oracle success | symbolic realizing cand | proxy realizing cand | symbolic beats proxy | proxy preserved clean | proxy semantic collateral |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in source["summary"]:
            lines.append(
                f"| {row['task']} | {row['num_frames']} | {fmt(row['symbolic_oracle_success'])} | {fmt(row['proxy_oracle_success'])} | "
                f"{fmt(row['symbolic_realizing_candidate_rate'])} | {fmt(row['proxy_realizing_candidate_rate'])} | {fmt(row['symbolic_beats_proxy'])} | "
                f"{fmt(row['proxy_preserved_clean_rate'])} | {fmt(row['proxy_total_semantic_collateral'])} |"
            )
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
