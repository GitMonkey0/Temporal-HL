#!/usr/bin/env python3
"""Slice audit for transition-conditioned hand-motion donor pairs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from tools.build_transition_conditioned_hand_motion_audit import (
    TASKS,
    FLEX_THRESHOLD,
    best_proxy_attrs,
    build_pair_bank,
    canonical,
    fmt,
    pair_distance,
    pair_realizes_target,
    pick_best_proxy_pair,
    pick_best_symbolic_pair,
    source_cluster_decoded,
)
from tools.build_geometry_locality_audit import frame_geom
from tools.build_learned_token_proxy_report import build_semantic_frame_vocab, overlap_labels
from tools.build_local_edit_audit import contiguous_runs, eligible_value, frame_attrs


ROOT = Path("/opt/tiger/hand")
GEN = ROOT / "experiments/generated"
SUM = GEN / "summary_tables"


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def slice_names(seq_name: str, task: str) -> list[str]:
    canon = canonical(seq_name)
    out = ["all"]
    if "Occlusion" in canon and "Finger_Occlusions" not in canon:
        out.append("occlusion")
    if "Finger_Occlusions" in canon:
        out.append("finger_occlusion")
    if "Wrist_ROM" in canon:
        out.append("wrist_rom")
    if "Interaction" in canon:
        out.append("interaction")
    else:
        out.append("noninteraction")
    if task.startswith("right_hand_motion"):
        out.append("right")
    if task.startswith("left_hand_motion"):
        out.append("left")
    if "No_Occlusion" in canon:
        out.append("no_occlusion")
    return out


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["source"], row["task"], row["slice"])].append(row)
    out = []
    for (source, task, slice_name), items in sorted(grouped.items()):
        n = len(items)
        sym_avail = sum(row["symbolic_pair_available"] for row in items)
        prox_avail = sum(row["proxy_pair_available"] for row in items)
        out.append(
            {
                "source": source,
                "task": task,
                "slice": slice_name,
                "num_frames": n,
                "symbolic_pair_available_rate": sym_avail / n,
                "proxy_pair_available_rate": prox_avail / n,
                "symbolic_pair_distance": sum(row["symbolic_pair_distance"] for row in items if row["symbolic_pair_available"]) / max(sym_avail, 1),
                "proxy_pair_distance": sum(row["proxy_pair_distance"] for row in items if row["proxy_pair_available"]) / max(prox_avail, 1),
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

    raw_rows = []
    for source_name, decoded in [("semantic_frame", semantic_decoded), ("continuous_frame", continuous_decoded)]:
        for sequence in test_data["sequences"]:
            if canonical(sequence["seq_name"]) not in labels and sequence["seq_name"] not in labels:
                continue
            frames = sequence["frames"]
            for task_field, task_target in TASKS:
                task = f"{task_field}->{task_target}"
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
                        sym_avail = int(sym is not None)
                        prox_avail = int(prox is not None)
                        sym_dist = pair_distance(test_prev_geom, test_curr_geom, sym, task_field) if sym is not None else 0.0
                        prox_dist = pair_distance(test_prev_geom, test_curr_geom, prox, task_field) if prox is not None else 0.0
                        proxy_clean = 0 if proxy_attrs is None else int(sum(1 for field in [] if False) == 0)
                        # preserved-clean is measured from the conditioned attrs, same as the parent audit
                        if proxy_attrs is None:
                            proxy_preserved_clean = 0
                            proxy_total_semantic_collateral = 0
                        else:
                            from tools.build_transition_conditioned_hand_motion_audit import preserved_fields
                            proxy_preserved_clean = int(all(proxy_attrs[field] == curr_attrs[field] for field in preserved_fields(task_field)))
                            from tools.build_local_edit_audit import TRACKED_FIELDS
                            proxy_total_semantic_collateral = sum(
                                1 for field in TRACKED_FIELDS if field != task_field and proxy_attrs[field] != curr_attrs[field]
                            )
                        for slice_name in slice_names(sequence["seq_name"], task):
                            raw_rows.append(
                                {
                                    "source": source_name,
                                    "seq_name": sequence["seq_name"],
                                    "task": task,
                                    "slice": slice_name,
                                    "symbolic_pair_available": sym_avail,
                                    "proxy_pair_available": prox_avail,
                                    "symbolic_pair_distance": sym_dist,
                                    "proxy_pair_distance": prox_dist,
                                    "symbolic_beats_proxy": int(sym is not None and prox is not None and sym_dist <= prox_dist),
                                    "proxy_preserved_clean": proxy_preserved_clean,
                                    "proxy_total_semantic_collateral": proxy_total_semantic_collateral,
                                }
                            )

    summary = summarize(raw_rows)
    payload = {
        "artifacts": {
            "train_json": str(GEN / "temporal_hl_val.json"),
            "test_json": str(GEN / "temporal_hl_test.json"),
            "transition_conditioned_hand_motion_audit": str(GEN / "transition_conditioned_hand_motion_audit.json"),
        },
        "flex_threshold": FLEX_THRESHOLD,
        "summary": summary,
    }

    out_json = GEN / "transition_conditioned_hand_motion_slice_audit.json"
    out_md = SUM / "transition_conditioned_hand_motion_slice_audit.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Transition-Conditioned Hand-Motion Slice Audit",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "| source | task | slice | frames | sym avail | proxy avail | sym dist | proxy dist | sym beats proxy | proxy clean | proxy collateral |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        if row["slice"] in {"all", "occlusion", "finger_occlusion", "interaction", "right", "left", "wrist_rom"}:
            lines.append(
                f"| {row['source']} | {row['task']} | {row['slice']} | {row['num_frames']} | "
                f"{fmt(row['symbolic_pair_available_rate'])} | {fmt(row['proxy_pair_available_rate'])} | {fmt(row['symbolic_pair_distance'])} | "
                f"{fmt(row['proxy_pair_distance'])} | {fmt(row['symbolic_beats_proxy_rate'])} | {fmt(row['proxy_preserved_clean_rate'])} | {fmt(row['proxy_total_semantic_collateral'])} |"
            )
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
