#!/usr/bin/env python3
"""Generalization study for pair-guided reranking across interaction slices."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from tools.build_transition_conditioned_symbolic_editor import (
    GEN,
    SUM,
    build_pair_bank,
    build_semantic_frame_vocab,
    canonical,
    frame_geom,
    grouped_motif_signature,
    load_json,
    overlap_labels,
)
from tools.build_geometry_locality_audit import hand_delta
from tools.build_local_edit_audit import contiguous_runs, eligible_value, frame_attrs
from tools.build_weak_slice_joint_editor_prototype import current_grouped_signature, fmt
from tools.build_weak_slice_topk_joint_search import evaluate_joint
from tools.build_weak_slice_split_donor_prototype import right_relaxed_candidates


TASKS = [
    ("left_hand_motion", "closing"),
    ("left_hand_motion", "opening"),
    ("right_hand_motion", "closing"),
    ("right_hand_motion", "opening"),
]

LEVEL = {"still": 0, "local": 1, "active": 2}
INTERACTION_VALUES = ["approach", "separate", "steady", "unknown"]
MOTION_VALUES = ["none", "start", "opening", "closing", "mixed", "steady", "unknown"]
DEPTHS = (5, 10)
REFERENCE_METHODS = {
    "base_top5": 5,
    "base_top10": 10,
    "base_top20": 20,
}


def parse_grouped(group: str):
    parts = group.split("|")
    if len(parts) != 6:
        return None, []
    return parts[0], parts[1:]


def one_hot(value: str, vocab: list[str]) -> list[float]:
    return [1.0 if value == item else 0.0 for item in vocab]


def opposite_hand_name(task_field: str) -> str:
    return "right" if task_field == "left_hand_motion" else "left"


def candidate_pool_for_task(pair_bank, task_field, task_target, curr_attrs, prev_geom, curr_geom):
    if task_field == "right_hand_motion":
        return right_relaxed_candidates(
            pair_bank,
            curr_attrs,
            prev_geom,
            curr_geom,
            task_field=task_field,
            task_target=task_target,
        )

    out = []
    for row in pair_bank:
        attrs = row["curr_attrs"]
        if attrs[task_field] != curr_attrs[task_field]:
            continue
        if attrs["hand_type"] != curr_attrs["hand_type"]:
            continue
        if attrs["interaction_motion"] != curr_attrs["interaction_motion"]:
            continue
        if attrs["right_hand_motion"] != curr_attrs["right_hand_motion"]:
            continue
        out.append(row)
    out.sort(
        key=lambda row: (
            hand_delta(curr_geom["left"], row["curr_geom"]["left"]) + hand_delta(prev_geom["left"], row["prev_geom"]["left"]),
            row["seq_name"],
            row["curr_frame_idx"],
        )
    )
    return out


def relaxed_left_family_candidates_with_meta(pair_bank, current_opp_group, curr_attrs, prev_geom, curr_geom, opposite_hand: str, max_profile_distance: int = 1):
    target_motion, target_profile = parse_grouped(current_opp_group)
    out = []
    for row in pair_bank:
        attrs = row["curr_attrs"]
        if attrs["hand_type"] != curr_attrs["hand_type"]:
            continue
        if attrs["interaction_motion"] != curr_attrs["interaction_motion"]:
            continue
        donor_opp = row["curr_frame"].get(opposite_hand)
        if donor_opp is None:
            continue
        donor_group = grouped_motif_signature(
            str(donor_opp.get("hand_motion", "unknown")),
            list(donor_opp.get("transition_labels", [])),
        )
        donor_motion, donor_profile = parse_grouped(donor_group)
        if donor_motion != target_motion:
            continue
        if len(donor_profile) != len(target_profile):
            continue
        family_dist = sum(abs(LEVEL[a] - LEVEL[b]) for a, b in zip(donor_profile, target_profile))
        if family_dist > max_profile_distance:
            continue
        donor_curr_geom = row["curr_geom"]
        curr_cross = curr_geom.get("cross_hand_distance")
        donor_cross = donor_curr_geom.get("cross_hand_distance")
        cross_delta_to_current = 0.0
        if curr_cross is not None and donor_cross is not None:
            cross_delta_to_current = abs(float(curr_cross) - float(donor_cross))
        out.append(
            {
                "row": row,
                "family_dist": family_dist,
                "target_profile": [LEVEL[x] for x in target_profile],
                "donor_profile": [LEVEL[x] for x in donor_profile],
                "curr_opp_delta": hand_delta(curr_geom[opposite_hand], donor_curr_geom[opposite_hand]),
                "prev_opp_delta": hand_delta(prev_geom[opposite_hand], donor_curr_geom[opposite_hand]),
                "cross_delta_to_current": cross_delta_to_current,
                "curr_cross_distance": 0.0 if curr_cross is None else float(curr_cross),
                "donor_cross_distance": 0.0 if donor_cross is None else float(donor_cross),
                "interaction_motion_value": str(curr_attrs.get("interaction_motion", "unknown")),
                "other_hand_motion": str(curr_attrs.get(f"{opposite_hand}_hand_motion", "unknown")),
            }
        )
    out.sort(key=lambda item: (item["family_dist"], item["row"]["seq_name"], item["row"]["curr_frame_idx"]))
    for rank, item in enumerate(out):
        item["raw_rank"] = rank
    return out


def target_right_features(target_row, task_field, prev_geom, curr_geom):
    hand_name = "right" if task_field == "right_hand_motion" else "left"
    donor_curr_geom = target_row["curr_geom"]
    curr_cross = curr_geom.get("cross_hand_distance")
    donor_cross = donor_curr_geom.get("cross_hand_distance")
    cross_delta_to_current = 0.0
    if curr_cross is not None and donor_cross is not None:
        cross_delta_to_current = abs(float(curr_cross) - float(donor_cross))
    donor_hand = target_row["curr_frame"].get(hand_name)
    donor_group = "unknown"
    if donor_hand is not None:
        donor_group = grouped_motif_signature(
            str(donor_hand.get("hand_motion", "unknown")),
            list(donor_hand.get("transition_labels", [])),
        )
    motion, profile = parse_grouped(donor_group)
    return {
        "curr_target_delta": hand_delta(curr_geom[hand_name], donor_curr_geom[hand_name]),
        "prev_target_delta": hand_delta(prev_geom[hand_name], donor_curr_geom[hand_name]),
        "target_cross_delta_to_current": cross_delta_to_current,
        "target_donor_cross_distance": 0.0 if donor_cross is None else float(donor_cross),
        "target_motion": motion or "unknown",
        "target_profile": [LEVEL.get(x, 0) for x in profile] if profile else [0, 0, 0, 0, 0],
    }


def pair_feature_vector(opp_item, target_item, opp_pool_size: int, target_pool_size: int):
    pair_cross_gap = abs(opp_item["donor_cross_distance"] - target_item["target_donor_cross_distance"])
    feats = [
        float(opp_item["family_dist"]),
        float(opp_item["curr_opp_delta"]),
        float(opp_item["prev_opp_delta"]),
        float(opp_item["cross_delta_to_current"]),
        float(opp_item["curr_cross_distance"]),
        float(opp_item["donor_cross_distance"]),
        float(opp_item["raw_rank"]),
        float(opp_pool_size),
        float(target_pool_size),
        float(target_item["curr_target_delta"]),
        float(target_item["prev_target_delta"]),
        float(target_item["target_cross_delta_to_current"]),
        float(target_item["target_donor_cross_distance"]),
        float(pair_cross_gap),
        float(abs(opp_item["cross_delta_to_current"] - target_item["target_cross_delta_to_current"])),
    ]
    feats.extend(float(x) for x in opp_item["target_profile"])
    feats.extend(float(x) for x in opp_item["donor_profile"])
    feats.extend(float(x) for x in target_item["target_profile"])
    feats.extend(one_hot(opp_item["interaction_motion_value"], INTERACTION_VALUES))
    feats.extend(one_hot(opp_item["other_hand_motion"], MOTION_VALUES))
    feats.extend(one_hot(target_item["target_motion"], MOTION_VALUES))
    return feats


def collect_slice_frames(data, task_field: str, task_target: str):
    rows = []
    for sequence in data["sequences"]:
        frames = sequence["frames"]
        for start, end, value in contiguous_runs(frames, task_field):
            if end - start < 3 or not eligible_value(task_field, value, task_target):
                continue
            for idx in range(max(start, 1), end):
                prev_frame = frames[idx - 1]
                curr_frame = frames[idx]
                if curr_frame.get("hand_type") != "interacting":
                    continue
                rows.append(
                    {
                        "seq_name": sequence["seq_name"],
                        "prev_frame": prev_frame,
                        "curr_frame": curr_frame,
                        "prev_geom": frame_geom(prev_frame),
                        "curr_geom": frame_geom(curr_frame),
                        "curr_attrs": frame_attrs(curr_frame),
                        "current_opp_group": current_grouped_signature(curr_frame.get(opposite_hand_name(task_field))),
                    }
                )
    return rows


def build_examples(frames, pair_bank, task_field: str, task_target: str):
    xs = []
    ys = []
    meta = defaultdict(int)
    opp_hand = opposite_hand_name(task_field)
    for entry in frames:
        prev_frame = entry["prev_frame"]
        curr_frame = entry["curr_frame"]
        prev_geom = entry["prev_geom"]
        curr_geom = entry["curr_geom"]
        curr_attrs = entry["curr_attrs"]
        target_pool = [
            row for row in candidate_pool_for_task(pair_bank, task_field, task_target, curr_attrs, prev_geom, curr_geom)
            if not (row["seq_name"] == entry["seq_name"] and row["curr_frame_idx"] == curr_frame["frame_idx"])
        ]
        opp_pool = [
            item for item in relaxed_left_family_candidates_with_meta(pair_bank, entry["current_opp_group"], curr_attrs, prev_geom, curr_geom, opp_hand, 1)
            if not (item["row"]["seq_name"] == entry["seq_name"] and item["row"]["curr_frame_idx"] == curr_frame["frame_idx"])
        ]
        if not target_pool or not opp_pool:
            continue
        opp_pool_size = len(opp_pool)
        target_pool_size = len(target_pool)
        target_meta = {id(row): target_right_features(row, task_field, prev_geom, curr_geom) for row in target_pool}
        for opp_item in opp_pool:
            for target_row in target_pool:
                if task_field == "right_hand_motion":
                    res = evaluate_joint(prev_frame, curr_frame, prev_geom, target_row, opp_item["row"])
                else:
                    res = evaluate_joint(prev_frame, curr_frame, prev_geom, opp_item["row"], target_row)
                xs.append(pair_feature_vector(opp_item, target_meta[id(target_row)], opp_pool_size, target_pool_size))
                ys.append(int(res["joint_score"]))
                meta["pairs"] += 1
                meta["positives"] += int(res["joint_score"])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int32), meta


def summarize(rows, prefix):
    avail_key = f"{prefix}_available"
    joint_key = f"{prefix}_joint_score"
    right_key = f"{prefix}_right_grouped_match"
    left_key = f"{prefix}_left_preserve"
    n = len(rows)
    avail = sum(row[avail_key] for row in rows)
    return {
        "num_frames": n,
        "available_rate": avail / n,
        "joint_score_on_available": sum(row[joint_key] for row in rows if row[avail_key]) / max(avail, 1),
        "right_grouped_match_on_available": sum(row[right_key] for row in rows if row[avail_key]) / max(avail, 1),
        "left_preserve_on_available": sum(row[left_key] for row in rows if row[avail_key]) / max(avail, 1),
        "joint_hit_rate_overall": sum(row[joint_key] for row in rows) / n,
    }


def run_task(train_frames, test_frames, pair_bank, task_field: str, task_target: str):
    x_train, y_train, meta = build_examples(train_frames, pair_bank, task_field, task_target)
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    pos_weight = 1.0 if pos == 0 else max(1.0, neg / max(pos, 1))
    sample_weight = np.where(y_train == 1, pos_weight, 1.0).astype(np.float32)

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=5,
        max_iter=250,
        min_samples_leaf=30,
        random_state=0,
    )
    model.fit(x_train, y_train, sample_weight=sample_weight)

    opp_hand = opposite_hand_name(task_field)
    rows = []
    for entry in test_frames:
        prev_frame = entry["prev_frame"]
        curr_frame = entry["curr_frame"]
        prev_geom = entry["prev_geom"]
        curr_geom = entry["curr_geom"]
        curr_attrs = entry["curr_attrs"]
        target_pool = candidate_pool_for_task(pair_bank, task_field, task_target, curr_attrs, prev_geom, curr_geom)
        opp_pool = relaxed_left_family_candidates_with_meta(pair_bank, entry["current_opp_group"], curr_attrs, prev_geom, curr_geom, opp_hand, 1)

        rec = {
            "seq_name": entry["seq_name"],
            "frame_idx": curr_frame["frame_idx"],
            "interaction_motion_value": str(curr_frame.get("interaction_motion", "unknown")),
        }

        for name, depth in REFERENCE_METHODS.items():
            best = None
            if target_pool and opp_pool:
                for target_row in target_pool:
                    for opp_item in opp_pool[:depth]:
                        if task_field == "right_hand_motion":
                            res = evaluate_joint(prev_frame, curr_frame, prev_geom, target_row, opp_item["row"])
                        else:
                            res = evaluate_joint(prev_frame, curr_frame, prev_geom, opp_item["row"], target_row)
                        key = (res["joint_score"], res["left_preserve"], res["right_grouped_match"])
                        if best is None or key > best[0]:
                            best = (key, res)
            res = None if best is None else best[1]
            rec[f"{name}_available"] = int(res is not None)
            rec[f"{name}_joint_score"] = 0 if res is None else res["joint_score"]
            rec[f"{name}_right_grouped_match"] = 0 if res is None else res["right_grouped_match"]
            rec[f"{name}_left_preserve"] = 0 if res is None else res["left_preserve"]

        if target_pool and opp_pool:
            opp_pool_size = len(opp_pool)
            target_pool_size = len(target_pool)
            target_meta = {id(row): target_right_features(row, task_field, prev_geom, curr_geom) for row in target_pool}
            scored_opp = []
            for opp_item in opp_pool:
                feats = np.asarray(
                    [pair_feature_vector(opp_item, target_meta[id(target_row)], opp_pool_size, target_pool_size) for target_row in target_pool],
                    dtype=np.float32,
                )
                score = float(model.predict_proba(feats)[:, 1].max())
                scored_opp.append((score, opp_item))
            ranked_opp = [item for _, item in sorted(scored_opp, key=lambda pair: pair[0], reverse=True)]
        else:
            ranked_opp = []

        for depth in DEPTHS:
            prefix = f"pairguided_top{depth}"
            best = None
            if target_pool and ranked_opp:
                for target_row in target_pool:
                    for opp_item in ranked_opp[:depth]:
                        if task_field == "right_hand_motion":
                            res = evaluate_joint(prev_frame, curr_frame, prev_geom, target_row, opp_item["row"])
                        else:
                            res = evaluate_joint(prev_frame, curr_frame, prev_geom, opp_item["row"], target_row)
                        key = (res["joint_score"], res["left_preserve"], res["right_grouped_match"])
                        if best is None or key > best[0]:
                            best = (key, res)
            res = None if best is None else best[1]
            rec[f"{prefix}_available"] = int(res is not None)
            rec[f"{prefix}_joint_score"] = 0 if res is None else res["joint_score"]
            rec[f"{prefix}_right_grouped_match"] = 0 if res is None else res["right_grouped_match"]
            rec[f"{prefix}_left_preserve"] = 0 if res is None else res["left_preserve"]

        rows.append(rec)

    prefixes = list(REFERENCE_METHODS.keys()) + [f"pairguided_top{depth}" for depth in DEPTHS]
    summary = {prefix: summarize(rows, prefix) for prefix in prefixes}
    return {
        "training_stats": {
            "num_examples": int(len(y_train)),
            "num_positive": pos,
            "num_negative": neg,
            "positive_weight": float(pos_weight),
            **{k: int(v) for k, v in meta.items()},
        },
        "summary": summary,
        "rows": rows,
    }


def main():
    train_data = load_json(GEN / "temporal_hl_val.json")
    test_data = load_json(GEN / "temporal_hl_test.json")
    labels = overlap_labels(train_data, test_data)
    semantic_vocab = build_semantic_frame_vocab(train_data, labels)
    pair_bank = build_pair_bank(train_data, labels, semantic_vocab)

    task_results = {}
    ranking_rows = []
    for task_field, task_target in TASKS:
        train_frames = [
            row for row in collect_slice_frames(train_data, task_field, task_target)
            if canonical(row["seq_name"]) in labels or row["seq_name"] in labels
        ]
        test_frames = [
            row for row in collect_slice_frames(test_data, task_field, task_target)
            if canonical(row["seq_name"]) in labels or row["seq_name"] in labels
        ]
        result = run_task(train_frames, test_frames, pair_bank, task_field, task_target)
        task_name = f"{task_field}->{task_target}"
        task_results[task_name] = result
        for method, stats in result["summary"].items():
            ranking_rows.append({"task": task_name, "method": method, **stats})

    out_json = GEN / "pairguided_reranker_multislice.json"
    out_md = SUM / "pairguided_reranker_multislice.md"
    payload = {
        "artifacts": {
            "train_json": str(GEN / "temporal_hl_val.json"),
            "test_json": str(GEN / "temporal_hl_test.json"),
        },
        "focus": {
            "tasks": [f"{field}->{target}" for field, target in TASKS],
            "slice": "interaction only",
            "goal": "test whether pair-guided reranking generalizes beyond the original weak slice",
        },
        "task_results": task_results,
        "ranking_rows": ranking_rows,
    }
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Pair-Guided Reranker Multi-Slice Study",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Focus: interaction-only hand-motion edit slices.",
        "",
    ]
    for task_name, result in task_results.items():
        lines.extend(
            [
                f"## {task_name}",
                "",
                "### Training Stats",
                "",
                "| metric | value |",
                "| --- | ---: |",
            ]
        )
        for key, value in result["training_stats"].items():
            lines.append(f"| {key} | {value} |")
        lines.extend(
            [
                "",
                "### Method Summary",
                "",
                "| method | avail | joint on avail | right match | left preserve | joint overall |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for method, stats in result["summary"].items():
            lines.append(
                f"| {method} | {fmt(stats['available_rate'])} | {fmt(stats['joint_score_on_available'])} | "
                f"{fmt(stats['right_grouped_match_on_available'])} | {fmt(stats['left_preserve_on_available'])} | "
                f"{fmt(stats['joint_hit_rate_overall'])} |"
            )
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
