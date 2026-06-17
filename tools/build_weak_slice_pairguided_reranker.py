#!/usr/bin/env python3
"""Pair-guided learned reranker for the weak interaction slice."""

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
from tools.build_weak_slice_joint_editor_prototype import (
    TASK_FIELD,
    TASK_TARGET,
    current_grouped_signature,
    fmt,
)
from tools.build_weak_slice_topk_joint_search import evaluate_joint
from tools.build_weak_slice_split_donor_prototype import right_relaxed_candidates


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


def left_family_candidates_with_meta(pair_bank, current_left_group, curr_attrs, prev_geom, curr_geom, max_profile_distance: int = 1):
    target_motion, target_profile = parse_grouped(current_left_group)
    out = []
    for row in pair_bank:
        attrs = row["curr_attrs"]
        if attrs["hand_type"] != curr_attrs["hand_type"]:
            continue
        if attrs["interaction_motion"] != curr_attrs["interaction_motion"]:
            continue
        donor_left = row["curr_frame"].get("left")
        if donor_left is None:
            continue
        donor_group = grouped_motif_signature(
            str(donor_left.get("hand_motion", "unknown")),
            list(donor_left.get("transition_labels", [])),
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
                "curr_left_delta": hand_delta(curr_geom["left"], donor_curr_geom["left"]),
                "prev_left_delta": hand_delta(prev_geom["left"], donor_curr_geom["left"]),
                "cross_delta_to_current": cross_delta_to_current,
                "curr_cross_distance": 0.0 if curr_cross is None else float(curr_cross),
                "donor_cross_distance": 0.0 if donor_cross is None else float(donor_cross),
                "interaction_motion_value": str(curr_attrs.get("interaction_motion", "unknown")),
                "other_hand_motion": str(curr_attrs.get("left_hand_motion", "unknown")),
            }
        )
    out.sort(
        key=lambda item: (
            item["family_dist"],
            item["row"]["seq_name"],
            item["row"]["curr_frame_idx"],
        )
    )
    for rank, item in enumerate(out):
        item["raw_rank"] = rank
    return out


def right_features(right_row, prev_geom, curr_geom):
    donor_curr_geom = right_row["curr_geom"]
    curr_cross = curr_geom.get("cross_hand_distance")
    donor_cross = donor_curr_geom.get("cross_hand_distance")
    cross_delta_to_current = 0.0
    if curr_cross is not None and donor_cross is not None:
        cross_delta_to_current = abs(float(curr_cross) - float(donor_cross))
    donor_right = right_row["curr_frame"].get("right")
    donor_group = "unknown"
    if donor_right is not None:
        donor_group = grouped_motif_signature(
            str(donor_right.get("hand_motion", "unknown")),
            list(donor_right.get("transition_labels", [])),
        )
    motion, profile = parse_grouped(donor_group)
    return {
        "curr_right_delta": hand_delta(curr_geom["right"], donor_curr_geom["right"]),
        "prev_right_delta": hand_delta(prev_geom["right"], donor_curr_geom["right"]),
        "right_cross_delta_to_current": cross_delta_to_current,
        "right_donor_cross_distance": 0.0 if donor_cross is None else float(donor_cross),
        "right_motion": motion or "unknown",
        "right_profile": [LEVEL.get(x, 0) for x in profile] if profile else [0, 0, 0, 0, 0],
    }


def pair_feature_vector(left_item, right_item, relaxed_pool_size: int, right_pool_size: int):
    pair_cross_gap = abs(left_item["donor_cross_distance"] - right_item["right_donor_cross_distance"])
    feats = [
        float(left_item["family_dist"]),
        float(left_item["curr_left_delta"]),
        float(left_item["prev_left_delta"]),
        float(left_item["cross_delta_to_current"]),
        float(left_item["curr_cross_distance"]),
        float(left_item["donor_cross_distance"]),
        float(left_item["raw_rank"]),
        float(relaxed_pool_size),
        float(right_pool_size),
        float(right_item["curr_right_delta"]),
        float(right_item["prev_right_delta"]),
        float(right_item["right_cross_delta_to_current"]),
        float(right_item["right_donor_cross_distance"]),
        float(pair_cross_gap),
        float(abs(left_item["cross_delta_to_current"] - right_item["right_cross_delta_to_current"])),
    ]
    feats.extend(float(x) for x in left_item["target_profile"])
    feats.extend(float(x) for x in left_item["donor_profile"])
    feats.extend(float(x) for x in right_item["right_profile"])
    feats.extend(one_hot(left_item["interaction_motion_value"], INTERACTION_VALUES))
    feats.extend(one_hot(left_item["other_hand_motion"], MOTION_VALUES))
    feats.extend(one_hot(right_item["right_motion"], MOTION_VALUES))
    return feats


def collect_weak_frames(data):
    rows = []
    for sequence in data["sequences"]:
        frames = sequence["frames"]
        for start, end, value in contiguous_runs(frames, TASK_FIELD):
            if end - start < 3 or not eligible_value(TASK_FIELD, value, TASK_TARGET):
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
                        "current_left_group": current_grouped_signature(curr_frame.get("left")),
                    }
                )
    return rows


def build_pair_examples(train_frames, pair_bank):
    xs = []
    ys = []
    meta = defaultdict(int)
    for entry in train_frames:
        prev_frame = entry["prev_frame"]
        curr_frame = entry["curr_frame"]
        prev_geom = entry["prev_geom"]
        curr_geom = entry["curr_geom"]
        curr_attrs = entry["curr_attrs"]
        current_left_group = entry["current_left_group"]
        right_pool = [
            row for row in right_relaxed_candidates(pair_bank, curr_attrs, prev_geom, curr_geom)
            if not (row["seq_name"] == entry["seq_name"] and row["curr_frame_idx"] == curr_frame["frame_idx"])
        ]
        left_items = [
            item for item in left_family_candidates_with_meta(pair_bank, current_left_group, curr_attrs, prev_geom, curr_geom, max_profile_distance=1)
            if not (item["row"]["seq_name"] == entry["seq_name"] and item["row"]["curr_frame_idx"] == curr_frame["frame_idx"])
        ]
        if not right_pool or not left_items:
            continue
        relaxed_pool_size = len(left_items)
        right_pool_size = len(right_pool)
        right_meta = {id(row): right_features(row, prev_geom, curr_geom) for row in right_pool}
        for left_item in left_items:
            for right_row in right_pool:
                res = evaluate_joint(prev_frame, curr_frame, prev_geom, right_row, left_item["row"])
                xs.append(pair_feature_vector(left_item, right_meta[id(right_row)], relaxed_pool_size, right_pool_size))
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


def summarize_by_subtype(rows, prefix):
    avail_key = f"{prefix}_available"
    joint_key = f"{prefix}_joint_score"
    by = defaultdict(list)
    for row in rows:
        by[(row["other_hand_motion"], row["interaction_motion_value"])].append(row)
    out = []
    for key, items in sorted(by.items()):
        other_hand_motion, interaction_motion_value = key
        avail = sum(row[avail_key] for row in items)
        out.append(
            {
                "other_hand_motion": other_hand_motion,
                "interaction_motion_value": interaction_motion_value,
                "num_frames": len(items),
                "available_rate": avail / len(items),
                "joint_score_on_available": sum(row[joint_key] for row in items if row[avail_key]) / max(avail, 1),
                "joint_hit_rate_overall": sum(row[joint_key] for row in items) / len(items),
            }
        )
    return out


def main():
    train_data = load_json(GEN / "temporal_hl_val.json")
    test_data = load_json(GEN / "temporal_hl_test.json")
    labels = overlap_labels(train_data, test_data)
    semantic_vocab = build_semantic_frame_vocab(train_data, labels)
    pair_bank = build_pair_bank(train_data, labels, semantic_vocab)

    train_frames = [
        row for row in collect_weak_frames(train_data)
        if canonical(row["seq_name"]) in labels or row["seq_name"] in labels
    ]
    test_frames = [
        row for row in collect_weak_frames(test_data)
        if canonical(row["seq_name"]) in labels or row["seq_name"] in labels
    ]

    x_train, y_train, meta = build_pair_examples(train_frames, pair_bank)
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

    rows = []
    for entry in test_frames:
        prev_frame = entry["prev_frame"]
        curr_frame = entry["curr_frame"]
        prev_geom = entry["prev_geom"]
        curr_geom = entry["curr_geom"]
        curr_attrs = entry["curr_attrs"]
        current_left_group = entry["current_left_group"]
        right_pool = right_relaxed_candidates(pair_bank, curr_attrs, prev_geom, curr_geom)
        left_items = left_family_candidates_with_meta(pair_bank, current_left_group, curr_attrs, prev_geom, curr_geom, max_profile_distance=1)

        rec = {
            "seq_name": entry["seq_name"],
            "frame_idx": curr_frame["frame_idx"],
            "other_hand_motion": "none" if curr_frame.get("left") is None else str(curr_frame["left"].get("hand_motion", "unknown")),
            "interaction_motion_value": str(curr_frame.get("interaction_motion", "unknown")),
            "right_pool_size": len(right_pool),
            "relaxed_left_pool_size": len(left_items),
        }

        for name, depth in REFERENCE_METHODS.items():
            best = None
            if right_pool and left_items:
                for right_row in right_pool:
                    for item in left_items[:depth]:
                        res = evaluate_joint(prev_frame, curr_frame, prev_geom, right_row, item["row"])
                        key = (res["joint_score"], res["left_preserve"], res["right_grouped_match"])
                        if best is None or key > best[0]:
                            best = (key, res)
            res = None if best is None else best[1]
            rec[f"{name}_available"] = int(res is not None)
            rec[f"{name}_joint_score"] = 0 if res is None else res["joint_score"]
            rec[f"{name}_right_grouped_match"] = 0 if res is None else res["right_grouped_match"]
            rec[f"{name}_left_preserve"] = 0 if res is None else res["left_preserve"]

        if right_pool and left_items:
            relaxed_pool_size = len(left_items)
            right_pool_size = len(right_pool)
            right_meta = {id(row): right_features(row, prev_geom, curr_geom) for row in right_pool}
            scored_left = []
            for item in left_items:
                pair_feats = np.asarray(
                    [pair_feature_vector(item, right_meta[id(right_row)], relaxed_pool_size, right_pool_size) for right_row in right_pool],
                    dtype=np.float32,
                )
                score = float(model.predict_proba(pair_feats)[:, 1].max())
                scored_left.append((score, item))
            ranked_left = [item for _, item in sorted(scored_left, key=lambda pair: pair[0], reverse=True)]
        else:
            ranked_left = []

        for depth in DEPTHS:
            prefix = f"pairguided_top{depth}"
            best = None
            if right_pool and ranked_left:
                for right_row in right_pool:
                    for item in ranked_left[:depth]:
                        res = evaluate_joint(prev_frame, curr_frame, prev_geom, right_row, item["row"])
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
    ranked = sorted(
        ({"method": prefix, **summary[prefix]} for prefix in prefixes),
        key=lambda row: (
            row["joint_hit_rate_overall"],
            row["joint_score_on_available"],
            row["available_rate"],
        ),
        reverse=True,
    )
    subtype_summary = {prefix: summarize_by_subtype(rows, prefix) for prefix in prefixes}

    payload = {
        "artifacts": {
            "train_json": str(GEN / "temporal_hl_val.json"),
            "test_json": str(GEN / "temporal_hl_test.json"),
        },
        "focus": {
            "task": f"{TASK_FIELD}->{TASK_TARGET}",
            "slice": "interaction only",
            "goal": "use pair-level supervision to rank left candidates under the current right pool",
            "reference_methods": REFERENCE_METHODS,
            "pairguided_depths": list(DEPTHS),
        },
        "training_stats": {
            "num_examples": int(len(y_train)),
            "num_positive": pos,
            "num_negative": neg,
            "positive_weight": float(pos_weight),
            **{k: int(v) for k, v in meta.items()},
        },
        "summary": summary,
        "ranked_methods": ranked,
        "subtype_summary": subtype_summary,
        "rows": rows,
    }

    out_json = GEN / "weak_slice_pairguided_reranker.json"
    out_md = SUM / "weak_slice_pairguided_reranker.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Weak Slice Pair-Guided Reranker",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Focus: `right_hand_motion -> opening` on interaction frames only.",
        "",
        "Train a pair-level classifier on `(right donor, left donor)` success,",
        "then rank left candidates by their best predicted pair score under the",
        "current right pool.",
        "",
        "## Training Stats",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key, value in payload["training_stats"].items():
        lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "## Method Ranking",
            "",
            "| method | avail | joint on avail | right match | left preserve | joint overall |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in ranked:
        lines.append(
            f"| {row['method']} | {fmt(row['available_rate'])} | "
            f"{fmt(row['joint_score_on_available'])} | "
            f"{fmt(row['right_grouped_match_on_available'])} | "
            f"{fmt(row['left_preserve_on_available'])} | "
            f"{fmt(row['joint_hit_rate_overall'])} |"
        )

    for prefix in prefixes:
        lines.extend(
            [
                "",
                f"## By Subtype: {prefix}",
                "",
                "| other hand motion | interaction motion | frames | avail | joint on avail | joint overall |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in subtype_summary[prefix]:
            lines.append(
                f"| {row['other_hand_motion']} | {row['interaction_motion_value']} | "
                f"{row['num_frames']} | {fmt(row['available_rate'])} | "
                f"{fmt(row['joint_score_on_available'])} | "
                f"{fmt(row['joint_hit_rate_overall'])} |"
            )

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
