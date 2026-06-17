#!/usr/bin/env python3
"""Train small PyTorch symbolic classifiers with channel-wise learned fusion."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_counter(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in counter.items()}


def canonical_label(seq_name: str) -> str:
    fixes = {
        "ROM07_RT_Finger_Occlusions": "ROM07_Rt_Finger_Occlusions",
        "ROM08_LT_Finger_Occlusions": "ROM08_Lt_Finger_Occlusions",
    }
    return fixes.get(seq_name, seq_name)


def pack_list(values):
    if values is None:
        return "none"
    return "|".join(str(v) for v in values)


def event_group_key(frame: dict[str, object]) -> str:
    parts = [f"hand_type={frame.get('hand_type', 'unknown')}"]
    for hand_name in ("right", "left"):
        hand_record = frame.get(hand_name)
        if hand_record is None:
            parts.append(f"{hand_name}=none")
            continue
        parts.append(f"{hand_name}:state={pack_list(hand_record.get('token_labels'))}")
        parts.append(f"{hand_name}:motion={hand_record.get('hand_motion', 'unknown')}")
    parts.append(f"interaction={frame.get('interaction_motion', 'unknown')}")
    return ";".join(parts)


def event_duration_bucket(num_frames: int) -> str:
    if num_frames <= 1:
        return "instant"
    if num_frames <= 3:
        return "short"
    if num_frames <= 7:
        return "medium"
    return "long"


def frame_event_token_set(frame: dict[str, object]) -> set[str]:
    tokens = {f"hand_type={frame.get('hand_type', 'unknown')}"}
    for hand_name in ("right", "left"):
        hand_record = frame.get(hand_name)
        if hand_record is None:
            tokens.add(f"{hand_name}=none")
            continue
        for token in hand_record.get("token_labels", []):
            tokens.add(f"{hand_name}:state:{token}")
        tokens.add(f"{hand_name}:motion:{hand_record.get('hand_motion', 'unknown')}")
    tokens.add(f"interaction={frame.get('interaction_motion', 'unknown')}")
    return tokens


def segment_event_signature(frames: list[dict[str, object]]) -> str:
    if not frames:
        return "event::empty"
    anchor = frames[-1]
    event = frame_event_token_set(anchor)
    event.add(f"event::duration::{event_duration_bucket(len(frames))}")
    return "||".join(sorted(event))


def iter_hand_records(frame: dict[str, object]):
    for hand_name in ("right", "left"):
        record = frame.get(hand_name)
        if record is not None:
            yield record


def overlap_labels(train_data: dict[str, object], test_data: dict[str, object]) -> set[str]:
    train_labels = {canonical_label(seq["seq_name"]) for seq in train_data["sequences"]}
    test_labels = {canonical_label(seq["seq_name"]) for seq in test_data["sequences"]}
    return train_labels & test_labels


def extract_channel_features(
    frames: list[dict[str, object]],
    include_temporal: bool,
    include_wrist_features: bool = False,
    include_duration_features: bool = False,
    include_event_features: bool = False,
    include_coordination_features: bool = False,
    include_occlusion_features: bool = False,
    include_finger_detail_features: bool = False,
    include_finger_shape_features: bool = False,
):
    edge_order = [
        ("thumb", 3), ("thumb", 2), ("thumb", 1), ("thumb", 0),
        ("index", 3), ("index", 2), ("index", 1), ("index", 0),
        ("middle", 3), ("middle", 2), ("middle", 1), ("middle", 0),
        ("ring", 3), ("ring", 2), ("ring", 1), ("ring", 0),
        ("pinky", 3), ("pinky", 2), ("pinky", 1), ("pinky", 0),
    ]
    state_hist: Counter[str] = Counter()
    transition_hist: Counter[str] = Counter()
    hand_motion_hist: Counter[str] = Counter()
    interaction_hist: Counter[str] = Counter()
    frame_indices = []
    non_stay_transitions = 0
    major_transitions = 0
    total_transition_labels = 0
    active_hand_motion = 0
    total_hand_motion = 0
    interaction_changes = 0
    total_interaction = 0
    proximal_nonstay = 0
    proximal_major = 0
    proximal_angles = []
    distal_nonstay = 0
    distal_major = 0
    distal_angles = []
    proximal_coherent_frames = 0
    temporal_hand_frames = 0
    proximal_indices = [3, 7, 11, 15, 19]
    hand_state_signatures = {"right": [], "left": []}
    hand_active_signatures = {"right": [], "left": []}
    state_persistence_hist: Counter[str] = Counter()
    activity_persistence_hist: Counter[str] = Counter()
    interaction_persistence_hist: Counter[str] = Counter()
    interaction_activity_persistence_hist: Counter[str] = Counter()
    event_hist: Counter[str] = Counter()
    event_boundary_hist: Counter[str] = Counter()
    event_run_lengths: list[int] = []
    prev_event_key = None
    current_event_segment: list[dict[str, object]] = []
    previous_event_signature = None
    coordination_sync_nonstay = 0
    coordination_sync_same = 0
    coordination_sync_opposite = 0
    coordination_pair_count = 0
    coordination_motion_same = 0
    coordination_motion_opposite = 0
    coordination_motion_known = 0
    coordination_interacting_frames = 0
    cross_distance_values: list[float] = []
    cross_distance_deltas: list[float] = []
    prev_cross_distance = None
    flexion_gap_values: list[float] = []
    thumb_gap_alignment: list[float] = []
    occlusion_stats = defaultdict(list)
    finger_shape_stats = defaultdict(list)

    for frame in frames:
        if include_temporal and include_event_features:
            current_key = event_group_key(frame)
            if prev_event_key is None or current_key == prev_event_key:
                current_event_segment.append(frame)
            else:
                event_signature = segment_event_signature(current_event_segment)
                event_hist[f"event::{event_signature}"] += 1
                event_run_lengths.append(len(current_event_segment))
                if previous_event_signature is not None:
                    event_boundary_hist[
                        f"event_boundary::{previous_event_signature}=>{event_signature}"
                    ] += 1
                previous_event_signature = event_signature
                current_event_segment = [frame]
            prev_event_key = current_key
        frame_indices.append(frame["frame_idx"])
        interaction = frame.get("interaction_motion")
        if interaction not in (None, "unknown"):
            interaction_hist[f"interaction::{interaction}"] += 1
            total_interaction += 1
            if interaction != "steady":
                interaction_changes += 1
        if include_temporal and include_coordination_features:
            if frame.get("hand_type") == "interacting":
                coordination_interacting_frames += 1
            cross_distance = frame.get("cross_hand_distance")
            if cross_distance is not None:
                cross_distance = float(cross_distance)
                cross_distance_values.append(cross_distance)
                if prev_cross_distance is not None:
                    cross_distance_deltas.append(cross_distance - prev_cross_distance)
                prev_cross_distance = cross_distance
            else:
                prev_cross_distance = None
        interaction_persist = frame.get("interaction_persistence_label")
        if interaction_persist not in (None, "unknown"):
            interaction_persistence_hist[
                f"tempo::interaction_persist::{interaction_persist}"
            ] += 1
        interaction_active_persist = frame.get("interaction_activity_persistence_label")
        if interaction_active_persist not in (None, "unknown"):
            interaction_activity_persistence_hist[
                f"tempo::interaction_activity_persist::{interaction_active_persist}"
            ] += 1
        for hand_name in ("right", "left"):
            hand_record = frame.get(hand_name)
            validity = frame.get(f"{hand_name}_validity")
            if include_occlusion_features:
                if validity is None:
                    occlusion_stats[f"{hand_name}::missing"].append(1.0)
                    occlusion_stats[f"{hand_name}::present"].append(0.0)
                else:
                    occlusion_stats[f"{hand_name}::missing"].append(0.0)
                    occlusion_stats[f"{hand_name}::present"].append(1.0)
                    for field in (
                        "valid_joint_ratio",
                        "valid_edge_ratio",
                        "full_finger_ratio",
                        "any_finger_ratio",
                        "tip_valid_ratio",
                        "distal_edge_valid_ratio",
                        "proximal_edge_valid_ratio",
                    ):
                        occlusion_stats[f"{hand_name}::{field}"].append(
                            float(validity.get(field, 0.0))
                        )
                    occlusion_stats[f"{hand_name}::basis_valid"].append(
                        1.0 if validity.get("basis_valid", False) else 0.0
                    )
            if hand_record is None:
                continue
            for token in hand_record["token_labels"]:
                state_hist[f"state::{token}"] += 1
            if include_finger_detail_features:
                for (finger_name, idx_in_finger), token in zip(
                    edge_order, hand_record["token_labels"]
                ):
                    band = "distal" if idx_in_finger <= 1 else "proximal"
                    state_hist[
                        f"state_detail::{hand_name}:{finger_name}:{band}::{token}"
                    ] += 1
            state_persist = hand_record.get("state_persistence_label")
            if state_persist not in (None, "unknown"):
                state_persistence_hist[
                    f"tempo::{hand_name}:state_persist::{state_persist}"
                ] += 1
            activity_persist = hand_record.get("activity_persistence_label")
            if activity_persist not in (None, "unknown"):
                activity_persistence_hist[
                    f"tempo::{hand_name}:activity_persist::{activity_persist}"
                ] += 1
            if include_duration_features:
                hand_state_signatures[hand_name].append(tuple(hand_record["token_labels"]))
            if include_finger_shape_features:
                flex = hand_record.get("flexion_scores", {})
                if flex:
                    flex_values = [float(v) for v in flex.values()]
                    if flex_values:
                        finger_shape_stats[f"{hand_name}::flex_mean"].append(
                            statistics.mean(flex_values)
                        )
                        finger_shape_stats[f"{hand_name}::flex_std"].append(
                            statistics.pstdev(flex_values) if len(flex_values) > 1 else 0.0
                        )
                        finger_shape_stats[f"{hand_name}::flex_range"].append(
                            max(flex_values) - min(flex_values)
                        )
                    nonthumb = [float(flex[name]) for name in ("index", "middle", "ring", "pinky") if name in flex]
                    if "thumb" in flex and nonthumb:
                        finger_shape_stats[f"{hand_name}::thumb_gap"].append(
                            float(flex["thumb"]) - statistics.mean(nonthumb)
                        )
                tokens = hand_record.get("token_labels", [])
                if len(tokens) == 20:
                    distal_tokens = [tokens[idx] for idx in (3, 7, 11, 15, 19)]
                    proximal_tokens = [tokens[idx] for idx in (0, 4, 8, 12, 16)]
                    finger_shape_stats[f"{hand_name}::distal_unique_ratio"].append(
                        len(set(distal_tokens)) / 5.0
                    )
                    finger_shape_stats[f"{hand_name}::proximal_unique_ratio"].append(
                        len(set(proximal_tokens)) / 5.0
                    )
                    finger_shape_stats[f"{hand_name}::distal_neutral_ratio"].append(
                        sum(token == "x+0_y+1_z+0" for token in distal_tokens) / 5.0
                    )
                    finger_shape_stats[f"{hand_name}::proximal_neutral_ratio"].append(
                        sum(token == "x+0_y+1_z+0" for token in proximal_tokens) / 5.0
                    )
            if include_temporal:
                for token in hand_record["transition_labels"]:
                    if token == "start":
                        continue
                    transition_hist[f"transition::{token}"] += 1
                    total_transition_labels += 1
                    if token != "stay":
                        non_stay_transitions += 1
                    if token == "major_shift":
                        major_transitions += 1
                if include_finger_detail_features:
                    for (finger_name, idx_in_finger), token in zip(
                        edge_order, hand_record.get("transition_labels", [])
                    ):
                        if token == "start":
                            continue
                        band = "distal" if idx_in_finger <= 1 else "proximal"
                        transition_hist[
                            f"transition_detail::{hand_name}:{finger_name}:{band}::{token}"
                        ] += 1
                motion = hand_record.get("hand_motion")
                if motion not in (None, "start", "unknown"):
                    hand_motion_hist[f"hand_motion::{motion}"] += 1
                    total_hand_motion += 1
                    if motion != "steady":
                        active_hand_motion += 1
                if include_duration_features:
                    transition_labels = hand_record.get("transition_labels", [])
                    active = any(
                        token not in ("start", "stay") for token in transition_labels
                    )
                    hand_active_signatures[hand_name].append(active)
                labels = hand_record.get("transition_labels", [])
                angles = hand_record.get("transition_angles_deg", [])
                if include_wrist_features and labels and angles:
                    temporal_hand_frames += 1
                    prox_labels = [labels[idx] for idx in proximal_indices]
                    prox_angles = [angles[idx] for idx in proximal_indices]
                    dist_labels = [
                        labels[idx] for idx in range(len(labels)) if idx not in proximal_indices
                    ]
                    dist_angles = [
                        angles[idx] for idx in range(len(angles)) if idx not in proximal_indices
                    ]
                    prox_nonstay_local = sum(label != "stay" for label in prox_labels)
                    dist_nonstay_local = sum(label != "stay" for label in dist_labels)
                    proximal_nonstay += prox_nonstay_local
                    distal_nonstay += dist_nonstay_local
                    proximal_major += sum(label == "major_shift" for label in prox_labels)
                    distal_major += sum(label == "major_shift" for label in dist_labels)
                    proximal_angles.extend(prox_angles)
                    distal_angles.extend(dist_angles)
                    active_prox_labels = [label for label in prox_labels if label != "stay"]
                    if len(active_prox_labels) >= 3 and len(set(active_prox_labels)) == 1:
                        proximal_coherent_frames += 1
        if include_temporal and include_coordination_features:
            right_record = frame.get("right")
            left_record = frame.get("left")
            if right_record is not None and left_record is not None:
                right_trans = right_record.get("transition_labels", [])
                left_trans = left_record.get("transition_labels", [])
                pair_count = min(len(right_trans), len(left_trans))
                for idx in range(pair_count):
                    rt = right_trans[idx]
                    lt = left_trans[idx]
                    if "start" in (rt, lt):
                        continue
                    coordination_pair_count += 1
                    right_active = rt != "stay"
                    left_active = lt != "stay"
                    if right_active and left_active:
                        coordination_sync_nonstay += 1
                        if rt == lt:
                            coordination_sync_same += 1
                        else:
                            coordination_sync_opposite += 1
                right_motion = right_record.get("hand_motion")
                left_motion = left_record.get("hand_motion")
                motion_set = {right_motion, left_motion}
                if all(m not in (None, "unknown", "start") for m in motion_set):
                    coordination_motion_known += 1
                    if right_motion == left_motion:
                        coordination_motion_same += 1
                    elif motion_set == {"opening", "closing"}:
                        coordination_motion_opposite += 1
                right_flex = right_record.get("flexion_scores", {})
                left_flex = left_record.get("flexion_scores", {})
                common = [name for name in ("thumb", "index", "middle", "ring", "pinky") if name in right_flex and name in left_flex]
                if common:
                    diffs = [abs(float(right_flex[name]) - float(left_flex[name])) for name in common]
                    flexion_gap_values.append(statistics.mean(diffs))
                if "thumb" in right_flex and "thumb" in left_flex:
                    thumb_gap_alignment.append(abs(float(right_flex["thumb"]) - float(left_flex["thumb"])))

    channels = {"state": normalize_counter(state_hist)}
    if not include_temporal:
        channels.update(
            {"transition": {}, "hand_motion": {}, "interaction": {}, "event": {}, "tempo": {}}
        )
        return channels

    if include_event_features and current_event_segment:
        event_signature = segment_event_signature(current_event_segment)
        event_hist[f"event::{event_signature}"] += 1
        event_run_lengths.append(len(current_event_segment))
        if previous_event_signature is not None:
            event_boundary_hist[
                f"event_boundary::{previous_event_signature}=>{event_signature}"
            ] += 1

    frame_deltas = [b - a for a, b in zip(frame_indices, frame_indices[1:]) if b > a]
    effective_stride = statistics.median(frame_deltas) if frame_deltas else 0.0
    num_frames = len(frame_indices)
    steps = max(num_frames - 1, 1)
    duration_units = max(steps * effective_stride, 1.0)
    channels.update(
        {
            "transition": normalize_counter(transition_hist),
            "hand_motion": normalize_counter(hand_motion_hist),
            "interaction": normalize_counter(interaction_hist),
            "event": {},
            "tempo": {
                "tempo::effective_stride": effective_stride / 18.0,
                "tempo::frames": num_frames / 128.0,
                "tempo::transition_density": non_stay_transitions / duration_units,
                "tempo::major_ratio": (
                    major_transitions / total_transition_labels
                    if total_transition_labels
                    else 0.0
                ),
                "tempo::hand_change_ratio": (
                    active_hand_motion / total_hand_motion if total_hand_motion else 0.0
                ),
                "tempo::interaction_change_ratio": (
                    interaction_changes / total_interaction if total_interaction else 0.0
                ),
            },
        }
    )
    if include_event_features:
        channels["event"].update(normalize_counter(event_hist))
        channels["event"].update(normalize_counter(event_boundary_hist))
        channels["tempo"].update(
            {
                "tempo::event_count_ratio": sum(event_hist.values()) / max(num_frames, 1),
                "tempo::event_run_mean": (
                    statistics.mean(event_run_lengths) / max(num_frames, 1)
                    if event_run_lengths
                    else 0.0
                ),
                "tempo::event_run_max": (
                    max(event_run_lengths) / max(num_frames, 1) if event_run_lengths else 0.0
                ),
                "tempo::event_run_std": (
                    statistics.pstdev(event_run_lengths) / max(num_frames, 1)
                    if len(event_run_lengths) > 1
                    else 0.0
                ),
                "tempo::event_boundary_rate": (
                    sum(event_boundary_hist.values()) / max(num_frames - 1, 1)
                ),
            }
        )
    if include_coordination_features:
        channels["tempo"].update(
            {
                "tempo::coord::sync_nonstay_ratio": (
                    coordination_sync_nonstay / max(coordination_pair_count, 1)
                ),
                "tempo::coord::sync_same_ratio": (
                    coordination_sync_same / max(coordination_pair_count, 1)
                ),
                "tempo::coord::sync_opposite_ratio": (
                    coordination_sync_opposite / max(coordination_pair_count, 1)
                ),
                "tempo::coord::motion_same_ratio": (
                    coordination_motion_same / max(coordination_motion_known, 1)
                ),
                "tempo::coord::motion_opposite_ratio": (
                    coordination_motion_opposite / max(coordination_motion_known, 1)
                ),
                "tempo::coord::interacting_ratio": (
                    coordination_interacting_frames / max(num_frames, 1)
                ),
                "tempo::coord::cross_dist_mean": (
                    statistics.mean(cross_distance_values) / 200.0 if cross_distance_values else 0.0
                ),
                "tempo::coord::cross_dist_min": (
                    min(cross_distance_values) / 200.0 if cross_distance_values else 0.0
                ),
                "tempo::coord::cross_dist_range": (
                    (max(cross_distance_values) - min(cross_distance_values)) / 200.0
                    if len(cross_distance_values) > 1
                    else 0.0
                ),
                "tempo::coord::cross_delta_mean": (
                    statistics.mean(cross_distance_deltas) / 50.0 if cross_distance_deltas else 0.0
                ),
                "tempo::coord::cross_delta_abs_mean": (
                    statistics.mean(abs(x) for x in cross_distance_deltas) / 50.0
                    if cross_distance_deltas
                    else 0.0
                ),
                "tempo::coord::flex_gap_mean": (
                    statistics.mean(flexion_gap_values) if flexion_gap_values else 0.0
                ),
                "tempo::coord::thumb_gap_mean": (
                    statistics.mean(thumb_gap_alignment) if thumb_gap_alignment else 0.0
                ),
            }
        )
    if include_wrist_features:
        channels["tempo"].update(
            {
                "tempo::proximal_nonstay_ratio": (
                    proximal_nonstay / max(proximal_nonstay + distal_nonstay, 1)
                ),
                "tempo::proximal_major_ratio": (
                    proximal_major / max(proximal_nonstay, 1)
                ),
                "tempo::proximal_angle_mean": (
                    statistics.mean(proximal_angles) / 90.0 if proximal_angles else 0.0
                ),
                "tempo::distal_angle_mean": (
                    statistics.mean(distal_angles) / 90.0 if distal_angles else 0.0
                ),
                "tempo::proximal_angle_ratio": (
                    (statistics.mean(proximal_angles) / max(statistics.mean(distal_angles), 1e-6))
                    if proximal_angles and distal_angles
                    else 0.0
                ),
                "tempo::proximal_coherence_ratio": (
                    proximal_coherent_frames / temporal_hand_frames if temporal_hand_frames else 0.0
                ),
            }
        )
    if include_duration_features:
        def run_lengths(items):
            runs = []
            prev = None
            count = 0
            for item in items:
                if prev is None or item == prev:
                    count += 1
                else:
                    runs.append(count)
                    count = 1
                prev = item
            if count > 0:
                runs.append(count)
            return runs

        static_runs = []
        active_runs = []
        state_changes = 0
        total_states = 0
        unique_ratio_values = []
        for hand_name in ("right", "left"):
            state_seq = hand_state_signatures[hand_name]
            if state_seq:
                runs = run_lengths(state_seq)
                static_runs.extend(runs)
                state_changes += max(len(runs) - 1, 0)
                total_states += len(state_seq)
                unique_ratio_values.append(len(set(state_seq)) / max(len(state_seq), 1))
            active_seq = hand_active_signatures[hand_name]
            if active_seq:
                active_runs.extend(run_lengths(active_seq))
        channels["tempo"].update(
            {
                "tempo::state_run_mean": (
                    statistics.mean(static_runs) / max(num_frames, 1) if static_runs else 0.0
                ),
                "tempo::state_run_max": (
                    max(static_runs) / max(num_frames, 1) if static_runs else 0.0
                ),
                "tempo::active_run_mean": (
                    statistics.mean(active_runs) / max(num_frames, 1) if active_runs else 0.0
                ),
                "tempo::active_run_max": (
                    max(active_runs) / max(num_frames, 1) if active_runs else 0.0
                ),
                "tempo::state_change_rate": (
                    state_changes / max(total_states - len(("right", "left")), 1)
                    if total_states > 0
                    else 0.0
                ),
                "tempo::state_uniqueness": (
                    statistics.mean(unique_ratio_values) if unique_ratio_values else 0.0
                ),
            }
        )
    if state_persistence_hist:
        channels["tempo"].update(normalize_counter(state_persistence_hist))
    if activity_persistence_hist:
        channels["tempo"].update(normalize_counter(activity_persistence_hist))
    if interaction_persistence_hist:
        channels["tempo"].update(normalize_counter(interaction_persistence_hist))
    if interaction_activity_persistence_hist:
        channels["tempo"].update(
            normalize_counter(interaction_activity_persistence_hist)
        )
    if include_occlusion_features:
        for key, values in occlusion_stats.items():
            channels["tempo"][f"tempo::occ::{key}"] = (
                statistics.mean(values) if values else 0.0
            )
    if include_finger_shape_features:
        for key, values in finger_shape_stats.items():
            channels["tempo"][f"tempo::fshape::{key}"] = (
                statistics.mean(values) if values else 0.0
            )
    return channels


def make_window_samples(
    dataset: dict[str, object],
    allowed_labels: set[str],
    include_temporal: bool,
    window_size: int,
    stride: int,
    include_wrist_features: bool = False,
) -> list[dict[str, object]]:
    samples = []
    for seq_idx, sequence in enumerate(dataset["sequences"]):
        label = canonical_label(sequence["seq_name"])
        if label not in allowed_labels:
            continue
        frames = sequence["frames"]
        if len(frames) < window_size:
            starts = [0]
        else:
            starts = list(range(0, len(frames) - window_size + 1, stride))
            if starts[-1] != len(frames) - window_size:
                starts.append(len(frames) - window_size)
        for window_idx, start in enumerate(starts):
            window = frames[start : start + window_size]
            channels = extract_channel_features(
                window, include_temporal, include_wrist_features
            )
            samples.append(
                {
                    "sample_id": f"{seq_idx}:{window_idx}",
                    "seq_name": sequence["seq_name"],
                    "label": label,
                    "channels": channels,
                }
            )
    return samples


def split_train_holdout(
    samples: list[dict[str, object]], holdout_fraction: float, seed: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample["label"]].append(sample)
    rng = random.Random(seed)
    train, holdout = [], []
    for label, items in grouped.items():
        items = list(items)
        rng.shuffle(items)
        holdout_n = max(1, int(round(len(items) * holdout_fraction)))
        holdout.extend(items[:holdout_n])
        train.extend(items[holdout_n:])
    return train, holdout


def subsample_by_fraction(
    samples: list[dict[str, object]], fraction: float, seed: int
) -> list[dict[str, object]]:
    if fraction >= 0.999:
        return list(samples)
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample["label"]].append(sample)
    rng = random.Random(seed)
    subset = []
    for label, items in grouped.items():
        items = list(items)
        rng.shuffle(items)
        keep = max(1, int(round(len(items) * fraction)))
        subset.extend(items[:keep])
    return subset


def build_channel_vocab(samples: list[dict[str, object]], channel_name: str) -> list[str]:
    vocab = set()
    for sample in samples:
        vocab.update(sample["channels"][channel_name].keys())
    return sorted(vocab)


def vectorize_channel(samples: list[dict[str, object]], channel_name: str, vocab: list[str]) -> np.ndarray:
    index = {key: idx for idx, key in enumerate(vocab)}
    mat = np.zeros((len(samples), len(vocab)), dtype=np.float32)
    for row_idx, sample in enumerate(samples):
        for key, value in sample["channels"][channel_name].items():
            if key in index:
                mat[row_idx, index[key]] = value
    return mat


@dataclass
class EncodedSet:
    state: np.ndarray
    transition: np.ndarray
    hand_motion: np.ndarray
    interaction: np.ndarray
    tempo: np.ndarray
    labels: np.ndarray
    seq_names: list[str]


def encode_samples(
    samples: list[dict[str, object]],
    vocabs: dict[str, list[str]],
    label_to_idx: dict[str, int],
) -> EncodedSet:
    return EncodedSet(
        state=vectorize_channel(samples, "state", vocabs["state"]),
        transition=vectorize_channel(samples, "transition", vocabs["transition"]),
        hand_motion=vectorize_channel(samples, "hand_motion", vocabs["hand_motion"]),
        interaction=vectorize_channel(samples, "interaction", vocabs["interaction"]),
        tempo=vectorize_channel(samples, "tempo", vocabs["tempo"]),
        labels=np.asarray([label_to_idx[s["label"]] for s in samples], dtype=np.int64),
        seq_names=[s["seq_name"] for s in samples],
    )


class ChannelDataset(Dataset):
    def __init__(self, encoded: EncodedSet):
        self.encoded = encoded

    def __len__(self):
        return len(self.encoded.labels)

    def __getitem__(self, idx: int):
        return {
            "state": torch.from_numpy(self.encoded.state[idx]),
            "transition": torch.from_numpy(self.encoded.transition[idx]),
            "hand_motion": torch.from_numpy(self.encoded.hand_motion[idx]),
            "interaction": torch.from_numpy(self.encoded.interaction[idx]),
            "tempo": torch.from_numpy(self.encoded.tempo[idx]),
            "label": torch.tensor(self.encoded.labels[idx], dtype=torch.long),
        }


class BranchEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StateOnlyNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.state_encoder = BranchEncoder(state_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        state = self.state_encoder(batch["state"])
        logits = self.classifier(state)
        return logits, {"state": state}


class TemporalFusionNet(nn.Module):
    def __init__(
        self,
        dims: dict[str, int],
        hidden_dim: int,
        gate_hidden_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.branch_names = ["state", "transition", "hand_motion", "interaction", "tempo"]
        self.encoders = nn.ModuleDict(
            {name: BranchEncoder(dims[name], hidden_dim) for name in self.branch_names}
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * len(self.branch_names), gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(gate_hidden_dim, len(self.branch_names)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        branch_feats = {name: self.encoders[name](batch[name]) for name in self.branch_names}
        stacked = torch.stack([branch_feats[name] for name in self.branch_names], dim=1)
        gate_input = torch.cat([branch_feats[name] for name in self.branch_names], dim=-1)
        gate_logits = self.gate(gate_input)
        gate_weights = torch.softmax(gate_logits, dim=-1)
        fused = (stacked * gate_weights.unsqueeze(-1)).sum(dim=1)
        logits = self.classifier(fused)
        aux = {"gate_weights": gate_weights.detach()}
        aux.update(branch_feats)
        return logits, aux


class TemporalLogitFusionNet(nn.Module):
    def __init__(
        self,
        dims: dict[str, int],
        hidden_dim: int,
        gate_hidden_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.branch_names = ["state", "transition", "hand_motion", "interaction", "tempo"]
        self.encoders = nn.ModuleDict(
            {name: BranchEncoder(dims[name], hidden_dim) for name in self.branch_names}
        )
        self.branch_heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, num_classes),
                )
                for name in self.branch_names
            }
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * len(self.branch_names), gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(gate_hidden_dim, len(self.branch_names)),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        branch_feats = {name: self.encoders[name](batch[name]) for name in self.branch_names}
        branch_logits = {name: self.branch_heads[name](branch_feats[name]) for name in self.branch_names}
        gate_input = torch.cat([branch_feats[name] for name in self.branch_names], dim=-1)
        gate_weights = torch.softmax(self.gate(gate_input), dim=-1)
        stacked_logits = torch.stack([branch_logits[name] for name in self.branch_names], dim=1)
        fused_logits = (stacked_logits * gate_weights.unsqueeze(-1)).sum(dim=1)
        aux = {"gate_weights": gate_weights.detach()}
        aux.update(branch_feats)
        return fused_logits, aux


def aggregate_sequence_predictions(
    probs: np.ndarray,
    pred_labels: np.ndarray,
    labels: list[str],
    seq_names: list[str],
    method: str,
) -> tuple[list[str], list[str]]:
    by_probs = defaultdict(list)
    by_votes = defaultdict(list)
    gt = {}
    for prob, pred, seq_name, label in zip(probs, pred_labels, seq_names, labels):
        by_probs[seq_name].append(prob)
        by_votes[seq_name].append(int(pred))
        gt[seq_name] = label
    y_true, y_pred = [], []
    for seq_name in sorted(by_probs):
        probs_arr = np.asarray(by_probs[seq_name], dtype=np.float64)
        if method == "mean_log_prob":
            agg = np.mean(np.log(np.clip(probs_arr, 1e-8, 1.0)), axis=0)
            pred_idx = int(np.argmax(agg))
        elif method == "vote":
            pred_idx = Counter(by_votes[seq_name]).most_common(1)[0][0]
        else:
            agg = np.mean(probs_arr, axis=0)
            pred_idx = int(np.argmax(agg))
        y_true.append(gt[seq_name])
        y_pred.append(labels[pred_idx])
    return y_true, y_pred


def evaluate_model(
    model: nn.Module,
    encoded: EncodedSet,
    label_names: list[str],
    device: torch.device,
    aggregation: str,
) -> tuple[float, float, dict[str, float]]:
    model.eval()
    dataset = ChannelDataset(encoded)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    probs_all, preds_all, labels_all = [], [], []
    gate_sums = None
    gate_count = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("label")
            logits, aux = model(batch)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            probs_all.append(probs.cpu().numpy())
            preds_all.append(preds.cpu().numpy())
            labels_all.append(labels.cpu().numpy())
            if "gate_weights" in aux:
                gates = aux["gate_weights"].cpu().numpy()
                gate_sums = gates.sum(axis=0) if gate_sums is None else gate_sums + gates.sum(axis=0)
                gate_count += gates.shape[0]
    probs = np.concatenate(probs_all, axis=0)
    preds = np.concatenate(preds_all, axis=0)
    labels_idx = np.concatenate(labels_all, axis=0)
    window_acc = float((preds == labels_idx).mean())
    y_true, y_pred = aggregate_sequence_predictions(
        probs, preds, label_names, encoded.seq_names, aggregation
    )
    seq_acc = float(np.mean([a == b for a, b in zip(y_true, y_pred)]))
    gate_means = {}
    if gate_sums is not None and gate_count > 0:
        for name, value in zip(["state", "transition", "hand_motion", "interaction", "tempo"], gate_sums / gate_count):
            gate_means[name] = float(value)
    return window_acc, seq_acc, gate_means


def train_once(
    train_encoded: EncodedSet,
    holdout_encoded: EncodedSet,
    test_encoded: EncodedSet,
    label_names: list[str],
    mode: str,
    device: torch.device,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    aggregation: str,
    seed: int,
    fusion_type: str,
) -> dict[str, object]:
    set_seed(seed)
    dims = {
        "state": train_encoded.state.shape[1],
        "transition": train_encoded.transition.shape[1],
        "hand_motion": train_encoded.hand_motion.shape[1],
        "interaction": train_encoded.interaction.shape[1],
        "tempo": train_encoded.tempo.shape[1],
    }
    num_classes = len(label_names)
    if mode == "state":
        model = StateOnlyNet(dims["state"], hidden_dim, num_classes)
    else:
        if fusion_type == "logit":
            model = TemporalLogitFusionNet(dims, hidden_dim, hidden_dim, num_classes)
        else:
            model = TemporalFusionNet(dims, hidden_dim, hidden_dim, num_classes)
    model.to(device)

    loader = DataLoader(ChannelDataset(train_encoded), batch_size=64, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_holdout = -1.0
    best_state = None
    patience = 20
    patience_left = patience
    for _ in range(epochs):
        model.train()
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("label")
            optimizer.zero_grad()
            logits, _ = model(batch)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        _, holdout_seq, _ = evaluate_model(
            model, holdout_encoded, label_names, device, aggregation
        )
        if holdout_seq > best_holdout:
            best_holdout = holdout_seq
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    train_window, train_seq, gate_means = evaluate_model(
        model, train_encoded, label_names, device, aggregation
    )
    holdout_window, holdout_seq, _ = evaluate_model(
        model, holdout_encoded, label_names, device, aggregation
    )
    test_window, test_seq, test_gates = evaluate_model(
        model, test_encoded, label_names, device, aggregation
    )
    return {
        "train_window_accuracy": train_window,
        "train_sequence_accuracy": train_seq,
        "holdout_window_accuracy": holdout_window,
        "holdout_sequence_accuracy": holdout_seq,
        "test_window_accuracy": test_window,
        "test_sequence_accuracy": test_seq,
        "gate_means_train": gate_means,
        "gate_means_test": test_gates,
    }


def summarize_results(results: list[dict[str, object]]) -> dict[str, float]:
    def mean_std(key: str):
        values = [r[key] for r in results]
        return statistics.mean(values), (statistics.pstdev(values) if len(values) > 1 else 0.0)

    out = {}
    for key in [
        "train_window_accuracy",
        "train_sequence_accuracy",
        "holdout_window_accuracy",
        "holdout_sequence_accuracy",
        "test_window_accuracy",
        "test_sequence_accuracy",
    ]:
        mean, std = mean_std(key)
        out[f"{key}_mean"] = mean
        out[f"{key}_std"] = std
    gate_keys = set()
    for r in results:
        gate_keys.update(r.get("gate_means_test", {}).keys())
    if gate_keys:
        out["gate_means_test"] = {
            key: statistics.mean(r["gate_means_test"].get(key, 0.0) for r in results)
            for key in sorted(gate_keys)
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/generated/symbolic_torch_summary.json"),
    )
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--fusion-type",
        choices=["embedding", "logit"],
        default="logit",
        help="Temporal branch fusion style for the learned model.",
    )
    parser.add_argument(
        "--aggregation",
        choices=["mean_prob", "mean_log_prob", "vote"],
        default="mean_log_prob",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = load_json(args.train_json)
    test_data = load_json(args.test_json)
    labels = sorted(overlap_labels(train_data, test_data))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    payload = {
        "device": str(device),
        "window_size": args.window_size,
        "stride": args.stride,
        "fractions": args.fractions,
        "seeds": args.seeds,
        "aggregation": args.aggregation,
        "hidden_dim": args.hidden_dim,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "fusion_type": args.fusion_type,
        "state_only": {"summary": {}, "results": {}},
        "temporal_hl": {"summary": {}, "results": {}},
    }

    for mode_key, include_temporal in [("state_only", False), ("temporal_hl", True)]:
        full_train = make_window_samples(
            train_data, set(labels), include_temporal, args.window_size, args.stride
        )
        full_test = make_window_samples(
            test_data, set(labels), include_temporal, args.window_size, args.stride
        )
        mode_results = {}
        for fraction in args.fractions:
            fraction_runs = []
            for seed in args.seeds:
                train_fraction = subsample_by_fraction(full_train, fraction, seed)
                train_subset, holdout_subset = split_train_holdout(
                    train_fraction, args.holdout_fraction, seed
                )
                combined = train_subset + holdout_subset + full_test
                vocabs = {
                    name: build_channel_vocab(combined, name)
                    for name in ["state", "transition", "hand_motion", "interaction", "tempo"]
                }
                train_encoded = encode_samples(train_subset, vocabs, label_to_idx)
                holdout_encoded = encode_samples(holdout_subset, vocabs, label_to_idx)
                test_encoded = encode_samples(full_test, vocabs, label_to_idx)
                fraction_runs.append(
                    train_once(
                        train_encoded=train_encoded,
                        holdout_encoded=holdout_encoded,
                        test_encoded=test_encoded,
                        label_names=labels,
                        mode="temporal" if include_temporal else "state",
                        device=device,
                        hidden_dim=args.hidden_dim,
                        epochs=args.epochs,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        aggregation=args.aggregation,
                        seed=seed,
                        fusion_type=args.fusion_type,
                    )
                )
            payload[mode_key]["results"][str(fraction)] = fraction_runs
            payload[mode_key]["summary"][str(fraction)] = summarize_results(fraction_runs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"output: {args.output}")
    for mode_key in ["state_only", "temporal_hl"]:
        print(f"\n{mode_key}")
        for fraction in map(str, args.fractions):
            item = payload[mode_key]["summary"][fraction]
            print(
                f"fraction={fraction} "
                f"test_seq={item['test_sequence_accuracy_mean']:.4f}±{item['test_sequence_accuracy_std']:.4f} "
                f"test_win={item['test_window_accuracy_mean']:.4f}±{item['test_window_accuracy_std']:.4f}"
            )
            if "gate_means_test" in item:
                print(f"gate_means={item['gate_means_test']}")


if __name__ == "__main__":
    main()
