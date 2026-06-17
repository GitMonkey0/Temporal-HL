#!/usr/bin/env python3
"""Build deterministic frame-wise HL and temporal-HL labels from InterHand.

This script derives a compact symbolic sequence representation from InterHand
3D joints. It is meant to support strong experiment design for temporal HL:

- per-frame hand state tokens over 26 discrete directions
- per-transition motion tokens over regional vectors
- coarse hand-level open/close trends
- coarse cross-hand approach/separate trends
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


RIGHT = {
    "thumb": [0, 1, 2, 3],
    "index": [4, 5, 6, 7],
    "middle": [8, 9, 10, 11],
    "ring": [12, 13, 14, 15],
    "pinky": [16, 17, 18, 19],
    "wrist": 20,
}

LEFT = {
    "thumb": [21, 22, 23, 24],
    "index": [25, 26, 27, 28],
    "middle": [29, 30, 31, 32],
    "ring": [33, 34, 35, 36],
    "pinky": [37, 38, 39, 40],
    "wrist": 41,
}

EDGE_ORDER = [
    ("thumb", 3),
    ("thumb", 2),
    ("thumb", 1),
    ("thumb", 0),
    ("index", 3),
    ("index", 2),
    ("index", 1),
    ("index", 0),
    ("middle", 3),
    ("middle", 2),
    ("middle", 1),
    ("middle", 0),
    ("ring", 3),
    ("ring", 2),
    ("ring", 1),
    ("ring", 0),
    ("pinky", 3),
    ("pinky", 2),
    ("pinky", 1),
    ("pinky", 0),
]

FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]


def duration_bucket(units: float | None) -> str:
    if units is None or units <= 0:
        return "start"
    if units <= 18:
        return "instant"
    if units <= 54:
        return "short"
    if units <= 126:
        return "medium"
    return "long"


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def vec_sub(a: list[float], b: list[float]) -> list[float]:
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def dot(a: list[float], b: list[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(a: list[float]) -> float:
    return math.sqrt(max(dot(a, a), 1e-12))


def normalize(a: list[float]) -> list[float]:
    n = norm(a)
    return [a[0] / n, a[1] / n, a[2] / n]


def cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def orthogonalize(vec: list[float], axis: list[float]) -> list[float]:
    projection = dot(vec, axis)
    return [vec[i] - projection * axis[i] for i in range(3)]


def angle_deg(a: list[float], b: list[float]) -> float:
    value = dot(normalize(a), normalize(b))
    value = max(-1.0, min(1.0, value))
    return math.degrees(math.acos(value))


def build_direction_codebook() -> list[dict[str, object]]:
    tokens = []
    token_id = 0
    for x in (-1, 0, 1):
        for y in (-1, 0, 1):
            for z in (-1, 0, 1):
                if x == 0 and y == 0 and z == 0:
                    continue
                label = f"x{x:+d}_y{y:+d}_z{z:+d}"
                tokens.append(
                    {
                        "id": token_id,
                        "label": label,
                        "center": normalize([float(x), float(y), float(z)]),
                    }
                )
                token_id += 1
    return tokens


CODEBOOK = build_direction_codebook()


def valid_joint_ids(joint_valid: list[list[bool]]) -> set[int]:
    return {idx for idx, flag in enumerate(joint_valid) if bool(flag[0])}


def get_hand_spec(hand_name: str) -> dict[str, object]:
    return RIGHT if hand_name == "right" else LEFT


def summarize_hand_validity(valid_ids: set[int], hand_name: str) -> dict[str, object]:
    spec = get_hand_spec(hand_name)
    joint_ids = [spec["wrist"]]
    for finger_name in FINGER_NAMES:
        joint_ids.extend(spec[finger_name])
    valid_joint_count = sum(joint_id in valid_ids for joint_id in joint_ids)

    edge_pairs = []
    for finger_name, idx_in_finger in EDGE_ORDER:
        child = spec[finger_name][idx_in_finger]
        parent = parent_index(hand_name, finger_name, idx_in_finger)
        edge_pairs.append((finger_name, idx_in_finger, parent, child))
    valid_edge_count = sum(
        parent in valid_ids and child in valid_ids for _, _, parent, child in edge_pairs
    )

    finger_full_valid = 0
    finger_any_valid = 0
    distal_valid = 0
    proximal_valid = 0
    tip_valid = 0
    for finger_name in FINGER_NAMES:
        joints = spec[finger_name]
        valid_joint_flags = [joint_id in valid_ids for joint_id in joints]
        if all(valid_joint_flags):
            finger_full_valid += 1
        if any(valid_joint_flags):
            finger_any_valid += 1
        if joints[0] in valid_ids:
            tip_valid += 1
        distal_valid += int(joints[0] in valid_ids and joints[1] in valid_ids)
        proximal_valid += int(joints[2] in valid_ids and joints[3] in valid_ids)

    basis_required = {spec["wrist"], spec["index"][-1], spec["middle"][-1], spec["pinky"][-1]}
    basis_valid = basis_required.issubset(valid_ids)
    return {
        "valid_joint_count": valid_joint_count,
        "valid_joint_ratio": round(valid_joint_count / len(joint_ids), 6),
        "valid_edge_count": valid_edge_count,
        "valid_edge_ratio": round(valid_edge_count / len(edge_pairs), 6),
        "full_finger_count": finger_full_valid,
        "full_finger_ratio": round(finger_full_valid / len(FINGER_NAMES), 6),
        "any_finger_count": finger_any_valid,
        "any_finger_ratio": round(finger_any_valid / len(FINGER_NAMES), 6),
        "tip_valid_count": tip_valid,
        "tip_valid_ratio": round(tip_valid / len(FINGER_NAMES), 6),
        "distal_edge_valid_count": distal_valid,
        "distal_edge_valid_ratio": round(distal_valid / len(FINGER_NAMES), 6),
        "proximal_edge_valid_count": proximal_valid,
        "proximal_edge_valid_ratio": round(proximal_valid / len(FINGER_NAMES), 6),
        "basis_valid": basis_valid,
    }


def frame_has_hand(hand_name: str, frame_item: dict[str, object]) -> bool:
    hand_type = frame_item["hand_type"]
    if hand_name == "right":
        return hand_type in {"right", "interacting"}
    return hand_type in {"left", "interacting"}


def build_local_basis(
    coords: list[list[float]], valid_ids: set[int], hand_name: str
) -> tuple[list[float], list[float], list[float]] | None:
    spec = get_hand_spec(hand_name)
    wrist = spec["wrist"]
    index_base = spec["index"][-1]
    middle_base = spec["middle"][-1]
    pinky_base = spec["pinky"][-1]
    required = {wrist, index_base, middle_base, pinky_base}
    if not required.issubset(valid_ids):
        return None

    origin = coords[wrist]
    y_axis = normalize(vec_sub(coords[middle_base], origin))
    palm_span = vec_sub(coords[pinky_base], coords[index_base])
    z_axis = orthogonalize(palm_span, y_axis)
    if norm(z_axis) < 1e-6:
        return None
    z_axis = normalize(z_axis)
    x_axis = cross(y_axis, z_axis)
    if norm(x_axis) < 1e-6:
        return None
    x_axis = normalize(x_axis)
    z_axis = normalize(cross(x_axis, y_axis))
    return x_axis, y_axis, z_axis


def parent_index(hand_name: str, finger_name: str, idx_in_finger: int) -> int:
    spec = get_hand_spec(hand_name)
    if idx_in_finger == 3:
        return spec["wrist"]
    return spec[finger_name][idx_in_finger + 1]


def vector_to_local(
    vec: list[float], basis: tuple[list[float], list[float], list[float]]
) -> list[float]:
    x_axis, y_axis, z_axis = basis
    return [dot(vec, x_axis), dot(vec, y_axis), dot(vec, z_axis)]


def quantize_direction(vec: list[float]) -> tuple[int, str]:
    unit = normalize(vec)
    best = max(CODEBOOK, key=lambda item: dot(unit, item["center"]))  # type: ignore[arg-type]
    return int(best["id"]), str(best["label"])


def compute_finger_flexion(
    coords: list[list[float]], valid_ids: set[int], hand_name: str
) -> dict[str, float]:
    spec = get_hand_spec(hand_name)
    wrist = spec["wrist"]
    middle_base = spec["middle"][-1]
    if wrist not in valid_ids or middle_base not in valid_ids:
        return {}
    palm_scale = norm(vec_sub(coords[middle_base], coords[wrist]))
    if palm_scale < 1e-6:
        return {}

    scores = {}
    for finger_name in FINGER_NAMES:
        finger_ids = spec[finger_name]
        tip = finger_ids[0]
        if tip not in valid_ids:
            continue
        scores[finger_name] = norm(vec_sub(coords[tip], coords[wrist])) / palm_scale
    return scores


def summarize_flexion_transition(
    prev_scores: dict[str, float], curr_scores: dict[str, float], threshold: float
) -> str:
    deltas = []
    for finger_name in FINGER_NAMES:
        if finger_name in prev_scores and finger_name in curr_scores:
            deltas.append(curr_scores[finger_name] - prev_scores[finger_name])
    if not deltas:
        return "unknown"
    pos = sum(delta > threshold for delta in deltas)
    neg = sum(delta < -threshold for delta in deltas)
    if pos == 0 and neg == 0:
        return "steady"
    if pos >= max(2, neg + 1):
        return "opening"
    if neg >= max(2, pos + 1):
        return "closing"
    return "mixed"


def min_cross_hand_distance(
    coords: list[list[float]], valid_ids: set[int]
) -> float | None:
    right_ids = [0, 4, 8, 12, 16, 20]
    left_ids = [21, 25, 29, 33, 37, 41]
    usable_right = [joint_id for joint_id in right_ids if joint_id in valid_ids]
    usable_left = [joint_id for joint_id in left_ids if joint_id in valid_ids]
    if not usable_right or not usable_left:
        return None
    best = None
    for right_id in usable_right:
        for left_id in usable_left:
            distance = norm(vec_sub(coords[right_id], coords[left_id]))
            best = distance if best is None else min(best, distance)
    return best


def summarize_interaction_transition(
    prev_distance: float | None, curr_distance: float | None, threshold: float
) -> str:
    if prev_distance is None or curr_distance is None:
        return "unknown"
    delta = curr_distance - prev_distance
    if abs(delta) <= threshold:
        return "steady"
    return "separate" if delta > 0 else "approach"


def frame_to_hl(
    coords: list[list[float]], valid_ids: set[int], hand_name: str
) -> dict[str, object] | None:
    basis = build_local_basis(coords, valid_ids, hand_name)
    if basis is None:
        return None

    token_ids = []
    token_labels = []
    raw_local_vectors = []
    spec = get_hand_spec(hand_name)
    for finger_name, idx_in_finger in EDGE_ORDER:
        child = spec[finger_name][idx_in_finger]
        parent = parent_index(hand_name, finger_name, idx_in_finger)
        if child not in valid_ids or parent not in valid_ids:
            return None
        local_vec = vector_to_local(vec_sub(coords[child], coords[parent]), basis)
        token_id, token_label = quantize_direction(local_vec)
        token_ids.append(token_id)
        token_labels.append(token_label)
        raw_local_vectors.append([round(v, 6) for v in normalize(local_vec)])

    return {
        "token_ids": token_ids,
        "token_labels": token_labels,
        "local_vectors": raw_local_vectors,
        "flexion_scores": compute_finger_flexion(coords, valid_ids, hand_name),
    }


def transition_label(
    prev_vectors: list[list[float]],
    curr_vectors: list[list[float]],
    prev_tokens: list[int],
    curr_tokens: list[int],
    stay_deg: float,
    minor_deg: float,
) -> tuple[list[str], list[float]]:
    labels = []
    angles = []
    for prev_vec, curr_vec, prev_token, curr_token in zip(
        prev_vectors, curr_vectors, prev_tokens, curr_tokens
    ):
        angle = angle_deg(prev_vec, curr_vec)
        angles.append(round(angle, 4))
        if angle <= stay_deg and prev_token == curr_token:
            labels.append("stay")
        elif angle <= minor_deg:
            labels.append("minor_shift")
        else:
            labels.append("major_shift")
    return labels, angles


def build_sequence_index(data: dict[str, object]) -> dict[tuple[int, str], dict[str, object]]:
    sequences: dict[tuple[int, str], dict[str, object]] = {}
    for image in data["images"]:
        seq_key = (int(image["capture"]), str(image["seq_name"]))
        record = sequences.setdefault(
            seq_key,
            {
                "subject": int(image["subject"]),
                "frames": set(),
                "cameras": set(),
            },
        )
        record["frames"].add(int(image["frame_idx"]))
        record["cameras"].add(str(image["camera"]))
    return sequences


def annotate_segment_duration_labels(
    frames: list[dict[str, object]], stats: dict[str, object]
) -> None:
    hand_segment_hist: Counter[str] = stats.setdefault("hand_segment_duration_hist", Counter())
    hand_activity_segment_hist: Counter[str] = stats.setdefault(
        "hand_activity_segment_hist", Counter()
    )
    interaction_segment_hist: Counter[str] = stats.setdefault(
        "interaction_segment_duration_hist", Counter()
    )
    interaction_activity_segment_hist: Counter[str] = stats.setdefault(
        "interaction_activity_segment_hist", Counter()
    )

    def annotate_runs(values, setter, hist: Counter[str], prefix: str = ""):
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and values[end] == values[start]:
                end += 1
            run_frames = frames[start:end]
            if len(run_frames) <= 1:
                units = 0
            else:
                units = int(run_frames[-1]["frame_idx"]) - int(run_frames[0]["frame_idx"])
            label = duration_bucket(units)
            for idx in range(start, end):
                setter(idx, units, label)
            hist[f"{prefix}{label}"] += end - start
            start = end

    for hand_name in ("right", "left"):
        state_values = []
        activity_values = []
        for frame in frames:
            hand_record = frame.get(hand_name)
            state_values.append(
                None if hand_record is None else tuple(hand_record.get("token_ids", []))
            )
            activity_values.append(
                None
                if hand_record is None
                else hand_record.get("activity_persistence_label", "missing")
            )

        annotate_runs(
            state_values,
            lambda idx, units, label, hand_name=hand_name: frames[idx][hand_name].update(  # type: ignore[index]
                {
                    "state_segment_duration_units": units,
                    "state_segment_duration_label": label,
                }
            )
            if frames[idx].get(hand_name) is not None
            else None,
            hand_segment_hist,
            prefix=f"{hand_name}::",
        )
        annotate_runs(
            activity_values,
            lambda idx, units, label, hand_name=hand_name: frames[idx][hand_name].update(  # type: ignore[index]
                {
                    "activity_segment_duration_units": units,
                    "activity_segment_duration_label": label,
                }
            )
            if frames[idx].get(hand_name) is not None
            else None,
            hand_activity_segment_hist,
            prefix=f"{hand_name}::",
        )

    interaction_values = [frame.get("interaction_motion") for frame in frames]
    interaction_active_values = [
        frame.get("interaction_activity_persistence_label", "missing") for frame in frames
    ]
    annotate_runs(
        interaction_values,
        lambda idx, units, label: frames[idx].update(
            {
                "interaction_segment_duration_units": units,
                "interaction_segment_duration_label": label,
            }
        ),
        interaction_segment_hist,
    )
    annotate_runs(
        interaction_active_values,
        lambda idx, units, label: frames[idx].update(
            {
                "interaction_activity_segment_duration_units": units,
                "interaction_activity_segment_duration_label": label,
            }
        ),
        interaction_activity_segment_hist,
    )


def export_sequences(
    data: dict[str, object],
    joints: dict[str, object],
    stay_deg: float,
    minor_deg: float,
    flex_threshold: float,
    approach_threshold: float,
) -> tuple[dict[str, object], dict[str, object]]:
    seq_index = build_sequence_index(data)
    exported = {"direction_codebook": CODEBOOK, "sequences": []}
    stats = {
        "num_sequences": 0,
        "num_frames": 0,
        "num_valid_hand_frames": Counter(),
        "state_token_hist": Counter(),
        "transition_token_hist": Counter(),
        "hand_motion_hist": Counter(),
        "interaction_motion_hist": Counter(),
        "state_persistence_hist": Counter(),
        "activity_persistence_hist": Counter(),
        "interaction_persistence_hist": Counter(),
        "interaction_activity_persistence_hist": Counter(),
    }

    for (capture, seq_name), meta in sorted(seq_index.items()):
        capture_frames = joints.get(str(capture), {})
        frames = []
        prev_right = None
        prev_left = None
        prev_cross_distance = None
        prev_state_signature = {"right": None, "left": None}
        prev_state_change_frame = {"right": None, "left": None}
        prev_active_flag = {"right": None, "left": None}
        prev_active_change_frame = {"right": None, "left": None}
        prev_interaction_state = None
        prev_interaction_state_change_frame = None
        prev_interaction_active = None
        prev_interaction_active_change_frame = None

        for frame_idx in sorted(meta["frames"]):
            frame_item = capture_frames.get(str(frame_idx))
            if frame_item is None:
                continue
            coords = frame_item["world_coord"]
            valid_ids = valid_joint_ids(frame_item["joint_valid"])
            frame_record = {
                "frame_idx": frame_idx,
                "hand_type": frame_item["hand_type"],
                "hand_type_valid": bool(frame_item["hand_type_valid"]),
            }

            current_right = None
            if frame_has_hand("right", frame_item):
                frame_record["right_validity"] = summarize_hand_validity(valid_ids, "right")
                current_right = frame_to_hl(coords, valid_ids, "right")
                if current_right is not None:
                    stats["num_valid_hand_frames"]["right"] += 1
                    stats["state_token_hist"].update(current_right["token_labels"])
                    if prev_right is None:
                        current_right["transition_labels"] = ["start"] * 20
                        current_right["transition_angles_deg"] = [0.0] * 20
                        current_right["hand_motion"] = "start"
                    else:
                        labels, angles = transition_label(
                            prev_right["local_vectors"],
                            current_right["local_vectors"],
                            prev_right["token_ids"],
                            current_right["token_ids"],
                            stay_deg,
                            minor_deg,
                        )
                        current_right["transition_labels"] = labels
                        current_right["transition_angles_deg"] = angles
                        current_right["hand_motion"] = summarize_flexion_transition(
                            prev_right["flexion_scores"],
                            current_right["flexion_scores"],
                            flex_threshold,
                        )
                        stats["transition_token_hist"].update(labels)
                        stats["hand_motion_hist"][current_right["hand_motion"]] += 1
                    state_signature = tuple(current_right["token_ids"])
                    if state_signature != prev_state_signature["right"]:
                        prev_state_signature["right"] = state_signature
                        prev_state_change_frame["right"] = frame_idx
                    state_units = (
                        None
                        if prev_state_change_frame["right"] is None
                        else frame_idx - prev_state_change_frame["right"]
                    )
                    current_right["state_persistence_units"] = state_units
                    current_right["state_persistence_label"] = duration_bucket(state_units)
                    active_flag = any(
                        token not in ("start", "stay")
                        for token in current_right["transition_labels"]
                    )
                    if active_flag != prev_active_flag["right"]:
                        prev_active_flag["right"] = active_flag
                        prev_active_change_frame["right"] = frame_idx
                    active_units = (
                        None
                        if prev_active_change_frame["right"] is None
                        else frame_idx - prev_active_change_frame["right"]
                    )
                    current_right["activity_persistence_units"] = active_units
                    current_right["activity_persistence_label"] = duration_bucket(active_units)
                    stats["state_persistence_hist"][
                        f"right::{current_right['state_persistence_label']}"
                    ] += 1
                    stats["activity_persistence_hist"][
                        f"right::{current_right['activity_persistence_label']}"
                    ] += 1
                else:
                    prev_state_signature["right"] = None
                    prev_state_change_frame["right"] = None
                    prev_active_flag["right"] = None
                    prev_active_change_frame["right"] = None
                frame_record["right"] = current_right
            else:
                frame_record["right"] = None
                frame_record["right_validity"] = None
                prev_right = None
                prev_state_signature["right"] = None
                prev_state_change_frame["right"] = None
                prev_active_flag["right"] = None
                prev_active_change_frame["right"] = None

            current_left = None
            if frame_has_hand("left", frame_item):
                frame_record["left_validity"] = summarize_hand_validity(valid_ids, "left")
                current_left = frame_to_hl(coords, valid_ids, "left")
                if current_left is not None:
                    stats["num_valid_hand_frames"]["left"] += 1
                    stats["state_token_hist"].update(current_left["token_labels"])
                    if prev_left is None:
                        current_left["transition_labels"] = ["start"] * 20
                        current_left["transition_angles_deg"] = [0.0] * 20
                        current_left["hand_motion"] = "start"
                    else:
                        labels, angles = transition_label(
                            prev_left["local_vectors"],
                            current_left["local_vectors"],
                            prev_left["token_ids"],
                            current_left["token_ids"],
                            stay_deg,
                            minor_deg,
                        )
                        current_left["transition_labels"] = labels
                        current_left["transition_angles_deg"] = angles
                        current_left["hand_motion"] = summarize_flexion_transition(
                            prev_left["flexion_scores"],
                            current_left["flexion_scores"],
                            flex_threshold,
                        )
                        stats["transition_token_hist"].update(labels)
                        stats["hand_motion_hist"][current_left["hand_motion"]] += 1
                    state_signature = tuple(current_left["token_ids"])
                    if state_signature != prev_state_signature["left"]:
                        prev_state_signature["left"] = state_signature
                        prev_state_change_frame["left"] = frame_idx
                    state_units = (
                        None
                        if prev_state_change_frame["left"] is None
                        else frame_idx - prev_state_change_frame["left"]
                    )
                    current_left["state_persistence_units"] = state_units
                    current_left["state_persistence_label"] = duration_bucket(state_units)
                    active_flag = any(
                        token not in ("start", "stay")
                        for token in current_left["transition_labels"]
                    )
                    if active_flag != prev_active_flag["left"]:
                        prev_active_flag["left"] = active_flag
                        prev_active_change_frame["left"] = frame_idx
                    active_units = (
                        None
                        if prev_active_change_frame["left"] is None
                        else frame_idx - prev_active_change_frame["left"]
                    )
                    current_left["activity_persistence_units"] = active_units
                    current_left["activity_persistence_label"] = duration_bucket(active_units)
                    stats["state_persistence_hist"][
                        f"left::{current_left['state_persistence_label']}"
                    ] += 1
                    stats["activity_persistence_hist"][
                        f"left::{current_left['activity_persistence_label']}"
                    ] += 1
                else:
                    prev_state_signature["left"] = None
                    prev_state_change_frame["left"] = None
                    prev_active_flag["left"] = None
                    prev_active_change_frame["left"] = None
                frame_record["left"] = current_left
            else:
                frame_record["left"] = None
                frame_record["left_validity"] = None
                prev_left = None
                prev_state_signature["left"] = None
                prev_state_change_frame["left"] = None
                prev_active_flag["left"] = None
                prev_active_change_frame["left"] = None

            cross_distance = min_cross_hand_distance(coords, valid_ids)
            frame_record["interaction_motion"] = summarize_interaction_transition(
                prev_cross_distance, cross_distance, approach_threshold
            )
            if frame_record["interaction_motion"] != "unknown":
                stats["interaction_motion_hist"][frame_record["interaction_motion"]] += 1
            if frame_record["interaction_motion"] != prev_interaction_state:
                prev_interaction_state = frame_record["interaction_motion"]
                prev_interaction_state_change_frame = frame_idx
            interaction_state_units = (
                None
                if prev_interaction_state_change_frame is None
                else frame_idx - prev_interaction_state_change_frame
            )
            frame_record["interaction_persistence_units"] = interaction_state_units
            frame_record["interaction_persistence_label"] = duration_bucket(
                interaction_state_units
            )
            stats["interaction_persistence_hist"][
                frame_record["interaction_persistence_label"]
            ] += 1
            interaction_active = frame_record["interaction_motion"] not in (
                "unknown",
                "start",
                "steady",
            )
            if interaction_active != prev_interaction_active:
                prev_interaction_active = interaction_active
                prev_interaction_active_change_frame = frame_idx
            interaction_active_units = (
                None
                if prev_interaction_active_change_frame is None
                else frame_idx - prev_interaction_active_change_frame
            )
            frame_record["interaction_activity_persistence_units"] = (
                interaction_active_units
            )
            frame_record["interaction_activity_persistence_label"] = duration_bucket(
                interaction_active_units
            )
            stats["interaction_activity_persistence_hist"][
                frame_record["interaction_activity_persistence_label"]
            ] += 1
            frame_record["cross_hand_distance"] = (
                round(cross_distance, 4) if cross_distance is not None else None
            )

            prev_right = current_right
            prev_left = current_left
            prev_cross_distance = cross_distance
            frames.append(frame_record)

        if not frames:
            continue
        annotate_segment_duration_labels(frames, stats)
        exported["sequences"].append(
            {
                "capture": capture,
                "seq_name": seq_name,
                "subject": meta["subject"],
                "num_cameras": len(meta["cameras"]),
                "num_frames": len(frames),
                "frames": frames,
            }
        )
        stats["num_sequences"] += 1
        stats["num_frames"] += len(frames)

    stats["num_valid_hand_frames"] = dict(stats["num_valid_hand_frames"])
    stats["state_token_hist"] = dict(stats["state_token_hist"].most_common())
    stats["transition_token_hist"] = dict(stats["transition_token_hist"].most_common())
    stats["hand_motion_hist"] = dict(stats["hand_motion_hist"].most_common())
    stats["interaction_motion_hist"] = dict(stats["interaction_motion_hist"].most_common())
    stats["state_persistence_hist"] = dict(
        stats["state_persistence_hist"].most_common()
    )
    stats["activity_persistence_hist"] = dict(
        stats["activity_persistence_hist"].most_common()
    )
    stats["interaction_persistence_hist"] = dict(
        stats["interaction_persistence_hist"].most_common()
    )
    stats["interaction_activity_persistence_hist"] = dict(
        stats["interaction_activity_persistence_hist"].most_common()
    )
    stats["hand_segment_duration_hist"] = dict(
        stats["hand_segment_duration_hist"].most_common()
    )
    stats["hand_activity_segment_hist"] = dict(
        stats["hand_activity_segment_hist"].most_common()
    )
    stats["interaction_segment_duration_hist"] = dict(
        stats["interaction_segment_duration_hist"].most_common()
    )
    stats["interaction_activity_segment_hist"] = dict(
        stats["interaction_activity_segment_hist"].most_common()
    )
    stats["num_state_bins_used"] = len(stats["state_token_hist"])
    stats["num_transition_bins_used"] = len(stats["transition_token_hist"])
    return exported, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annot-root",
        type=Path,
        default=Path("/opt/tiger/InterHand/annotations/machine_annot"),
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/generated/temporal_hl_val.json"),
    )
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--stay-deg", type=float, default=10.0)
    parser.add_argument("--minor-deg", type=float, default=35.0)
    parser.add_argument("--flex-threshold", type=float, default=0.03)
    parser.add_argument("--approach-threshold", type=float, default=8.0)
    args = parser.parse_args()

    data_path = args.annot_root / f"InterHand2.6M_{args.split}_data.json"
    joint_path = args.annot_root / f"InterHand2.6M_{args.split}_joint_3d.json"
    data = load_json(data_path)
    joints = load_json(joint_path)

    exported, stats = export_sequences(
        data=data,
        joints=joints,
        stay_deg=args.stay_deg,
        minor_deg=args.minor_deg,
        flex_threshold=args.flex_threshold,
        approach_threshold=args.approach_threshold,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(exported, f)

    summary_path = args.summary_output
    if summary_path is None:
        summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(stats, f, indent=2)

    print(f"split: {args.split}")
    print(f"output: {args.output}")
    print(f"summary_output: {summary_path}")
    print(f"num_sequences: {stats['num_sequences']}")
    print(f"num_frames: {stats['num_frames']}")
    print(f"num_valid_hand_frames: {stats['num_valid_hand_frames']}")
    print(f"num_state_bins_used: {stats['num_state_bins_used']}")
    print(f"num_transition_bins_used: {stats['num_transition_bins_used']}")
    print(f"top_state_tokens: {list(stats['state_token_hist'].items())[:10]}")
    print(f"transition_token_hist: {stats['transition_token_hist']}")
    print(f"hand_motion_hist: {stats['hand_motion_hist']}")
    print(f"interaction_motion_hist: {stats['interaction_motion_hist']}")


if __name__ == "__main__":
    main()
