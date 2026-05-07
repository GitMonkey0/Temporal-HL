from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Tuple

import numpy as np


HAND_PARENTS: Tuple[int, ...] = (
    0,
    1,
    2,
    3,
    0,
    5,
    6,
    7,
    0,
    9,
    10,
    11,
    0,
    13,
    14,
    15,
    0,
    17,
    18,
    19,
)

RIGHT_SLICE = slice(0, 21)
LEFT_SLICE = slice(21, 42)
IGNORE_INDEX = -100
HOLD_TOKEN = 26


def _build_direction_codebook() -> np.ndarray:
    vectors: List[np.ndarray] = []
    for x, y, z in product((-1.0, 0.0, 1.0), repeat=3):
        if x == 0.0 and y == 0.0 and z == 0.0:
            continue
        v = np.array([x, y, z], dtype=np.float32)
        v /= np.linalg.norm(v) + 1e-8
        vectors.append(v)
    return np.stack(vectors, axis=0)


DIRECTION_CODEBOOK = _build_direction_codebook()


@dataclass
class TemporalHLLabels:
    coords: np.ndarray
    joint_valid: np.ndarray
    static_tokens: np.ndarray
    motion_tokens: np.ndarray
    keyframe_mask: np.ndarray
    hand_type: List[str]


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(norm, eps, None)


def quantize_direction(vectors: np.ndarray) -> np.ndarray:
    unit = normalize(vectors)
    scores = unit @ DIRECTION_CODEBOOK.T
    return scores.argmax(axis=-1).astype(np.int64)


def _hand_region_vectors(
    hand_coords: np.ndarray,
    hand_valid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    vectors = []
    valid = []
    for child_idx, parent_idx in enumerate(HAND_PARENTS, start=1):
        child = hand_coords[:, child_idx]
        parent = hand_coords[:, parent_idx]
        vec = child - parent
        is_valid = hand_valid[:, child_idx] & hand_valid[:, parent_idx]
        vectors.append(vec)
        valid.append(is_valid)
    return np.stack(vectors, axis=1), np.stack(valid, axis=1)


def compute_region_vectors(
    coords: np.ndarray,
    joint_valid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    right_vec, right_valid = _hand_region_vectors(coords[:, RIGHT_SLICE], joint_valid[:, RIGHT_SLICE])
    left_vec, left_valid = _hand_region_vectors(coords[:, LEFT_SLICE], joint_valid[:, LEFT_SLICE])
    vectors = np.concatenate([right_vec, left_vec], axis=1)
    valid = np.concatenate([right_valid, left_valid], axis=1)
    return vectors, valid


def compute_static_tokens(region_vectors: np.ndarray, valid: np.ndarray) -> np.ndarray:
    tokens = quantize_direction(region_vectors)
    tokens = np.where(valid, tokens, IGNORE_INDEX)
    return tokens.astype(np.int64)


def compute_motion_tokens(
    region_vectors: np.ndarray,
    valid: np.ndarray,
    hold_threshold: float = 0.20,
) -> np.ndarray:
    unit_vectors = normalize(region_vectors)
    deltas = np.zeros_like(unit_vectors)
    deltas[1:] = unit_vectors[1:] - unit_vectors[:-1]

    pair_valid = np.zeros_like(valid)
    pair_valid[1:] = valid[1:] & valid[:-1]

    magnitudes = np.linalg.norm(deltas, axis=-1)
    direction_tokens = quantize_direction(deltas)
    tokens = np.where(magnitudes < hold_threshold, HOLD_TOKEN, direction_tokens)
    tokens[0] = HOLD_TOKEN
    tokens = np.where(pair_valid | (np.arange(tokens.shape[0])[:, None] == 0), tokens, IGNORE_INDEX)
    return tokens.astype(np.int64)


def smooth_signal(values: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    if kernel_size <= 1:
        return values
    radius = kernel_size // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    return np.convolve(padded, kernel, mode="valid")


def detect_keyframes(
    region_vectors: np.ndarray,
    valid: np.ndarray,
    smoothing: int = 5,
    min_gap: int = 2,
) -> np.ndarray:
    unit_vectors = normalize(region_vectors)
    delta = np.linalg.norm(unit_vectors[1:] - unit_vectors[:-1], axis=-1)
    pair_valid = valid[1:] & valid[:-1]

    frame_energy = np.zeros(region_vectors.shape[0], dtype=np.float32)
    if delta.size > 0:
        denom = np.maximum(pair_valid.sum(axis=1), 1)
        frame_energy[1:] = (delta * pair_valid).sum(axis=1) / denom
    smoothed = smooth_signal(frame_energy, kernel_size=smoothing)
    threshold = np.median(smoothed)

    keyframes = np.zeros(region_vectors.shape[0], dtype=np.int64)
    keyframes[0] = 1
    keyframes[-1] = 1

    last_idx = 0
    for idx in range(1, len(smoothed) - 1):
        is_local_min = smoothed[idx] <= smoothed[idx - 1] and smoothed[idx] <= smoothed[idx + 1]
        if is_local_min and smoothed[idx] <= threshold and (idx - last_idx) >= min_gap:
            keyframes[idx] = 1
            last_idx = idx
    return keyframes


def extract_temporal_hl(
    coords: np.ndarray,
    joint_valid: np.ndarray,
    hand_type: List[str],
) -> TemporalHLLabels:
    region_vectors, valid = compute_region_vectors(coords, joint_valid)
    static_tokens = compute_static_tokens(region_vectors, valid)
    motion_tokens = compute_motion_tokens(region_vectors, valid)
    keyframe_mask = detect_keyframes(region_vectors, valid)
    return TemporalHLLabels(
        coords=coords.astype(np.float32),
        joint_valid=joint_valid.astype(np.float32),
        static_tokens=static_tokens,
        motion_tokens=motion_tokens,
        keyframe_mask=keyframe_mask.astype(np.int64),
        hand_type=hand_type,
    )


def summarize_tokens(tokens: np.ndarray) -> Dict[int, int]:
    values, counts = np.unique(tokens[tokens >= 0], return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}

