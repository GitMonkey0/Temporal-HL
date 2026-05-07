from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from temporal_hl.notation import DIRECTION_CODEBOOK, HAND_PARENTS, HOLD_TOKEN, LEFT_SLICE, RIGHT_SLICE


@dataclass
class ReconstructionResult:
    reconstructed: np.ndarray
    mpjpe: float
    valid_ratio: float


def _mean_bone_lengths(hand_coords: np.ndarray, hand_valid: np.ndarray) -> np.ndarray:
    lengths = []
    for child_idx, parent_idx in enumerate(HAND_PARENTS, start=1):
        child = hand_coords[:, child_idx]
        parent = hand_coords[:, parent_idx]
        valid = hand_valid[:, child_idx] & hand_valid[:, parent_idx]
        bone = np.linalg.norm(child - parent, axis=-1)
        bone = bone[valid]
        lengths.append(float(bone.mean()) if bone.size > 0 else 0.05)
    return np.asarray(lengths, dtype=np.float32)


def estimate_bone_lengths(coords: np.ndarray, joint_valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    right = _mean_bone_lengths(coords[:, RIGHT_SLICE], joint_valid[:, RIGHT_SLICE] > 0.5)
    left = _mean_bone_lengths(coords[:, LEFT_SLICE], joint_valid[:, LEFT_SLICE] > 0.5)
    return right, left


def reconstruct_from_static_tokens(
    static_tokens: np.ndarray,
    coords: np.ndarray,
    joint_valid: np.ndarray,
) -> ReconstructionResult:
    coords = coords.astype(np.float32)
    joint_valid = joint_valid.astype(bool)
    recon = np.zeros_like(coords, dtype=np.float32)
    recon[:, 0] = coords[:, 0]
    recon[:, 21] = coords[:, 21]

    bone_r, bone_l = estimate_bone_lengths(coords, joint_valid)
    _reconstruct_hand(
        recon[:, RIGHT_SLICE],
        static_tokens[:, :20],
        bone_r,
        root=coords[:, 0],
    )
    _reconstruct_hand(
        recon[:, LEFT_SLICE],
        static_tokens[:, 20:],
        bone_l,
        root=coords[:, 21],
    )

    mask = joint_valid[..., None]
    error = np.linalg.norm((recon - coords) * mask, axis=-1)
    denom = np.maximum(joint_valid.sum(), 1)
    mpjpe = float(error.sum() / denom)
    valid_ratio = float(joint_valid.mean())
    return ReconstructionResult(reconstructed=recon, mpjpe=mpjpe, valid_ratio=valid_ratio)


def _reconstruct_hand(
    out: np.ndarray,
    static_tokens: np.ndarray,
    bone_lengths: np.ndarray,
    root: np.ndarray,
) -> None:
    out[:, 0] = root
    for child_idx, parent_idx in enumerate(HAND_PARENTS, start=1):
        token = np.clip(static_tokens[:, child_idx - 1], 0, len(DIRECTION_CODEBOOK) - 1)
        direction = DIRECTION_CODEBOOK[token]
        out[:, child_idx] = out[:, parent_idx] + direction * bone_lengths[child_idx - 1]


def reconstruct_from_static_and_motion(
    static_tokens: np.ndarray,
    motion_tokens: np.ndarray,
    coords: np.ndarray,
    joint_valid: np.ndarray,
    motion_scale: float = 0.15,
) -> ReconstructionResult:
    base = reconstruct_from_static_tokens(static_tokens, coords, joint_valid)
    recon = base.reconstructed.copy()
    motion_dir = np.zeros((motion_tokens.shape[0], motion_tokens.shape[1], 3), dtype=np.float32)
    valid_motion = (motion_tokens >= 0) & (motion_tokens != HOLD_TOKEN)
    motion_indices = np.clip(motion_tokens, 0, len(DIRECTION_CODEBOOK) - 1)
    motion_dir[valid_motion] = DIRECTION_CODEBOOK[motion_indices[valid_motion]]

    for hand_offset in (0, 20):
        for local_idx, parent_idx in enumerate(HAND_PARENTS, start=1):
            global_idx = hand_offset + local_idx
            vec_idx = hand_offset + local_idx - 1
            recon[:, global_idx] += motion_scale * motion_dir[:, vec_idx]

    mask = joint_valid[..., None]
    error = np.linalg.norm((recon - coords) * mask, axis=-1)
    denom = np.maximum(joint_valid.sum(), 1)
    mpjpe = float(error.sum() / denom)
    valid_ratio = float(joint_valid.mean())
    return ReconstructionResult(reconstructed=recon, mpjpe=mpjpe, valid_ratio=valid_ratio)

