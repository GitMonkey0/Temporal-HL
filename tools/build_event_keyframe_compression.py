#!/usr/bin/env python3
"""Event-level temporal compression on the frame-count axis: curvature-aware
keyframe selection (RDP) + slerp interpolation, vs uniform subsampling.

Motivation. The other tools all compress the *per-frame* bit cost (how many bits
to describe one frame). This tool is orthogonal: it compresses the *number of
frames*. Hand motion is piecewise-smooth, so most frames are nearly predictable
from their neighbors by interpolation; only the curvature-bearing frames (where
the direction path bends) carry new information. Keeping those "event" keyframes
and reconstructing the rest is exactly the symbolic-music pianoroll->event move,
applied to the temporal axis of HL.

Method:
  * RDP keyframe selection on each bone's direction path: a Ramer-Douglas-Peucker
    style recursion that keeps a frame when the slerp of its neighbors deviates
    from it by more than a tolerance; the tolerance is tuned per track to hit a
    target keep-ratio. The selection criterion is *curvature* of the path.
  * slerp reconstruction: dropped frames are filled by spherical-linear
    interpolation between the surrounding kept keyframes.
  * baseline: uniform frame subsampling at the SAME keep-ratio, slerp-decoded.

Insight recorded in the note: the keyframe criterion must MATCH the decoder. A
slerp decoder pairs with a curvature criterion (this tool); a hold/zero-order
decoder would instead pair with a drift criterion. Mismatching them wastes
keyframes. Runs PER DATASET (HanCo and InterHand handled separately), because
their frame rates and motion statistics differ.

    python tools/build_event_keyframe_compression.py \
        --datasets interhand:/opt/tiger/InterHand/annotations/machine_annot \
        --split test --keep-ratios 0.45 0.57 \
        --out experiments/event_keyframe_compression_<date>.json

Pure numpy + stdlib; reuses build_temporal_hl (the canonical encoder) for zero
convention drift. The per-track direction sequences come from frame_to_hl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import build_temporal_hl as hl


def slerp(a, b, t):
    """Spherical linear interpolation between unit vectors a,b (each [3]) at t."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    cos = float(np.clip(a @ b, -1.0, 1.0))
    omega = np.arccos(cos)
    if omega < 1e-6:
        return a
    so = np.sin(omega)
    return (np.sin((1 - t) * omega) / so) * a + (np.sin(t * omega) / so) * b


def slerp_reconstruct(path, keep_idx):
    """Reconstruct a [T,3] unit path from the kept keyframe indices via slerp."""
    T = len(path)
    keep_idx = sorted(set(keep_idx) | {0, T - 1})
    recon = np.empty_like(path)
    for a, b in zip(keep_idx[:-1], keep_idx[1:]):
        recon[a] = path[a]
        recon[b] = path[b]
        for i in range(a + 1, b):
            t = (i - a) / (b - a)
            recon[i] = slerp(path[a], path[b], t)
    recon[0] = path[0]
    recon[-1] = path[-1]
    return recon / (np.linalg.norm(recon, axis=1, keepdims=True) + 1e-12)


def rdp_keyframes(path, tol):
    """RDP-style keyframe selection on a unit-vector path. Keeps a frame when the
    slerp of its bracketing keyframes deviates from the true frame by > tol (rad).
    Returns sorted kept indices (curvature criterion, slerp-consistent)."""
    T = len(path)
    keep = {0, T - 1}
    stack = [(0, T - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        # max angular deviation of slerp(a,b) from the true path on (a,b)
        worst_i, worst_d = -1, -1.0
        for i in range(a + 1, b):
            t = (i - a) / (b - a)
            approx = slerp(path[a], path[b], t)
            d = np.arccos(float(np.clip(approx @ path[i], -1.0, 1.0)))
            if d > worst_d:
                worst_d, worst_i = d, i
        if worst_d > tol and worst_i > 0:
            keep.add(worst_i)
            stack.append((a, worst_i))
            stack.append((worst_i, b))
    return sorted(keep)


def keyframes_to_keep_ratio(path, target_ratio, iters=18):
    """Binary-search the RDP tolerance so the kept fraction ~ target_ratio."""
    T = len(path)
    lo, hi = 0.0, np.pi
    best = list(range(T))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        keep = rdp_keyframes(path, mid)
        ratio = len(keep) / T
        if ratio > target_ratio:
            lo = mid  # need fewer keyframes -> larger tolerance
        else:
            hi = mid
        best = keep
    return best


def uniform_keyframes(T, target_ratio):
    k = max(2, int(round(T * target_ratio)))
    return sorted(set(np.linspace(0, T - 1, k).round().astype(int).tolist()))


def track_error(path, keep_idx):
    recon = slerp_reconstruct(path, keep_idx)
    cos = np.clip((recon * path).sum(1), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)).mean())


def load_tracks(joint_path: Path, max_frames):
    """Per (capture, hand, bone) unit-direction paths [T,3] from frame_to_hl."""
    joints = hl.load_json(joint_path)
    n_bones = len(hl.EDGE_ORDER)
    tracks = []
    seen = 0
    for _capture, frames in joints.items():
        if not isinstance(frames, dict):
            continue
        ordered = sorted(((k, v) for k, v in frames.items()
                          if isinstance(v, dict) and "world_coord" in v),
                         key=lambda kv: _safe_int(kv[0]))
        per_hand = {"right": [], "left": []}
        for _idx, frame_item in ordered:
            coords = frame_item["world_coord"]
            valid_ids = hl.valid_joint_ids(frame_item["joint_valid"])
            for hand_name in ("right", "left"):
                if not hl.frame_has_hand(hand_name, frame_item):
                    continue
                rec = hl.frame_to_hl(coords, valid_ids, hand_name)
                if rec is None:
                    continue
                lv = np.asarray(rec["local_vectors"], dtype=np.float64)
                per_hand[hand_name].append(lv / (np.linalg.norm(lv, axis=1, keepdims=True) + 1e-12))
            seen += 1
            if max_frames is not None and seen >= max_frames:
                ordered = []  # stop after flushing below
                break
        for h in ("right", "left"):
            seq = per_hand[h]
            if len(seq) >= 4:
                arr = np.asarray(seq)            # [T,20,3]
                for bone in range(n_bones):
                    tracks.append(arr[:, bone, :])
        if max_frames is not None and seen >= max_frames:
            break
    return tracks


def _safe_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def _resolve_joint_path(root: Path, split: str, dataset: str) -> Path | None:
    """Find a per-split joint file. InterHand uses InterHand2.6M_<split>_joint_3d
    .json; for other datasets (e.g. HanCo) try a <dataset>_<split>_joint_3d.json
    sibling, falling back to the InterHand name so the tool stays runnable."""
    cands = [
        root / f"{dataset}_{split}_joint_3d.json",
        root / split / f"{dataset}_{split}_joint_3d.json",
        root / f"InterHand2.6M_{split}_joint_3d.json",
        root / split / f"InterHand2.6M_{split}_joint_3d.json",
    ]
    for c in cands:
        if c.exists():
            return c
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+",
                    default=["interhand:/opt/tiger/InterHand/annotations/machine_annot"],
                    help="list of name:annot_root entries; run each separately")
    ap.add_argument("--split", default="test")
    ap.add_argument("--keep-ratios", type=float, nargs="+", default=[0.45, 0.57])
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    all_rows = []
    for spec in args.datasets:
        name, _, root = spec.partition(":")
        root = Path(root)
        jp = _resolve_joint_path(root, args.split, name)
        if jp is None:
            print(f"[{name}] no joint file under {root} for split={args.split}; skip")
            continue
        print(f"[{name}] loading {jp} ...", flush=True)
        tracks = load_tracks(jp, args.max_frames)
        print(f"[{name}] {len(tracks)} bone tracks")
        if not tracks:
            continue

        # accuracy floor: keep all frames (lossless slerp == identity)
        floor = float(np.mean([track_error(t, list(range(len(t)))) for t in tracks]))
        for ratio in args.keep_ratios:
            kf_err, uni_err = [], []
            for t in tracks:
                kf = keyframes_to_keep_ratio(t, ratio)
                uni = uniform_keyframes(len(t), ratio)
                kf_err.append(track_error(t, kf))
                uni_err.append(track_error(t, uni))
            all_rows.append({
                "dataset": name, "keep_ratio": float(ratio),
                "keyframe_rdp_slerp_deg": round(float(np.mean(kf_err)), 4),
                "uniform_subsample_deg": round(float(np.mean(uni_err)), 4),
                "floor_deg": round(floor, 4),
            })

    print(f"\n{'dataset':12s} {'keep':>5s} {'rdp+slerp°':>11s} "
          f"{'uniform°':>9s} {'floor°':>7s}")
    for r in all_rows:
        print(f"{r['dataset']:12s} {r['keep_ratio']:5.2f} "
              f"{r['keyframe_rdp_slerp_deg']:11.2f} "
              f"{r['uniform_subsample_deg']:9.2f} {r['floor_deg']:7.2f}")
    print("\ncurvature-aware keyframes should beat uniform subsampling at every "
          "rate; the criterion (curvature) is matched to the decoder (slerp).")

    result = {"split": args.split, "rows": all_rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
