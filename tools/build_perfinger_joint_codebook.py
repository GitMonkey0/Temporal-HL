#!/usr/bin/env python3
"""Per-finger *joint* direction quantization (anatomy-aware), vs independent
per-bone codes, at matched accuracy.

Motivation. The data-adaptive codebook
(`build_adaptive_direction_codebook.py`) fits the direction distribution, but it
still treats every bone independently and ignores the *structure* of the hand. A
finger is not four independent directions: its four bones are strongly coupled
(when the finger curls, all four bend together). Spending a separate code budget
on each bone therefore pays repeatedly for information one bone already implies
about its neighbors.

This tool quantizes a whole finger *jointly*: it concatenates the finger's four
local bone directions into a single 12-D vector and learns one codebook over
those finger configurations, so one token denotes an entire finger pose
(extended / slightly bent / curled). It compares, at matched bits-per-bone:

  * independent per-bone k-means   (each bone its own 3-D codebook)
  * per-finger joint k-means        (one 12-D codebook per finger)

Expected mechanism: because the four bones are correlated, the joint codebook
reaches the same reconstruction accuracy at a much lower rate (roughly one third
of the bits), and each code is a readable finger configuration. This is the
"respect anatomy" step on top of "fit the data".

    python tools/build_perfinger_joint_codebook.py \
        --annot-root <InterHand annotations> --fit-split train --eval-split test \
        --kbone 16 26 --kfinger 64 128 256 \
        --out experiments/perfinger_joint_<date>.json

Pure numpy + stdlib; reuses the canonical HL encoder in build_temporal_hl and
the spherical k-means / metrics in build_adaptive_direction_codebook, so there is
zero convention drift from the HL labels this repo produces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import build_temporal_hl as hl
import build_adaptive_direction_codebook as adb

# EDGE_ORDER is finger-major (thumb x4, index x4, ... pinky x4), so the 20
# local vectors of a frame reshape to [5 fingers, 4 bones, 3].
N_FINGERS = len(hl.FINGER_NAMES)
BONES_PER_FINGER = len(hl.EDGE_ORDER) // N_FINGERS  # 4


def load_finger_blocks(annot_root: Path, split: str, max_frames):
    """Return a list of 5 arrays, blocks[f] = [M_f, 4, 3] unit bone directions
    for finger f, collected over both hands and all frames."""
    fname = f"InterHand2.6M_{split}_joint_3d.json"
    for cand in (annot_root / fname, annot_root / split / fname):
        if cand.exists():
            joint_path = cand
            break
    else:
        raise FileNotFoundError(f"{fname} not under {annot_root} (flat or /{split}/)")
    joints = hl.load_json(joint_path)
    blocks = [[] for _ in range(N_FINGERS)]
    seen = 0
    for _capture, frames in joints.items():
        if not isinstance(frames, dict):
            continue
        for _idx, frame_item in frames.items():
            if not isinstance(frame_item, dict) or "world_coord" not in frame_item:
                continue
            coords = frame_item["world_coord"]
            valid_ids = hl.valid_joint_ids(frame_item["joint_valid"])
            for hand_name in ("right", "left"):
                if not hl.frame_has_hand(hand_name, frame_item):
                    continue
                rec = hl.frame_to_hl(coords, valid_ids, hand_name)
                if rec is None:
                    continue
                lv = np.asarray(rec["local_vectors"], dtype=np.float64)   # [20,3]
                lv = lv / (np.linalg.norm(lv, axis=1, keepdims=True) + 1e-12)
                grid = lv.reshape(N_FINGERS, BONES_PER_FINGER, 3)
                for f in range(N_FINGERS):
                    blocks[f].append(grid[f])
            seen += 1
            if max_frames is not None and seen >= max_frames:
                return [np.asarray(b) for b in blocks]
    return [np.asarray(b) for b in blocks]


def euclid_kmeans(X, k, iters=100, seed=0, tol=1e-6):
    """Plain Lloyd k-means in R^d (here d=12, a product of 4 unit spheres)."""
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), size=k, replace=False)].copy()
    prev = np.inf
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1) if len(X) * k < 4_000_000 \
            else _chunked_assign_dist(X, C)
        assign = np.argmin(d, axis=1)
        newC = C.copy()
        for j in range(k):
            m = assign == j
            if m.any():
                newC[j] = X[m].mean(0)
        inertia = float(np.take_along_axis(d, assign[:, None], 1).mean())
        if abs(prev - inertia) < tol:
            C = newC
            break
        prev, C = inertia, newC
    return C


def _chunked_assign_dist(X, C, chunk=20000):
    out = np.empty((len(X), len(C)))
    for s in range(0, len(X), chunk):
        e = s + chunk
        out[s:e] = ((X[s:e, None, :] - C[None, :, :]) ** 2).sum(-1)
    return out


def per_bone_independent(block_train, block_eval, k):
    """4 separate 3-D codebooks (one per bone in the finger). Returns
    (mean angular deg over the 4 bones, mean bits/bone)."""
    angs, bits = [], []
    for b in range(BONES_PER_FINGER):
        Vt = block_train[:, b, :]
        Ve = block_eval[:, b, :]
        C = adb.spherical_kmeans(Vt, k, init="kmeans++")
        angs.append(adb.angular_error_deg(Ve, C))
        bits.append(adb.empirical_entropy_bits(Ve, C))
    return float(np.mean(angs)), float(np.mean(bits))


def per_finger_joint(block_train, block_eval, k):
    """One 12-D codebook for the whole finger. Returns (mean angular deg over the
    4 reconstructed bones, bits per bone = code entropy / 4)."""
    Xt = block_train.reshape(len(block_train), -1)
    Xe = block_eval.reshape(len(block_eval), -1)
    C = euclid_kmeans(Xt, k)
    d = _chunked_assign_dist(Xe, C)
    assign = np.argmin(d, axis=1)
    recon = C[assign].reshape(len(Xe), BONES_PER_FINGER, 3)
    recon = recon / (np.linalg.norm(recon, axis=2, keepdims=True) + 1e-12)
    cos = np.clip((recon * block_eval).sum(-1), -1.0, 1.0)
    ang = float(np.degrees(np.arccos(cos)).mean())
    counts = np.bincount(assign, minlength=k)
    p = counts[counts > 0] / counts.sum()
    bits_per_bone = float(-(p * np.log2(p)).sum()) / BONES_PER_FINGER
    return ang, bits_per_bone


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annot-root", type=Path,
                    default=Path("/opt/tiger/InterHand/annotations/machine_annot"))
    ap.add_argument("--fit-split", default="train")
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--kbone", type=int, nargs="+", default=[16, 26])
    ap.add_argument("--kfinger", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    print(f"[fit ] loading {args.fit_split} ...", flush=True)
    btr = load_finger_blocks(args.annot_root, args.fit_split, args.max_frames)
    print(f"[eval] loading {args.eval_split} ...", flush=True)
    bev = load_finger_blocks(args.annot_root, args.eval_split, args.max_frames)
    print(f"[data] per-finger instances (train): "
          f"{[len(x) for x in btr]}\n")

    # aggregate across the 5 fingers, weighting by eval instance count
    def aggregate(fn, k):
        a_acc, b_acc, w_acc = 0.0, 0.0, 0
        for f in range(N_FINGERS):
            if len(btr[f]) < k or len(bev[f]) == 0:
                continue
            a, b = fn(btr[f], bev[f], k)
            w = len(bev[f])
            a_acc += a * w
            b_acc += b * w
            w_acc += w
        return a_acc / w_acc, b_acc / w_acc

    rows = []
    for k in args.kbone:
        a, b = aggregate(per_bone_independent, k)
        rows.append({"method": f"per_bone_independent_k{k}", "bits_per_bone": b,
                     "angular_deg": a})
    for k in args.kfinger:
        a, b = aggregate(per_finger_joint, k)
        rows.append({"method": f"per_finger_joint_k{k}", "bits_per_bone": b,
                     "angular_deg": a})

    rows.sort(key=lambda r: r["bits_per_bone"])
    print(f"{'method':28s} {'bits/bone':>10s} {'angular°':>9s}")
    for r in rows:
        print(f"{r['method']:28s} {r['bits_per_bone']:10.2f} {r['angular_deg']:9.2f}")

    result = {"fit_split": args.fit_split, "eval_split": args.eval_split, "rows": rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
