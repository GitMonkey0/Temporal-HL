#!/usr/bin/env python3
"""Fit a data-adaptive direction codebook for HL and compare it to the fixed
HL-26 cube-surface codebook, at a matched code budget.

Motivation. HL quantizes every regional bone direction to the nearest of 26
fixed directions on a cube surface (build_temporal_hl.CODEBOOK). Those 26 codes
are spaced uniformly, but real finger-bone directions are extremely concentrated
(most bones point "forward" along the canonical +y axis). A uniform codebook
therefore spends most of its codes where there is little data, so its mean
quantization angle is larger than it needs to be for 26 codes.

This tool reuses the *exact* HL encoder in build_temporal_hl (build_local_basis
-> vector_to_local -> the 26-direction quantizer), collects every canonical local
bone direction it produces, then fits K unit centroids with spherical k-means and
reports mean angular quantization error for:

  * HL-26 fixed codebook            (baseline)
  * k-means K=26                    (data-adaptive, same budget -> "is the
                                      *layout* of HL-26 optimal for 26 codes?")
  * k-means K=26 initialized at HL-26 (the "build on HL" variant: keep HL's
                                      symbolic centers, let them drift to data)
  * a sweep of K                    (the rate-distortion curve)

It does NOT modify HL or its encoder; it measures how much fixed-uniform spacing
costs and what an adaptive layout recovers, as evidence for a temporal-HL
representation-design section.

    python tools/build_adaptive_direction_codebook.py \
        --annot-root /opt/tiger/InterHand/annotations/machine_annot \
        --split train --ks 12 16 26 40 64 \
        --out experiments/adaptive_codebook_<date>.json

Pure numpy + stdlib; reuses build_temporal_hl for the encoding so there is zero
convention drift from the HL labels this repo already produces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import build_temporal_hl as hl


def load_local_directions(annot_root: Path, split: str, max_frames: int | None):
    """Every canonical local bone direction HL quantizes, over both hands.

    Returns (vectors [N,3] float64 unit, edge_idx [N] int in 0..19). Uses
    frame_to_hl so the directions are identical to the HL token inputs.
    """
    fname = f"InterHand2.6M_{split}_joint_3d.json"
    # support both the flat repo layout (annot_root/<file>) and a per-split
    # subdir layout (annot_root/<split>/<file>).
    for cand in (annot_root / fname, annot_root / split / fname):
        if cand.exists():
            joint_path = cand
            break
    else:
        raise FileNotFoundError(f"{fname} not under {annot_root} (flat or /{split}/)")
    joints = hl.load_json(joint_path)
    vecs, edges = [], []
    seen = 0
    for _capture, frames in joints.items():
        if not isinstance(frames, dict):
            continue
        for _frame_idx, frame_item in frames.items():
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
                lv = rec["local_vectors"]            # 20 unit vectors in local frame
                for e, v in enumerate(lv):
                    vecs.append(v)
                    edges.append(e)
            seen += 1
            if max_frames is not None and seen >= max_frames:
                return _finish(vecs, edges)
    return _finish(vecs, edges)


def _finish(vecs, edges):
    if not vecs:
        return np.zeros((0, 3)), np.zeros((0,), dtype=np.int64)
    V = np.asarray(vecs, dtype=np.float64)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    return V, np.asarray(edges, dtype=np.int64)


def hl26_codebook() -> np.ndarray:
    """The 26 fixed HL directions, as a [26,3] unit array (same order as the
    repo's CODEBOOK)."""
    return np.asarray([c["center"] for c in hl.CODEBOOK], dtype=np.float64)


def _kpp_init(V, k, rng):
    n = len(V)
    centers = [V[int(rng.integers(n))]]
    closest = 1.0 - V @ centers[0]
    for _ in range(1, k):
        p = np.clip(closest, 0, None)
        s = p.sum()
        idx = int(rng.integers(n)) if s <= 0 else int(rng.choice(n, p=p / s))
        centers.append(V[idx])
        closest = np.minimum(closest, 1.0 - V @ centers[-1])
    return np.asarray(centers)


def spherical_kmeans(V, k, init="kmeans++", iters=100, seed=0, tol=1e-6):
    """k unit centroids maximizing mean cosine to assigned points.
    init: 'kmeans++' | 'hl26' (k must be 26)."""
    rng = np.random.default_rng(seed)
    if init == "hl26":
        if k != 26:
            raise ValueError("hl26 init requires k=26")
        C = hl26_codebook().copy()
    else:
        C = _kpp_init(V, k, rng)
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    prev = np.inf
    for _ in range(iters):
        assign = np.argmax(V @ C.T, axis=1)
        newC = C.copy()
        for j in range(k):
            m = assign == j
            if m.any():
                newC[j] = V[m].sum(0)
        newC = newC / (np.linalg.norm(newC, axis=1, keepdims=True) + 1e-12)
        inertia = float((1.0 - (V * newC[assign]).sum(1)).mean())
        if abs(prev - inertia) < tol:
            C = newC
            break
        prev, C = inertia, newC
    return C


def angular_error_deg(V, C) -> float:
    """Mean angle (deg) from each vector to its nearest center."""
    cos = np.clip((V @ C.T).max(1), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)).mean())


def empirical_entropy_bits(V, C) -> float:
    assign = np.argmax(V @ C.T, axis=1)
    counts = np.bincount(assign, minlength=C.shape[0])
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annot-root", type=Path,
                    default=Path("/opt/tiger/InterHand/annotations/machine_annot"))
    ap.add_argument("--fit-split", default="train",
                    help="split to FIT the adaptive codebook on")
    ap.add_argument("--eval-split", default="test",
                    help="split to EVALUATE all codebooks on (held-out)")
    ap.add_argument("--ks", type=int, nargs="+", default=[12, 16, 26, 40, 64])
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    print(f"[fit ] loading {args.fit_split} ...", flush=True)
    Vfit, _ = load_local_directions(args.annot_root, args.fit_split, args.max_frames)
    print(f"[fit ] {len(Vfit)} local bone directions")
    if args.eval_split == args.fit_split:
        Vev = Vfit
        print("[eval] reusing fit split (no held-out split given)")
    else:
        print(f"[eval] loading {args.eval_split} ...", flush=True)
        Vev, _ = load_local_directions(args.annot_root, args.eval_split, args.max_frames)
    print(f"[eval] {len(Vev)} local bone directions\n")

    hlcb = hl26_codebook()
    rows = []
    rows.append({"name": "hl26", "K": 26, "bits": empirical_entropy_bits(Vev, hlcb),
                 "angular_deg": angular_error_deg(Vev, hlcb)})
    # HL-init adaptive at the matched K=26 budget
    Chl = spherical_kmeans(Vfit, 26, init="hl26", seed=args.seed)
    rows.append({"name": "kmeans26_hlinit", "K": 26,
                 "bits": empirical_entropy_bits(Vev, Chl),
                 "angular_deg": angular_error_deg(Vev, Chl)})
    for k in args.ks:
        C = spherical_kmeans(Vfit, k, init="kmeans++", seed=args.seed)
        rows.append({"name": f"kmeans{k}", "K": int(k),
                     "bits": empirical_entropy_bits(Vev, C),
                     "angular_deg": angular_error_deg(Vev, C)})

    rows.sort(key=lambda r: r["bits"])
    print(f"{'codebook':18s} {'K':>4s} {'bits/vec':>9s} {'angular°':>9s}")
    for r in rows:
        print(f"{r['name']:18s} {r['K']:4d} {r['bits']:9.2f} {r['angular_deg']:9.2f}")

    hl26 = next(r for r in rows if r["name"] == "hl26")
    km26 = next((r for r in rows if r["name"] == "kmeans26"), None)
    if km26:
        drop = 100 * (hl26["angular_deg"] - km26["angular_deg"]) / hl26["angular_deg"]
        print(f"\nmatched K=26: HL-26 {hl26['angular_deg']:.2f}° -> "
              f"adaptive {km26['angular_deg']:.2f}°  ({drop:.0f}% lower)")

    result = {"fit_split": args.fit_split, "eval_split": args.eval_split,
              "n_fit": int(len(Vfit)), "n_eval": int(len(Vev)),
              "seed": args.seed, "rows": rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
