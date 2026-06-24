#!/usr/bin/env python3
"""Hierarchical coarse + residual direction codes (anatomy-aware, high fidelity).

Motivation. The per-finger joint code
(`build_perfinger_joint_codebook.py`) is compact because one token summarizes a
whole finger, but a single finger code has an accuracy ceiling: it must average
over the exact pose of all four bones. This tool adds a second layer. A per-finger
*coarse* code captures "which finger pose," and a small per-bone *residual* code
refines "by how much," correcting the coarse reconstruction of each bone.

The two layers are near-independent — the coarse code already explains most of
the finger configuration, so the residual carries little extra (the
coarse->residual mutual information is small), which is exactly why a few residual
bits go a long way. The result is the high-fidelity operating point: lower error
than a flat per-bone codebook at a lower rate, while keeping an anatomically
grounded coarse layer (whole-finger pose) with a direct physical meaning.

  * flat per-bone k-means          (reference: one 3-D code per bone)
  * coarse only (per-finger joint)  (the previous tool, no residual)
  * coarse + per-bone residual      (this tool, at a few residual budgets)

    python tools/build_hierarchical_coarse_residual.py \
        --annot-root <InterHand annotations> --fit-split train --eval-split test \
        --kcoarse 128 --kres 8 16 32 \
        --out experiments/hierarchical_coarse_residual_<date>.json

Pure numpy + stdlib; reuses build_perfinger_joint_codebook (loader + 12-D
k-means) and build_adaptive_direction_codebook (spherical k-means + metrics), so
nothing here re-derives the HL encoding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import build_temporal_hl as hl
import build_adaptive_direction_codebook as adb
import build_perfinger_joint_codebook as pf

N_FINGERS = pf.N_FINGERS
BPF = pf.BONES_PER_FINGER  # 4


def _entropy_bits(assign, k):
    counts = np.bincount(assign, minlength=k)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def coarse_recon(block, C):
    """Assign each finger instance to its nearest 12-D coarse code and return the
    (unit) reconstructed 4 bone directions plus the assignment."""
    X = block.reshape(len(block), -1)
    assign = np.argmin(pf._chunked_assign_dist(X, C), axis=1)
    recon = C[assign].reshape(len(block), BPF, 3)
    recon = recon / (np.linalg.norm(recon, axis=2, keepdims=True) + 1e-12)
    return recon, assign


def flat_per_bone(btr, bev, k):
    """Reference: independent 3-D k-means per bone, aggregated."""
    a, b, w = 0.0, 0.0, 0
    for f in range(N_FINGERS):
        if len(btr[f]) < k or len(bev[f]) == 0:
            continue
        af, bf = pf.per_bone_independent(btr[f], bev[f], k)
        a += af * len(bev[f]); b += bf * len(bev[f]); w += len(bev[f])
    return a / w, b / w


def hierarchical(btr, bev, kcoarse, kres):
    """coarse per-finger code + per-bone residual code. Returns (angular deg,
    bits/bone)."""
    a_acc, bit_acc, w_acc = 0.0, 0.0, 0
    for f in range(N_FINGERS):
        if len(btr[f]) < kcoarse or len(bev[f]) == 0:
            continue
        Ctr = btr[f]
        Cev = bev[f]
        coarse = pf.euclid_kmeans(Ctr.reshape(len(Ctr), -1), kcoarse)
        rec_tr, a_tr = coarse_recon(Ctr, coarse)
        rec_ev, a_ev = coarse_recon(Cev, coarse)
        coarse_bits = _entropy_bits(a_ev, kcoarse)

        # per-bone residual codebooks fit on train residuals
        ang_bones, res_bits_bones = [], []
        for b in range(BPF):
            res_tr = Ctr[:, b, :] - rec_tr[:, b, :]
            res_ev = Cev[:, b, :] - rec_ev[:, b, :]
            Cr = pf.euclid_kmeans(res_tr, kres)
            assign = np.argmin(pf._chunked_assign_dist(res_ev, Cr), axis=1)
            fixed = rec_ev[:, b, :] + Cr[assign]
            fixed = fixed / (np.linalg.norm(fixed, axis=1, keepdims=True) + 1e-12)
            cos = np.clip((fixed * Cev[:, b, :]).sum(1), -1.0, 1.0)
            ang_bones.append(float(np.degrees(np.arccos(cos)).mean()))
            res_bits_bones.append(_entropy_bits(assign, kres))

        ang = float(np.mean(ang_bones))
        bits_per_bone = coarse_bits / BPF + float(np.mean(res_bits_bones))
        w = len(Cev)
        a_acc += ang * w; bit_acc += bits_per_bone * w; w_acc += w
    return a_acc / w_acc, bit_acc / w_acc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annot-root", type=Path,
                    default=Path("/opt/tiger/InterHand/annotations/machine_annot"))
    ap.add_argument("--fit-split", default="train")
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--kcoarse", type=int, default=128)
    ap.add_argument("--kres", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--flat-ks", type=int, nargs="+", default=[26, 64])
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    print(f"[fit ] loading {args.fit_split} ...", flush=True)
    btr = pf.load_finger_blocks(args.annot_root, args.fit_split, args.max_frames)
    print(f"[eval] loading {args.eval_split} ...", flush=True)
    bev = pf.load_finger_blocks(args.annot_root, args.eval_split, args.max_frames)

    rows = []
    for k in args.flat_ks:
        a, b = flat_per_bone(btr, bev, k)
        rows.append({"method": f"flat_per_bone_k{k}", "bits_per_bone": b, "angular_deg": a})
    # coarse-only point (per-finger joint at kcoarse)
    ca, cb, cw = 0.0, 0.0, 0
    for f in range(N_FINGERS):
        if len(btr[f]) < args.kcoarse or len(bev[f]) == 0:
            continue
        aa, bb = pf.per_finger_joint(btr[f], bev[f], args.kcoarse)
        ca += aa * len(bev[f]); cb += bb * len(bev[f]); cw += len(bev[f])
    rows.append({"method": f"coarse_only_k{args.kcoarse}",
                 "bits_per_bone": cb / cw, "angular_deg": ca / cw})
    for kr in args.kres:
        a, b = hierarchical(btr, bev, args.kcoarse, kr)
        rows.append({"method": f"coarse{args.kcoarse}+residual_k{kr}",
                     "bits_per_bone": b, "angular_deg": a})

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
