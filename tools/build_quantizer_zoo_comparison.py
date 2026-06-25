#!/usr/bin/env python3
"""Matched-bits comparison of standard quantizers on HL directions: FSQ / PQ /
VQ / RVQ / BSQ, in both raw-3D and spherical-(theta,phi) space.

Motivation. The adaptive codebook tools use plain k-means (a 1-layer VQ). Before
claiming that is the right family, this tool places the HL direction quantization
problem next to the standard quantizer zoo, at MATCHED bit budgets, so the choice
is grounded rather than assumed:

  * VQ   - single-layer k-means (the learned-VQ proxy; our baseline)
  * RVQ  - residual VQ: multiple k-means stages, each coding the previous residual
  * PQ   - product quantization: split the vector into sub-blocks, a codebook each
  * FSQ  - finite scalar quantization: round each coordinate to a small level set
  * BSQ  - binary spherical quantization: sign bits on a (random) projection,
           then re-normalize to the sphere

Each is evaluated in TWO domains: the raw 3-D unit direction, and the 2-D
spherical-coordinate (theta, phi) parameterization. The (theta,phi) domain has
coordinate singularities at the poles, so scalar/grid quantizers that look natural
there should actually do WORSE than the same method in raw 3-D — a concrete reason
HL-style direction work should stay in the ambient 3-D space.

Expected mechanism: single-layer VQ (k-means) is competitive with a well-tuned
learned VQ; RVQ pushes high-fidelity reconstruction below a couple of degrees;
raw-3D beats spherical-coord at matched bits; BSQ is dominated.

    python tools/build_quantizer_zoo_comparison.py \
        --annot-root <InterHand annotations> --fit-split train --eval-split test \
        --bits 6 \
        --out experiments/quantizer_zoo_comparison_<date>.json

Pure numpy + stdlib; reuses build_temporal_hl + build_adaptive_direction_codebook
(its spherical k-means + metrics) so there is zero convention drift. All the extra
quantizers (PQ/FSQ/RVQ/BSQ) are implemented compactly here in numpy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import build_temporal_hl as hl
import build_adaptive_direction_codebook as adb


def _angular(recon, V):
    recon = recon / (np.linalg.norm(recon, axis=1, keepdims=True) + 1e-12)
    cos = np.clip((recon * V).sum(1), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)).mean())


def to_sphere(V):
    """Raw 3-D unit vectors -> (theta, phi): theta in [0,pi] elevation from +Y,
    phi azimuth in [-pi,pi]. Returns [N,2]."""
    y = np.clip(V[:, 1], -1.0, 1.0)
    theta = np.arccos(y)
    phi = np.arctan2(V[:, 2], V[:, 0])
    return np.stack([theta, phi], axis=1)


def from_sphere(S):
    theta, phi = S[:, 0], S[:, 1]
    s = np.sin(theta)
    return np.stack([s * np.cos(phi), np.cos(theta), s * np.sin(phi)], axis=1)


def euclid_kmeans(X, k, iters=80, seed=0, tol=1e-6):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), size=min(k, len(X)), replace=False)].copy()
    prev = np.inf
    for _ in range(iters):
        d = _cdist(X, C)
        assign = np.argmin(d, axis=1)
        newC = C.copy()
        for j in range(len(C)):
            m = assign == j
            if m.any():
                newC[j] = X[m].mean(0)
        inertia = float(np.take_along_axis(d, assign[:, None], 1).mean())
        if abs(prev - inertia) < tol:
            C = newC
            break
        prev, C = inertia, newC
    return C


def _cdist(X, C, chunk=20000):
    out = np.empty((len(X), len(C)))
    for s in range(0, len(X), chunk):
        e = s + chunk
        out[s:e] = ((X[s:e, None, :] - C[None, :, :]) ** 2).sum(-1)
    return out


# ---- quantizers in a generic vector space (X is [N,d]) -----------------------

def q_vq(Xt, Xe, k, seed=0):
    C = euclid_kmeans(Xt, k, seed=seed)
    a = np.argmin(_cdist(Xe, C), axis=1)
    return C[a], float(np.log2(len(C)))


def q_rvq(Xt, Xe, stages, k_per_stage, seed=0):
    """Residual VQ: each stage codes the residual of the previous reconstruction."""
    rec_t = np.zeros_like(Xt)
    rec_e = np.zeros_like(Xe)
    bits = 0.0
    for s in range(stages):
        res_t = Xt - rec_t
        C = euclid_kmeans(res_t, k_per_stage, seed=seed + s)
        a_t = np.argmin(_cdist(res_t, C), axis=1)
        a_e = np.argmin(_cdist(Xe - rec_e, C), axis=1)
        rec_t = rec_t + C[a_t]
        rec_e = rec_e + C[a_e]
        bits += float(np.log2(k_per_stage))
    return rec_e, bits


def q_pq(Xt, Xe, n_sub, k_sub, seed=0):
    """Product quantization: split the d dims into n_sub blocks, a codebook each."""
    d = Xt.shape[1]
    bounds = np.array_split(np.arange(d), n_sub)
    rec = np.empty_like(Xe)
    bits = 0.0
    for bi, cols in enumerate(bounds):
        if len(cols) == 0:
            continue
        C = euclid_kmeans(Xt[:, cols], k_sub, seed=seed + bi)
        a = np.argmin(_cdist(Xe[:, cols], C), axis=1)
        rec[:, cols] = C[a]
        bits += float(np.log2(k_sub))
    return rec, bits


def q_fsq(Xt, Xe, levels):
    """Finite scalar quantization: round each coordinate to `levels` uniform bins
    spanning its observed train range. bits = d * log2(levels)."""
    lo = Xt.min(0)
    hi = Xt.max(0)
    span = np.where(hi - lo > 1e-9, hi - lo, 1.0)
    g = np.clip(np.round((Xe - lo) / span * (levels - 1)), 0, levels - 1)
    rec = lo + g / (levels - 1) * span
    return rec, float(Xt.shape[1] * np.log2(levels))


def q_bsq(Xt, Xe, n_bits, seed=0):
    """Binary spherical quantization: project onto n_bits fixed directions, take
    sign bits, decode as the (normalized) sum of signed directions."""
    rng = np.random.default_rng(seed)
    d = Xt.shape[1]
    P = rng.standard_normal((d, n_bits))
    P = P / (np.linalg.norm(P, axis=0, keepdims=True) + 1e-12)
    bits_e = np.sign(Xe @ P)
    bits_e[bits_e == 0] = 1.0
    rec = bits_e @ P.T
    return rec, float(n_bits)


def evaluate_domain(Vt, Ve, domain, total_bits, seed):
    """Run every quantizer in `domain` ('raw3d' or 'sphere') at ~total_bits, and
    return reconstruction angular error (always measured back in 3-D)."""
    if domain == "raw3d":
        Xt, Xe = Vt, Ve
        back = lambda R: R
    else:
        Xt, Xe = to_sphere(Vt), to_sphere(Ve)
        back = from_sphere

    K = int(round(2 ** total_bits))
    rows = []

    rec, b = q_vq(Xt, Xe, K, seed)
    rows.append(("VQ", b, _angular(back(rec), Ve)))

    # RVQ: split the budget into stages of ~3 bits each
    stages = max(2, int(round(total_bits / 3)))
    k_stage = int(round(2 ** (total_bits / stages)))
    rec, b = q_rvq(Xt, Xe, stages, max(2, k_stage), seed)
    rows.append((f"RVQ_{stages}x{max(2, k_stage)}", b, _angular(back(rec), Ve)))

    # PQ: one block per dim, sharing the budget
    d = Xt.shape[1]
    k_sub = int(round(2 ** (total_bits / d)))
    rec, b = q_pq(Xt, Xe, d, max(2, k_sub), seed)
    rows.append((f"PQ_{d}x{max(2, k_sub)}", b, _angular(back(rec), Ve)))

    # FSQ: per-coordinate levels to match the budget
    levels = int(round(2 ** (total_bits / d)))
    rec, b = q_fsq(Xt, Xe, max(2, levels))
    rows.append((f"FSQ_L{max(2, levels)}", b, _angular(back(rec), Ve)))

    # BSQ: n_bits sign bits
    rec, b = q_bsq(Xt, Xe, int(round(total_bits)), seed)
    rows.append((f"BSQ_{int(round(total_bits))}b", b, _angular(back(rec), Ve)))

    return [{"method": m, "domain": domain, "bits": float(bb),
             "angular_deg": a} for (m, bb, a) in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annot-root", type=Path,
                    default=Path("/opt/tiger/InterHand/annotations/machine_annot"))
    ap.add_argument("--fit-split", default="train")
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--bits", type=float, default=6.0,
                    help="target total bits per direction for every quantizer")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    print(f"[fit ] loading {args.fit_split} ...", flush=True)
    Vt, _ = adb.load_local_directions(args.annot_root, args.fit_split, args.max_frames)
    print(f"[eval] loading {args.eval_split} ...", flush=True)
    Ve, _ = adb.load_local_directions(args.annot_root, args.eval_split, args.max_frames)
    print(f"[data] fit={len(Vt)} eval={len(Ve)} at ~{args.bits:.0f} bits/dir\n")

    rows = []
    for domain in ("raw3d", "sphere"):
        rows.extend(evaluate_domain(Vt, Ve, domain, args.bits, args.seed))

    rows.sort(key=lambda r: (r["domain"], r["angular_deg"]))
    print(f"{'method':14s} {'domain':7s} {'bits':>6s} {'angular°':>9s}")
    for r in rows:
        print(f"{r['method']:14s} {r['domain']:7s} {r['bits']:6.2f} "
              f"{r['angular_deg']:9.2f}")
    print("\nraw-3D vs spherical-coord at matched bits isolates the cost of the "
          "(theta,phi) pole singularities; RVQ is the high-fidelity end.")

    result = {"fit_split": args.fit_split, "eval_split": args.eval_split,
              "target_bits": float(args.bits), "seed": args.seed, "rows": rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
