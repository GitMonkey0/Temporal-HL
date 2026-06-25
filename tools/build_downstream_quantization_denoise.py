#!/usr/bin/env python3
"""Downstream denoising probe: quantization as an implicit denoiser, and whether
a per-finger (anatomy-prior) codebook denoises better than a per-bone one.

Motivation. A discrete codebook is a prior: snapping a noisy direction to the
nearest codeword pulls it toward the manifold of poses the hand actually takes.
If the codewords are whole-finger configurations (per-finger joint codes), that
prior is anatomical — a noisy bone is corrected not just toward "some plausible
direction" but toward "a direction consistent with the rest of its finger". This
probe measures that denoising directly.

Protocol: take clean GT local bone directions, add synthetic angular noise at a
sweep of sigma (degrees), quantize the noisy directions (snap to nearest
prototype), and measure the angular error of the quantized result vs the clean GT.
The "correction" is (error of noisy input) - (error of quantized output): a
positive number means quantization moved the noisy direction CLOSER to clean. It
compares:

  * per_bone   - each bone snapped independently to its own 3-D codebook
  * per_finger - each finger's 4 bones snapped jointly to a 12-D codebook (the
                 codeword is a real finger pose -> anatomical prior)

Expected mechanism: at small noise both help little; as sigma grows the
per-finger codebook corrects more (its joint prior rejects anatomically
implausible per-bone noise), so at sigma >= ~15 deg the average correction is
clearly positive and grows with the noise level.

    python tools/build_downstream_quantization_denoise.py \
        --annot-root <InterHand annotations> --fit-split train --eval-split test \
        --sigmas 5 10 15 20 30 --kbone 26 --kfinger 128 \
        --out experiments/downstream_quantization_denoise_<date>.json

Pure numpy + stdlib; reuses build_temporal_hl + build_adaptive_direction_codebook
(+ build_perfinger_joint_codebook) for zero convention drift.
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


def add_angular_noise(V, sigma_deg, rng):
    """Perturb each unit vector by a random rotation of ~sigma_deg (tangent-space
    Gaussian then renormalize). V: [...,3]."""
    sigma = np.radians(sigma_deg)
    noise = rng.standard_normal(V.shape) * sigma
    # project noise onto the tangent plane of each V, then step + renormalize
    radial = (noise * V).sum(-1, keepdims=True) * V
    tangent = noise - radial
    out = V + tangent
    return out / (np.linalg.norm(out, axis=-1, keepdims=True) + 1e-12)


def _ang(a, b):
    cos = np.clip((a * b).sum(-1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def per_bone_denoise(blocks_tr, blocks_ev, k, sigma, seed):
    """Fit a per-bone 3-D codebook; snap noisy eval bones; return mean correction
    (deg) and mean residual error (deg), aggregated over fingers and bones."""
    rng = np.random.default_rng(seed)
    corr_acc, err_acc, w = 0.0, 0.0, 0
    for f in range(N_FINGERS):
        if len(blocks_tr[f]) < k or len(blocks_ev[f]) == 0:
            continue
        for b in range(BPF):
            Vt = blocks_tr[f][:, b, :]
            Ve = blocks_ev[f][:, b, :]
            C = adb.spherical_kmeans(Vt, k, init="kmeans++", seed=seed)
            noisy = add_angular_noise(Ve, sigma, rng)
            snapped = C[np.argmax(noisy @ C.T, axis=1)]
            snapped = snapped / (np.linalg.norm(snapped, axis=1, keepdims=True) + 1e-12)
            noisy_err = _ang(noisy, Ve).mean()
            snap_err = _ang(snapped, Ve).mean()
            wgt = len(Ve)
            corr_acc += (noisy_err - snap_err) * wgt
            err_acc += snap_err * wgt
            w += wgt
    return (corr_acc / w, err_acc / w) if w else (0.0, 0.0)


def per_finger_denoise(blocks_tr, blocks_ev, k, sigma, seed):
    """Fit a per-finger 12-D codebook; snap each finger's noisy 4 bones jointly;
    return mean correction (deg) and residual error (deg), aggregated."""
    rng = np.random.default_rng(seed)
    corr_acc, err_acc, w = 0.0, 0.0, 0
    for f in range(N_FINGERS):
        if len(blocks_tr[f]) < k or len(blocks_ev[f]) == 0:
            continue
        Xt = blocks_tr[f].reshape(len(blocks_tr[f]), -1)
        C = pf.euclid_kmeans(Xt, k, seed=seed)
        clean = blocks_ev[f]                       # [M,4,3]
        noisy = add_angular_noise(clean, sigma, rng)
        Xe = noisy.reshape(len(noisy), -1)
        a = np.argmin(pf._chunked_assign_dist(Xe, C), axis=1)
        snapped = C[a].reshape(len(noisy), BPF, 3)
        snapped = snapped / (np.linalg.norm(snapped, axis=2, keepdims=True) + 1e-12)
        noisy_err = _ang(noisy, clean).mean()
        snap_err = _ang(snapped, clean).mean()
        wgt = len(clean)
        corr_acc += (noisy_err - snap_err) * wgt
        err_acc += snap_err * wgt
        w += wgt
    return (corr_acc / w, err_acc / w) if w else (0.0, 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annot-root", type=Path,
                    default=Path("/opt/tiger/InterHand/annotations/machine_annot"))
    ap.add_argument("--fit-split", default="train")
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--sigmas", type=float, nargs="+", default=[5, 10, 15, 20, 30])
    ap.add_argument("--kbone", type=int, default=26)
    ap.add_argument("--kfinger", type=int, default=128)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    print(f"[fit ] loading {args.fit_split} ...", flush=True)
    btr = pf.load_finger_blocks(args.annot_root, args.fit_split, args.max_frames)
    print(f"[eval] loading {args.eval_split} ...", flush=True)
    bev = pf.load_finger_blocks(args.annot_root, args.eval_split, args.max_frames)

    rows = []
    for sigma in args.sigmas:
        cb, eb = per_bone_denoise(btr, bev, args.kbone, sigma, args.seed)
        cf, ef = per_finger_denoise(btr, bev, args.kfinger, sigma, args.seed)
        rows.append({
            "sigma_deg": float(sigma),
            "per_bone_correction_deg": round(cb, 4),
            "per_finger_correction_deg": round(cf, 4),
            "per_bone_residual_deg": round(eb, 4),
            "per_finger_residual_deg": round(ef, 4),
        })

    print(f"{'sigma°':>7s} {'perbone_corr°':>14s} {'perfinger_corr°':>16s}")
    for r in rows:
        print(f"{r['sigma_deg']:7.1f} {r['per_bone_correction_deg']:14.2f} "
              f"{r['per_finger_correction_deg']:16.2f}")
    print("\npositive correction = quantization pulled the noisy direction toward "
          "clean; per-finger's anatomical prior should correct more as sigma grows.")

    result = {"fit_split": args.fit_split, "eval_split": args.eval_split,
              "kbone": int(args.kbone), "kfinger": int(args.kfinger),
              "seed": args.seed, "rows": rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
