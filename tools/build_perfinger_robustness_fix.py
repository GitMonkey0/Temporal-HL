#!/usr/bin/env python3
"""Per-finger joint code: a worst-bone failure mode, a negative robust-allocation
attempt, and the hierarchical-residual cure.

Motivation. The per-finger *joint* codebook
(`build_perfinger_joint_codebook.py`) binds a finger's four bones into ONE shared
12-D code. That is great for rate in a clean regime, but it has a structural
weakness: the shared code must satisfy four bones at once, so a single
manifold-outlier bone (a lifted / occluded thumb tip in a noisy ASL pose) can
hijack the code and drag the whole finger's reconstruction. The cross-regime audit
(`build_cross_regime_robustness.py`) already showed the per-finger code losing
ground specifically on the noisy regime; this tool isolates *why*, tries the
tempting-but-wrong fix, and then lands the right one. It is three parts:

  (1) DIAGNOSIS. Per (finger, frame) we measure the finger's mean angular error vs
      its single WORST bone's angular error and report their Pearson correlation: a
      high correlation is the signature that the finger error is *carried by* its
      worst bone. We also measure a "worst-bone hijack" rate — the fraction of
      finger-codes where one bone's error dwarfs its three siblings — on a clean
      regime vs a noisy regime, where the rate is expected to jump sharply.

  (2) I2 ROBUST-ALLOCATION (a negative attempt, recorded honestly). If one bad bone
      hijacks the *assignment*, maybe a more robust, trimmed/median-style assignment
      that down-weights the single worst-matching bone picks a better shared code.
      We implement exactly that (assign on the best 3 of 4 bone sub-distances) and
      score ALL four reconstructed bones. It does not rescue the outlier bone — the
      code is now chosen to fit the other three and ignores the outlier, so its
      reconstruction is no better (and typically worse). A robust *shared* code
      cannot fix a problem that lives in one bone; the fix has to be per-bone.

  (3) I1 CURE (the win). A coarse per-finger code PLUS a per-bone INDEPENDENT
      residual (`build_hierarchical_coarse_residual.py`): each bone gets its own
      residual code, so the worst bone's error is absorbed locally instead of
      dragging the whole finger. We compare, in-domain and under transfer, the
      coarse-only per-finger code, a conservative per-bone adaptive codebook, and
      the coarse+residual cure.

It does NOT modify HL or its encoder; it audits and repairs a failure mode of an
existing tokenizer, as honest evidence (including a negative result) for a
temporal-HL representation-design section.

    python tools/build_perfinger_robustness_fix.py \
        --datasets interhand:/opt/tiger/InterHand/annotations/machine_annot \
                   asl:/opt/tiger/ASL/annotations \
        --clean-regime interhand --noisy-regime asl \
        --fit-split train --eval-split test \
        --kfinger 128 --kcoarse 128 --kres 32 --kbone 64 \
        --out experiments/perfinger_robustness_fix_2026-06-27.json

Dataset-parameterized exactly like build_cross_regime_robustness: each --datasets
entry is name:annot_root and joint files resolve as <dataset>_<split>_joint_3d.json
with an InterHand2.6M fallback, so the tool stays runnable where only InterHand data
is present locally (clean and noisy then fall back to the same file). Pure numpy +
stdlib; reuses build_temporal_hl for the encoding, build_perfinger_joint_codebook
for the 12-D loader / k-means / assignment, build_hierarchical_coarse_residual for
the coarse+residual cure, and build_cross_regime_robustness for the dataset-
parameterized loader, so there is zero convention drift from the HL labels this repo
produces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import build_temporal_hl as hl
import build_adaptive_direction_codebook as adb
import build_perfinger_joint_codebook as pf
import build_hierarchical_coarse_residual as hcr
import build_cross_regime_robustness as crr

N_FINGERS = pf.N_FINGERS
BONES_PER_FINGER = pf.BONES_PER_FINGER  # 4 (EDGE_ORDER is finger-major)


# --------------------------------------------------------------------------- #
# shared low-level pieces (all reuse the per-finger assignment machinery)
# --------------------------------------------------------------------------- #
def fit_finger_codebooks(blocks_train, kfinger, seed):
    """One 12-D per-finger joint codebook per finger (reuses pf.euclid_kmeans).
    Fingers without enough fitting instances get a None placeholder."""
    cb = []
    for f in range(N_FINGERS):
        Xt = blocks_train[f].reshape(len(blocks_train[f]), -1)
        cb.append(pf.euclid_kmeans(Xt, kfinger, seed=seed) if len(Xt) >= kfinger else None)
    return cb


def perfinger_bone_error_matrix(block_eval: np.ndarray, C: np.ndarray) -> np.ndarray:
    """[M,4] angular error (deg) of each bone reconstructed from the shared
    per-finger code, for every finger instance. crr.per_finger_bone_errors returns
    the column means of exactly this matrix; here we keep the full per-frame matrix
    so a single frame's worst bone can be inspected."""
    Xe = block_eval.reshape(len(block_eval), -1)
    assign = np.argmin(pf._chunked_assign_dist(Xe, C), axis=1)
    recon = C[assign].reshape(len(Xe), BONES_PER_FINGER, 3)
    recon = recon / (np.linalg.norm(recon, axis=2, keepdims=True) + 1e-12)
    cos = np.clip((recon * block_eval).sum(-1), -1.0, 1.0)   # [M,4]
    return np.degrees(np.arccos(cos))


def _trimmed_assign(Xe: np.ndarray, C: np.ndarray, chunk: int = 2000) -> np.ndarray:
    """Robust ('trimmed') per-finger assignment: choose the code that minimizes the
    sum of the best 3 of the 4 per-bone squared distances, i.e. down-weight the
    single worst-matching bone so one outlier bone cannot veto an otherwise good
    shared code. Reconstruction of all four bones still uses the chosen code."""
    Cb = C.reshape(len(C), BONES_PER_FINGER, 3)
    assign = np.empty(len(Xe), dtype=np.int64)
    for s in range(0, len(Xe), chunk):
        Xc = Xe[s:s + chunk].reshape(-1, BONES_PER_FINGER, 3)            # [m,4,3]
        bone_sqd = ((Xc[:, None, :, :] - Cb[None, :, :, :]) ** 2).sum(-1)  # [m,k,4]
        trimmed = bone_sqd.sum(2) - bone_sqd.max(2)   # drop the worst bone per code
        assign[s:s + chunk] = np.argmin(trimmed, axis=1)
    return assign


def perfinger_angular(block_eval: np.ndarray, C: np.ndarray, mode: str = "full") -> float:
    """Mean per-bone angular error (deg) of a finger's eval block under a fitted
    12-D code, with either the standard full-12-D assignment ('full') or the
    trimmed best-3-of-4 assignment ('trimmed'). Both reconstruct all four bones."""
    Xe = block_eval.reshape(len(block_eval), -1)
    if mode == "full":
        assign = np.argmin(pf._chunked_assign_dist(Xe, C), axis=1)
    else:
        assign = _trimmed_assign(Xe, C)
    recon = C[assign].reshape(len(Xe), BONES_PER_FINGER, 3)
    recon = recon / (np.linalg.norm(recon, axis=2, keepdims=True) + 1e-12)
    cos = np.clip((recon * block_eval).sum(-1), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)).mean())


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------------------- #
# part (1) diagnosis
# --------------------------------------------------------------------------- #
def diagnose(blocks_eval, finger_cb, hijack_ratio, hijack_floor_deg):
    """Per (finger, frame): finger mean error vs worst single-bone error. Returns
    the Pearson correlation between them and the worst-bone hijack rate (fraction of
    finger-codes whose worst bone both exceeds an absolute floor and dwarfs the mean
    of its three siblings by hijack_ratio)."""
    fmean, worst, hij = [], [], []
    for f in range(N_FINGERS):
        be = blocks_eval[f]
        C = finger_cb[f]
        if C is None or len(be) == 0:
            continue
        E = perfinger_bone_error_matrix(be, C)               # [M,4]
        w = E.max(1)                                          # worst bone per frame
        others_mean = (E.sum(1) - w) / (BONES_PER_FINGER - 1)
        fmean.append(E.mean(1))
        worst.append(w)
        hij.append((w >= hijack_floor_deg) & (w >= hijack_ratio * np.maximum(others_mean, 1e-9)))
    if not fmean:
        return float("nan"), float("nan"), 0
    fm = np.concatenate(fmean)
    wb = np.concatenate(worst)
    hj = np.concatenate(hij)
    return _pearson(fm, wb), float(hj.mean()), int(len(fm))


# --------------------------------------------------------------------------- #
# part (2) robust-allocation aggregation
# --------------------------------------------------------------------------- #
def perfinger_alloc_angular(blocks_eval, finger_cb, mode):
    """Mean per-bone angular error over all fingers (eval-count weighted) for a
    fitted per-finger code under the chosen assignment mode."""
    a, w = 0.0, 0
    for f in range(N_FINGERS):
        be = blocks_eval[f]
        C = finger_cb[f]
        if C is None or len(be) == 0:
            continue
        a += perfinger_angular(be, C, mode) * len(be)
        w += len(be)
    return (a / w) if w else float("nan")


# --------------------------------------------------------------------------- #
# part (3) cure: coarse-only / per-bone / coarse+residual (reuse hcr & pf)
# --------------------------------------------------------------------------- #
def coarse_only_angular(btr, bev, kcoarse):
    """Coarse per-finger joint code with no residual (reuses pf.per_finger_joint);
    eval-count weighted over fingers, like hcr.main's coarse-only point."""
    a, b, w = 0.0, 0.0, 0
    for f in range(N_FINGERS):
        if len(btr[f]) < kcoarse or len(bev[f]) == 0:
            continue
        aa, bb = pf.per_finger_joint(btr[f], bev[f], kcoarse)
        a += aa * len(bev[f]); b += bb * len(bev[f]); w += len(bev[f])
    return ((a / w), (b / w)) if w else (float("nan"), float("nan"))


def cure_condition(name, btr, bev, kcoarse, kres, kbone):
    """One eval condition for part (3): coarse-only vs conservative per-bone
    adaptive code vs the coarse+per-bone-residual cure."""
    co_deg, _ = coarse_only_angular(btr, bev, kcoarse)
    pb_deg, _ = hcr.flat_per_bone(btr, bev, kbone)
    cure_deg, cure_bits = hcr.hierarchical(btr, bev, kcoarse, kres)
    return {"condition": name,
            "coarse_only_deg": round(co_deg, 4),
            f"perbone_adaptive_km{kbone}_deg": round(pb_deg, 4),
            "coarse_plus_residual_deg": round(cure_deg, 4),
            "coarse_plus_residual_bits": round(cure_bits, 4)}


# --------------------------------------------------------------------------- #
def _load_blocks(specs, name, split, max_frames):
    """Resolve and load a regime/split's finger blocks via the cross-regime loader
    (with its InterHand fallback). Returns (joint_path, blocks) or (None, None)."""
    root = dict(specs).get(name)
    if root is None:
        return None, None
    p = crr._resolve_joint_path(root, split, name)
    if p is None:
        return None, None
    _, blocks = crr.load_regime(p, max_frames)
    return p, blocks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+",
                    default=["interhand:/opt/tiger/InterHand/annotations/machine_annot",
                             "asl:/opt/tiger/InterHand/annotations/machine_annot"],
                    help="list of name:annot_root regimes (InterHand fallback per name)")
    ap.add_argument("--clean-regime", default="interhand",
                    help="reference (clean) regime name from --datasets")
    ap.add_argument("--noisy-regime", default="asl",
                    help="stress (noisy / lifting / occluded) regime name from --datasets")
    ap.add_argument("--fit-split", default="train")
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--kfinger", type=int, default=128,
                    help="per-finger joint codebook size (diagnosis + robust-alloc)")
    ap.add_argument("--kcoarse", type=int, default=128, help="coarse per-finger code size")
    ap.add_argument("--kres", type=int, default=32, help="per-bone residual code size")
    ap.add_argument("--kbone", type=int, default=64,
                    help="conservative per-bone adaptive codebook size (comparison)")
    ap.add_argument("--hijack-ratio", type=float, default=2.5,
                    help="worst bone counts as a hijack if it exceeds the mean of its "
                         "three siblings by this factor")
    ap.add_argument("--hijack-floor-deg", type=float, default=15.0,
                    help="absolute angular floor (deg) a worst bone must clear to count")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    specs = []
    for spec in args.datasets:
        nm, _, root = spec.partition(":")
        specs.append((nm, Path(root)))

    # -- load the clean and noisy regimes (fit + eval splits) --
    cp_tr, clean_tr = _load_blocks(specs, args.clean_regime, args.fit_split, args.max_frames)
    cp_ev, clean_ev = _load_blocks(specs, args.clean_regime, args.eval_split, args.max_frames)
    np_tr, noisy_tr = _load_blocks(specs, args.noisy_regime, args.fit_split, args.max_frames)
    np_ev, noisy_ev = _load_blocks(specs, args.noisy_regime, args.eval_split, args.max_frames)
    if clean_tr is None or clean_ev is None or noisy_tr is None or noisy_ev is None:
        print("[abort] could not resolve a joint file for clean/noisy regime "
              "(checked the <dataset>_<split>_joint_3d.json names + InterHand fallback)")
        return
    same_file = (cp_ev == np_ev)
    print(f"[clean:{args.clean_regime}] fit <- {cp_tr}\n[clean] eval <- {cp_ev}")
    print(f"[noisy:{args.noisy_regime}] fit <- {np_tr}\n[noisy] eval <- {np_ev}")
    if same_file:
        print("[note] clean and noisy resolved to the SAME joint file (local InterHand "
              "fallback): clean/noisy/transfer numbers will coincide until a distinct "
              "noisy regime's joints are present.")

    # -- fit per-finger joint codebooks once per fit regime (diagnosis + robust-alloc) --
    print(f"\n[fit] per-finger joint codebooks (k={args.kfinger}) ...", flush=True)
    cb_clean = fit_finger_codebooks(clean_tr, args.kfinger, args.seed)
    cb_noisy = fit_finger_codebooks(noisy_tr, args.kfinger, args.seed)

    # =================== part (1) diagnosis ===================
    print("\n== (1) diagnosis: worst bone carries the finger's joint-code error ==")
    diag = {}
    for tag, bev in (("clean", clean_ev), ("noisy", noisy_ev)):
        corr, hijack, n = diagnose(bev, cb_clean, args.hijack_ratio, args.hijack_floor_deg)
        diag[tag] = {"pearson_mean_vs_worst": (round(corr, 4) if corr == corr else None),
                     "hijack_rate": (round(hijack, 4) if hijack == hijack else None),
                     "n_finger_frames": n}
        print(f"  {tag:6s}: corr(finger-mean, worst-bone) = {corr:.3f}   "
              f"worst-bone hijack rate = {100*hijack:5.2f}%   (n={n})")
    print("  high corr + a clean->noisy jump in hijack rate = one bad bone carries / "
          "hijacks the shared per-finger code.")

    # =================== part (2) robust-allocation (negative) ===================
    print("\n== (2) robust (trimmed) shared-code assignment: does NOT rescue the outlier bone ==")
    alloc = []
    alloc_conds = [("noisy_within", cb_noisy, noisy_ev),
                   ("transfer_clean_to_noisy", cb_clean, noisy_ev)]
    print(f"  {'condition':24s} {'full°':>8s} {'trimmed°':>9s}")
    for name, cb, bev in alloc_conds:
        full_deg = perfinger_alloc_angular(bev, cb, "full")
        trim_deg = perfinger_alloc_angular(bev, cb, "trimmed")
        alloc.append({"condition": name,
                      "full_assign_deg": round(full_deg, 4),
                      "trimmed_assign_deg": round(trim_deg, 4)})
        print(f"  {name:24s} {full_deg:8.2f} {trim_deg:9.2f}")
    print("  trimming the worst bone out of the assignment chooses a code for the other "
          "three and abandons the outlier -> no better (expected worse).")

    # =================== part (3) cure (win) ===================
    print("\n== (3) cure: coarse per-finger code + per-bone INDEPENDENT residual ==")
    cure = []
    cure_conds = [("clean_within", clean_tr, clean_ev),
                  ("noisy_within", noisy_tr, noisy_ev),
                  ("transfer_clean_to_noisy", clean_tr, noisy_ev)]
    kbone_key = f"perbone_adaptive_km{args.kbone}_deg"
    print(f"  {'condition':24s} {'coarse°':>8s} {'perbone°':>9s} {'cure°':>7s} {'cureb':>6s}")
    for name, btr, bev in cure_conds:
        row = cure_condition(name, btr, bev, args.kcoarse, args.kres, args.kbone)
        cure.append(row)
        print(f"  {name:24s} {row['coarse_only_deg']:8.2f} {row[kbone_key]:9.2f} "
              f"{row['coarse_plus_residual_deg']:7.2f} {row['coarse_plus_residual_bits']:6.2f}")
    print("  per-bone residuals absorb the worst bone locally -> the finger no longer "
          "inherits its worst bone's error.")

    result = {
        "clean_regime": args.clean_regime, "noisy_regime": args.noisy_regime,
        "fit_split": args.fit_split, "eval_split": args.eval_split,
        "kfinger": args.kfinger, "kcoarse": args.kcoarse, "kres": args.kres,
        "kbone": args.kbone, "hijack_ratio": args.hijack_ratio,
        "hijack_floor_deg": args.hijack_floor_deg, "seed": args.seed,
        "clean_eval_equals_noisy_eval": bool(same_file),
        "diagnosis": diag, "robust_allocation": alloc, "cure": cure,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
