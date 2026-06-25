#!/usr/bin/env python3
"""Closed-loop decoding audit for chained/relative codes: does error accumulate?

Motivation. The hierarchical and relative schemes
(`build_hierarchical_coarse_residual.py`, `build_relative_direction_codes.py`)
decode a bone from a *reference* (its parent bone, or the coarse reconstruction).
When those tools are measured, the reference is the ORACLE ground-truth value. In
real decoding the reference is itself a *quantized* reconstruction, so each bone
is decoded from an already-lossy ancestor. Errors could therefore accumulate
along the finger chain (wrist -> proximal -> ... -> tip), and an experiment that
always uses the oracle reference would hide that drift.

This tool runs the decode in *closed loop*: it walks each finger from the most
proximal bone outward, and decodes every bone from the QUANTIZED reconstruction of
its parent, exactly as a real decoder must. It measures the per-depth angular
error and compares:

  * oracle-reference     (each bone decoded from GT parent; upper bound)
  * closed-loop          (each bone decoded from the quantized parent; realistic)

and, for the budget allocation across chain depths:

  * equal K per depth    (same codebook size at every bone position)
  * data-driven K        (more codes to the depths that carry more residual
                          variance; same total budget)

Expected mechanism: because each residual codebook is fit on the *closed-loop*
reference distribution and the per-bone residuals are small, the end-of-chain
degradation should stay small (within a usability red line of a few degrees) and
NOT blow up; and a data-driven split of a fixed K budget across depths should beat
an equal split.

    python tools/build_closed_loop_chain_audit.py \
        --annot-root <InterHand annotations> --fit-split train --eval-split test \
        --ktotal 64 \
        --out experiments/closed_loop_chain_audit_<date>.json

Pure numpy + stdlib; reuses build_temporal_hl + build_adaptive_direction_codebook
(+ the per-finger loader in build_perfinger_joint_codebook) for zero drift.
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
BPF = pf.BONES_PER_FINGER  # 4; depth 0 = most proximal, depth 3 = tip


def _basis_from_axis(axis):
    a = axis / (np.linalg.norm(axis) + 1e-12)
    helper = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(a, helper)
    u = u / (np.linalg.norm(u) + 1e-12)
    w = np.cross(a, u)
    return np.stack([a, u, w], axis=0)


def _depth_order(block):
    """Reorder a finger block [M,4,3] from proximal->tip. EDGE_ORDER lists bones
    tip-first (idx 3,2,1,0), so reverse the bone axis to walk the kinematic chain
    from the wrist outward."""
    return block[:, ::-1, :]


def _residual_codebook(ref_dir, tgt_dir, k, seed=0):
    """Fit a spherical k-means codebook on the target direction expressed in the
    frame aligned to the reference direction (one row per sample)."""
    res = np.stack([_basis_from_axis(ref_dir[i]) @ tgt_dir[i]
                    for i in range(len(tgt_dir))], axis=0)
    res = res / (np.linalg.norm(res, axis=1, keepdims=True) + 1e-12)
    return adb.spherical_kmeans(res, k, init="kmeans++", seed=seed), res


def chain_eval(btr, bev, k_per_depth, seed=0):
    """Walk the finger chain proximal->tip. At each depth, fit a residual code on
    the (closed-loop) parent reference and decode. Returns per-depth angular error
    for both oracle-reference and closed-loop decoding, aggregated over fingers."""
    oracle = np.zeros(BPF)
    closed = np.zeros(BPF)
    weight = 0
    for f in range(N_FINGERS):
        if len(btr[f]) < max(k_per_depth) or len(bev[f]) == 0:
            continue
        tr = _depth_order(btr[f])   # [M,4,3] proximal->tip
        ev = _depth_order(bev[f])
        w = len(ev)
        weight += w

        # parent reference for depth 0 is the wrist-forward axis (canonical +Y in
        # the local frame); for deeper bones it is the previous bone.
        root = np.tile(np.array([0.0, 1.0, 0.0]), (len(tr), 1))
        root_ev = np.tile(np.array([0.0, 1.0, 0.0]), (len(ev), 1))

        oracle_ref_tr, oracle_ref_ev = root, root_ev
        closed_ref_tr, closed_ref_ev = root.copy(), root_ev.copy()

        for d in range(BPF):
            k = k_per_depth[d]
            tgt_tr, tgt_ev = tr[:, d, :], ev[:, d, :]

            # --- oracle: residual fit & decode against GT parent
            C_o, _ = _residual_codebook(oracle_ref_tr, tgt_tr, k, seed)
            res_ev_o = np.stack([_basis_from_axis(oracle_ref_ev[i]) @ tgt_ev[i]
                                 for i in range(len(tgt_ev))], axis=0)
            res_ev_o = res_ev_o / (np.linalg.norm(res_ev_o, axis=1, keepdims=True) + 1e-12)
            a_o = np.argmax(res_ev_o @ C_o.T, axis=1)
            recon_o = np.stack([_basis_from_axis(oracle_ref_ev[i]).T @ C_o[a_o[i]]
                                for i in range(len(tgt_ev))], axis=0)
            recon_o = recon_o / (np.linalg.norm(recon_o, axis=1, keepdims=True) + 1e-12)
            cos_o = np.clip((recon_o * tgt_ev).sum(1), -1.0, 1.0)
            oracle[d] += float(np.degrees(np.arccos(cos_o)).mean()) * w

            # --- closed loop: residual fit & decode against the QUANTIZED parent
            C_c, _ = _residual_codebook(closed_ref_tr, tgt_tr, k, seed)
            res_ev_c = np.stack([_basis_from_axis(closed_ref_ev[i]) @ tgt_ev[i]
                                 for i in range(len(tgt_ev))], axis=0)
            res_ev_c = res_ev_c / (np.linalg.norm(res_ev_c, axis=1, keepdims=True) + 1e-12)
            a_c = np.argmax(res_ev_c @ C_c.T, axis=1)
            recon_c = np.stack([_basis_from_axis(closed_ref_ev[i]).T @ C_c[a_c[i]]
                                for i in range(len(tgt_ev))], axis=0)
            recon_c = recon_c / (np.linalg.norm(recon_c, axis=1, keepdims=True) + 1e-12)
            cos_c = np.clip((recon_c * tgt_ev).sum(1), -1.0, 1.0)
            closed[d] += float(np.degrees(np.arccos(cos_c)).mean()) * w

            # advance references: oracle uses GT, closed loop uses its own recon
            recon_tr = _decode_train(closed_ref_tr, tgt_tr, C_c)
            oracle_ref_tr, oracle_ref_ev = tgt_tr, tgt_ev
            closed_ref_tr, closed_ref_ev = recon_tr, recon_c

    if weight == 0:
        return np.zeros(BPF), np.zeros(BPF)
    return oracle / weight, closed / weight


def _decode_train(ref_dir, tgt_dir, C):
    """Closed-loop reconstruction on the train split (to feed the next depth)."""
    res = np.stack([_basis_from_axis(ref_dir[i]) @ tgt_dir[i]
                    for i in range(len(tgt_dir))], axis=0)
    res = res / (np.linalg.norm(res, axis=1, keepdims=True) + 1e-12)
    a = np.argmax(res @ C.T, axis=1)
    recon = np.stack([_basis_from_axis(ref_dir[i]).T @ C[a[i]]
                      for i in range(len(tgt_dir))], axis=0)
    return recon / (np.linalg.norm(recon, axis=1, keepdims=True) + 1e-12)


def _data_driven_alloc(btr, ktotal):
    """Split a total per-finger-chain code budget across the 4 depths in
    proportion to each depth's directional variance (more codes where motion is
    richer). Returns a list of 4 ints summing to ~ktotal (each >= 2)."""
    var = np.zeros(BPF)
    for f in range(N_FINGERS):
        if len(btr[f]) == 0:
            continue
        b = _depth_order(btr[f])
        for d in range(BPF):
            mean = b[:, d, :].mean(0)
            var[d] += 1.0 - np.linalg.norm(mean)  # spread around the mean dir
    if var.sum() <= 0:
        return [max(2, ktotal // BPF)] * BPF
    share = var / var.sum()
    alloc = np.maximum(2, np.round(share * ktotal).astype(int))
    return [int(x) for x in alloc]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annot-root", type=Path,
                    default=Path("/opt/tiger/InterHand/annotations/machine_annot"))
    ap.add_argument("--fit-split", default="train")
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--ktotal", type=int, default=64,
                    help="total code budget per finger chain (split across 4 depths)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    print(f"[fit ] loading {args.fit_split} ...", flush=True)
    btr = pf.load_finger_blocks(args.annot_root, args.fit_split, args.max_frames)
    print(f"[eval] loading {args.eval_split} ...", flush=True)
    bev = pf.load_finger_blocks(args.annot_root, args.eval_split, args.max_frames)

    equal = [max(2, args.ktotal // BPF)] * BPF
    driven = _data_driven_alloc(btr, args.ktotal)
    print(f"[alloc] equal={equal}  data_driven={driven}\n")

    rows = []
    for name, alloc in (("equal_K", equal), ("data_driven_K", driven)):
        oracle, closed = chain_eval(btr, bev, alloc, seed=args.seed)
        rows.append({
            "allocation": name, "k_per_depth": [int(x) for x in alloc],
            "oracle_per_depth_deg": [round(float(x), 4) for x in oracle],
            "closed_per_depth_deg": [round(float(x), 4) for x in closed],
            "oracle_end_deg": round(float(oracle[-1]), 4),
            "closed_end_deg": round(float(closed[-1]), 4),
            "chain_end_degradation_deg": round(float(closed[-1] - oracle[-1]), 4),
        })

    print(f"{'allocation':16s} {'oracle_end°':>11s} {'closed_end°':>11s} "
          f"{'degradation°':>12s}")
    for r in rows:
        print(f"{r['allocation']:16s} {r['oracle_end_deg']:11.2f} "
              f"{r['closed_end_deg']:11.2f} {r['chain_end_degradation_deg']:12.2f}")
    print("\nclosed-loop end-of-chain degradation is the realistic decoding cost; "
          "a small, bounded gap means error does NOT accumulate down the chain.")

    result = {"fit_split": args.fit_split, "eval_split": args.eval_split,
              "ktotal": int(args.ktotal), "seed": args.seed, "rows": rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
