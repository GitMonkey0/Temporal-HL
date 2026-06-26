#!/usr/bin/env python3
"""Cross-regime robustness audit of the main HL tokenizers, plus a per-finger
failure-mode diagnostic.

Motivation. Three tokenizers are now on the table: the fixed HL-26 cube-surface
codebook (`build_temporal_hl`), the data-adaptive per-bone spherical k-means
codebook (`build_adaptive_direction_codebook`), and the anatomy-aware per-finger
*joint* codebook (`build_perfinger_joint_codebook`). They all look good when fit
and scored on clean InterHand two-hand captures. The robustness question is
whether the ordering survives a *change of capture regime*: single-hand FreiHAND,
free single-hand HanCo sequences, and noisy ASL sign-language poses with heavy
self-occlusion. A representation section needs to know which gains are universal
and which are regime-specific.

This tool fits the two data-driven codebooks ONCE on a reference regime, then
APPLIES every tokenizer unchanged to each regime's held-out directions and reports
mean reconstruction angular error (deg) and rate (bits/bone). On top of the
headline numbers it runs a per-finger *failure-mode* diagnostic for the per-finger
joint code: within each finger the four bones share ONE code, so a single badly
reconstructed bone (e.g. a lifted/occluded thumb tip) can drag the whole finger's
shared code. For every (regime, finger) we record the WORST single bone's angular
error and the finger's joint-code mean error, then report their correlation across
all (regime, finger) pairs. A high correlation is the signature of the
"worst bone hijacks the shared per-finger code" failure mode.

It does NOT modify HL or its encoder; it audits how the main tokenizers behave off
their training regime, as honest evidence (including a negative result) for a
temporal-HL representation-design section.

    python tools/build_cross_regime_robustness.py \
        --datasets interhand:/opt/tiger/InterHand/annotations/machine_annot \
                   freihand:/opt/tiger/FreiHAND/annotations \
                   hanco:/opt/tiger/HanCo/annotations \
                   asl:/opt/tiger/ASL/annotations \
        --fit-dataset interhand --fit-split train --eval-split test \
        --k 26 --kfinger 128 \
        --out experiments/cross_regime_robustness_<date>.json

Dataset-parameterized like build_event_keyframe_compression: each --datasets entry
is name:annot_root, and joint files resolve as <dataset>_<split>_joint_3d.json with
an InterHand2.6M fallback so the tool stays runnable where only InterHand data is
present locally. Pure numpy + stdlib; reuses build_temporal_hl for the encoding,
build_adaptive_direction_codebook for the per-bone codebook/metrics, and
build_perfinger_joint_codebook for the per-finger joint codebook, so there is zero
convention drift from the HL labels this repo already produces.
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
BONES_PER_FINGER = pf.BONES_PER_FINGER


def _resolve_joint_path(root: Path, split: str, dataset: str) -> Path | None:
    """Find a per-split joint file. InterHand uses InterHand2.6M_<split>_joint_3d
    .json; for other regimes (FreiHAND / HanCo / ASL) try a <dataset>_<split>_
    joint_3d.json sibling, falling back to the InterHand name so the tool stays
    runnable."""
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


def load_regime(joint_path: Path, max_frames):
    """One pass over a regime's joints, returning BOTH views of the same canonical
    HL directions:
      V       [N,3] float64 unit  -- every local bone direction (per-bone tokenizers)
      blocks  list of 5 arrays, blocks[f] = [M,4,3] -- finger-grouped bones, in the
              finger-major EDGE_ORDER layout (per-finger joint tokenizer)
    """
    joints = hl.load_json(joint_path)
    vecs, edges = [], []
    blocks = [[] for _ in range(N_FINGERS)]
    seen = 0

    def _flush():
        V, _ = adb._finish(vecs, edges)
        return V, [np.asarray(b) for b in blocks]

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
                lv = np.asarray(rec["local_vectors"], dtype=np.float64)
                lv = lv / (np.linalg.norm(lv, axis=1, keepdims=True) + 1e-12)
                for e, v in enumerate(lv):
                    vecs.append(v)
                    edges.append(e)
                grid = lv.reshape(N_FINGERS, BONES_PER_FINGER, 3)
                for f in range(N_FINGERS):
                    blocks[f].append(grid[f])
            seen += 1
            if max_frames is not None and seen >= max_frames:
                return _flush()
    return _flush()


def per_finger_bone_errors(block_eval: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Apply a fitted per-finger joint codebook C ([k,12]) to a finger's eval
    blocks ([M,4,3]); return the mean angular error (deg) of EACH of the 4 bones
    reconstructed from the shared code. Reuses the perfinger assignment machinery."""
    Xe = block_eval.reshape(len(block_eval), -1)
    d = pf._chunked_assign_dist(Xe, C)
    assign = np.argmin(d, axis=1)
    recon = C[assign].reshape(len(Xe), BONES_PER_FINGER, 3)
    recon = recon / (np.linalg.norm(recon, axis=2, keepdims=True) + 1e-12)
    cos = np.clip((recon * block_eval).sum(-1), -1.0, 1.0)   # [M,4]
    return np.degrees(np.arccos(cos)).mean(0)               # [4]


def per_finger_bits_per_bone(block_eval: np.ndarray, C: np.ndarray) -> float:
    Xe = block_eval.reshape(len(block_eval), -1)
    assign = np.argmin(pf._chunked_assign_dist(Xe, C), axis=1)
    counts = np.bincount(assign, minlength=len(C))
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum()) / BONES_PER_FINGER


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+",
                    default=["interhand:/opt/tiger/InterHand/annotations/machine_annot",
                             "freihand:/opt/tiger/InterHand/annotations/machine_annot",
                             "hanco:/opt/tiger/InterHand/annotations/machine_annot",
                             "asl:/opt/tiger/InterHand/annotations/machine_annot"],
                    help="list of name:annot_root regimes; tokenizers applied to each")
    ap.add_argument("--fit-dataset", default=None,
                    help="regime to FIT the data-driven codebooks on "
                         "(default: first --datasets entry)")
    ap.add_argument("--fit-split", default="train")
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--k", type=int, default=26,
                    help="adaptive per-bone codebook size (matched to HL-26)")
    ap.add_argument("--kfinger", type=int, default=128,
                    help="per-finger joint codebook size")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    specs = []
    for spec in args.datasets:
        name, _, root = spec.partition(":")
        specs.append((name, Path(root)))
    fit_name = args.fit_dataset or specs[0][0]

    # -- fit the two data-driven codebooks ONCE, on the reference regime --
    fit_root = dict(specs)[fit_name]
    fp = _resolve_joint_path(fit_root, args.fit_split, fit_name)
    if fp is None:
        print(f"[fit:{fit_name}] no joint file under {fit_root}; abort")
        return
    print(f"[fit:{fit_name}] <- {fp}", flush=True)
    Vfit, blocks_fit = load_regime(fp, args.max_frames)
    if len(Vfit) == 0:
        print(f"[fit:{fit_name}] no directions decoded; abort")
        return
    print(f"[fit:{fit_name}] {len(Vfit)} directions; fitting codebooks ...", flush=True)
    hlcb = adb.hl26_codebook()
    adapt_cb = adb.spherical_kmeans(Vfit, args.k, init="kmeans++", seed=args.seed)
    finger_cb = []
    for f in range(N_FINGERS):
        Xt = blocks_fit[f].reshape(len(blocks_fit[f]), -1)
        finger_cb.append(pf.euclid_kmeans(Xt, args.kfinger, seed=args.seed))

    # -- apply every tokenizer to each regime's held-out directions --
    rows = []
    diag = []            # (regime, finger, joint_mean_deg, worst_bone_deg)
    for name, root in specs:
        ep = _resolve_joint_path(root, args.eval_split, name)
        if ep is None:
            print(f"[{name}] no eval joint file under {root}; skip")
            continue
        print(f"[{name}] eval <- {ep}", flush=True)
        Ve, blocks_e = load_regime(ep, args.max_frames)
        if len(Ve) == 0:
            print(f"[{name}] no directions decoded; skip")
            continue

        hl26_deg = adb.angular_error_deg(Ve, hlcb)
        adapt_deg = adb.angular_error_deg(Ve, adapt_cb)
        adapt_bits = adb.empirical_entropy_bits(Ve, adapt_cb)
        hl26_bits = adb.empirical_entropy_bits(Ve, hlcb)

        # per-finger joint: weighted mean angular error + the failure-mode diagnostic
        f_ang, f_bits, w = [], [], []
        for f in range(N_FINGERS):
            be = blocks_e[f]
            if len(be) == 0:
                continue
            bone_err = per_finger_bone_errors(be, finger_cb[f])   # [4]
            joint_mean = float(bone_err.mean())
            worst_bone = float(bone_err.max())
            f_ang.append(joint_mean)
            f_bits.append(per_finger_bits_per_bone(be, finger_cb[f]))
            w.append(len(be))
            diag.append({"regime": name, "finger": hl.FINGER_NAMES[f],
                         "joint_mean_deg": round(joint_mean, 4),
                         "worst_bone_deg": round(worst_bone, 4)})
        w = np.asarray(w, dtype=np.float64)
        finger_deg = float(np.average(f_ang, weights=w)) if len(w) else float("nan")
        finger_bits = float(np.average(f_bits, weights=w)) if len(w) else float("nan")

        rows.append({"regime": name, "n_eval": int(len(Ve)),
                     "hl26_deg": round(hl26_deg, 4), "hl26_bits": round(hl26_bits, 4),
                     "adaptive_deg": round(adapt_deg, 4),
                     "adaptive_bits": round(adapt_bits, 4),
                     "perfinger_deg": round(finger_deg, 4),
                     "perfinger_bits": round(finger_bits, 4)})

    # -- report --
    print("\n== reconstruction angular error per tokenizer, per regime ==")
    print(f"{'regime':12s} {'HL26°':>7s} {'adapt°':>7s} {'pfing°':>7s} "
          f"{'HL26b':>6s} {'adptb':>6s} {'pfingb':>6s}")
    for r in rows:
        print(f"{r['regime']:12s} {r['hl26_deg']:7.2f} {r['adaptive_deg']:7.2f} "
              f"{r['perfinger_deg']:7.2f} {r['hl26_bits']:6.2f} "
              f"{r['adaptive_bits']:6.2f} {r['perfinger_bits']:6.2f}")

    print("\n== per-finger joint failure-mode diagnostic ==")
    print(f"{'regime':12s} {'finger':8s} {'joint_mean°':>11s} {'worst_bone°':>11s}")
    for d in diag:
        print(f"{d['regime']:12s} {d['finger']:8s} "
              f"{d['joint_mean_deg']:11.2f} {d['worst_bone_deg']:11.2f}")

    corr = float("nan")
    if len(diag) >= 2:
        jm = np.asarray([d["joint_mean_deg"] for d in diag])
        wb = np.asarray([d["worst_bone_deg"] for d in diag])
        if jm.std() > 1e-9 and wb.std() > 1e-9:
            corr = float(np.corrcoef(jm, wb)[0, 1])
    print(f"\ncorr(worst-bone error, finger joint-code error) over "
          f"{len(diag)} (regime,finger) pairs = {corr:.3f}")
    print("a high positive correlation is the signature of one bad bone hijacking "
          "the finger's shared per-finger code -> motivates an independent-residual fix.")

    result = {
        "fit_dataset": fit_name, "fit_split": args.fit_split,
        "eval_split": args.eval_split, "k": args.k, "kfinger": args.kfinger,
        "seed": args.seed, "rows": rows, "perfinger_diagnostic": diag,
        "worst_bone_vs_joint_corr": (round(corr, 4) if corr == corr else None),
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
