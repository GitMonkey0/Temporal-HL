#!/usr/bin/env python3
"""Sweep adaptive-codebook variants at matched K: k-means / k-medoids / region.

Motivation. The data-adaptive direction codebook
(`build_adaptive_direction_codebook.py`) fits K unit centroids with spherical
k-means. Two questions remain before we trust it as the representation backbone:

  1. Are the learned centers *readable*? Spherical k-means centers are averaged
     directions, not directions the hand actually takes. A spherical *k-medoids*
     variant constrains every center to be an actually observed bone direction,
     so a code book entry is a real, namable pose — at (hopefully) no accuracy
     cost.
  2. Is "one codebook for all bones" the right granularity? A *region-aware*
     variant fits a separate codebook per finger (the five fingers are the
     natural structural regions of the hand) and spends the budget where each
     region's directions actually live.

This tool reuses the canonical HL encoder (`build_temporal_hl.frame_to_hl`) and
the directions/metrics in `build_adaptive_direction_codebook.py`, and compares,
at a matched per-bone code budget K:

  * kmeans       (spherical k-means, the adaptive baseline)
  * kmedoids     (spherical k-medoids; centers are observed directions)
  * region_aware (one spherical-k-means codebook per finger)

It reports mean reconstruction angular error and an interpretability note
(k-medoids centers are real directions, so each is printable as an observed
bone direction). Expected direction: k-medoids ~ k-means at matched K, while the
region-aware per-finger codebooks reach the same accuracy at a clearly lower
effective rate — which is exactly the structural signal that motivates per-finger
joint quantization (`build_perfinger_joint_codebook.py`).

    python tools/build_adaptive_codebook_sweep.py \
        --annot-root <InterHand annotations> --fit-split train --eval-split test \
        --ks 16 26 40 \
        --out experiments/adaptive_codebook_sweep_<date>.json

Pure numpy + stdlib; reuses build_temporal_hl + build_adaptive_direction_codebook
so there is zero convention drift from the HL labels this repo produces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import build_temporal_hl as hl
import build_adaptive_direction_codebook as adb


def spherical_kmedoids(V, k, iters=50, seed=0):
    """k unit medoids: like spherical k-means but each center is constrained to
    be one of the observed directions (a real bone direction, hence readable).

    Init with k-means++ on cosine distance, then alternate assignment and, per
    cluster, pick the member maximizing total cosine to the cluster (the medoid).
    Returns a [k,3] array whose rows are exact observed unit directions.
    """
    rng = np.random.default_rng(seed)
    n = len(V)
    # k-means++ style seeding on cosine distance, but keep the chosen rows.
    idx0 = int(rng.integers(n))
    med_idx = [idx0]
    closest = 1.0 - V @ V[idx0]
    for _ in range(1, k):
        p = np.clip(closest, 0, None)
        s = p.sum()
        j = int(rng.integers(n)) if s <= 0 else int(rng.choice(n, p=p / s))
        med_idx.append(j)
        closest = np.minimum(closest, 1.0 - V @ V[j])
    med_idx = list(dict.fromkeys(med_idx))  # de-dup while preserving order
    while len(med_idx) < k:
        med_idx.append(int(rng.integers(n)))
    C = V[med_idx].copy()
    for _ in range(iters):
        assign = np.argmax(V @ C.T, axis=1)
        new_idx = []
        for j in range(k):
            members = np.where(assign == j)[0]
            if len(members) == 0:
                new_idx.append(int(rng.integers(n)))
                continue
            sub = V[members]
            # medoid = member with max summed cosine to the cluster
            sim = sub @ sub.T
            best = members[int(np.argmax(sim.sum(1)))]
            new_idx.append(int(best))
        new_idx_arr = np.asarray(new_idx)
        if np.array_equal(new_idx_arr, np.asarray(med_idx)):
            med_idx = new_idx
            break
        med_idx = new_idx
        C = V[med_idx].copy()
    return V[med_idx].copy()


def per_finger_directions(annot_root: Path, split: str, max_frames):
    """The HL local bone directions split by finger: returns a list of 5 arrays,
    each [M_f, 3] unit, using the same loader convention as the other tools."""
    fname = f"InterHand2.6M_{split}_joint_3d.json"
    for cand in (annot_root / fname, annot_root / split / fname):
        if cand.exists():
            joint_path = cand
            break
    else:
        raise FileNotFoundError(f"{fname} not under {annot_root} (flat or /{split}/)")
    joints = hl.load_json(joint_path)
    n_fingers = len(hl.FINGER_NAMES)
    bones_per_finger = len(hl.EDGE_ORDER) // n_fingers
    per_finger = [[] for _ in range(n_fingers)]
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
                lv = np.asarray(rec["local_vectors"], dtype=np.float64).reshape(
                    n_fingers, bones_per_finger, 3)
                for f in range(n_fingers):
                    per_finger[f].append(lv[f])
            seen += 1
            if max_frames is not None and seen >= max_frames:
                return [_stack_unit(b) for b in per_finger]
    return [_stack_unit(b) for b in per_finger]


def _stack_unit(blocks):
    if not blocks:
        return np.zeros((0, 3))
    V = np.concatenate([np.asarray(b) for b in blocks], axis=0)
    return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annot-root", type=Path,
                    default=Path("/opt/tiger/InterHand/annotations/machine_annot"))
    ap.add_argument("--fit-split", default="train")
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--ks", type=int, nargs="+", default=[16, 26, 40])
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    print(f"[fit ] loading {args.fit_split} ...", flush=True)
    Vfit, _ = adb.load_local_directions(args.annot_root, args.fit_split, args.max_frames)
    print(f"[fit ] {len(Vfit)} local bone directions")
    print(f"[eval] loading {args.eval_split} ...", flush=True)
    Vev, _ = adb.load_local_directions(args.annot_root, args.eval_split, args.max_frames)
    print(f"[eval] {len(Vev)} local bone directions")

    print(f"[fit ] loading {args.fit_split} per-finger ...", flush=True)
    pf_fit = per_finger_directions(args.annot_root, args.fit_split, args.max_frames)
    print(f"[eval] loading {args.eval_split} per-finger ...", flush=True)
    pf_ev = per_finger_directions(args.annot_root, args.eval_split, args.max_frames)
    n_fingers = len(hl.FINGER_NAMES)

    rows = []
    medoid_examples = {}
    for k in args.ks:
        # spherical k-means (adaptive baseline)
        Ckm = adb.spherical_kmeans(Vfit, k, init="kmeans++", seed=args.seed)
        rows.append({"method": f"kmeans_k{k}", "K": int(k),
                     "bits_per_bone": adb.empirical_entropy_bits(Vev, Ckm),
                     "angular_deg": adb.angular_error_deg(Vev, Ckm)})

        # spherical k-medoids (centers are observed -> readable)
        Cmd = spherical_kmedoids(Vfit, k, seed=args.seed)
        rows.append({"method": f"kmedoids_k{k}", "K": int(k),
                     "bits_per_bone": adb.empirical_entropy_bits(Vev, Cmd),
                     "angular_deg": adb.angular_error_deg(Vev, Cmd)})
        medoid_examples[f"k{k}"] = [[round(float(x), 4) for x in c] for c in Cmd[:5]]

        # region-aware: one codebook per finger, eval per finger then aggregate.
        a_acc, b_acc, w_acc = 0.0, 0.0, 0
        for f in range(n_fingers):
            if len(pf_fit[f]) < k or len(pf_ev[f]) == 0:
                continue
            Cf = adb.spherical_kmeans(pf_fit[f], k, init="kmeans++", seed=args.seed)
            w = len(pf_ev[f])
            a_acc += adb.angular_error_deg(pf_ev[f], Cf) * w
            b_acc += adb.empirical_entropy_bits(pf_ev[f], Cf) * w
            w_acc += w
        if w_acc:
            rows.append({"method": f"region_aware_k{k}", "K": int(k),
                         "bits_per_bone": b_acc / w_acc, "angular_deg": a_acc / w_acc})

    rows.sort(key=lambda r: (r["K"], r["method"]))
    print(f"\n{'method':20s} {'K':>4s} {'bits/bone':>10s} {'angular°':>9s}")
    for r in rows:
        print(f"{r['method']:20s} {r['K']:4d} {r['bits_per_bone']:10.2f} "
              f"{r['angular_deg']:9.2f}")
    print("\nk-medoids centers are real observed bone directions (readable); "
          "first few per K are recorded in the JSON `medoid_examples`.")

    result = {"fit_split": args.fit_split, "eval_split": args.eval_split,
              "n_fit": int(len(Vfit)), "n_eval": int(len(Vev)),
              "seed": args.seed, "rows": rows, "medoid_examples": medoid_examples}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
