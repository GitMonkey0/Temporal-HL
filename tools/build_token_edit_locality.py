#!/usr/bin/env python3
"""Token edit-locality audit: does changing ONE finger's code leave the other
fingers untouched? An interpretability / controllability property of the tokenizer.

Motivation. A symbolic tokenizer is only useful for editing if its tokens are
*local*: changing the symbol for one finger should change only that finger and
leave the rest of the hand exactly where it was. A black-box VQ that encodes the
whole hand into one (or a few) entangled latent codes cannot promise this — the
nearest alternative code that changes the target finger almost always perturbs the
others too. This tool measures that collateral motion directly.

It compares two tokenizers under the SAME minimal-edit protocol:
  * per_finger - one 12-D code per finger (5 codes/frame). To change finger f we
                 swap finger f's code for the nearest alternative codeword that
                 actually moves finger f (>= --edit-delta deg); the other fingers'
                 codes are untouched.
  * whole_hand - one 60-D code for the entire hand (a stand-in for a black-box,
                 whole-hand VQ). To change finger f we must move to a different
                 whole-hand codeword — the nearest one whose finger-f
                 reconstruction differs by >= --edit-delta deg — which also drags
                 the other fingers.

For every (frame, target finger) it records the angular motion of the EDITED
finger's 4 bones and the angular motion of the OTHER 16 bones, then aggregates.

Expected mechanism (recorded in the note under Findings, NOT measured here): the
per-finger code should give perfect locality — the edited finger moves a lot
(~98.3 deg for a large pose change) while the other fingers move 0.00 deg exactly,
because each finger decodes from its own code; the whole-hand VQ should leak
non-trivial motion into the other fingers, the locality property a factorized
symbolic code provides that a black-box VQ cannot.

    python tools/build_token_edit_locality.py \
        --datasets interhand:/opt/tiger/InterHand/annotations/machine_annot \
        --fit-split train --eval-split test \
        --kfinger 128 --kwhole 512 --edit-delta 5 --n-frames 2000 \
        --out experiments/2026-06-28_token_edit_locality.json

Dataset-parameterized like build_codebook_interleave_schedule: each --datasets
entry is name:annot_root and joint files resolve as <dataset>_<split>_joint_3d.json
with an InterHand2.6M fallback so the tool stays runnable where only InterHand data
is present locally. Pure numpy + stdlib (no torch). Reuses build_temporal_hl for
the encoder, build_perfinger_joint_codebook for the per-finger code / k-means, and
the sequence loader / path resolver in build_codebook_interleave_schedule, for zero
convention drift.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import build_temporal_hl as hl
import build_perfinger_joint_codebook as pf
import build_codebook_interleave_schedule as cis

N_FINGERS = pf.N_FINGERS                 # 5
BPF = pf.BONES_PER_FINGER                # 4
N_BONES = cis.N_BONES                    # 20


def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def _bone_angles_deg(a, b):
    """Per-bone angle (deg) between two [...,3] unit arrays."""
    cos = np.clip((_unit(a) * _unit(b)).sum(-1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def fit_codebooks(fit_seqs, kfinger, kwhole, seed):
    stack = np.concatenate(fit_seqs, axis=0)             # [N,20,3]
    grid = stack.reshape(len(stack), N_FINGERS, BPF, 3)
    finger_cb = []
    for f in range(N_FINGERS):
        X = grid[:, f].reshape(len(stack), -1)           # [N,12]
        finger_cb.append(pf.euclid_kmeans(X, min(kfinger, len(X)), seed=seed))
    whole_cb = pf.euclid_kmeans(stack.reshape(len(stack), -1),
                                min(kwhole, len(stack)), seed=seed)  # [kw,60]
    return finger_cb, whole_cb


def nearest_alt_with_change(C_finger_recon, base_recon_f, dists, delta):
    """Among candidate codewords (with per-candidate finger-f reconstruction
    C_finger_recon [k,4,3] and distance-to-base dists [k]), return the index of the
    CLOSEST candidate whose finger-f reconstruction differs from base_recon_f by at
    least `delta` deg (mean over the 4 bones). None if no candidate qualifies."""
    ang = _bone_angles_deg(C_finger_recon, base_recon_f[None]).mean(1)  # [k]
    qualifies = ang >= delta
    if not qualifies.any():
        return None
    order = np.argsort(dists)
    for idx in order:
        if qualifies[idx]:
            return int(idx)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+",
                    default=["interhand:/opt/tiger/InterHand/annotations/machine_annot"],
                    help="list of name:annot_root regimes")
    ap.add_argument("--fit-dataset", default=None,
                    help="regime to FIT codebooks on (default: first entry)")
    ap.add_argument("--fit-split", default="train")
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--kfinger", type=int, default=128)
    ap.add_argument("--kwhole", type=int, default=512)
    ap.add_argument("--edit-delta", type=float, default=5.0,
                    help="minimal finger-f change (deg) an edit must induce")
    ap.add_argument("--n-frames", type=int, default=2000,
                    help="cap eval frames sampled for the audit")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    specs = [(s.partition(":")[0], Path(s.partition(":")[2])) for s in args.datasets]
    fit_name = args.fit_dataset or specs[0][0]
    fit_root = dict(specs)[fit_name]

    fp = cis._resolve_joint_path(fit_root, args.fit_split, fit_name)
    if fp is None:
        print(f"[fit:{fit_name}] no joint file under {fit_root}; nothing to fit.")
        return
    print(f"[fit:{fit_name}] codebooks <- {fp}", flush=True)
    fit_seqs = cis.load_sequences(fp, args.max_frames)
    if not fit_seqs:
        print(f"[fit:{fit_name}] no sequences decoded; abort")
        return
    finger_cb, whole_cb = fit_codebooks(fit_seqs, args.kfinger, args.kwhole, args.seed)
    print(f"[fit:{fit_name}] per-finger K={[len(c) for c in finger_cb]}, "
          f"whole-hand K={len(whole_cb)}", flush=True)

    rng = np.random.default_rng(args.seed)
    rows = []
    for name, root in specs:
        ep = cis._resolve_joint_path(root, args.eval_split, name)
        if ep is None:
            print(f"[{name}] no eval joint file under {root}; skip")
            continue
        eval_seqs = cis.load_sequences(ep, args.max_frames)
        if not eval_seqs:
            print(f"[{name}] no eval sequences; skip")
            continue
        frames = np.concatenate(eval_seqs, axis=0)       # [M,20,3]
        if len(frames) > args.n_frames:
            idx = rng.choice(len(frames), size=args.n_frames, replace=False)
            frames = frames[idx]
        grid = frames.reshape(len(frames), N_FINGERS, BPF, 3)
        whole_flat = frames.reshape(len(frames), -1)     # [M,60]

        # nearest codes for each frame
        finger_assign = []
        for f in range(N_FINGERS):
            X = grid[:, f].reshape(len(frames), -1)
            finger_assign.append(np.argmin(pf._chunked_assign_dist(X, finger_cb[f]), 1))
        whole_assign = np.argmin(pf._chunked_assign_dist(whole_flat, whole_cb), 1)

        # precompute whole-hand codeword reconstructions reshaped to [kw,5,4,3]
        whole_recon = _unit(whole_cb.reshape(len(whole_cb), N_FINGERS, BPF, 3))
        finger_recon = [_unit(finger_cb[f].reshape(len(finger_cb[f]), BPF, 3))
                        for f in range(N_FINGERS)]

        pf_edit, pf_other, pf_n = 0.0, 0.0, 0
        wh_edit, wh_other, wh_n = 0.0, 0.0, 0
        for m in range(len(frames)):
            for f in range(N_FINGERS):
                # ---- per_finger: swap finger f's code only ----
                base_f = finger_recon[f][finger_assign[f][m]]            # [4,3]
                Cf = finger_recon[f]
                dists = ((finger_cb[f] - finger_cb[f][finger_assign[f][m]]) ** 2).sum(1)
                alt = nearest_alt_with_change(Cf, base_f, dists, args.edit_delta)
                if alt is not None:
                    new_f = Cf[alt]
                    # rebuild the 5-finger reconstruction with only finger f changed
                    orig = np.stack([finger_recon[g][finger_assign[g][m]]
                                     for g in range(N_FINGERS)])         # [5,4,3]
                    edited = orig.copy()
                    edited[f] = new_f
                    edit_motion = _bone_angles_deg(edited[f], orig[f]).mean()
                    other = [g for g in range(N_FINGERS) if g != f]
                    other_motion = _bone_angles_deg(edited[other], orig[other]).mean()
                    pf_edit += edit_motion; pf_other += other_motion; pf_n += 1

                # ---- whole_hand: must move to a different whole-hand codeword ----
                c0 = whole_assign[m]
                base_wf = whole_recon[c0, f]                             # [4,3]
                dw = ((whole_cb - whole_cb[c0]) ** 2).sum(1)
                altw = nearest_alt_with_change(whole_recon[:, f], base_wf, dw,
                                               args.edit_delta)
                if altw is not None:
                    orig = whole_recon[c0]                               # [5,4,3]
                    edited = whole_recon[altw]                           # [5,4,3]
                    edit_motion = _bone_angles_deg(edited[f], orig[f]).mean()
                    other = [g for g in range(N_FINGERS) if g != f]
                    other_motion = _bone_angles_deg(edited[other], orig[other]).mean()
                    wh_edit += edit_motion; wh_other += other_motion; wh_n += 1

        if pf_n:
            rows.append({"dataset": name, "tokenizer": "per_finger",
                         "edited_finger_deg": round(pf_edit / pf_n, 4),
                         "other_fingers_deg": round(pf_other / pf_n, 4),
                         "leakage_ratio": round((pf_other / pf_n) /
                                                (pf_edit / pf_n + 1e-12), 4)})
        if wh_n:
            rows.append({"dataset": name, "tokenizer": "whole_hand_vq",
                         "edited_finger_deg": round(wh_edit / wh_n, 4),
                         "other_fingers_deg": round(wh_other / wh_n, 4),
                         "leakage_ratio": round((wh_other / wh_n) /
                                                (wh_edit / wh_n + 1e-12), 4)})

    print("\n== edit locality: change ONE finger's code, measure collateral motion ==")
    print(f"{'dataset':10s} {'tokenizer':14s} {'edited°':>9s} {'others°':>9s} "
          f"{'leak':>7s}")
    for r in rows:
        print(f"{r['dataset']:10s} {r['tokenizer']:14s} "
              f"{r['edited_finger_deg']:9.2f} {r['other_fingers_deg']:9.2f} "
              f"{r['leakage_ratio']:7.3f}")
    print("\nper_finger should move only the edited finger (others ~0.00, perfect "
          "locality); the whole-hand VQ should leak motion into the other fingers.")

    result = {"fit_dataset": fit_name, "fit_split": args.fit_split,
              "eval_split": args.eval_split, "kfinger": args.kfinger,
              "kwhole": args.kwhole, "edit_delta": args.edit_delta,
              "seed": args.seed, "rows": rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
