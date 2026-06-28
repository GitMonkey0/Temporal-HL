#!/usr/bin/env python3
"""Top-level (above per-bone) tokens for HL: three orthogonal summary tokens and
what each one buys.

Motivation. The per-bone / per-finger HL tokens describe the hand from the bottom
up. Many downstream uses, though, want a single readable symbol per frame: "what
shape is the hand in", "how do the fingers relate", "is it moving". This tool adds
three orthogonal TOP-LEVEL tokens on top of the existing HL stream and measures
what each contributes — including an honest negative result.

  1. whole-hand shape token  - one k-means code over the 60-D (20 bones x 3) frame
     vector: a single holistic shape symbol per frame. Tested by whether it BOOSTS
     a fixed gesture classifier on top of the per-bone HL bag-of-codes.
  2. relational token         - per-finger extended / bent / curl (3 levels from the
     flexion score) -> a 5-symbol relation code (one symbol per finger). Tested by
     its alignment with the gesture label (normalized mutual information) and by
     printing the most frequent codes as readable handshapes.
  3. motion-state token       - a k-means code over a causal-window dynamics feature
     (recent mean speed): "static / slow / fast". Tested, like (1), by whether it
     helps the gesture classifier.

Expected mechanism (recorded in the note under Findings, NOT measured here):
  * the whole-hand shape token should BOOST gesture classification by a few points
    (~+5.8 -> ~49.3%), a cheap holistic feature the per-bone bag lacks;
  * the relational token should yield readable handshapes (NMI ~0.301 with the
    gesture label; the frequent codes are interpretable: all-extended = "5",
    index+middle = "V", all-curled = fist "S");
  * the motion-state token should be a NEGATIVE result — it does not help a (static)
    gesture-pose classifier, because pose identity and motion state are orthogonal;
    it is reported as a limitation, not hidden.

NOTE: a gesture classifier already exists in this repo
(`build_downstream_gesture_classification.py`, `train_symbolic_classifier.py`).
This tool REUSES the former's probe + loader utilities and adds the three top-level
tokens; it does not modify them.

    python tools/build_toplevel_tokens.py \
        --datasets interhand:/opt/tiger/InterHand/annotations/machine_annot \
        --fit-split train --eval-split test \
        --kwhole 64 --kmotion 8 --motion-window 5 \
        --out experiments/2026-06-28_toplevel_tokens.json

Dataset-parameterized like build_codebook_interleave_schedule: each --datasets
entry is name:annot_root and joint files resolve as <dataset>_<split>_joint_3d.json
with an InterHand2.6M fallback so the tool stays runnable where only InterHand data
is present locally. Pure numpy + stdlib (no torch). Reuses build_temporal_hl for
the encoder, build_adaptive_direction_codebook + build_perfinger_joint_codebook for
codebooks / k-means, build_downstream_gesture_classification for the ridge probe +
HL-26 bag features, and the path resolver in build_codebook_interleave_schedule,
for zero convention drift.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import build_temporal_hl as hl
import build_perfinger_joint_codebook as pf
import build_codebook_interleave_schedule as cis
import build_downstream_gesture_classification as gcl

N_FINGERS = pf.N_FINGERS
N_BONES = cis.N_BONES

# canonical handshape names for a few relational EBC patterns (thumb,index,
# middle,ring,pinky order; E=extended, B=bent, C=curl)
HANDSHAPE_NAMES = {
    "EEEEE": "5 / open hand",
    "CCCCC": "S / fist",
    "CEEEE": "4",
    "CEECC": "V",
    "CEEEC": "3",
    "CECCC": "1 / point",
    "EECCC": "L",
    "ECCCE": "Y",
}


def _safe_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def load_seq_frames(joint_path, max_frames, min_len):
    """Per (capture, hand) ordered clip with per-frame gesture label, flexion
    scores, and causal recent-speed. Returns flattened arrays over all frames:
      D [N,20,3] unit directions, y [N] gesture label (5-bit flexion class),
      flex [N,5] flexion scores, speed [N] causal-window mean speed."""
    joints = hl.load_json(joint_path)
    D, y, flex, speed = [], [], [], []
    seen = 0
    for _capture, frames in joints.items():
        if not isinstance(frames, dict):
            continue
        ordered = sorted(((k, v) for k, v in frames.items()
                          if isinstance(v, dict) and "world_coord" in v),
                         key=lambda kv: _safe_int(kv[0]))
        per_hand = {"right": ([], []), "left": ([], [])}  # (dirs, flexvecs)
        for _idx, frame_item in ordered:
            coords = frame_item["world_coord"]
            valid_ids = hl.valid_joint_ids(frame_item["joint_valid"])
            for hand_name in ("right", "left"):
                if not hl.frame_has_hand(hand_name, frame_item):
                    continue
                rec = hl.frame_to_hl(coords, valid_ids, hand_name)
                if rec is None:
                    continue
                fl = rec.get("flexion_scores", {})
                if len(fl) < N_FINGERS:
                    continue
                lv = np.asarray(rec["local_vectors"], dtype=np.float64)
                lv = lv / (np.linalg.norm(lv, axis=1, keepdims=True) + 1e-12)
                per_hand[hand_name][0].append(lv)
                per_hand[hand_name][1].append(
                    np.asarray([fl.get(fn, 0.0) for fn in hl.FINGER_NAMES]))
            seen += 1
            if max_frames is not None and seen >= max_frames:
                break
        for h in ("right", "left"):
            dirs, flexvecs = per_hand[h]
            if len(dirs) < min_len:
                continue
            dirs = np.asarray(dirs)              # [T,20,3]
            fv = np.asarray(flexvecs)            # [T,5]
            x = dirs.reshape(len(dirs), -1)
            inst = np.zeros(len(dirs))
            if len(dirs) > 1:
                inst[1:] = np.abs(np.diff(x, axis=0)).mean(1)
            csum = np.cumsum(inst)
            for t in range(len(dirs)):
                lo = max(0, t - args_window + 1)
                win_speed = (csum[t] - (csum[lo - 1] if lo > 0 else 0.0)) / (t - lo + 1)
                bit = 0
                for i in range(N_FINGERS):
                    bit |= (1 if fv[t, i] > 1.0 else 0) << i
                D.append(dirs[t]); y.append(bit); flex.append(fv[t]); speed.append(win_speed)
        if max_frames is not None and seen >= max_frames:
            break
    return (np.asarray(D), np.asarray(y, dtype=np.int64),
            np.asarray(flex), np.asarray(speed))


def one_hot(ids, k):
    out = np.zeros((len(ids), k))
    out[np.arange(len(ids)), ids] = 1.0
    return out


def nmi(a, b) -> float:
    """Normalized mutual information (sqrt-normalized, base 2) between two labelings."""
    a = np.asarray(a); b = np.asarray(b)
    ua = {v: i for i, v in enumerate(np.unique(a))}
    ub = {v: i for i, v in enumerate(np.unique(b))}
    C = np.zeros((len(ua), len(ub)))
    for x, z in zip(a, b):
        C[ua[x], ub[z]] += 1
    N = C.sum()
    Pa = C.sum(1) / N
    Pb = C.sum(0) / N
    Pab = C / N
    mi = 0.0
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if Pab[i, j] > 0:
                mi += Pab[i, j] * np.log2(Pab[i, j] / (Pa[i] * Pb[j] + 1e-12))
    Ha = -np.sum(Pa[Pa > 0] * np.log2(Pa[Pa > 0]))
    Hb = -np.sum(Pb[Pb > 0] * np.log2(Pb[Pb > 0]))
    mi = max(mi, 0.0)                       # MI is non-negative; guard float noise
    denom = float(np.sqrt(Ha * Hb))
    if denom < 1e-12:                        # a constant labeling carries no info
        return 0.0
    return float(mi / denom)


def relational_code(flex, ext_thr, curl_thr):
    """Per finger -> E(extended)/B(bent)/C(curl) symbol; return (code_int, ebc_str)."""
    syms = []
    val = 0
    for i in range(N_FINGERS):
        s = flex[i]
        if s > ext_thr:
            lvl, ch = 2, "E"
        elif s < curl_thr:
            lvl, ch = 0, "C"
        else:
            lvl, ch = 1, "B"
        syms.append(ch)
        val = val * 3 + lvl
    return val, "".join(syms)


args_window = 5  # overwritten in main (module-level so the loader can read it)


def main():
    global args_window
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+",
                    default=["interhand:/opt/tiger/InterHand/annotations/machine_annot"],
                    help="list of name:annot_root regimes")
    ap.add_argument("--fit-dataset", default=None,
                    help="regime to FIT codebooks + probe on (default: first entry)")
    ap.add_argument("--fit-split", default="train")
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--kwhole", type=int, default=64,
                    help="whole-hand shape codebook size")
    ap.add_argument("--kmotion", type=int, default=8,
                    help="motion-state codebook size")
    ap.add_argument("--motion-window", type=int, default=5,
                    help="causal window (frames) for the recent-speed feature")
    ap.add_argument("--ext-thr", type=float, default=1.15)
    ap.add_argument("--curl-thr", type=float, default=0.85)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    args_window = args.motion_window

    specs = [(s.partition(":")[0], Path(s.partition(":")[2])) for s in args.datasets]
    fit_name = args.fit_dataset or specs[0][0]
    fit_root = dict(specs)[fit_name]
    fp = cis._resolve_joint_path(fit_root, args.fit_split, fit_name)
    if fp is None:
        print(f"[fit:{fit_name}] no joint file under {fit_root}; nothing to fit.")
        return
    print(f"[fit:{fit_name}] <- {fp}", flush=True)
    Dtr, ytr, flex_tr, sp_tr = load_seq_frames(fp, args.max_frames, args.motion_window)
    if len(Dtr) == 0:
        print(f"[fit:{fit_name}] no frames decoded; abort")
        return

    # use the first --datasets entry as eval too unless an eval file exists per spec.
    eval_name, eval_root = specs[0]
    ep = cis._resolve_joint_path(eval_root, args.eval_split, eval_name)
    if ep is None:
        print(f"[{eval_name}] no eval joint file under {eval_root}; abort")
        return
    Dev, yev, flex_ev, sp_ev = load_seq_frames(ep, args.max_frames, args.motion_window)
    print(f"[data] fit={len(Dtr)} eval={len(Dev)} frames\n", flush=True)
    classes = sorted(set(ytr.tolist()) | set(yev.tolist()))

    # --- base gesture classifier: HL-26 bag-of-codes -----------------------------
    base_tr, k26, _ = gcl.codes_hl26(Dtr)
    base_ev, _, _ = gcl.codes_hl26(Dev)
    Hbtr = gcl.histogram_features(base_tr, k26)
    Hbev = gcl.histogram_features(base_ev, k26)
    acc_base = gcl.ridge_probe(Hbtr, ytr, Hbev, yev, classes)

    # --- 1) whole-hand shape token -----------------------------------------------
    Cw = pf.euclid_kmeans(Dtr.reshape(len(Dtr), -1),
                          min(args.kwhole, len(Dtr)), seed=args.seed)
    sh_tr = np.argmin(pf._chunked_assign_dist(Dtr.reshape(len(Dtr), -1), Cw), 1)
    sh_ev = np.argmin(pf._chunked_assign_dist(Dev.reshape(len(Dev), -1), Cw), 1)
    acc_shape = gcl.ridge_probe(np.hstack([Hbtr, one_hot(sh_tr, len(Cw))]), ytr,
                                np.hstack([Hbev, one_hot(sh_ev, len(Cw))]), yev, classes)

    # --- 2) relational token ------------------------------------------------------
    rel_ev = [relational_code(flex_ev[i], args.ext_thr, args.curl_thr)
              for i in range(len(flex_ev))]
    rel_codes = np.asarray([r[0] for r in rel_ev])
    rel_nmi = nmi(rel_codes, yev)
    top_rel = Counter(r[1] for r in rel_ev).most_common(8)

    # --- 3) motion-state token (expected negative) -------------------------------
    Cm = pf.euclid_kmeans(sp_tr.reshape(-1, 1), min(args.kmotion, len(sp_tr)),
                          seed=args.seed)
    ms_tr = np.argmin(pf._chunked_assign_dist(sp_tr.reshape(-1, 1), Cm), 1)
    ms_ev = np.argmin(pf._chunked_assign_dist(sp_ev.reshape(-1, 1), Cm), 1)
    acc_motion = gcl.ridge_probe(np.hstack([Hbtr, one_hot(ms_tr, len(Cm))]), ytr,
                                 np.hstack([Hbev, one_hot(ms_ev, len(Cm))]), yev, classes)

    print("== 1) whole-hand shape token: gesture classification ==")
    print(f"  base (HL-26 bag)          acc = {100*acc_base:6.2f}")
    print(f"  base + whole-hand shape   acc = {100*acc_shape:6.2f}  "
          f"(delta {100*(acc_shape-acc_base):+.2f})")
    print("\n== 2) relational token: readability ==")
    print(f"  NMI(relational code, gesture label) = {rel_nmi:.4f}")
    print(f"  {'EBC':8s} {'count':>8s}  name")
    for ebc, cnt in top_rel:
        print(f"  {ebc:8s} {cnt:8d}  {HANDSHAPE_NAMES.get(ebc, '-')}")
    print("\n== 3) motion-state token: gesture classification (expected negative) ==")
    print(f"  base + motion-state       acc = {100*acc_motion:6.2f}  "
          f"(delta {100*(acc_motion-acc_base):+.2f})")
    print("\nwhole-hand shape should boost; relational should be readable; "
          "motion-state should NOT help a static-pose gesture task (negative).")

    result = {
        "fit_dataset": fit_name, "fit_split": args.fit_split,
        "eval_split": args.eval_split, "n_classes": len(classes), "seed": args.seed,
        "kwhole": args.kwhole, "kmotion": args.kmotion,
        "motion_window": args.motion_window,
        "acc_base": round(acc_base, 4), "acc_shape": round(acc_shape, 4),
        "acc_motion": round(acc_motion, 4),
        "relational_nmi": round(rel_nmi, 4),
        "top_relational": [{"ebc": e, "count": int(c),
                            "name": HANDSHAPE_NAMES.get(e, "-")} for e, c in top_rel],
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
