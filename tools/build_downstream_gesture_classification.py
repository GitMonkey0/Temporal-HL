#!/usr/bin/env python3
"""Downstream discriminative probe: gesture classification on token streams from
different tokenizers (HL-26 vs data-adaptive k-means vs per-finger joint).

Motivation. All the intrinsic experiments measure reconstruction angular error.
A representation can have low reconstruction error and still be a poor *symbolic*
substrate for downstream models. This probe closes that loop with a simple,
fixed classifier trained on the discrete token streams each tokenizer emits, and
asks which tokenizer yields the most discriminable codes for a gesture-label task.

It compares three tokenizers at the per-frame, per-hand level:
  * hl26          - the fixed 26-direction HL codebook (baseline)
  * adaptive_kmeans - a data-adaptive spherical k-means codebook (per bone)
  * per_finger    - one joint code per finger (anatomy-aware, ~half the bit-rate)

A gesture label per frame is derived from the discretized hand-pose signature
(here, the per-finger flexion pattern from `compute_finger_flexion`, binned into
extended/curled), so the task is well-defined without external annotations and is
identical across tokenizers. The classifier is a plain bag-of-codes multinomial
logistic-regression-style linear probe trained with closed-form ridge on one-hot
code histograms — self-contained numpy, no torch / sklearn dependency required.

Expected mechanism: the data-adaptive codebook should add a few points of
accuracy over HL-26 (it spends codes where the data is), and the per-finger code
should match or beat HL-26 at roughly half the bit-rate (its codes are whole
finger poses, which are exactly what a gesture is made of).

NOTE: a related classifier tool already exists in this repo
(`train_symbolic_classifier.py` / `eval_symbolic_channel_variants_classifier.py`).
This is a clean, self-contained scoped entry point for the tokenizer comparison;
it does not modify or call those.

    python tools/build_downstream_gesture_classification.py \
        --annot-root <InterHand annotations> --fit-split train --eval-split test \
        --kadaptive 26 --kfinger 128 \
        --out experiments/downstream_gesture_classification_<date>.json

Pure numpy + stdlib; reuses build_temporal_hl + build_adaptive_direction_codebook
(+ the per-finger loader/codebook in build_perfinger_joint_codebook) for zero
convention drift.
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


def load_frames(annot_root: Path, split: str, max_frames):
    """Per-frame, per-hand records: 20 local directions + a gesture label derived
    from the per-finger flexion pattern (extended/curled per finger -> 0..31)."""
    fname = f"InterHand2.6M_{split}_joint_3d.json"
    for cand in (annot_root / fname, annot_root / split / fname):
        if cand.exists():
            joint_path = cand
            break
    else:
        raise FileNotFoundError(f"{fname} not under {annot_root} (flat or /{split}/)")
    joints = hl.load_json(joint_path)
    dirs, labels = [], []
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
                flex = rec.get("flexion_scores", {})
                if len(flex) < N_FINGERS:
                    continue
                lv = np.asarray(rec["local_vectors"], dtype=np.float64)
                lv = lv / (np.linalg.norm(lv, axis=1, keepdims=True) + 1e-12)
                # gesture label: per-finger extended(1)/curled(0) -> 5-bit class
                bit = 0
                for i, fn in enumerate(hl.FINGER_NAMES):
                    bit |= (1 if flex.get(fn, 0.0) > 1.0 else 0) << i
                dirs.append(lv)
                labels.append(bit)
            seen += 1
            if max_frames is not None and seen >= max_frames:
                return np.asarray(dirs), np.asarray(labels, dtype=np.int64)
    return np.asarray(dirs), np.asarray(labels, dtype=np.int64)


def codes_hl26(D):
    C = adb.hl26_codebook()
    flat = D.reshape(-1, 3)
    a = np.argmax(flat @ C.T, axis=1).reshape(len(D), -1)
    return a, len(C), float(D.shape[1] * np.log2(len(C)))


def codes_adaptive(Dtr, Dev, k, seed=0):
    C = adb.spherical_kmeans(Dtr.reshape(-1, 3), k, init="kmeans++", seed=seed)
    at = np.argmax(Dtr.reshape(-1, 3) @ C.T, axis=1).reshape(len(Dtr), -1)
    ae = np.argmax(Dev.reshape(-1, 3) @ C.T, axis=1).reshape(len(Dev), -1)
    return at, ae, k, float(Dtr.shape[1] * np.log2(k))


def codes_per_finger(Dtr, Dev, k, seed=0):
    """One joint 12-D code per finger; 5 codes per frame, bits = 5*log2(k)/20
    per bone -> bit-rate per frame = 5*log2(k)."""
    Xtr = Dtr.reshape(len(Dtr), N_FINGERS, BPF, 3)
    Xev = Dev.reshape(len(Dev), N_FINGERS, BPF, 3)
    at = np.empty((len(Dtr), N_FINGERS), dtype=np.int64)
    ae = np.empty((len(Dev), N_FINGERS), dtype=np.int64)
    for f in range(N_FINGERS):
        Ct = pf.euclid_kmeans(Xtr[:, f].reshape(len(Dtr), -1), k, seed=seed)
        at[:, f] = np.argmin(pf._chunked_assign_dist(
            Xtr[:, f].reshape(len(Dtr), -1), Ct), axis=1)
        ae[:, f] = np.argmin(pf._chunked_assign_dist(
            Xev[:, f].reshape(len(Dev), -1), Ct), axis=1)
    return at, ae, k, float(N_FINGERS * np.log2(k))


def histogram_features(codes, k):
    """One-hot code histogram per frame (bag of codes), L1-normalized."""
    n, c = codes.shape
    feat = np.zeros((n, k))
    for j in range(c):
        feat[np.arange(n), codes[:, j]] += 1.0
    return feat / (feat.sum(1, keepdims=True) + 1e-12)


def ridge_probe(Xtr, ytr, Xev, yev, classes, lam=1.0):
    """Closed-form one-vs-rest ridge classifier (a linear probe). Returns top-1
    accuracy on the eval split."""
    Y = np.zeros((len(ytr), len(classes)))
    cls_index = {c: i for i, c in enumerate(classes)}
    for i, y in enumerate(ytr):
        Y[i, cls_index[y]] = 1.0
    Xb = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    W = np.linalg.solve(A, Xb.T @ Y)
    Xeb = np.hstack([Xev, np.ones((len(Xev), 1))])
    pred = np.argmax(Xeb @ W, axis=1)
    pred_lab = np.asarray(classes)[pred]
    return float((pred_lab == yev).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annot-root", type=Path,
                    default=Path("/opt/tiger/InterHand/annotations/machine_annot"))
    ap.add_argument("--fit-split", default="train")
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--kadaptive", type=int, default=26)
    ap.add_argument("--kfinger", type=int, default=128)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    print(f"[fit ] loading {args.fit_split} ...", flush=True)
    Dtr, ytr = load_frames(args.annot_root, args.fit_split, args.max_frames)
    print(f"[eval] loading {args.eval_split} ...", flush=True)
    Dev, yev = load_frames(args.annot_root, args.eval_split, args.max_frames)
    print(f"[data] fit={len(Dtr)} eval={len(Dev)} frames\n")

    classes = sorted(set(ytr.tolist()) | set(yev.tolist()))
    rows = []

    a_tr, k26, bits26 = codes_hl26(Dtr)
    a_ev, _, _ = codes_hl26(Dev)
    acc = ridge_probe(histogram_features(a_tr, k26), ytr,
                      histogram_features(a_ev, k26), yev, classes)
    rows.append({"tokenizer": "hl26", "K": k26, "bits_per_frame": bits26,
                 "accuracy": round(acc, 4)})

    at, ae, kad, bitsad = codes_adaptive(Dtr, Dev, args.kadaptive, args.seed)
    acc = ridge_probe(histogram_features(at, kad), ytr,
                      histogram_features(ae, kad), yev, classes)
    rows.append({"tokenizer": f"adaptive_kmeans_k{kad}", "K": kad,
                 "bits_per_frame": bitsad, "accuracy": round(acc, 4)})

    at, ae, kf, bitsf = codes_per_finger(Dtr, Dev, args.kfinger, args.seed)
    acc = ridge_probe(histogram_features(at, kf), ytr,
                      histogram_features(ae, kf), yev, classes)
    rows.append({"tokenizer": f"per_finger_k{kf}", "K": kf,
                 "bits_per_frame": bitsf, "accuracy": round(acc, 4)})

    print(f"{'tokenizer':22s} {'K':>5s} {'bits/frame':>11s} {'acc':>7s}")
    for r in rows:
        print(f"{r['tokenizer']:22s} {r['K']:5d} {r['bits_per_frame']:11.2f} "
              f"{100*r['accuracy']:7.2f}")
    print("\nadaptive should add a few points over HL-26; per-finger should match "
          "or beat HL-26 at roughly half the bit-rate.")

    result = {"fit_split": args.fit_split, "eval_split": args.eval_split,
              "n_classes": len(classes), "seed": args.seed, "rows": rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
