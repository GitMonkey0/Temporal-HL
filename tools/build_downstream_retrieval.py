#!/usr/bin/env python3
"""Training-free sequence retrieval on token streams from different tokenizers.

Motivation. A discrete tokenization unlocks a capability continuous coordinates do
not have for free: training-free symbolic matching. Once a motion clip is a string
of tokens, two clips can be compared by bag-of-tokens overlap, shared n-grams, or
token edit / warping distance — no learned encoder, no training. This probe asks
how retrievable a tokenizer's streams are: given a query clip, can we pull back a
clip of the SAME gesture purely by symbolic similarity?

It compares three tokenizers (hl26 / adaptive_kmeans / per_finger) under three
training-free, discrete-only matchers, plus a continuous reference:
  * bag    - per-(slot,code) token histogram, cosine similarity
  * ngram  - consecutive frame-word bigram sets, Jaccard similarity
  * dtw    - dynamic time warping with per-frame Hamming (token-mismatch) cost
  * float_dtw (reference) - DTW with L2 cost on the raw float directions: the ONLY
             matcher continuous coordinates support; bag / ngram need discrete
             symbols and are undefined on floats. This is the concrete sense in
             which symbolic matching is a discrete-only capability.

A self-supervised gesture label per clip (majority per-frame flexion signature,
the same 5-bit extended/curled code used by the gesture-classification probe)
defines relevance. Retrieval is leave-one-out precision@1 (does the nearest OTHER
clip share the query's gesture?), reported as an absolute score AND as a multiple
of the random-pick baseline (the expected same-label rate of the gallery).

Expected mechanism (recorded in the note under Findings, NOT measured here):
symbolic retrieval should reach many times the random baseline (up to ~31x), with
the per-finger bag-of-tokens matcher strongest (~25.4% precision@1), because a
whole-finger code names exactly the configuration that defines a handshape; the
float_dtw reference shows the continuous baseline can warp but cannot do the
bag / n-gram edit-style matching that drives the gains.

NOTE: a learned symbolic-retrieval stack already exists in this repo
(`eval_sequence_symbolic_retrieval.py`, `eval_symbolic_retrieval.py`). This is a
clean, self-contained, TRAINING-FREE scoped entry point for the tokenizer
comparison and does not modify or call those.

    python tools/build_downstream_retrieval.py \
        --datasets interhand:/opt/tiger/InterHand/annotations/machine_annot \
        --eval-split test --kadaptive 64 --kfinger 128 \
        --matchers bag ngram dtw --max-seqs 400 \
        --out experiments/2026-06-28_downstream_retrieval.json

Dataset-parameterized like build_codebook_interleave_schedule: each --datasets
entry is name:annot_root and joint files resolve as <dataset>_<split>_joint_3d.json
with an InterHand2.6M fallback so the tool stays runnable where only InterHand data
is present locally. Pure numpy + stdlib (no torch). Reuses build_temporal_hl for
the encoder, build_adaptive_direction_codebook / build_perfinger_joint_codebook for
codebooks, build_downstream_forecast for the tokenizer classes, and the path
resolver in build_codebook_interleave_schedule, for zero convention drift.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import build_temporal_hl as hl
import build_codebook_interleave_schedule as cis
import build_downstream_forecast as fc

N_FINGERS = fc.N_FINGERS
N_BONES = fc.N_BONES


def _safe_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def load_labeled_sequences(joint_path, max_frames, min_len, max_len_keep):
    """Per (capture, hand) ordered clip: (clean dir seq [T,20,3], gesture label).
    Gesture label = the majority per-frame 5-bit extended/curled flexion code, the
    same self-supervised signature used by the gesture-classification probe."""
    joints = hl.load_json(joint_path)
    out = []
    seen = 0
    for _capture, frames in joints.items():
        if not isinstance(frames, dict):
            continue
        ordered = sorted(((k, v) for k, v in frames.items()
                          if isinstance(v, dict) and "world_coord" in v),
                         key=lambda kv: _safe_int(kv[0]))
        per_hand = {"right": [], "left": []}
        per_flex = {"right": [], "left": []}
        for _idx, frame_item in ordered:
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
                bit = 0
                for i, fn in enumerate(hl.FINGER_NAMES):
                    bit |= (1 if flex.get(fn, 0.0) > 1.0 else 0) << i
                per_hand[hand_name].append(lv)
                per_flex[hand_name].append(bit)
            seen += 1
            if max_frames is not None and seen >= max_frames:
                break
        for h in ("right", "left"):
            if len(per_hand[h]) >= min_len:
                seq = np.asarray(per_hand[h])
                if max_len_keep is not None and len(seq) > max_len_keep:
                    step = int(np.ceil(len(seq) / max_len_keep))
                    seq = seq[::step]
                label = Counter(per_flex[h]).most_common(1)[0][0]
                out.append((seq, int(label)))
        if max_frames is not None and seen >= max_frames:
            break
    return out


# --------------------------------------------------------------------------- #
# discrete matchers (operate on token id arrays [T, n_tok])
# --------------------------------------------------------------------------- #
def bag_feature(ids, K):
    """Per-(slot,code) histogram, L2-normalized -> cosine similarity is a dot."""
    n_tok = ids.shape[1]
    feat = np.zeros(n_tok * K)
    for j in range(n_tok):
        h = np.bincount(ids[:, j], minlength=K)
        feat[j * K:(j + 1) * K] = h
    n = np.linalg.norm(feat)
    return feat / (n + 1e-12)


def ngram_set(ids):
    """Set of consecutive frame-word bigrams (each frame-word = its token tuple)."""
    words = [tuple(int(v) for v in row) for row in ids]
    return set(zip(words[:-1], words[1:])) if len(words) > 1 else set(words)


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / (len(a | b) + 1e-12)


def dtw_cost(A, B, frame_cost):
    """Generic DTW; frame_cost(i,j) gives the local cost between A[i] and B[j]."""
    Ta, Tb = len(A), len(B)
    D = np.full((Ta + 1, Tb + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, Ta + 1):
        for j in range(1, Tb + 1):
            c = frame_cost(A[i - 1], B[j - 1])
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[Ta, Tb] / (Ta + Tb)


def hamming_frame(a, b):
    return float(np.mean(a != b))


def l2_frame(a, b):
    return float(np.linalg.norm(a.reshape(-1) - b.reshape(-1)))


def leave_one_out_p1(scores, labels, higher_is_better):
    """scores[i,j] = similarity/distance from i to j (diag ignored). precision@1 =
    fraction of queries whose nearest OTHER clip shares the gesture label."""
    N = len(labels)
    labels = np.asarray(labels)
    hit = 0
    for i in range(N):
        s = scores[i].copy()
        s[i] = -np.inf if higher_is_better else np.inf
        j = int(np.argmax(s) if higher_is_better else np.argmin(s))
        hit += int(labels[j] == labels[i])
    return hit / N


def random_p1(labels):
    """Expected same-label rate of a random other-clip pick (the chance baseline)."""
    labels = np.asarray(labels)
    N = len(labels)
    if N < 2:
        return float("nan")
    acc = 0.0
    counts = Counter(labels.tolist())
    for i in range(N):
        acc += (counts[labels[i]] - 1) / (N - 1)
    return acc / N


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+",
                    default=["interhand:/opt/tiger/InterHand/annotations/machine_annot"],
                    help="list of name:annot_root regimes")
    ap.add_argument("--fit-dataset", default=None,
                    help="regime to FIT codebooks on (default: first entry)")
    ap.add_argument("--fit-split", default="train")
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--kadaptive", type=int, default=64)
    ap.add_argument("--kfinger", type=int, default=128)
    ap.add_argument("--matchers", nargs="+", default=["bag", "ngram", "dtw"])
    ap.add_argument("--with-float-dtw", action="store_true", default=True,
                    help="also score the continuous-DTW reference baseline")
    ap.add_argument("--min-len", type=int, default=4)
    ap.add_argument("--max-len-keep", type=int, default=48,
                    help="downsample long clips to this many frames (DTW is O(T^2))")
    ap.add_argument("--max-seqs", type=int, default=400,
                    help="cap clips per dataset (leave-one-out is O(N^2))")
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
    tokenizers = [
        fc.HL26Tokenizer(),
        fc.AdaptiveTokenizer(fit_seqs, args.kadaptive, seed=args.seed),
        fc.PerFingerTokenizer(fit_seqs, args.kfinger, seed=args.seed),
    ]

    rng = np.random.default_rng(args.seed)
    rows = []
    float_rows = []
    for name, root in specs:
        ep = cis._resolve_joint_path(root, args.eval_split, name)
        if ep is None:
            print(f"[{name}] no eval joint file under {root}; skip")
            continue
        clips = load_labeled_sequences(ep, args.max_frames, args.min_len,
                                       args.max_len_keep)
        if len(clips) < 2:
            print(f"[{name}] <2 clips; skip")
            continue
        if len(clips) > args.max_seqs:
            idx = rng.choice(len(clips), size=args.max_seqs, replace=False)
            clips = [clips[i] for i in idx]
        seqs = [c[0] for c in clips]
        labels = [c[1] for c in clips]
        N = len(seqs)
        rnd = random_p1(labels)
        print(f"[{name}] {N} clips, {len(set(labels))} gesture labels; "
              f"random p@1 = {rnd:.4f}", flush=True)

        # continuous reference: float DTW on raw directions
        if args.with_float_dtw:
            Dmat = np.zeros((N, N))
            for i in range(N):
                for j in range(i + 1, N):
                    d = dtw_cost(seqs[i], seqs[j], l2_frame)
                    Dmat[i, j] = Dmat[j, i] = d
            p1 = leave_one_out_p1(Dmat, labels, higher_is_better=False)
            float_rows.append({"dataset": name, "matcher": "float_dtw",
                               "precision_at_1": round(p1, 4), "random": round(rnd, 4),
                               "ratio_vs_random": round(p1 / rnd, 2) if rnd else None})

        for tok in tokenizers:
            ids_list = tok.encode(seqs)
            for matcher in args.matchers:
                if matcher == "bag":
                    F = np.stack([bag_feature(ids, tok.K) for ids in ids_list])
                    S = F @ F.T
                    p1 = leave_one_out_p1(S, labels, higher_is_better=True)
                elif matcher == "ngram":
                    grams = [ngram_set(ids) for ids in ids_list]
                    S = np.zeros((N, N))
                    for i in range(N):
                        for j in range(i + 1, N):
                            S[i, j] = S[j, i] = jaccard(grams[i], grams[j])
                    p1 = leave_one_out_p1(S, labels, higher_is_better=True)
                elif matcher == "dtw":
                    Dm = np.zeros((N, N))
                    for i in range(N):
                        for j in range(i + 1, N):
                            Dm[i, j] = Dm[j, i] = dtw_cost(ids_list[i], ids_list[j],
                                                           hamming_frame)
                    p1 = leave_one_out_p1(Dm, labels, higher_is_better=False)
                else:
                    raise ValueError(matcher)
                rows.append({"dataset": name, "tokenizer": tok.name,
                             "matcher": matcher, "precision_at_1": round(p1, 4),
                             "random": round(rnd, 4),
                             "ratio_vs_random": round(p1 / rnd, 2) if rnd else None})

    print("\n== leave-one-out retrieval precision@1 (discrete matchers) ==")
    print(f"{'dataset':10s} {'tokenizer':16s} {'matcher':7s} {'p@1':>7s} "
          f"{'random':>7s} {'xrand':>6s}")
    for r in rows:
        xr = f"{r['ratio_vs_random']:6.1f}" if r["ratio_vs_random"] is not None else f"{'-':>6s}"
        print(f"{r['dataset']:10s} {r['tokenizer']:16s} {r['matcher']:7s} "
              f"{100*r['precision_at_1']:7.2f} {100*r['random']:7.2f} {xr}")
    if float_rows:
        print("\n== continuous reference (float_dtw; bag/ngram undefined on floats) ==")
        for r in float_rows:
            xr = f"{r['ratio_vs_random']:6.1f}" if r["ratio_vs_random"] is not None else f"{'-':>6s}"
            print(f"{r['dataset']:10s} {'(raw float)':16s} {r['matcher']:9s} "
                  f"{100*r['precision_at_1']:7.2f} {100*r['random']:7.2f} {xr}")
    print("\nsymbolic matching should reach many x random; per_finger bag should be "
          "strongest. bag/ngram are discrete-only (undefined on continuous floats).")

    result = {"fit_dataset": fit_name, "eval_split": args.eval_split,
              "kadaptive": args.kadaptive, "kfinger": args.kfinger,
              "matchers": args.matchers, "seed": args.seed,
              "rows": rows, "float_rows": float_rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
