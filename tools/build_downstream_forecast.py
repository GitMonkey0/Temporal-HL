#!/usr/bin/env python3
"""Downstream motion-forecasting probe on token streams from different tokenizers.

Motivation. Reconstruction angular error is an *intrinsic* property of a codebook;
it says how faithfully a tokenizer can copy a pose, not how *predictable* the
resulting token stream is. A representation that compresses well can still be a
poor substrate for a sequence model: if its tokens scramble the smooth temporal
structure of motion, a forecaster learns less from them. This probe closes that
loop with a single-step forecasting task and trains the SAME predictor family on
the discrete token streams each tokenizer emits.

It compares three tokenizers at the per-frame, per-hand level:
  * hl26            - the fixed 26-direction HL codebook (20 tokens/frame)
  * adaptive_kmeans - a data-adaptive per-bone spherical k-means codebook
                      (20 tokens/frame, larger K -> lower reconstruction error)
  * per_finger      - one joint 12-D code per finger (5 tokens/frame, the most
                      compressed stream)

and three predictors, all trained to a MATCHED budget on each tokenizer's stream:
  * copy_last - predict next frame = current frame (the no-motion reference;
                pure numpy, always computed)
  * gru / lstm - recurrent next-token models over the per-frame token blocks
  * ar         - a small causal-self-attention (autoregressive) next-token model

Every predictor emits next-frame token ids; we decode them back to directions
through the SAME tokenizer's codebook and report the single-step forecast angular
error (deg) against the held-out CLEAN next-frame directions, plus the relative
improvement over copy_last. Decoding through each tokenizer's own codebook is what
makes the comparison fair: a tokenizer is charged for both its prediction quality
and its reconstruction coarseness.

Expected mechanism (recorded in the note under Findings, NOT measured here): the
data-adaptive codebook should forecast best; only the learned AR model should beat
copy_last by a clear margin; and the per-finger stream — though it compresses the
most — should forecast WORSE, because a coarse whole-finger code throws away the
fine bone-level structure a forecaster relies on. "compression-optimal is not
generation-optimal."

NOTE: sequence models already exist in this repo (`train_temporal_hl_sequence.py`,
`train_joint_sequence_student.py`). This is a clean, self-contained scoped entry
point for the tokenizer comparison and does not modify or call those.

    python tools/build_downstream_forecast.py \
        --datasets interhand:/opt/tiger/InterHand/annotations/machine_annot \
        --fit-dataset interhand --fit-split train --eval-split test \
        --kadaptive 64 --kfinger 128 \
        --predictors copy_last gru lstm ar --train-steps 4000 \
        --out experiments/2026-06-28_downstream_forecast.json

Dataset-parameterized like build_codebook_interleave_schedule: each --datasets
entry is name:annot_root and joint files resolve as <dataset>_<split>_joint_3d.json
with an InterHand2.6M fallback so the tool stays runnable where only InterHand data
is present locally. Pure numpy + stdlib for the tokenizers and copy_last; the
learned predictors use torch (imported lazily) and are simply skipped (-> pending)
when torch is unavailable. Reuses build_temporal_hl for the encoder,
build_adaptive_direction_codebook for the per-bone codebook,
build_perfinger_joint_codebook for the per-finger code, and the sequence loader /
path resolver in build_codebook_interleave_schedule, so there is zero convention
drift from the HL labels this repo produces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import build_temporal_hl as hl
import build_adaptive_direction_codebook as adb
import build_perfinger_joint_codebook as pf
import build_codebook_interleave_schedule as cis

N_FINGERS = pf.N_FINGERS                 # 5
BPF = pf.BONES_PER_FINGER                # 4
N_BONES = cis.N_BONES                    # 20


# --------------------------------------------------------------------------- #
# tokenizers: each exposes encode (clean dirs -> token ids) + decode (ids ->
# unit reconstruction). A "stream" is a list of per-sequence [T, n_tok] id arrays.
# --------------------------------------------------------------------------- #
class HL26Tokenizer:
    name = "hl26"
    n_tok = N_BONES

    def __init__(self):
        self.C = adb.hl26_codebook()         # [26,3]
        self.K = len(self.C)

    def encode(self, seqs):
        out = []
        for s in seqs:
            flat = s.reshape(-1, 3)
            ids = np.argmax(flat @ self.C.T, axis=1).reshape(len(s), N_BONES)
            out.append(ids.astype(np.int64))
        return out

    def decode(self, ids):
        rec = self.C[ids]                    # [T,20,3]
        return rec / (np.linalg.norm(rec, axis=2, keepdims=True) + 1e-12)


class AdaptiveTokenizer:
    name = "adaptive_kmeans"

    def __init__(self, fit_seqs, k, seed=0):
        self.K = k
        self.n_tok = N_BONES
        self.codebooks = cis.fit_perbone_codebooks(fit_seqs, k, seed=seed)  # 20 x [k,3]

    def encode(self, seqs):
        out = []
        for s in seqs:
            ids = np.zeros((len(s), N_BONES), dtype=np.int64)
            for b in range(N_BONES):
                ids[:, b] = np.argmax(s[:, b, :] @ self.codebooks[b].T, axis=1)
            out.append(ids)
        return out

    def decode(self, ids):
        T = len(ids)
        rec = np.zeros((T, N_BONES, 3))
        for b in range(N_BONES):
            rec[:, b, :] = self.codebooks[b][ids[:, b]]
        return rec / (np.linalg.norm(rec, axis=2, keepdims=True) + 1e-12)


class PerFingerTokenizer:
    name = "per_finger"

    def __init__(self, fit_seqs, k, seed=0):
        self.K = k
        self.n_tok = N_FINGERS
        stack = np.concatenate(fit_seqs, axis=0).reshape(-1, N_FINGERS, BPF, 3)
        self.codebooks = []                  # 5 x [k,12]
        for f in range(N_FINGERS):
            Xt = stack[:, f].reshape(len(stack), -1)
            kk = min(k, len(Xt))
            self.codebooks.append(pf.euclid_kmeans(Xt, kk, seed=seed))

    def encode(self, seqs):
        out = []
        for s in seqs:
            grid = s.reshape(len(s), N_FINGERS, BPF, 3)
            ids = np.zeros((len(s), N_FINGERS), dtype=np.int64)
            for f in range(N_FINGERS):
                X = grid[:, f].reshape(len(s), -1)
                ids[:, f] = np.argmin(pf._chunked_assign_dist(X, self.codebooks[f]), axis=1)
            out.append(ids)
        return out

    def decode(self, ids):
        T = len(ids)
        rec = np.zeros((T, N_FINGERS, BPF, 3))
        for f in range(N_FINGERS):
            rec[:, f] = self.codebooks[f][ids[:, f]].reshape(T, BPF, 3)
        rec = rec.reshape(T, N_BONES, 3)
        return rec / (np.linalg.norm(rec, axis=2, keepdims=True) + 1e-12)


# --------------------------------------------------------------------------- #
# forecasting metric
# --------------------------------------------------------------------------- #
def _mean_angular_deg(pred_dirs, clean_dirs) -> float:
    """Mean per-bone angle (deg) between two [.,20,3] arrays (already unit)."""
    cos = np.clip((pred_dirs * clean_dirs).sum(-1), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)).mean())


def copy_last_error(tok, eval_ids, eval_seqs) -> float:
    """Predict next frame = current frame, decode through the tokenizer, score vs
    the clean next-frame directions. Pure numpy, always computable."""
    num, den = 0.0, 0
    for ids, clean in zip(eval_ids, eval_seqs):
        if len(ids) < 2:
            continue
        pred = tok.decode(ids[:-1])          # frame t-1 used to predict frame t
        target = clean[1:]
        target = target / (np.linalg.norm(target, axis=2, keepdims=True) + 1e-12)
        cos = np.clip((pred * target).sum(-1), -1.0, 1.0)
        num += float(np.degrees(np.arccos(cos)).sum())
        den += cos.size
    return num / den if den else float("nan")


def train_and_forecast(tok, train_ids, eval_ids, eval_seqs, arch,
                       train_steps, win, seed):
    """Train one next-token model (gru / lstm / ar) on the tokenizer's stream to a
    matched budget, then single-step-forecast each eval frame from its past, decode,
    and score against the clean next-frame directions. Needs torch; returns None if
    torch is unavailable so the numpy paths stay runnable."""
    try:
        import torch
        import torch.nn as nn
    except Exception:
        return None
    n_tok, K = tok.n_tok, tok.K
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    demb, H = 16, 64

    class Pred(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.ModuleList([nn.Embedding(K, demb) for _ in range(n_tok)])
            self.arch = arch
            if arch == "gru":
                self.rnn = nn.GRU(n_tok * demb, H, batch_first=True)
            elif arch == "lstm":
                self.rnn = nn.LSTM(n_tok * demb, H, batch_first=True)
            elif arch == "ar":
                self.proj = nn.Linear(n_tok * demb, H)
                self.pos = nn.Embedding(4096, H)
                layer = nn.TransformerEncoderLayer(H, nhead=4, dim_feedforward=2 * H,
                                                   batch_first=True)
                self.enc = nn.TransformerEncoder(layer, num_layers=1)
            else:
                raise ValueError(arch)
            self.heads = nn.ModuleList([nn.Linear(H, K) for _ in range(n_tok)])

        def backbone(self, frames):          # frames [B,L,n_tok] long -> [B,L,H]
            B, L, _ = frames.shape
            fe = torch.cat([self.emb[i](frames[:, :, i]) for i in range(n_tok)], -1)
            if self.arch in ("gru", "lstm"):
                out, _ = self.rnn(fe)
                return out
            x = self.proj(fe) + self.pos(torch.arange(L))[None]
            mask = torch.triu(torch.ones(L, L) * float("-inf"), diagonal=1)
            return self.enc(x, mask=mask)

        def logits(self, frames):            # [B,L,n_tok,K] (predict NEXT at each pos)
            h = self.backbone(frames)
            return torch.stack([self.heads[i](h) for i in range(n_tok)], 2)

    model = Pred()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    windows = [c for c in train_ids if len(c) >= 2]
    if not windows:
        return None

    model.train()
    for _ in range(train_steps):
        batch = []
        for _ in range(min(32, len(windows))):
            c = windows[int(rng.integers(len(windows)))]
            L = min(win + 1, len(c))
            lo = int(rng.integers(0, len(c) - L + 1))
            batch.append(c[lo:lo + L])
        Lb = min(len(b) for b in batch)
        arr = np.stack([b[:Lb] for b in batch])           # [B,Lb,n_tok]
        ft = torch.from_numpy(arr).long()
        log = model.logits(ft[:, :-1])                    # predict frames 1..Lb-1
        tgt = ft[:, 1:]
        loss = sum(nn.functional.cross_entropy(
            log[:, :, i, :].reshape(-1, K), tgt[:, :, i].reshape(-1))
            for i in range(n_tok))
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    num, den = 0.0, 0
    with torch.no_grad():
        for ids, clean in zip(eval_ids, eval_seqs):
            if len(ids) < 2:
                continue
            ft = torch.from_numpy(ids[None]).long()
            log = model.logits(ft[:, :-1])[0]             # [T-1,n_tok,K]
            pred_ids = log.argmax(-1).numpy().astype(np.int64)  # predicts frames 1..T-1
            pred = tok.decode(pred_ids)
            target = clean[1:]
            target = target / (np.linalg.norm(target, axis=2, keepdims=True) + 1e-12)
            cos = np.clip((pred * target).sum(-1), -1.0, 1.0)
            num += float(np.degrees(np.arccos(cos)).sum())
            den += cos.size
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+",
                    default=["interhand:/opt/tiger/InterHand/annotations/machine_annot"],
                    help="list of name:annot_root regimes")
    ap.add_argument("--fit-dataset", default=None,
                    help="regime to FIT codebooks + predictors on (default: first)")
    ap.add_argument("--fit-split", default="train")
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--kadaptive", type=int, default=64)
    ap.add_argument("--kfinger", type=int, default=128)
    ap.add_argument("--predictors", nargs="+",
                    default=["copy_last", "gru", "lstm", "ar"])
    ap.add_argument("--train-steps", type=int, default=4000)
    ap.add_argument("--win", type=int, default=16)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    specs = [(s.partition(":")[0], Path(s.partition(":")[2])) for s in args.datasets]
    fit_name = args.fit_dataset or specs[0][0]
    fit_root = dict(specs)[fit_name]

    fp = cis._resolve_joint_path(fit_root, args.fit_split, fit_name)
    if fp is None:
        print(f"[fit:{fit_name}] no joint file under {fit_root}; nothing to fit. "
              "Pass real data to populate the forecast table.")
        return
    print(f"[fit:{fit_name}] <- {fp}", flush=True)
    fit_seqs = cis.load_sequences(fp, args.max_frames)
    if not fit_seqs:
        print(f"[fit:{fit_name}] no sequences decoded; abort")
        return
    print(f"[fit:{fit_name}] {len(fit_seqs)} clips; building tokenizers ...", flush=True)

    tokenizers = [
        HL26Tokenizer(),
        AdaptiveTokenizer(fit_seqs, args.kadaptive, seed=args.seed),
        PerFingerTokenizer(fit_seqs, args.kfinger, seed=args.seed),
    ]

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
        print(f"[{name}] eval <- {ep} ({len(eval_seqs)} clips)", flush=True)
        for tok in tokenizers:
            train_ids = tok.encode(fit_seqs)
            eval_ids = tok.encode(eval_seqs)
            copy_err = copy_last_error(tok, eval_ids, eval_seqs)
            row = {"dataset": name, "tokenizer": tok.name, "n_tok": tok.n_tok,
                   "K": tok.K, "copy_last_deg": round(copy_err, 4)}
            for arch in args.predictors:
                if arch == "copy_last":
                    continue
                err = train_and_forecast(tok, train_ids, eval_ids, eval_seqs, arch,
                                         args.train_steps, args.win, args.seed)
                row[f"{arch}_deg"] = (round(err, 4) if err is not None else None)
                if err is not None and copy_err == copy_err:
                    row[f"{arch}_vs_copy_pct"] = round(
                        100 * (copy_err - err) / copy_err, 2)
            rows.append(row)

    learned = [a for a in args.predictors if a != "copy_last"]
    print("\n== single-step forecast angular error (deg), lower is better ==")
    header = f"{'dataset':10s} {'tokenizer':16s} {'n_tok':>5s} {'copy':>7s} " + \
        " ".join(f"{a:>7s}" for a in learned)
    print(header)
    for r in rows:
        cells = " ".join(
            (f"{r.get(a + '_deg'):7.2f}" if r.get(a + "_deg") is not None
             else f"{'pend':>7s}") for a in learned)
        print(f"{r['dataset']:10s} {r['tokenizer']:16s} {r['n_tok']:5d} "
              f"{r['copy_last_deg']:7.2f} {cells}")
    print("\nadaptive_kmeans should forecast best; only the AR model should beat "
          "copy_last clearly; per_finger compresses most yet should forecast worse "
          "(compression-optimal != generation-optimal).")

    result = {"fit_dataset": fit_name, "fit_split": args.fit_split,
              "eval_split": args.eval_split, "kadaptive": args.kadaptive,
              "kfinger": args.kfinger, "predictors": args.predictors,
              "train_steps": args.train_steps, "seed": args.seed, "rows": rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
