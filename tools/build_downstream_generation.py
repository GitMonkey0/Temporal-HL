#!/usr/bin/env python3
"""Downstream generation probe: distribution-level motion generation on token
streams from different tokenizers, AR free-running vs masked (MoMask-style) decode.

Motivation. Forecasting scores a single next step; unconditional *generation* asks
the harder question — does a model trained on a tokenizer's stream produce whole
motion clips whose DISTRIBUTION matches real hand motion? A coarse codebook can
look fine on reconstruction yet generate jittery or sluggish motion, because the
information a generator needs (the fine velocity structure) was discarded at
quantization and cannot be recovered downstream. This probe trains a small
generator on each tokenizer's stream and compares the generated distribution to
held-out real motion.

It compares three tokenizers (hl26 / adaptive_kmeans / per_finger, reusing the
tokenizer classes in build_downstream_forecast) under two decoding regimes:
  * ar     - autoregressive free-running: sample a clip frame by frame, each frame
             conditioned only on generated past frames (exposure-bias-exposed)
  * masked - MoMask-style non-autoregressive: a bidirectional model fills a
             fixed-length clip by iterative confidence-based unmasking

and reports two distribution-level metrics against the held-out real clips:
  * FGD       - Frechet Gesture Distance (reuses the deterministic kinematic
                gesture-feature Gaussian + Frechet formula in
                build_codebook_interleave_schedule)
  * JS(speed) - Jensen-Shannon divergence between the per-frame speed histograms
                (mean |velocity| over the 60 direction channels) of generated vs
                real motion -- a direct, interpretable check on motion dynamics

Expected mechanism (recorded in the note under Findings, NOT measured here): the
data-adaptive / per-finger streams should generate motion whose speed distribution
is far closer to real (JS(speed) per_finger ~0.077 vs HL-26 ~0.338, ~4x), because
HL-26's 26 coarse directions cannot express small inter-frame velocity; masked
decoding should be <= AR for the adaptive tokenizers, but for HL-26 even masked
decoding cannot recover what was lost at quantization (its FGD stays high, ~2077).

NOTE: sequence generators already exist in this repo
(`train_temporal_hl_sequence.py`, `train_joint_sequence_student.py`). This is a
clean, self-contained scoped entry point for the tokenizer comparison and does not
modify or call those.

    python tools/build_downstream_generation.py \
        --datasets interhand:/opt/tiger/InterHand/annotations/machine_annot \
        --fit-dataset interhand --fit-split train --eval-split test \
        --kadaptive 64 --kfinger 128 --modes ar masked \
        --run-gen --train-steps 8000 --n-gen 64 \
        --out experiments/2026-06-28_downstream_generation.json

Dataset-parameterized like build_codebook_interleave_schedule: each --datasets
entry is name:annot_root and joint files resolve as <dataset>_<split>_joint_3d.json
with an InterHand2.6M fallback so the tool stays runnable where only InterHand data
is present locally. Tokenization + JS-histogram machinery are pure numpy; the
generators use torch (imported lazily) and are gated behind --run-gen, so the
measured FGD / JS cells stay PENDING (no numbers hardcoded) until a real run.
Reuses build_temporal_hl, build_adaptive_direction_codebook,
build_perfinger_joint_codebook, build_codebook_interleave_schedule (FGD + loader)
and build_downstream_forecast (tokenizers) for zero convention drift.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import build_codebook_interleave_schedule as cis
import build_downstream_forecast as fc

N_FINGERS = fc.N_FINGERS
N_BONES = fc.N_BONES


# --------------------------------------------------------------------------- #
# JS(speed): interpretable dynamics check (pure numpy)
# --------------------------------------------------------------------------- #
def clip_speeds(dir_clip: np.ndarray) -> np.ndarray:
    """Per-frame speed of one clip [T,20,3]: mean |velocity| over the 60 flattened
    direction channels."""
    x = dir_clip.reshape(len(dir_clip), -1)
    if len(x) < 2:
        return np.zeros((0,))
    return np.abs(np.diff(x, axis=0)).mean(1)


def _js(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base 2) between two discrete distributions."""
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / (b[mask] + 1e-12))))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def speed_js(real_clips, gen_clips, n_bins=40) -> float:
    rs = np.concatenate([clip_speeds(c) for c in real_clips]) if real_clips else np.zeros(0)
    gs = np.concatenate([clip_speeds(c) for c in gen_clips]) if gen_clips else np.zeros(0)
    if len(rs) == 0 or len(gs) == 0:
        return float("nan")
    hi = float(np.percentile(np.concatenate([rs, gs]), 99)) or 1.0
    edges = np.linspace(0.0, hi + 1e-9, n_bins + 1)
    ph, _ = np.histogram(rs, bins=edges)
    qh, _ = np.histogram(gs, bins=edges)
    return round(_js(ph.astype(float), qh.astype(float)), 4)


# --------------------------------------------------------------------------- #
# generators (torch, gated): AR free-running + masked MoMask-style decode
# --------------------------------------------------------------------------- #
def run_generation(tok, train_ids, eval_ids, real_clips, mode, train_steps,
                   win, n_gen, mask_rounds, seed):
    """Train a small generator on the tokenizer's stream and produce n_gen clips.
    mode='ar': autoregressive free-running (frame by frame on generated past).
    mode='masked': bidirectional model + iterative confidence unmasking (MoMask).
    Returns (fgd, js_speed) vs real_clips, or None if torch is unavailable."""
    try:
        import torch
        import torch.nn as nn
    except Exception:
        return None
    n_tok, K = tok.n_tok, tok.K
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    demb, H = 16, 96
    MASK = K  # extra vocab id for the masked slot (masked mode only)
    lengths = [len(c) for c in eval_ids if len(c) >= 2] or [32]

    class Net(nn.Module):
        def __init__(self, vocab):
            super().__init__()
            self.emb = nn.ModuleList([nn.Embedding(vocab, demb) for _ in range(n_tok)])
            self.proj = nn.Linear(n_tok * demb, H)
            self.pos = nn.Embedding(8192, H)
            layer = nn.TransformerEncoderLayer(H, nhead=4, dim_feedforward=2 * H,
                                               batch_first=True)
            self.enc = nn.TransformerEncoder(layer, num_layers=2)
            self.heads = nn.ModuleList([nn.Linear(H, K) for _ in range(n_tok)])

        def forward(self, frames, causal):   # frames [B,L,n_tok] -> [B,L,n_tok,K]
            B, L, _ = frames.shape
            fe = torch.cat([self.emb[i](frames[:, :, i]) for i in range(n_tok)], -1)
            x = self.proj(fe) + self.pos(torch.arange(L))[None]
            mask = (torch.triu(torch.ones(L, L) * float("-inf"), diagonal=1)
                    if causal else None)
            h = self.enc(x, mask=mask)
            return torch.stack([self.heads[i](h) for i in range(n_tok)], 2)

    def sample_window():
        c = train_ids[int(rng.integers(len(train_ids)))]
        L = min(win, len(c))
        lo = int(rng.integers(0, len(c) - L + 1))
        return c[lo:lo + L]

    train_seqs = [c for c in train_ids if len(c) >= 2]
    if not train_seqs:
        return None

    if mode == "ar":
        model = Net(K)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(train_steps):
            batch = [sample_window() for _ in range(min(32, len(train_seqs)))]
            Lb = min(len(b) for b in batch)
            arr = np.stack([b[:Lb] for b in batch])
            ft = torch.from_numpy(arr).long()
            log = model(ft[:, :-1], causal=True)
            tgt = ft[:, 1:]
            loss = sum(nn.functional.cross_entropy(
                log[:, :, i, :].reshape(-1, K), tgt[:, :, i].reshape(-1))
                for i in range(n_tok))
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        gen = []
        with torch.no_grad():
            for _ in range(n_gen):
                T = int(rng.choice(lengths))
                hist = train_seqs[int(rng.integers(len(train_seqs)))][:1].copy()
                while len(hist) < T:
                    ft = torch.from_numpy(hist[None]).long()
                    log = model(ft, causal=True)[0, -1]   # [n_tok,K] next frame
                    nxt = np.array([int(rng.choice(
                        K, p=torch.softmax(log[i], -1).numpy())) for i in range(n_tok)])
                    hist = np.concatenate([hist, nxt[None]], axis=0)
                gen.append(tok.decode(hist[:T]))
    elif mode == "masked":
        model = Net(K + 1)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(train_steps):
            batch = [sample_window() for _ in range(min(32, len(train_seqs)))]
            Lb = min(len(b) for b in batch)
            arr = np.stack([b[:Lb] for b in batch])
            ft = torch.from_numpy(arr).long()
            r = float(rng.uniform(0.2, 0.9))
            mask = torch.from_numpy(rng.random(ft.shape) < r)
            inp = ft.clone()
            inp[mask] = MASK
            log = model(inp, causal=False)
            loss = 0.0
            for i in range(n_tok):
                mi = mask[:, :, i].reshape(-1)
                if mi.any():
                    loss = loss + nn.functional.cross_entropy(
                        log[:, :, i, :].reshape(-1, K)[mi], ft[:, :, i].reshape(-1)[mi])
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        gen = []
        with torch.no_grad():
            for _ in range(n_gen):
                T = int(rng.choice(lengths))
                cur = np.full((T, n_tok), MASK, dtype=np.int64)
                conf = np.zeros((T, n_tok))
                masked = np.ones((T, n_tok), dtype=bool)
                for rnd in range(mask_rounds):
                    log = model(torch.from_numpy(cur[None]).long(), causal=False)[0]
                    keep_frac = (rnd + 1) / mask_rounds
                    for i in range(n_tok):
                        p = torch.softmax(log[:, i, :], -1).numpy()
                        ids = p.argmax(1)
                        cur[masked[:, i], i] = ids[masked[:, i]]
                        conf[:, i] = p.max(1)
                    flat_conf = conf.reshape(-1).copy()
                    n_keep = int(round(keep_frac * flat_conf.size))
                    order = np.argsort(-flat_conf)
                    keepset = set(order[:n_keep].tolist())
                    new_masked = np.ones(flat_conf.size, dtype=bool)
                    for idx in keepset:
                        new_masked[idx] = False
                    masked = new_masked.reshape(T, n_tok)
                    cur[masked] = MASK
                cur[cur == MASK] = 0
                gen.append(tok.decode(cur))
    else:
        raise ValueError(mode)

    fgd = float(cis.frechet_gesture_distance(real_clips, gen))
    js = speed_js(real_clips, gen)
    return round(fgd, 2), js


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+",
                    default=["interhand:/opt/tiger/InterHand/annotations/machine_annot"],
                    help="list of name:annot_root regimes")
    ap.add_argument("--fit-dataset", default=None)
    ap.add_argument("--fit-split", default="train")
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--kadaptive", type=int, default=64)
    ap.add_argument("--kfinger", type=int, default=128)
    ap.add_argument("--modes", nargs="+", default=["ar", "masked"])
    ap.add_argument("--run-gen", action="store_true",
                    help="train each generator and compute FGD + JS (needs torch); "
                         "off by default -> measured cells pending")
    ap.add_argument("--train-steps", type=int, default=8000)
    ap.add_argument("--win", type=int, default=24)
    ap.add_argument("--n-gen", type=int, default=64)
    ap.add_argument("--mask-rounds", type=int, default=8)
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
    print(f"[fit:{fit_name}] <- {fp}", flush=True)
    fit_seqs = cis.load_sequences(fp, args.max_frames)
    if not fit_seqs:
        print(f"[fit:{fit_name}] no sequences decoded; abort")
        return
    print(f"[fit:{fit_name}] {len(fit_seqs)} clips; building tokenizers ...", flush=True)
    tokenizers = [
        fc.HL26Tokenizer(),
        fc.AdaptiveTokenizer(fit_seqs, args.kadaptive, seed=args.seed),
        fc.PerFingerTokenizer(fit_seqs, args.kfinger, seed=args.seed),
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
            real_clips = [tok.decode(ids) for ids in eval_ids if len(ids) >= 2]
            for mode in args.modes:
                fgd = js = None
                if args.run_gen:
                    res = run_generation(tok, train_ids, eval_ids, real_clips, mode,
                                         args.train_steps, args.win, args.n_gen,
                                         args.mask_rounds, args.seed)
                    if res is not None:
                        fgd, js = res
                rows.append({"dataset": name, "tokenizer": tok.name, "mode": mode,
                             "K": tok.K, "fgd": fgd, "js_speed": js})

    print("\n== distribution-level generation (FGD lower better; JS(speed) lower better) ==")
    print(f"{'dataset':10s} {'tokenizer':16s} {'mode':7s} {'FGD':>10s} {'JS(speed)':>10s}")
    for r in rows:
        fcell = f"{r['fgd']:10.2f}" if r["fgd"] is not None else f"{'pending':>10s}"
        jcell = f"{r['js_speed']:10.4f}" if r["js_speed"] is not None else f"{'pending':>10s}"
        print(f"{r['dataset']:10s} {r['tokenizer']:16s} {r['mode']:7s} {fcell} {jcell}")
    if not args.run_gen:
        print("\nmeasured cells pending: pass --run-gen (+ data + --train-steps) to "
              "train each generator and compute FGD + JS(speed). adaptive / per_finger "
              "should match real speed far better than coarse HL-26.")

    result = {"fit_dataset": fit_name, "fit_split": args.fit_split,
              "eval_split": args.eval_split, "kadaptive": args.kadaptive,
              "kfinger": args.kfinger, "modes": args.modes,
              "ran_gen": bool(args.run_gen), "train_steps": args.train_steps,
              "seed": args.seed, "rows": rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
