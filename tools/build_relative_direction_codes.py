#!/usr/bin/env python3
"""Relative (parent / temporal) direction codes vs absolute, plus a gzip check.

Motivation. The adaptive codebook quantizes each bone direction *absolutely*, in
the canonical hand frame. But a bone direction is highly predictable from two
references already available at decode time:

  (i)  its PARENT bone (the proximal bone it hangs off) — fingers bend smoothly,
       so a bone rarely deviates far from its parent;
  (ii) the SAME bone in the PREVIOUS frame — hand motion is slow relative to
       frame rate, so frame-to-frame change is small.

If we encode the *residual* against such a reference and quantize the residual,
the residual distribution is far more concentrated than the absolute direction,
so the same angular accuracy costs far fewer bits. This is the standard
predictive-coding move, and it mirrors symbolic-music pianoroll modeling, where
note events are coded relative to a running context.

This tool reuses the canonical HL encoder and the spherical-k-means/metrics in
`build_adaptive_direction_codebook.py`, and compares three schemes at matched
accuracy:

  * absolute        (quantize the direction in the hand frame; the baseline)
  * parent_relative (quantize the direction expressed in a frame aligned to the
                     parent bone)
  * time_relative   (quantize the direction expressed in a frame aligned to the
                     same bone in the previous frame)

For each scheme it reports bits/direction (code entropy) and angular error, and
additionally gzip-compresses the emitted token-id stream. The point of the gzip
check: a well-decorrelated (temporal-relative) stream is already close to
incompressible, so general-purpose compression buys little on top of it —
"temporal-relative decorrelation == having already done the compression".

    python tools/build_relative_direction_codes.py \
        --annot-root <InterHand annotations> --fit-split train --eval-split test \
        --k 64 \
        --out experiments/relative_direction_codes_<date>.json

Pure numpy + stdlib (gzip from stdlib); reuses build_temporal_hl +
build_adaptive_direction_codebook for zero convention drift.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

import build_temporal_hl as hl
import build_adaptive_direction_codebook as adb

N_FINGERS = len(hl.FINGER_NAMES)
BONES_PER_FINGER = len(hl.EDGE_ORDER) // N_FINGERS  # 4


def _basis_from_axis(axis):
    """An orthonormal [3,3] basis whose first row is `axis` (unit). Any direction
    can be re-expressed in this frame; aligning the frame to a reference bone makes
    'pointing the same way as the reference' the canonical direction."""
    a = axis / (np.linalg.norm(axis) + 1e-12)
    helper = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(a, helper)
    u = u / (np.linalg.norm(u) + 1e-12)
    w = np.cross(a, u)
    return np.stack([a, u, w], axis=0)


def load_sequences(annot_root: Path, split: str, max_frames):
    """Yield per (capture, hand) ordered lists of [20,3] unit local-direction
    arrays, so temporal residuals (same bone, previous frame) are well-defined.

    Returns a list of [T,20,3] arrays (one per contiguous hand track).
    """
    fname = f"InterHand2.6M_{split}_joint_3d.json"
    for cand in (annot_root / fname, annot_root / split / fname):
        if cand.exists():
            joint_path = cand
            break
    else:
        raise FileNotFoundError(f"{fname} not under {annot_root} (flat or /{split}/)")
    joints = hl.load_json(joint_path)
    tracks = []
    seen = 0
    for _capture, frames in joints.items():
        if not isinstance(frames, dict):
            continue
        ordered = sorted(((k, v) for k, v in frames.items()
                          if isinstance(v, dict) and "world_coord" in v),
                         key=lambda kv: _safe_int(kv[0]))
        per_hand = {"right": [], "left": []}
        for _idx, frame_item in ordered:
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
                per_hand[hand_name].append(lv)
            seen += 1
            if max_frames is not None and seen >= max_frames:
                for h in ("right", "left"):
                    if len(per_hand[h]) >= 2:
                        tracks.append(np.asarray(per_hand[h]))
                return tracks
        for h in ("right", "left"):
            if len(per_hand[h]) >= 2:
                tracks.append(np.asarray(per_hand[h]))
    return tracks


def _safe_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def build_residuals(tracks, scheme: str):
    """Re-express each bone direction relative to a reference, returning a flat
    [N,3] unit array of residual directions ready for spherical k-means.

    scheme:
      * absolute        -> identity (the direction itself)
      * parent_relative -> direction in the frame aligned to its parent bone
      * time_relative   -> direction in the frame aligned to the same bone in t-1
    """
    out = []
    for trk in tracks:
        T = len(trk)
        grid = trk.reshape(T, N_FINGERS, BONES_PER_FINGER, 3)
        for t in range(T):
            for f in range(N_FINGERS):
                for b in range(BONES_PER_FINGER):
                    v = grid[t, f, b]
                    if scheme == "absolute":
                        out.append(v)
                    elif scheme == "parent_relative":
                        # parent = next bone toward the wrist (b+1); wrist for the
                        # most proximal bone -> fall back to absolute.
                        if b + 1 < BONES_PER_FINGER:
                            ref = grid[t, f, b + 1]
                        else:
                            ref = None
                        out.append(v if ref is None else _basis_from_axis(ref) @ v)
                    elif scheme == "time_relative":
                        if t == 0:
                            out.append(v)
                        else:
                            ref = grid[t - 1, f, b]
                            out.append(_basis_from_axis(ref) @ v)
                    else:
                        raise ValueError(scheme)
    V = np.asarray(out, dtype=np.float64)
    return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)


def gzip_bits_per_symbol(assign: np.ndarray) -> float:
    """gzip-compress the token-id stream and report bits/symbol. A decorrelated
    stream gzips poorly (-> close to the raw entropy), an autocorrelated one well."""
    if len(assign) == 0:
        return 0.0
    raw = assign.astype(np.uint16).tobytes()
    comp = gzip.compress(raw, compresslevel=9)
    return 8.0 * len(comp) / len(assign)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annot-root", type=Path,
                    default=Path("/opt/tiger/InterHand/annotations/machine_annot"))
    ap.add_argument("--fit-split", default="train")
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    print(f"[fit ] loading {args.fit_split} tracks ...", flush=True)
    tr_tracks = load_sequences(args.annot_root, args.fit_split, args.max_frames)
    print(f"[eval] loading {args.eval_split} tracks ...", flush=True)
    ev_tracks = load_sequences(args.annot_root, args.eval_split, args.max_frames)
    print(f"[data] tracks fit={len(tr_tracks)} eval={len(ev_tracks)}\n")

    rows = []
    for scheme in ("absolute", "parent_relative", "time_relative"):
        Vt = build_residuals(tr_tracks, scheme)
        Ve = build_residuals(ev_tracks, scheme)
        C = adb.spherical_kmeans(Vt, args.k, init="kmeans++", seed=args.seed)
        assign = np.argmax(Ve @ C.T, axis=1)
        rows.append({
            "scheme": scheme, "K": int(args.k),
            "bits_per_dir": adb.empirical_entropy_bits(Ve, C),
            "angular_deg": adb.angular_error_deg(Ve, C),
            "gzip_bits_per_dir": gzip_bits_per_symbol(assign),
        })

    print(f"{'scheme':18s} {'K':>4s} {'bits/dir':>9s} {'angular°':>9s} "
          f"{'gzip b/dir':>11s}")
    for r in rows:
        print(f"{r['scheme']:18s} {r['K']:4d} {r['bits_per_dir']:9.2f} "
              f"{r['angular_deg']:9.2f} {r['gzip_bits_per_dir']:11.2f}")
    print("\ngzip bits/dir ~ entropy bits/dir means the stream is already "
          "decorrelated: temporal-relative coding has done the compression.")

    result = {"fit_split": args.fit_split, "eval_split": args.eval_split,
              "K": int(args.k), "seed": args.seed, "rows": rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
