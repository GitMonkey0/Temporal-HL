#!/usr/bin/env python3
"""Verify the canonical per-frame HL-26 direction encoding.

`build_temporal_hl.frame_to_hl` is the canonical frame-wise encoder: it takes
hand joints, builds a canonical hand frame (origin at the wrist, +Y toward the
middle-finger base, palm plane from index/pinky bases), expresses each of the 20
regional vectors in that frame, and quantizes the direction to one of the 26
cuboid-surface symbols.

This script is an INDEPENDENT, self-contained re-derivation of that same
encoding. It reuses only the joint topology (`EDGE_ORDER`, `parent_index`) and
re-implements the codebook, the canonical basis, and the quantization from
scratch, then asserts the two agree token-for-token. It is a regression guard:
if the basis math, the codebook construction, or the quantization rule in
`build_temporal_hl.py` ever drifts, this test fails.

Run from the repository root:

    python -m tools.verify_hl_encoding                  # seeded synthetic frames
    python -m tools.verify_hl_encoding --joints PATH    # real InterHand joint_3d.json
    python -m tools.verify_hl_encoding --frames 5000    # more synthetic frames

Exit code is non-zero if any frame disagrees.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

from tools.build_temporal_hl import (
    EDGE_ORDER,
    frame_to_hl,
    get_hand_spec,
    parent_index,
)

# --------------------------------------------------------------------------
# Independent re-implementation of the HL-26 direction encoding.
# --------------------------------------------------------------------------


def build_codebook() -> list[tuple[float, float, float]]:
    """26 unit directions on the surface of a cuboid: {-1,0,1}^3 minus origin.

    Built with the same nested -1/0/1 iteration order as the canonical encoder,
    so direction id k denotes the same (x, y, z) lattice direction in both.
    """
    codebook = []
    for x in (-1, 0, 1):
        for y in (-1, 0, 1):
            for z in (-1, 0, 1):
                if x == 0 and y == 0 and z == 0:
                    continue
                n = math.sqrt(float(x * x + y * y + z * z))
                codebook.append((x / n, y / n, z / n))
    return codebook


CODEBOOK = build_codebook()


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _normalize(a):
    n = math.sqrt(max(_dot(a, a), 1e-12))
    return (a[0] / n, a[1] / n, a[2] / n)


def canonical_basis(coords, hand_name):
    """Origin + orthonormal (x, y, z) hand frame, or None if degenerate."""
    spec = get_hand_spec(hand_name)
    origin = coords[spec["wrist"]]
    y = _normalize(_sub(coords[spec["middle"][-1]], origin))
    span = _sub(coords[spec["pinky"][-1]], coords[spec["index"][-1]])
    proj = _dot(span, y)
    z = (span[0] - proj * y[0], span[1] - proj * y[1], span[2] - proj * y[2])
    if math.sqrt(_dot(z, z)) < 1e-6:
        return None
    z = _normalize(z)
    x = _cross(y, z)
    if math.sqrt(_dot(x, x)) < 1e-6:
        return None
    x = _normalize(x)
    z = _normalize(_cross(x, y))
    return origin, x, y, z


def quantize(unit):
    return max(range(len(CODEBOOK)), key=lambda k: _dot(unit, CODEBOOK[k]))


def encode_frame(coords, valid_ids, hand_name):
    """Independent HL-26 token ids in EDGE_ORDER, or None (mirrors frame_to_hl)."""
    basis = canonical_basis(coords, hand_name)
    if basis is None:
        return None
    origin, x, y, z = basis
    spec = get_hand_spec(hand_name)
    tokens = []
    for finger_name, idx_in_finger in EDGE_ORDER:
        child = spec[finger_name][idx_in_finger]
        parent = parent_index(hand_name, finger_name, idx_in_finger)
        if child not in valid_ids or parent not in valid_ids:
            return None
        vec = _sub(coords[child], coords[parent])
        local = (_dot(vec, x), _dot(vec, y), _dot(vec, z))
        tokens.append(quantize(_normalize(local)))
    return tokens


# --------------------------------------------------------------------------
# Frame sources.
# --------------------------------------------------------------------------


def synthetic_frames(num_frames, seed=0):
    """Seeded random 42-joint frames. Random gaussian coordinates are enough to
    exercise the basis/quantization equivalence (this tests code agreement, not
    anatomical realism)."""
    rng = random.Random(seed)
    for _ in range(num_frames):
        yield [(rng.gauss(0, 80), rng.gauss(0, 80), rng.gauss(0, 80)) for _ in range(42)]


def interhand_frames(path, max_frames):
    """Real InterHand world_coord frames from a *_joint_3d.json file."""
    with open(path) as f:
        data = json.load(f)
    count = 0
    for capture in sorted(data, key=lambda c: int(c)):
        frames = data[capture]
        for frame_idx in sorted(frames, key=lambda x: int(x)):
            coords = frames[frame_idx]["world_coord"]
            yield [tuple(p) for p in coords]
            count += 1
            if max_frames and count >= max_frames:
                return


# --------------------------------------------------------------------------
# Comparison.
# --------------------------------------------------------------------------


def run(frames, hands=("right", "left")):
    valid = set(range(42))
    stats = {h: [0, 0] for h in hands}      # [matches, total]
    mismatch_examples = []
    for coords in frames:
        for hand_name in hands:
            ref = frame_to_hl(coords, valid, hand_name)
            mine = encode_frame(coords, valid, hand_name)
            if ref is None or mine is None:
                continue
            ref_tokens = ref["token_ids"]
            for r in range(len(ref_tokens)):
                stats[hand_name][1] += 1
                if ref_tokens[r] == mine[r]:
                    stats[hand_name][0] += 1
                elif len(mismatch_examples) < 5:
                    mismatch_examples.append((hand_name, r, ref_tokens[r], mine[r]))
    return stats, mismatch_examples


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--joints", default=None,
                    help="InterHand *_joint_3d.json; if omitted, use synthetic frames")
    ap.add_argument("--frames", type=int, default=2000,
                    help="number of frames (synthetic) or cap (real data)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.joints:
        src = interhand_frames(args.joints, args.frames)
        label = f"InterHand {Path(args.joints).name} (<= {args.frames} frames)"
    else:
        src = synthetic_frames(args.frames, args.seed)
        label = f"{args.frames} seeded synthetic frames"

    stats, mismatches = run(src)
    print(f"HL-26 encoding consistency check  [{label}]")
    total_m = total_n = 0
    for hand_name, (m, n) in stats.items():
        total_m += m
        total_n += n
        pct = 100.0 * m / n if n else float("nan")
        print(f"  {hand_name:5s}: {m}/{n} tokens identical to frame_to_hl  ({pct:.2f}%)")
    overall = 100.0 * total_m / total_n if total_n else float("nan")
    print(f"  total: {total_m}/{total_n}  ({overall:.2f}%)")
    if mismatches:
        print("  first mismatches (hand, vector_idx, frame_to_hl, independent):")
        for ex in mismatches:
            print(f"    {ex}")
    ok = total_n > 0 and total_m == total_n
    print("RESULT:", "PASS — canonical encoding reproduced exactly" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
