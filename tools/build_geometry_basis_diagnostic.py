#!/usr/bin/env python3
"""Does *denser geometry* beat HL-26? A fixed-codebook control (negative result).

Motivation. Before fitting a data-adaptive codebook
(`build_adaptive_direction_codebook.py`), it is worth ruling out the simplest
explanation for HL-26's quantization error: maybe the 26 cube-surface directions
are just too few, and any *finer fixed grid* — or a better coordinate system —
would already fix it. This tool tests that hypothesis and finds it false.

It evaluates several **fixed, data-agnostic** direction codebooks on the exact
local bone directions the HL encoder produces (reusing
`build_adaptive_direction_codebook.load_local_directions`), at matched bits:

  * HL-26                       (the cube-surface baseline)
  * uniform (theta, phi) grid   (naive spherical, pole on +Z)
  * pole-corrected grid         (pole moved to +Y, equal-area in cos-theta)
  * Fibonacci near-uniform       (the best a uniform layout can do)

The point is the *diagnosis*: a naive uniform spherical grid is WORSE than HL-26,
because it puts its pole on the +Z axis — the sparsest direction for hand bones
(which concentrate on +Y) — so a large fraction of its cells fall where there is
no data. Moving the pole to +Y and using equal-area cells recovers most of the
gap and lands near Fibonacci sampling, but it is still a *uniform* codebook. The
lesson carried into the main tool: what advances the frontier is adapting to the
data distribution, not refining a fixed geometry.

    python tools/build_geometry_basis_diagnostic.py \
        --annot-root <InterHand annotations> --eval-split test \
        --out experiments/geometry_basis_<date>.json

Pure numpy + stdlib; no codebook is fit on data, so train/test is irrelevant —
every codebook here is constructed analytically and only *evaluated* on the
held-out directions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import build_temporal_hl as hl
import build_adaptive_direction_codebook as adb


def _normalize_rows(C: np.ndarray) -> np.ndarray:
    return C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)


def uniform_theta_phi_grid(n_theta: int, n_phi: int, pole_axis: str = "z") -> np.ndarray:
    """Uniform grid in (theta, phi): theta in (0,pi) elevation from `pole_axis`,
    phi azimuth around it. This is the *naive* spherical codebook."""
    thetas = (np.arange(n_theta) + 0.5) * np.pi / n_theta
    phis = (np.arange(n_phi)) * 2 * np.pi / n_phi
    pts = []
    for t in thetas:
        for p in phis:
            # build around +Z, then rotate the pole onto the requested axis
            v = np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
            if pole_axis == "y":
                v = np.array([v[0], v[2], v[1]])   # swap Z<->Y: pole -> +Y
            pts.append(v)
    return _normalize_rows(np.asarray(pts))


def equal_area_grid(n_theta: int, n_phi: int, pole_axis: str = "y") -> np.ndarray:
    """Equal-area cells: sample cos(theta) uniformly (not theta), pole on +Y."""
    cos_t = (np.arange(n_theta) + 0.5) * 2.0 / n_theta - 1.0   # uniform in [-1,1]
    phis = (np.arange(n_phi)) * 2 * np.pi / n_phi
    pts = []
    for c in cos_t:
        s = float(np.sqrt(max(0.0, 1.0 - c * c)))
        for p in phis:
            v = np.array([s * np.cos(p), s * np.sin(p), c])
            if pole_axis == "y":
                v = np.array([v[0], v[2], v[1]])
            pts.append(v)
    return _normalize_rows(np.asarray(pts))


def fibonacci_sphere(n: int) -> np.ndarray:
    """n near-uniform points on the sphere (the best a uniform layout can do)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    gold = np.pi * (1.0 + 5.0 ** 0.5)
    theta = gold * i
    return _normalize_rows(np.stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=1))


def dead_code_fraction(V: np.ndarray, C: np.ndarray) -> float:
    """Fraction of codes that are nearest to (almost) no data."""
    assign = np.argmax(V @ C.T, axis=1)
    used = np.bincount(assign, minlength=C.shape[0])
    return float((used == 0).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annot-root", type=Path,
                    default=Path("/opt/tiger/InterHand/annotations/machine_annot"))
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    print(f"[eval] loading {args.eval_split} ...", flush=True)
    V, _ = adb.load_local_directions(args.annot_root, args.eval_split, args.max_frames)
    print(f"[eval] {len(V)} held-out local bone directions\n")

    codebooks = {
        "hl26": adb.hl26_codebook(),
        "uniform_thetaphi_5x8_poleZ": uniform_theta_phi_grid(5, 8, pole_axis="z"),
        "uniform_thetaphi_5x8_poleY": uniform_theta_phi_grid(5, 8, pole_axis="y"),
        "equal_area_8x14_poleY": equal_area_grid(8, 14, pole_axis="y"),
        "fibonacci_128": fibonacci_sphere(128),
    }

    rows = []
    for name, C in codebooks.items():
        rows.append({
            "name": name, "K": int(C.shape[0]),
            "bits": adb.empirical_entropy_bits(V, C),
            "angular_deg": adb.angular_error_deg(V, C),
            "dead_code_frac": dead_code_fraction(V, C),
        })
    rows.sort(key=lambda r: r["bits"])

    print(f"{'codebook':30s} {'K':>4s} {'bits':>6s} {'angular°':>9s} {'dead%':>6s}")
    for r in rows:
        print(f"{r['name']:30s} {r['K']:4d} {r['bits']:6.2f} "
              f"{r['angular_deg']:9.2f} {100*r['dead_code_frac']:6.1f}")

    hl26 = next(r for r in rows if r["name"] == "hl26")
    naive = next(r for r in rows if r["name"] == "uniform_thetaphi_5x8_poleZ")
    print(f"\nnaive uniform spherical {naive['angular_deg']:.2f}° vs HL-26 "
          f"{hl26['angular_deg']:.2f}°  -> "
          f"{'WORSE' if naive['angular_deg'] > hl26['angular_deg'] else 'better'} "
          f"(pole on the sparse +Z axis wastes codes)")

    result = {"eval_split": args.eval_split, "n_eval": int(len(V)), "rows": rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
