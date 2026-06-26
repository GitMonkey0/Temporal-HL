#!/usr/bin/env python3
"""Cross-dataset transfer of the data-adaptive direction codebook: does the
adaptive layout capture *general* hand structure, or just one dataset's quirks?

Motivation. `build_adaptive_direction_codebook.py` shows that a spherical
k-means codebook beats the fixed HL-26 cube-surface codebook at a matched 26-code
budget, when fit and evaluated on the SAME dataset. A fair reviewer asks the next
question: is that gain real structure, or has the codebook simply memorised the
direction distribution of one capture setup? The honest test is *transfer* — fit
the codebook on dataset A, then APPLY it unchanged (no re-fit) to dataset B, and
vice-versa. If a codebook fit on A still beats HL-26 on B, the adaptive layout is
capturing the shared anatomy of how finger bones point, not a dataset artefact.

This tool reuses the *exact* HL encoder in build_temporal_hl (build_local_basis
-> vector_to_local -> the 26-direction quantizer) to collect canonical local bone
directions per dataset, then reuses build_adaptive_direction_codebook's spherical
k-means / angular-error / entropy helpers to build a small TRANSFER MATRIX of mean
reconstruction angular error:

  rows    = codebook source   (HL-26 fixed, adaptive@A, adaptive@B, ...)
  columns = evaluation dataset (A, B, ...)

  * the diagonal entries are each dataset's OWN-data fit (fit on its train split,
    scored on its held-out split) -> the best case for an adaptive codebook;
  * the off-diagonal entries are the TRANSFERRED codebooks (fit elsewhere) -> the
    generalisation test;
  * the HL-26 row is the fixed-codebook baseline on every dataset.

It does NOT modify HL or its encoder; it measures whether the adaptive layout
generalises across capture regimes, as evidence for a temporal-HL
representation-design section.

    python tools/build_cross_dataset_transfer.py \
        --datasets interhand:/opt/tiger/InterHand/annotations/machine_annot \
                   freihand:/opt/tiger/FreiHAND/annotations \
        --fit-split train --eval-split test --k 26 \
        --out experiments/cross_dataset_transfer_<date>.json

Dataset-parameterized like build_event_keyframe_compression: each --datasets entry
is name:annot_root, and joint files resolve as <dataset>_<split>_joint_3d.json with
an InterHand2.6M fallback so the tool stays runnable where only InterHand data is
present locally. Pure numpy + stdlib; reuses build_temporal_hl for the encoding and
build_adaptive_direction_codebook for the codebook machinery, so there is zero
convention drift from the HL labels this repo already produces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import build_temporal_hl as hl
import build_adaptive_direction_codebook as adb


def _resolve_joint_path(root: Path, split: str, dataset: str) -> Path | None:
    """Find a per-split joint file. InterHand uses InterHand2.6M_<split>_joint_3d
    .json; for other datasets (e.g. FreiHAND) try a <dataset>_<split>_joint_3d
    .json sibling, falling back to the InterHand name so the tool stays runnable."""
    cands = [
        root / f"{dataset}_{split}_joint_3d.json",
        root / split / f"{dataset}_{split}_joint_3d.json",
        root / f"InterHand2.6M_{split}_joint_3d.json",
        root / split / f"InterHand2.6M_{split}_joint_3d.json",
    ]
    for c in cands:
        if c.exists():
            return c
    return None


def load_directions(joint_path: Path, max_frames):
    """Every canonical local bone direction HL quantizes for this dataset, over
    both hands. Returns a [N,3] float64 unit array (frame_to_hl directions, so they
    are identical to the HL token inputs)."""
    joints = hl.load_json(joint_path)
    vecs, edges = [], []
    seen = 0
    for _capture, frames in joints.items():
        if not isinstance(frames, dict):
            continue
        for _frame_idx, frame_item in frames.items():
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
                for e, v in enumerate(rec["local_vectors"]):
                    vecs.append(v)
                    edges.append(e)
            seen += 1
            if max_frames is not None and seen >= max_frames:
                V, _ = adb._finish(vecs, edges)
                return V
    V, _ = adb._finish(vecs, edges)
    return V


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+",
                    default=["interhand:/opt/tiger/InterHand/annotations/machine_annot",
                             "freihand:/opt/tiger/InterHand/annotations/machine_annot"],
                    help="list of name:annot_root entries; codebook fit on each")
    ap.add_argument("--fit-split", default="train",
                    help="split to FIT each dataset's adaptive codebook on")
    ap.add_argument("--eval-split", default="test",
                    help="split to EVALUATE every codebook on (held-out)")
    ap.add_argument("--k", type=int, default=26,
                    help="adaptive codebook size (matched to HL-26 by default)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    # load fit + eval directions per dataset, and fit one adaptive codebook each
    names, fit_vecs, eval_vecs, codebooks = [], {}, {}, {}
    for spec in args.datasets:
        name, _, root = spec.partition(":")
        root = Path(root)
        fp = _resolve_joint_path(root, args.fit_split, name)
        ep = _resolve_joint_path(root, args.eval_split, name)
        if fp is None or ep is None:
            print(f"[{name}] missing fit/eval joint file under {root}; skip")
            continue
        print(f"[{name}] fit  <- {fp}", flush=True)
        Vf = load_directions(fp, args.max_frames)
        print(f"[{name}] eval <- {ep}", flush=True)
        Ve = load_directions(ep, args.max_frames)
        if len(Vf) == 0 or len(Ve) == 0:
            print(f"[{name}] no directions decoded; skip")
            continue
        print(f"[{name}] {len(Vf)} fit / {len(Ve)} eval directions", flush=True)
        names.append(name)
        fit_vecs[name] = Vf
        eval_vecs[name] = Ve
        codebooks[name] = adb.spherical_kmeans(Vf, args.k, init="kmeans++",
                                               seed=args.seed)

    if not names:
        print("no datasets loaded; nothing to do")
        return

    hlcb = adb.hl26_codebook()

    # transfer matrix: rows = codebook source, columns = eval dataset
    sources = ["hl26"] + [f"adaptive@{n}" for n in names]
    matrix = {}
    for src in sources:
        C = hlcb if src == "hl26" else codebooks[src.split("@", 1)[1]]
        matrix[src] = {n: adb.angular_error_deg(eval_vecs[n], C) for n in names}

    width = max(12, max(len(s) for s in sources))
    label = "source \\ eval"
    header = f"{label:{width}s} " + " ".join(f"{n:>12s}" for n in names)
    print("\n== transfer matrix: mean reconstruction angular error (deg) ==")
    print(header)
    for src in sources:
        cells = " ".join(f"{matrix[src][n]:12.2f}" for n in names)
        print(f"{src:{width}s} {cells}")

    # reduction of each TRANSFERRED adaptive codebook vs HL-26, per eval dataset
    print("\n== transferred adaptive vs HL-26 (negative = lower error) ==")
    reductions = []
    for n in names:
        base = matrix["hl26"][n]
        for src_name in names:
            if src_name == n:
                continue  # skip own-data fit; report the transfer cases
            adp = matrix[f"adaptive@{src_name}"][n]
            pct = 100.0 * (adp - base) / base if base > 0 else 0.0
            reductions.append({"fit_on": src_name, "eval_on": n,
                               "hl26_deg": round(base, 4),
                               "adaptive_deg": round(adp, 4),
                               "pct_vs_hl26": round(pct, 2)})
            print(f"  fit {src_name:>10s} -> eval {n:>10s}: "
                  f"HL-26 {base:6.2f}°  adaptive {adp:6.2f}°  ({pct:+.0f}%)")
    print("\na transferred adaptive codebook beating HL-26 (negative %) means the "
          "adaptive layout captures general hand structure, not one dataset's quirk.")

    result = {
        "fit_split": args.fit_split, "eval_split": args.eval_split,
        "k": args.k, "seed": args.seed, "datasets": names,
        "n_fit": {n: int(len(fit_vecs[n])) for n in names},
        "n_eval": {n: int(len(eval_vecs[n])) for n in names},
        "transfer_matrix_deg": {src: {n: round(matrix[src][n], 4) for n in names}
                                for src in sources},
        "transfer_vs_hl26": reductions,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
