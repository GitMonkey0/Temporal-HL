#!/usr/bin/env python3
"""Search simple correction benchmark subsets over pair-status and gap thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def get_rows(candidate_payload):
    subsets = {item["subset"]: item["rows"] for item in candidate_payload["subsets"]}
    if "changed" not in subsets:
        raise ValueError("Candidate payload is missing the `changed` subset.")
    return subsets["changed"]


def keep_pair_status(row, mode: str):
    if mode == "all":
        return True
    if mode == "unseen":
        return row["pair_status"] == "unseen"
    if mode == "seen_fix":
        return row["pair_status"] == "seen_fix"
    if mode == "seen_harm":
        return row["pair_status"] == "seen_harm"
    if mode == "not_seen_fix":
        return row["pair_status"] != "seen_fix"
    if mode == "not_seen_harm":
        return row["pair_status"] != "seen_harm"
    raise ValueError(f"Unknown pair-status mode: {mode}")


def keep_target_mode(row, mode: str):
    if mode == "all":
        return True
    if mode == "positive":
        return bool(row["would_fix"])
    if mode == "negative":
        return bool(row["would_harm"])
    if mode == "ambiguous":
        return (not row["would_fix"]) and (not row["would_harm"])
    if mode == "nonnegative":
        return not row["would_harm"]
    raise ValueError(f"Unknown target mode: {mode}")


def summarize(rows):
    old_correct = sum(int(r["original_correct"]) for r in rows)
    would_fix = sum(int(r["would_fix"]) for r in rows)
    would_harm = sum(int(r["would_harm"]) for r in rows)
    return {
        "rows": len(rows),
        "old_accuracy": old_correct / len(rows) if rows else 0.0,
        "would_fix": would_fix,
        "would_harm": would_harm,
        "pair_unseen": sum(int(r["pair_status"] == "unseen") for r in rows),
        "pair_seen_fix": sum(int(r["pair_status"] == "seen_fix") for r in rows),
        "pair_seen_harm": sum(int(r["pair_status"] == "seen_harm") for r in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument(
        "--pair-status-modes",
        type=str,
        default="all,unseen,seen_fix,seen_harm,not_seen_fix,not_seen_harm",
    )
    parser.add_argument(
        "--target-modes",
        type=str,
        default="all,positive,negative,ambiguous,nonnegative",
    )
    parser.add_argument(
        "--gap-thresholds",
        type=str,
        default="0.01,0.02,0.04,0.05,0.06,0.08",
    )
    parser.add_argument("--output", type=Path, default=Path("experiments/generated/correction_subset_frontier.json"))
    args = parser.parse_args()

    payload = load_json(args.candidate_json)
    rows = get_rows(payload)
    pair_modes = [x.strip() for x in args.pair_status_modes.split(",") if x.strip()]
    target_modes = [x.strip() for x in args.target_modes.split(",") if x.strip()]
    gap_thresholds = [float(x.strip()) for x in args.gap_thresholds.split(",") if x.strip()]

    frontier = []
    for pair_mode in pair_modes:
        for target_mode in target_modes:
            for gap_thr in gap_thresholds:
                subset = [
                    r for r in rows
                    if keep_pair_status(r, pair_mode)
                    and keep_target_mode(r, target_mode)
                    and r["ranking_top2_score_gap"] <= gap_thr
                ]
                info = summarize(subset)
                frontier.append(
                    {
                        "pair_status_mode": pair_mode,
                        "target_mode": target_mode,
                        "gap_threshold": gap_thr,
                        **info,
                    }
                )

    frontier.sort(
        key=lambda r: (
            -r["would_fix"],
            r["would_harm"],
            -r["pair_unseen"],
            -r["rows"],
            r["gap_threshold"],
            r["pair_status_mode"],
            r["target_mode"],
        )
    )

    out = {
        "candidate_json": str(args.candidate_json),
        "frontier": frontier,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(out, f, indent=2)

    print(f"output: {args.output}")
    for row in frontier[:20]:
        print(row)


if __name__ == "__main__":
    main()
