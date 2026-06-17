#!/usr/bin/env python3
"""Summarize method performance on pair-hard subsets derived from pair-rule outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def pair_status_map(pair_rule_payload):
    rows = pair_rule_payload["all_candidate_outcomes"]
    status = {}
    for row in rows:
        pair = tuple(row["pair"])
        train = [
            x for x in rows
            if x["seed"] != row["seed"] and tuple(x["pair"]) == pair
        ]
        fixes = sum(int(x["would_fix"]) for x in train)
        harms = sum(int(x["would_harm"]) for x in train)
        if fixes > harms and fixes > 0:
            tag = "seen_fix"
        elif harms >= fixes and harms > 0:
            tag = "seen_harm"
        else:
            tag = "unseen"
        status[(row["seed"], row["seq_name"])] = {
            "status": tag,
            "pair": list(pair),
            "would_fix": bool(row["would_fix"]),
            "would_harm": bool(row["would_harm"]),
            "original_prediction": row["original_prediction"],
            "retrieval_prediction": row["retrieval_prediction"],
        }
    return status


def load_method_rows(payload):
    if "all_candidate_outcomes" in payload:
        return payload["all_candidate_outcomes"]
    if "best_zero_harm" in payload:
        return payload["best_zero_harm"][0]["all_family_outcomes"]
    if "fixed_threshold_0.5" in payload:
        return payload["fixed_threshold_0.5"]["all_family_outcomes"]
    raise ValueError("Unsupported method payload format")


def summarize_subset(method_rows, status_map, subset_name, changed_only):
    totals = Counter()
    examples = []
    for row in method_rows:
        key = (row["seed"], row["seq_name"])
        info = status_map.get(key)
        if info is None:
            continue
        if info["status"] != subset_name:
            continue
        if changed_only and info["retrieval_prediction"] == info["original_prediction"]:
            continue
        totals["rows"] += 1
        totals["old_correct"] += int(row["original_correct"])
        totals["new_correct"] += int(row["new_correct"])
        totals["applied"] += int(row["apply_gate"])
        totals["would_fix"] += int(info["would_fix"])
        totals["would_harm"] += int(info["would_harm"])
        totals["improved"] += int((not row["original_correct"]) and row["new_correct"])
        totals["harmed"] += int(row["original_correct"] and (not row["new_correct"]))
        examples.append(
            {
                "seed": row["seed"],
                "seq_name": row["seq_name"],
                "pair": info["pair"],
                "would_fix": info["would_fix"],
                "would_harm": info["would_harm"],
                "original_correct": row["original_correct"],
                "new_correct": row["new_correct"],
                "apply_gate": row["apply_gate"],
                "original_prediction": info["original_prediction"],
                "retrieval_prediction": info["retrieval_prediction"],
                "new_prediction": row["new_prediction"],
            }
        )
    out = {
        "subset": subset_name,
        "changed_only": changed_only,
        "counts": dict(totals),
        "rows": examples,
    }
    if totals["rows"]:
        out["accuracy"] = {
            "old": totals["old_correct"] / totals["rows"],
            "new": totals["new_correct"] / totals["rows"],
            "delta": totals["new_correct"] / totals["rows"] - totals["old_correct"] / totals["rows"],
        }
    else:
        out["accuracy"] = {"old": 0.0, "new": 0.0, "delta": 0.0}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-rule-json", type=Path, required=True)
    parser.add_argument("--method-json", type=Path, nargs="+", required=True)
    parser.add_argument("--subset", type=str, default="unseen", choices=["unseen", "seen_fix", "seen_harm"])
    parser.add_argument("--changed-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("experiments/generated/pair_hard_subset_summary.json"))
    args = parser.parse_args()

    pair_payload = load_json(args.pair_rule_json)
    status_map = pair_status_map(pair_payload)
    methods = []
    for path in args.method_json:
        payload = load_json(path)
        rows = load_method_rows(payload)
        methods.append(
            {
                "method_json": str(path),
                **summarize_subset(rows, status_map, args.subset, args.changed_only),
            }
        )

    out = {
        "pair_rule_json": str(args.pair_rule_json),
        "subset": args.subset,
        "changed_only": args.changed_only,
        "methods": methods,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(out, f, indent=2)

    print(f"output: {args.output}")
    for item in methods:
        print(item["method_json"], item["accuracy"], item["counts"])


if __name__ == "__main__":
    main()
