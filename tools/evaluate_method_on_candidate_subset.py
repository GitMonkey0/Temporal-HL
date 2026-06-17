#!/usr/bin/env python3
"""Evaluate existing method outputs on a named candidate subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def load_method_rows(payload):
    if "all_candidate_outcomes" in payload:
        return payload["all_candidate_outcomes"]
    if "best_zero_harm" in payload:
        return payload["best_zero_harm"][0]["all_family_outcomes"]
    if "fixed_threshold_0.5" in payload:
        return payload["fixed_threshold_0.5"]["all_family_outcomes"]
    raise ValueError("Unsupported method payload format")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--subset", type=str, required=True)
    parser.add_argument("--method-json", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("experiments/generated/method_subset_eval.json"))
    args = parser.parse_args()

    candidate_payload = load_json(args.candidate_json)
    subset_rows = None
    for item in candidate_payload["subsets"]:
        if item["subset"] == args.subset:
            subset_rows = item["rows"]
            break
    if subset_rows is None:
        raise SystemExit(f"Subset not found: {args.subset}")
    keys = {(row["seed"], row["seq_name"]) for row in subset_rows}

    methods = []
    for path in args.method_json:
        payload = load_json(path)
        rows = [r for r in load_method_rows(payload) if (r["seed"], r["seq_name"]) in keys]
        old_correct = sum(int(r["original_correct"]) for r in rows)
        new_correct = sum(int(r["new_correct"]) for r in rows)
        applied = sum(int(r["apply_gate"]) for r in rows)
        improved = sum(int((not r["original_correct"]) and r["new_correct"]) for r in rows)
        harmed = sum(int(r["original_correct"] and (not r["new_correct"])) for r in rows)
        methods.append(
            {
                "method_json": str(path),
                "rows": len(rows),
                "old_accuracy": old_correct / len(rows) if rows else 0.0,
                "new_accuracy": new_correct / len(rows) if rows else 0.0,
                "delta": (new_correct - old_correct) / len(rows) if rows else 0.0,
                "num_applied": applied,
                "num_improved": improved,
                "num_harmed": harmed,
                "rows_detail": rows,
            }
        )

    out = {
        "candidate_json": str(args.candidate_json),
        "subset": args.subset,
        "methods": methods,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(out, f, indent=2)

    print(f"output: {args.output}")
    for item in methods:
        print(item["method_json"], item["rows"], item["old_accuracy"], item["new_accuracy"], item["delta"], item["num_applied"], item["num_improved"], item["num_harmed"])


if __name__ == "__main__":
    main()
