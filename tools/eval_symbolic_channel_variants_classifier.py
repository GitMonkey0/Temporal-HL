#!/usr/bin/env python3
"""Evaluate lightweight classifier performance for symbolic channel variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.train_symbolic_classifier import (
    load_json,
    make_window_samples,
    overlap_labels,
    run_experiment,
    summarize_results,
)


def parse_variants(spec: str):
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, weights = item.split("::", 1)
        vals = [float(x) for x in weights.split("+")]
        if len(vals) != 5:
            raise ValueError(f"Expected 5 weights for {name}, got {weights}")
        out.append(
            (
                name,
                {
                    "state": vals[0],
                    "transition": vals[1],
                    "hand_motion": vals[2],
                    "interaction": vals[3],
                    "tempo": vals[4],
                },
            )
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("experiments/generated/symbolic_channel_variants_classifier.json"))
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--c-value", type=float, default=16.0)
    parser.add_argument("--aggregation", choices=["mean_prob", "mean_log_prob", "vote"], default="mean_log_prob")
    parser.add_argument(
        "--variants",
        type=str,
        default=(
            "state_only::1+0+0+0+0,"
            "full_temporal::1+0.5+0.5+0.2+0.2,"
            "state_transition::1+0.5+0+0+0,"
            "state_motion::1+0+0.5+0+0"
        ),
    )
    args = parser.parse_args()

    train_data = load_json(args.train_json)
    test_data = load_json(args.test_json)
    labels = overlap_labels(train_data, test_data)

    results = {
        "labels": sorted(labels),
        "window_size": args.window_size,
        "stride": args.stride,
        "fractions": args.fractions,
        "seeds": args.seeds,
        "c_value": args.c_value,
        "aggregation": args.aggregation,
        "variants": {},
    }

    for name, weights in parse_variants(args.variants):
        mode = "temporal" if any(weights[k] > 0 for k in ("transition", "hand_motion", "interaction", "tempo")) else "state"
        train_samples = make_window_samples(
            train_data,
            labels,
            mode,
            args.window_size,
            args.stride,
            weights,
            include_wrist_features=False,
        )
        test_samples = make_window_samples(
            test_data,
            labels,
            mode,
            args.window_size,
            args.stride,
            weights,
            include_wrist_features=False,
        )
        runs = []
        for fraction in args.fractions:
            for seed in args.seeds:
                runs.append(
                    run_experiment(
                        train_samples=train_samples,
                        test_samples=test_samples,
                        fraction=fraction,
                        seed=seed,
                        c_value=args.c_value,
                        aggregation=args.aggregation,
                    )
                )
        results["variants"][name] = {
            "mode": mode,
            "weights": weights,
            "num_train_windows_full": len(train_samples),
            "num_test_windows": len(test_samples),
            "runs": runs,
            "summary": summarize_results(runs),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"output: {args.output}")
    for name, payload in results["variants"].items():
        print(name, payload["weights"])
        for frac, summ in payload["summary"].items():
            print(
                " fraction",
                frac,
                "seq",
                summ["sequence_accuracy_mean"],
                "win",
                summ["window_accuracy_mean"],
            )


if __name__ == "__main__":
    main()
