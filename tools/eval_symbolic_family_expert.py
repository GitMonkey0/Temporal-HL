#!/usr/bin/env python3
"""Evaluate a family-restricted multiclass expert over exported family score vectors."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tools.compare_symbolic_slices import slice_membership
from tools.eval_occlusion_late_correction import FAMILY_LABELS, build_gallery, build_query_events, retrieve_family_label
from tools.eval_sequence_symbolic_retrieval import canonical_label, load_json


def aggregate_slice_accuracy(rows):
    totals = Counter()
    correct = Counter()
    for row in rows:
        for group in slice_membership(row["seq_name"]):
            totals[group] += 1
            if row["correct"]:
                correct[group] += 1
    out = []
    for group in sorted(totals):
        out.append(
            {
                "slice": group,
                "total": totals[group],
                "correct": correct[group],
                "accuracy": correct[group] / max(totals[group], 1),
            }
        )
    return out


def compare_slices(old_rows, new_rows):
    old = {row["slice"]: row for row in aggregate_slice_accuracy(old_rows)}
    new = {row["slice"]: row for row in aggregate_slice_accuracy(new_rows)}
    out = []
    for key in sorted(set(old) | set(new)):
        old_acc = old.get(key, {}).get("accuracy", 0.0)
        new_acc = new.get(key, {}).get("accuracy", 0.0)
        out.append(
            {
                "slice": key,
                "old_accuracy": old_acc,
                "new_accuracy": new_acc,
                "delta": new_acc - old_acc,
            }
        )
    out.sort(key=lambda row: (-row["delta"], row["slice"]))
    return out


def parse_thresholds(raw: str):
    vals = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            vals.append(float(item))
    if not vals:
        raise ValueError("No thresholds provided")
    return sorted(set(vals))


def to_builtin(obj):
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def load_family_rows(paths):
    rows = []
    for path in paths:
        data = load_json(path)
        seed = data["seed"]
        for row in data["details"]["pretrained"]["sequence_rows"]:
            if "family_scores" not in row:
                raise KeyError(f"family_scores missing in {path}")
            pred = canonical_label(row["prediction"])
            if pred not in FAMILY_LABELS:
                continue
            rows.append(
                {
                    "seed": seed,
                    "seq_name": row["seq_name"],
                    "target": canonical_label(row["target"]),
                    "prediction": pred,
                    "correct": bool(row["correct"]),
                    "family_scores": {canonical_label(k): float(v) for k, v in row["family_scores"].items()},
                    "top3": row["top3"],
                }
            )
    return rows


def build_dataset(base_rows, query_events, gallery):
    retrieval_cache = {}
    dataset = []
    for row in base_rows:
        if row["seq_name"] not in query_events:
            continue
        if row["seq_name"] not in retrieval_cache:
            ranking = retrieve_family_label(query_events[row["seq_name"]], gallery)
            retrieval_cache[row["seq_name"]] = {
                canonical_label(label): float(score) for label, _, score in ranking
            }
        retrieval_scores = retrieval_cache[row["seq_name"]]
        feats = {}
        for label in FAMILY_LABELS:
            feats[f"main_{label}"] = row["family_scores"].get(label, -10.0)
            feats[f"ret_{label}"] = retrieval_scores.get(label, -10.0)
            feats[f"ret_minus_main_{label}"] = feats[f"ret_{label}"] - feats[f"main_{label}"]
        dataset.append(
            {
                "seed": row["seed"],
                "seq_name": row["seq_name"],
                "target": row["target"],
                "original_prediction": row["prediction"],
                "original_correct": row["correct"],
                "features": feats,
            }
        )
    return dataset


def feature_matrix(rows, feature_order):
    return np.asarray([[row["features"][name] for name in feature_order] for row in rows], dtype=np.float64)


def run_loso(dataset, feature_order, confidence_threshold):
    label_to_idx = {label: idx for idx, label in enumerate(FAMILY_LABELS)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    seeds = sorted({row["seed"] for row in dataset})
    outcomes = []
    for test_seed in seeds:
        train_rows = [row for row in dataset if row["seed"] != test_seed]
        test_rows = [row for row in dataset if row["seed"] == test_seed]
        x_train = feature_matrix(train_rows, feature_order)
        y_train = np.asarray([label_to_idx[row["target"]] for row in train_rows], dtype=np.int64)
        x_test = feature_matrix(test_rows, feature_order)

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        random_state=0,
                        class_weight="balanced",
                        solver="lbfgs",
                        max_iter=2000,
                    ),
                ),
            ]
        )
        model.fit(x_train, y_train)
        probs = model.predict_proba(x_test)
        pred_idx = probs.argmax(axis=1)
        pred_conf = probs.max(axis=1)
        for row, idx, conf in zip(test_rows, pred_idx, pred_conf):
            expert_label = idx_to_label[int(idx)]
            use_expert = float(conf) >= confidence_threshold
            new_prediction = expert_label if use_expert else row["original_prediction"]
            outcomes.append(
                {
                    "seed": row["seed"],
                    "seq_name": row["seq_name"],
                    "target": row["target"],
                    "original_prediction": row["original_prediction"],
                    "new_prediction": new_prediction,
                    "expert_prediction": expert_label,
                    "expert_confidence": float(conf),
                    "used_expert": use_expert,
                    "original_correct": row["original_correct"],
                    "new_correct": new_prediction == row["target"],
                }
            )
    return outcomes


def evaluate_threshold(dataset, feature_order, threshold):
    outcomes = run_loso(dataset, feature_order, threshold)
    old_rows = [{"seq_name": row["seq_name"], "correct": row["original_correct"]} for row in outcomes]
    new_rows = [{"seq_name": row["seq_name"], "correct": row["new_correct"]} for row in outcomes]
    improved = sum(1 for row in outcomes if (not row["original_correct"]) and row["new_correct"])
    harmed = sum(1 for row in outcomes if row["original_correct"] and (not row["new_correct"]))
    used = sum(1 for row in outcomes if row["used_expert"])
    family_old = sum(row["original_correct"] for row in outcomes) / max(len(outcomes), 1)
    family_new = sum(row["new_correct"] for row in outcomes) / max(len(outcomes), 1)
    return {
        "threshold": threshold,
        "family_accuracy": {
            "old": family_old,
            "new": family_new,
            "delta": family_new - family_old,
        },
        "counts": {
            "num_family_rows": len(outcomes),
            "num_used_expert": used,
            "num_improved": improved,
            "num_harmed": harmed,
        },
        "slice_delta_family_only": compare_slices(old_rows, new_rows),
        "used_cases": [row for row in outcomes if row["used_expert"]],
        "all_family_outcomes": outcomes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detailed-analysis", type=Path, nargs="+", required=True)
    parser.add_argument("--gallery-json", type=Path, required=True)
    parser.add_argument("--query-json", type=Path, required=True)
    parser.add_argument("--thresholds", type=str, default="0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/generated/symbolic_family_expert.json"),
    )
    args = parser.parse_args()

    base_rows = load_family_rows(args.detailed_analysis)
    gallery = build_gallery(load_json(args.gallery_json), set(FAMILY_LABELS))
    query_events = build_query_events(load_json(args.query_json), set(FAMILY_LABELS))
    dataset = build_dataset(base_rows, query_events, gallery)
    feature_order = sorted(dataset[0]["features"].keys())
    thresholds = parse_thresholds(args.thresholds)

    sweeps = [evaluate_threshold(dataset, feature_order, thr) for thr in thresholds]
    sweeps.sort(
        key=lambda row: (
            -row["family_accuracy"]["delta"],
            row["counts"]["num_harmed"],
            -row["counts"]["num_improved"],
            row["threshold"],
        )
    )
    payload = {
        "detailed_analysis": [str(path) for path in args.detailed_analysis],
        "gallery_json": str(args.gallery_json),
        "query_json": str(args.query_json),
        "feature_order": feature_order,
        "thresholds": thresholds,
        "family_dataset_size": len(dataset),
        "sweeps": sweeps,
        "fixed_threshold_0.5": next(row for row in sweeps if abs(row["threshold"] - 0.5) < 1e-9),
        "best_zero_harm": [row for row in sweeps if row["counts"]["num_harmed"] == 0][:10],
    }
    payload = to_builtin(payload)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)

    fixed = payload["fixed_threshold_0.5"]
    print(f"output: {args.output}")
    print(
        "fixed@0.5 family",
        fixed["family_accuracy"]["old"],
        "->",
        fixed["family_accuracy"]["new"],
        "delta",
        fixed["family_accuracy"]["delta"],
        "counts",
        fixed["counts"],
    )
    for row in payload["best_zero_harm"][:5]:
        print(
            "zero_harm",
            row["threshold"],
            "family",
            row["family_accuracy"]["old"],
            "->",
            row["family_accuracy"]["new"],
            "delta",
            row["family_accuracy"]["delta"],
            "counts",
            row["counts"],
        )


if __name__ == "__main__":
    main()
