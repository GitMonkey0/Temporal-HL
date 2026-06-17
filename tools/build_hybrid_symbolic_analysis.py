#!/usr/bin/env python3
"""Build a hybrid symbolic analysis by applying a validated gated correction."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from tools.eval_sequence_symbolic_retrieval import canonical_label
from tools.sweep_occlusion_late_correction_gate import should_apply


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def error_counts_from_predictions(sequence_consistency: dict[str, object], key: str):
    counts = Counter()
    for seq_name, item in sequence_consistency.items():
        target = canonical_label(seq_name)
        for pred in item[key]:
            prediction = canonical_label(pred["prediction"])
            if prediction != target:
                counts[(target, prediction)] += 1
    rows = []
    for (target, prediction), count in counts.items():
        rows.append(
            {
                "target": target,
                "prediction": prediction,
                "count": count,
            }
        )
    rows.sort(key=lambda row: (-row["count"], row["target"], row["prediction"]))
    return rows


def runs_from_sequence_consistency(sequence_consistency: dict[str, object]):
    seeds = set()
    for item in sequence_consistency.values():
        for pred in item["scratch_predictions"]:
            seeds.add(pred["seed"])
        for pred in item["pretrained_predictions"]:
            seeds.add(pred["seed"])
    runs = []
    for seed in sorted(seeds):
        scratch_total = 0
        scratch_correct = 0
        pretrained_total = 0
        pretrained_correct = 0
        scratch_errors = []
        pretrained_errors = []
        for seq_name, item in sorted(sequence_consistency.items()):
            target = canonical_label(seq_name)
            scratch_pred = next(pred for pred in item["scratch_predictions"] if pred["seed"] == seed)
            pretrained_pred = next(pred for pred in item["pretrained_predictions"] if pred["seed"] == seed)
            scratch_total += 1
            pretrained_total += 1
            scratch_correct += int(bool(scratch_pred["correct"]))
            pretrained_correct += int(bool(pretrained_pred["correct"]))
            if not scratch_pred["correct"]:
                scratch_errors.append(
                    {
                        "seq_name": seq_name,
                        "target": target,
                        "prediction": canonical_label(scratch_pred["prediction"]),
                        "correct": False,
                    }
                )
            if not pretrained_pred["correct"]:
                pretrained_errors.append(
                    {
                        "seq_name": seq_name,
                        "target": target,
                        "prediction": canonical_label(pretrained_pred["prediction"]),
                        "correct": False,
                    }
                )
        runs.append(
            {
                "seed": seed,
                "scratch_sequence_accuracy": scratch_correct / max(scratch_total, 1),
                "pretrained_sequence_accuracy": pretrained_correct / max(pretrained_total, 1),
                "scratch_errors": scratch_errors,
                "pretrained_errors": pretrained_errors,
            }
        )
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-json", type=Path, required=True)
    parser.add_argument("--late-correction-json", type=Path, required=True)
    parser.add_argument("--rule", type=str, default="change_only")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/generated/hybrid_symbolic_analysis.json"),
    )
    args = parser.parse_args()

    analysis = load_json(args.analysis_json)
    late = load_json(args.late_correction_json)

    family_rows = {}
    for row in late["family_rows"]:
        seq_name = row["seq_name"]
        seed = row["seed"]
        retrieval_top3 = row.get("retrieval_top3") or []
        top1_score = retrieval_top3[0][2] if len(retrieval_top3) >= 1 else None
        top2_score = retrieval_top3[1][2] if len(retrieval_top3) >= 2 else None
        margin = None
        if top1_score is not None and top2_score is not None:
            margin = float(top1_score - top2_score)
        original_prediction = canonical_label(row["original_prediction"])
        retrieval_prediction = canonical_label(row["corrected_prediction"])
        family_rows[(seq_name, seed)] = {
            "seq_name": seq_name,
            "seed": seed,
            "target": canonical_label(row["target"]),
            "original_prediction": original_prediction,
            "retrieval_prediction": retrieval_prediction,
            "changed_prediction": retrieval_prediction != original_prediction,
            "margin": margin,
            "retrieval_top3": retrieval_top3,
        }

    patched = json.loads(json.dumps(analysis))
    applied_cases = []
    for seq_name, item in patched["sequence_consistency"].items():
        target = canonical_label(seq_name)
        for pred in item["pretrained_predictions"]:
            key = (seq_name, pred["seed"])
            family = family_rows.get(key)
            if family is None:
                pred["prediction"] = canonical_label(pred["prediction"])
                pred["correct"] = canonical_label(pred["prediction"]) == target
                continue
            if should_apply(args.rule, family, args.threshold):
                new_prediction = family["retrieval_prediction"]
                old_prediction = canonical_label(pred["prediction"])
                pred["prediction"] = new_prediction
                pred["correct"] = new_prediction == target
                applied_cases.append(
                    {
                        "seq_name": seq_name,
                        "seed": pred["seed"],
                        "target": target,
                        "old_prediction": old_prediction,
                        "new_prediction": new_prediction,
                        "changed": new_prediction != old_prediction,
                        "improved": (old_prediction != target) and (new_prediction == target),
                        "margin": family["margin"],
                    }
                )
            else:
                pred["prediction"] = canonical_label(pred["prediction"])
                pred["correct"] = canonical_label(pred["prediction"]) == target

    patched["hybrid_patch"] = {
        "rule": args.rule,
        "threshold": args.threshold,
        "late_correction_json": str(args.late_correction_json),
        "analysis_json": str(args.analysis_json),
        "num_applied": len(applied_cases),
        "num_changed": sum(1 for row in applied_cases if row["changed"]),
        "num_improved": sum(1 for row in applied_cases if row["improved"]),
        "num_harmed": sum(
            1
            for row in applied_cases
            if row["old_prediction"] == row["target"] and row["new_prediction"] != row["target"]
        ),
        "applied_cases": applied_cases,
    }
    patched["runs"] = runs_from_sequence_consistency(patched["sequence_consistency"])
    patched["scratch_error_counts"] = error_counts_from_predictions(patched["sequence_consistency"], "scratch_predictions")
    patched["pretrained_error_counts"] = error_counts_from_predictions(
        patched["sequence_consistency"], "pretrained_predictions"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(patched, f, indent=2)

    pretrained_mean = sum(row["pretrained_sequence_accuracy"] for row in patched["runs"]) / max(len(patched["runs"]), 1)
    print(f"output: {args.output}")
    print("rule", args.rule, "threshold", args.threshold)
    print("applied", patched["hybrid_patch"]["num_applied"], "changed", patched["hybrid_patch"]["num_changed"])
    print("improved", patched["hybrid_patch"]["num_improved"], "harmed", patched["hybrid_patch"]["num_harmed"])
    print("pretrained_seq_mean", pretrained_mean)


if __name__ == "__main__":
    main()
