#!/usr/bin/env python3
"""Sweep gated variants of the occlusion-family late correction."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from tools.compare_symbolic_slices import slice_membership
from tools.eval_sequence_symbolic_retrieval import canonical_label


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


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
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("No thresholds provided")
    return sorted(set(values))


def build_base_rows(analysis):
    all_rows = []
    for seq_name, item in sorted(analysis["sequence_consistency"].items()):
        target_label = canonical_label(seq_name)
        for pred in item["pretrained_predictions"]:
            seed = pred["seed"]
            original_pred = canonical_label(pred["prediction"])
            row = {
                "seq_name": seq_name,
                "target": target_label,
                "prediction": original_pred,
                "correct": original_pred == target_label,
                "seed": seed,
            }
            all_rows.append(row)
    return all_rows


def enrich_family_rows(late_correction):
    out = {}
    for row in late_correction["family_rows"]:
        seq_name = row["seq_name"]
        target = canonical_label(row["target"])
        seed = row["seed"]
        original_prediction = canonical_label(row["original_prediction"])
        retrieval_prediction = canonical_label(row["corrected_prediction"])
        retrieval_top3 = row.get("retrieval_top3") or []
        top1_score = retrieval_top3[0][2] if len(retrieval_top3) >= 1 else None
        top2_score = retrieval_top3[1][2] if len(retrieval_top3) >= 2 else None
        margin = None
        if top1_score is not None and top2_score is not None:
            margin = float(top1_score - top2_score)
        enriched = dict(row)
        enriched["target"] = target
        enriched["margin"] = margin
        enriched["original_prediction"] = original_prediction
        enriched["retrieval_prediction"] = retrieval_prediction
        enriched["changed_prediction"] = retrieval_prediction != original_prediction
        out[(seq_name, seed)] = enriched
    return out


def should_apply(rule: str, family_row: dict[str, object], threshold: float):
    margin = family_row.get("margin")
    margin_ok = margin is not None and margin >= threshold
    changed = bool(family_row.get("changed_prediction"))
    if rule == "family_always":
        return True
    if rule == "margin_only":
        return margin_ok
    if rule == "change_only":
        return changed
    if rule == "change_and_margin":
        return changed and margin_ok
    raise ValueError(f"Unknown rule: {rule}")


def evaluate_rule(base_rows, family_rows, rule: str, threshold: float):
    old_rows = []
    new_rows = []
    family_eval = []
    improved = 0
    harmed = 0
    changed_cases = 0
    applied_cases = 0
    for row in base_rows:
        key = (row["seq_name"], row["seed"])
        old_rows.append(dict(row))
        new_row = dict(row)
        family_row = family_rows.get(key)
        applied = False
        if family_row is not None and should_apply(rule, family_row, threshold):
            new_row["prediction"] = canonical_label(family_row["retrieval_prediction"])
            new_row["correct"] = new_row["prediction"] == new_row["target"]
            applied = True
            applied_cases += 1
            if new_row["prediction"] != row["prediction"]:
                changed_cases += 1
        new_rows.append(new_row)

        if family_row is not None:
            old_correct = row["correct"]
            new_correct = new_row["correct"]
            family_eval.append(
                {
                    "seq_name": row["seq_name"],
                    "seed": row["seed"],
                    "target": row["target"],
                    "original_prediction": row["prediction"],
                    "new_prediction": new_row["prediction"],
                    "original_correct": old_correct,
                    "new_correct": new_correct,
                    "applied": applied,
                    "margin": family_row["margin"],
                    "changed_prediction": family_row["changed_prediction"],
                }
            )
            if (not old_correct) and new_correct:
                improved += 1
            if old_correct and (not new_correct):
                harmed += 1

    per_seed = []
    by_seed_old = defaultdict(list)
    by_seed_new = defaultdict(list)
    for old_row, new_row in zip(old_rows, new_rows):
        by_seed_old[old_row["seed"]].append(old_row)
        by_seed_new[new_row["seed"]].append(new_row)
    for seed in sorted(by_seed_old):
        old_acc = sum(row["correct"] for row in by_seed_old[seed]) / max(len(by_seed_old[seed]), 1)
        new_acc = sum(row["correct"] for row in by_seed_new[seed]) / max(len(by_seed_new[seed]), 1)
        per_seed.append(
            {
                "seed": seed,
                "old_accuracy": old_acc,
                "new_accuracy": new_acc,
                "delta": new_acc - old_acc,
            }
        )

    family_old_acc = sum(row["original_correct"] for row in family_eval) / max(len(family_eval), 1)
    family_new_acc = sum(row["new_correct"] for row in family_eval) / max(len(family_eval), 1)
    overall_old_acc = sum(row["correct"] for row in old_rows) / max(len(old_rows), 1)
    overall_new_acc = sum(row["correct"] for row in new_rows) / max(len(new_rows), 1)

    result = {
        "rule": rule,
        "threshold": threshold,
        "overall_accuracy": {
            "old": overall_old_acc,
            "new": overall_new_acc,
            "delta": overall_new_acc - overall_old_acc,
        },
        "family_accuracy": {
            "old": family_old_acc,
            "new": family_new_acc,
            "delta": family_new_acc - family_old_acc,
        },
        "counts": {
            "num_total_rows": len(old_rows),
            "num_family_rows": len(family_eval),
            "num_applied": applied_cases,
            "num_changed_prediction": changed_cases,
            "num_improved": improved,
            "num_harmed": harmed,
        },
        "per_seed_accuracy": per_seed,
        "slice_delta_overall": compare_slices(old_rows, new_rows),
        "slice_delta_family_only": compare_slices(
            [{"seq_name": row["seq_name"], "correct": row["original_correct"]} for row in family_eval],
            [{"seq_name": row["seq_name"], "correct": row["new_correct"]} for row in family_eval],
        ),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--late-correction-json", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.0,0.045,0.0475,0.05,0.0525,0.055,0.0575,0.06,0.0625,0.065,0.0675,0.07",
    )
    parser.add_argument(
        "--rules",
        type=str,
        default="family_always,margin_only,change_only,change_and_margin",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/generated/occlusion_late_correction_gate_sweep.json"),
    )
    args = parser.parse_args()

    late = load_json(args.late_correction_json)
    analysis = load_json(Path(late["analysis_json"]))
    thresholds = parse_thresholds(args.thresholds)
    rules = [item.strip() for item in args.rules.split(",") if item.strip()]

    base_rows = build_base_rows(analysis)
    family_rows = enrich_family_rows(late)

    sweeps = []
    for rule in rules:
        if rule in {"family_always", "change_only"}:
            candidates = [0.0]
        else:
            candidates = thresholds
        for threshold in candidates:
            sweeps.append(evaluate_rule(base_rows, family_rows, rule, threshold))

    best_by_delta = sorted(
        sweeps,
        key=lambda row: (
            -row["overall_accuracy"]["delta"],
            row["counts"]["num_harmed"],
            -row["counts"]["num_improved"],
            row["counts"]["num_applied"],
            row["rule"],
            row["threshold"],
        ),
    )
    best_zero_harm = [
        row
        for row in best_by_delta
        if row["counts"]["num_harmed"] == 0
    ]

    payload = {
        "late_correction_json": str(args.late_correction_json),
        "analysis_json": late["analysis_json"],
        "thresholds": thresholds,
        "rules": rules,
        "sweeps": sweeps,
        "best_overall_delta": best_by_delta[:10],
        "best_zero_harm": best_zero_harm[:10],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"output: {args.output}")
    for row in best_zero_harm[:5]:
        print(
            "zero_harm",
            row["rule"],
            row["threshold"],
            "overall",
            row["overall_accuracy"]["old"],
            "->",
            row["overall_accuracy"]["new"],
            "delta",
            row["overall_accuracy"]["delta"],
            "applied",
            row["counts"]["num_applied"],
            "changed",
            row["counts"]["num_changed_prediction"],
            "improved",
            row["counts"]["num_improved"],
        )


if __name__ == "__main__":
    main()
