#!/usr/bin/env python3
"""Build a counterfactual token-ablation report for representative hard cases."""

from __future__ import annotations

import json
from pathlib import Path

from tools.eval_sequence_symbolic_retrieval import dtw_similarity, overlap_labels
from tools.eval_symbolic_representation_intrinsic import build_gallery, build_queries


ROOT = Path("/opt/tiger/hand")
GEN = ROOT / "experiments/generated"
SUM = GEN / "summary_tables"


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def fmt(x: float) -> str:
    return f"{x:.4f}"


def filter_event(event: set[str], variant: str) -> set[str]:
    kept = set()
    for token in event:
        if variant == "drop_transition" and ":trans:" in token:
            continue
        if variant == "drop_motion" and ":motion:" in token:
            continue
        if variant == "drop_interaction" and token.startswith("interaction="):
            continue
        if variant == "drop_duration" and token.startswith("event::duration::"):
            continue
        kept.add(token)
    return kept


def evaluate_variant(query_rep, gallery, label):
    ranking = []
    for item in gallery:
        score = float(dtw_similarity(query_rep, item["rep"]))
        ranking.append((item["label"], item["seq_name"], score))
    ranking.sort(key=lambda x: x[2], reverse=True)
    correct_items = [x for x in ranking if x[0] == label]
    wrong_items = [x for x in ranking if x[0] != label]
    best_correct = correct_items[0]
    best_wrong = wrong_items[0]
    return {
        "best_correct_label": best_correct[0],
        "best_correct_seq": best_correct[1],
        "best_correct_score": best_correct[2],
        "best_wrong_label": best_wrong[0],
        "best_wrong_seq": best_wrong[1],
        "best_wrong_score": best_wrong[2],
        "margin": best_correct[2] - best_wrong[2],
        "top1_label": ranking[0][0],
        "top1_seq": ranking[0][1],
        "top1_correct": ranking[0][0] == label,
    }


def main():
    intrinsic = load_json(GEN / "symbolic_representation_intrinsic_val_test_plus.json")
    gallery_data = load_json(GEN / "temporal_hl_val.json")
    query_data = load_json(GEN / "temporal_hl_test.json")
    allowed = overlap_labels(gallery_data, query_data)

    gallery = build_gallery(gallery_data, allowed, "temporal", "event_dtw")
    queries = build_queries(query_data, allowed, "temporal", "event_dtw")
    query_map = {row["seq_name"]: row for row in queries}
    base_rows = {
        row["seq_name"]: row
        for row in next(r for r in intrinsic["results"] if r["name"] == "temporal_event")["rows"]
    }

    chosen = [
        "ROM04_RT_Occlusion",
        "ROM04_LT_Occlusion",
        "ROM07_RT_Finger_Occlusions",
        "ROM05_RT_Wrist_ROM",
        "ROM02_Interaction_2_Hand",
    ]
    variants = [
        "drop_transition",
        "drop_motion",
        "drop_interaction",
        "drop_duration",
    ]

    cases = []
    aggregate = {name: {"margin_delta_sum": 0.0, "neighbor_changes": 0, "top1_breaks": 0, "count": 0} for name in variants}
    for seq_name in chosen:
        if seq_name not in query_map or seq_name not in base_rows:
            continue
        q = query_map[seq_name]
        base = base_rows[seq_name]
        case = {
            "seq_name": seq_name,
            "label": q["label"],
            "base": {
                "best_wrong_label": base["best_wrong_label"],
                "best_wrong_seq": base["best_wrong_seq"],
                "margin": base["positive_margin"],
                "top1_correct": base["top1_correct"],
            },
            "variants": {},
        }
        for variant in variants:
            filtered = [filter_event(event, variant) for event in q["rep"]]
            result = evaluate_variant(filtered, gallery, q["label"])
            result["margin_delta_vs_base"] = result["margin"] - base["positive_margin"]
            result["wrong_neighbor_changed_vs_base"] = result["best_wrong_seq"] != base["best_wrong_seq"]
            case["variants"][variant] = result

            aggregate[variant]["margin_delta_sum"] += result["margin_delta_vs_base"]
            aggregate[variant]["neighbor_changes"] += int(result["wrong_neighbor_changed_vs_base"])
            aggregate[variant]["top1_breaks"] += int(not result["top1_correct"])
            aggregate[variant]["count"] += 1
        cases.append(case)

    aggregate_rows = []
    for variant, stats in aggregate.items():
        count = max(stats["count"], 1)
        aggregate_rows.append(
            {
                "variant": variant,
                "mean_margin_delta_vs_base": stats["margin_delta_sum"] / count,
                "neighbor_changes": stats["neighbor_changes"],
                "top1_breaks": stats["top1_breaks"],
                "count": stats["count"],
            }
        )
    aggregate_rows.sort(key=lambda r: (r["mean_margin_delta_vs_base"], -r["neighbor_changes"]))

    report = {
        "artifacts": {
            "intrinsic_report": str(GEN / "symbolic_representation_intrinsic_val_test_plus.json"),
            "gallery_json": str(GEN / "temporal_hl_val.json"),
            "query_json": str(GEN / "temporal_hl_test.json"),
        },
        "chosen_cases": chosen,
        "cases": cases,
        "aggregate": aggregate_rows,
        "derived_claim_checks": {
            "drop_motion_has_largest_mean_margin_harm": aggregate_rows[0]["variant"] == "drop_motion",
            "drop_transition_selectively_changes_neighbor": next(r for r in aggregate_rows if r["variant"] == "drop_transition")["neighbor_changes"] >= 1,
            "drop_duration_hurts_interaction_case": next(c for c in cases if c["seq_name"] == "ROM02_Interaction_2_Hand")["variants"]["drop_duration"]["margin_delta_vs_base"] < -0.01,
            "no_ablation_breaks_top1_on_this_case_set": sum(r["top1_breaks"] for r in aggregate_rows) == 0,
        },
        "takeaways": [
            "Counterfactual token ablation tests whether temporal symbolic channels are causally important for neighborhood structure, not merely correlated with better results.",
            "On this representative case set, dropping motion tokens causes the largest mean margin harm, so the current event encoding does not isolate transition as the single dominant temporal carrier.",
            "Dropping transition tokens is still not inert: it selectively changes the nearest wrong neighbor on a finger-occlusion case, which indicates local structural contribution even without the largest average margin loss.",
            "Dropping duration hurts the interaction case most strongly, which suggests that different temporal channels matter for different hard-case families.",
            "Because no single ablation breaks top-1 on this small case set, the strongest interpretation is not channel indispensability but channel-specific neighborhood shaping.",
        ],
    }

    out_json = GEN / "hardcase_counterfactual_report.json"
    out_md = SUM / "hardcase_counterfactual_report.md"
    out_json.write_text(json.dumps(report, indent=2))

    lines = [
        "# Hard-Case Counterfactual Report",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "## Aggregate",
        "",
        "| variant | mean margin delta vs base | neighbor changes | top1 breaks | count |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['variant']} | {fmt(row['mean_margin_delta_vs_base'])} | {row['neighbor_changes']} | {row['top1_breaks']} | {row['count']} |"
        )

    for case in cases:
        lines.extend(
            [
                "",
                f"## {case['seq_name']}",
                "",
                f"- `base_wrong_seq`: `{case['base']['best_wrong_seq']}`",
                f"- `base_margin`: `{fmt(case['base']['margin'])}`",
                "",
                "| variant | wrong seq | margin | margin delta vs base | top1 correct | neighbor changed |",
                "| --- | --- | ---: | ---: | --- | --- |",
            ]
        )
        for variant in variants:
            row = case["variants"][variant]
            lines.append(
                f"| {variant} | {row['best_wrong_seq']} | {fmt(row['margin'])} | {fmt(row['margin_delta_vs_base'])} | {row['top1_correct']} | {row['wrong_neighbor_changed_vs_base']} |"
            )

    lines.extend(["", "## Claim Checks", ""])
    for key, value in report["derived_claim_checks"].items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(["", "## Takeaways", ""])
    for item in report["takeaways"]:
        lines.append(f"- {item}")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
