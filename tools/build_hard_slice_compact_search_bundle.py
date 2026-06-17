#!/usr/bin/env python3
"""Build a compact evidence bundle for hard-slice compact search."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/opt/tiger/hand")
GEN = ROOT / "experiments/generated"
SUM = GEN / "summary_tables"


def load(name: str):
    return json.loads((GEN / name).read_text())


def fmt(x: float) -> str:
    return f"{x:.4f}"


def main():
    weak_topk = load("weak_slice_topk_joint_search.json")
    weak_full = load("weak_slice_full_joint_search.json")
    weak_scaling = load("weak_slice_relaxed_search_scaling.json")
    weak_learned = load("weak_slice_learned_reranker.json")
    weak_pairguided = load("weak_slice_pairguided_reranker.json")
    multislice = load("pairguided_reranker_multislice.json")

    opening_summary = {
        "baseline_single_donor_joint_overall": weak_topk["summary"]["baseline_joint_score"],
        "split_donor_joint_on_available": weak_topk["summary"]["split_joint_score"],
        "topk_exact_joint_on_available": weak_topk["summary"]["topk_joint_score"],
        "relaxed_top5_joint_overall": weak_scaling["summary"]["left_top_5"]["joint_hit_rate_overall"],
        "relaxed_top10_joint_overall": weak_scaling["summary"]["left_top_10"]["joint_hit_rate_overall"],
        "relaxed_top20_joint_overall": weak_scaling["summary"]["left_top_20"]["joint_hit_rate_overall"],
        "learned_top10_joint_overall": weak_learned["summary"]["learned_top10"]["joint_hit_rate_overall"],
        "pairguided_top5_joint_overall": weak_pairguided["summary"]["pairguided_top5"]["joint_hit_rate_overall"],
        "pairguided_top10_joint_overall": weak_pairguided["summary"]["pairguided_top10"]["joint_hit_rate_overall"],
    }

    hard_slice_rows = []
    for task in ("right_hand_motion->closing", "right_hand_motion->opening"):
        summary = multislice["task_results"][task]["summary"]
        hard_slice_rows.append(
            {
                "task": task,
                "base_top5": summary["base_top5"]["joint_hit_rate_overall"],
                "base_top10": summary["base_top10"]["joint_hit_rate_overall"],
                "base_top20": summary["base_top20"]["joint_hit_rate_overall"],
                "pairguided_top5": summary["pairguided_top5"]["joint_hit_rate_overall"],
                "pairguided_top10": summary["pairguided_top10"]["joint_hit_rate_overall"],
            }
        )

    easy_slice_rows = []
    for task in ("left_hand_motion->closing", "left_hand_motion->opening"):
        summary = multislice["task_results"][task]["summary"]
        easy_slice_rows.append(
            {
                "task": task,
                "base_top5": summary["base_top5"]["joint_hit_rate_overall"],
                "base_top10": summary["base_top10"]["joint_hit_rate_overall"],
                "base_top20": summary["base_top20"]["joint_hit_rate_overall"],
                "pairguided_top5": summary["pairguided_top5"]["joint_hit_rate_overall"],
                "pairguided_top10": summary["pairguided_top10"]["joint_hit_rate_overall"],
            }
        )

    reviewer_attack_map = {
        "not_just_more_search": {
            "evidence": [
                "pairguided_top10 == base_top20 on both hard right-hand interaction slices",
                "pairguided_top5 == base_top10 on right opening; pairguided_top5 > base_top5 on right closing",
            ]
        },
        "not_just_lookup_heuristic": {
            "evidence": [
                "coordination-aware hand-built reranks did not recover top20",
                "left-only learned reranker compresses top20->top10 only on the weak slice",
                "pair-guided learned reranker repeats the compression pattern across both hard right-hand slices",
            ]
        },
        "hard_slice_specific_not_universal": {
            "evidence": [
                "left-hand interaction slices are already easier and do not need the same mechanism",
                "pair-guided gains concentrate on the hard right-hand interaction regime",
            ]
        },
    }

    payload = {
        "focus": {
            "bundle": "hard-slice compact-search evidence",
            "primary_claim": "pair-guided learned reranking compresses search budget on hard right-hand interaction slices",
        },
        "weak_opening_progression": opening_summary,
        "hard_right_hand_slices": hard_slice_rows,
        "left_hand_reference_slices": easy_slice_rows,
        "source_artifacts": {
            "weak_slice_topk_joint_search": str(GEN / "weak_slice_topk_joint_search.json"),
            "weak_slice_full_joint_search": str(GEN / "weak_slice_full_joint_search.json"),
            "weak_slice_relaxed_search_scaling": str(GEN / "weak_slice_relaxed_search_scaling.json"),
            "weak_slice_learned_reranker": str(GEN / "weak_slice_learned_reranker.json"),
            "weak_slice_pairguided_reranker": str(GEN / "weak_slice_pairguided_reranker.json"),
            "pairguided_reranker_multislice": str(GEN / "pairguided_reranker_multislice.json"),
        },
        "reviewer_attack_map": reviewer_attack_map,
    }

    out_json = GEN / "hard_slice_compact_search_bundle.json"
    out_md = SUM / "hard_slice_compact_search_bundle.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Hard-Slice Compact Search Bundle",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Primary claim: pair-guided learned reranking compresses search budget on",
        "hard right-hand interaction slices without needing universal gains on all",
        "interaction slices.",
        "",
        "## Weak-Slice Progression: `right_hand_motion -> opening`",
        "",
        "| stage | overall joint hit |",
        "| --- | ---: |",
        f"| baseline single donor | {fmt(opening_summary['baseline_single_donor_joint_overall'])} |",
        f"| relaxed top-5 | {fmt(opening_summary['relaxed_top5_joint_overall'])} |",
        f"| relaxed top-10 | {fmt(opening_summary['relaxed_top10_joint_overall'])} |",
        f"| relaxed top-20 | {fmt(opening_summary['relaxed_top20_joint_overall'])} |",
        f"| learned top-10 | {fmt(opening_summary['learned_top10_joint_overall'])} |",
        f"| pair-guided top-5 | {fmt(opening_summary['pairguided_top5_joint_overall'])} |",
        f"| pair-guided top-10 | {fmt(opening_summary['pairguided_top10_joint_overall'])} |",
        "",
        "## Hard Right-Hand Slices",
        "",
        "| task | base top-5 | base top-10 | base top-20 | pair-guided top-5 | pair-guided top-10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in hard_slice_rows:
        lines.append(
            f"| {row['task']} | {fmt(row['base_top5'])} | {fmt(row['base_top10'])} | "
            f"{fmt(row['base_top20'])} | {fmt(row['pairguided_top5'])} | {fmt(row['pairguided_top10'])} |"
        )

    lines.extend(
        [
            "",
            "## Left-Hand Reference Slices",
            "",
            "| task | base top-5 | base top-10 | base top-20 | pair-guided top-5 | pair-guided top-10 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in easy_slice_rows:
        lines.append(
            f"| {row['task']} | {fmt(row['base_top5'])} | {fmt(row['base_top10'])} | "
            f"{fmt(row['base_top20'])} | {fmt(row['pairguided_top5'])} | {fmt(row['pairguided_top10'])} |"
        )

    lines.extend(
        [
            "",
            "## Reviewer Attack Map",
            "",
            "### Not Just More Search",
            "",
            "- `pairguided_top10 == base_top20` on both hard right-hand interaction slices.",
            "- `pairguided_top5` recovers most of the gap from `base_top5` to deeper search.",
            "",
            "### Not Just A Lookup Heuristic",
            "",
            "- Hand-built coordination reranks did not recover the deep-search gain.",
            "- Left-only learned reranking compressed only part of the budget.",
            "- Pair-guided supervision repeats the compression pattern across both hard right-hand slices.",
            "",
            "### Hard-Slice Specific, Not Universal",
            "",
            "- The mechanism matters most on hard right-hand interaction slices.",
            "- Easier left-hand interaction slices already respond well to plain search depth.",
        ]
    )

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
