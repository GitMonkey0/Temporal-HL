#!/usr/bin/env python3
"""Build a robustness report for grouped-concat + family-expert across fractions."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/opt/tiger/hand")
GEN = ROOT / "experiments/generated"
SUM = GEN / "summary_tables"
TOTAL_SEQ_ROWS = 39


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def fmt(x: float) -> str:
    return f"{x:.4f}"


def corrected_overall(base_seq_acc: float, num_improved: int, num_harmed: int, total_rows: int = TOTAL_SEQ_ROWS):
    base_correct = round(base_seq_acc * total_rows)
    new_correct = base_correct + num_improved - num_harmed
    return {
        "base_num_correct": base_correct,
        "new_num_correct": new_correct,
        "new_overall_sequence_accuracy": new_correct / total_rows,
    }


def select_best_zero_harm(payload: dict[str, object]):
    return payload["best_zero_harm"][0]


def main():
    base_025 = load_json(GEN / "symbolic_pretrain_grouped_concat_fraction025_pre_only_168_84_boost_rom05.json")
    base_05 = load_json(GEN / "symbolic_pretrain_grouped_concat_pre_only_168_84_boost_rom05.json")
    base_10 = base_05
    fx_025 = load_json(GEN / "grouped_concat_fraction025_family_expert.json")
    fx_05 = load_json(GEN / "grouped_concat_fraction05_family_expert.json")
    fx_10 = load_json(GEN / "grouped_concat_family_expert.json")

    rows = []
    configs = [
        ("0.25", base_025["summary"]["0.25"], select_best_zero_harm(fx_025), fx_025),
        ("0.5", base_05["summary"]["0.5"], select_best_zero_harm(fx_05), fx_05),
        ("1.0", base_10["summary"]["1.0"], select_best_zero_harm(fx_10), fx_10),
    ]

    for frac, base_summary, best, full_payload in configs:
        corr = corrected_overall(
            base_summary["pretrained_sequence_accuracy_mean"],
            best["counts"]["num_improved"],
            best["counts"]["num_harmed"],
        )
        stable_zero_harm = [row["threshold"] for row in full_payload["best_zero_harm"]]
        rows.append(
            {
                "fraction": frac,
                "base_pretrained_sequence_accuracy": base_summary["pretrained_sequence_accuracy_mean"],
                "base_pretrained_window_accuracy": base_summary["pretrained_window_accuracy_mean"],
                "best_zero_harm_threshold": best["threshold"],
                "stable_zero_harm_thresholds": stable_zero_harm,
                "family_accuracy_old": best["family_accuracy"]["old"],
                "family_accuracy_new": best["family_accuracy"]["new"],
                "num_family_rows": best["counts"]["num_family_rows"],
                "num_used_expert": best["counts"]["num_used_expert"],
                "num_improved": best["counts"]["num_improved"],
                "num_harmed": best["counts"]["num_harmed"],
                **corr,
            }
        )

    payload = {
        "artifacts": {
            "base_025": str(GEN / "symbolic_pretrain_grouped_concat_fraction025_pre_only_168_84_boost_rom05.json"),
            "base_05_10": str(GEN / "symbolic_pretrain_grouped_concat_pre_only_168_84_boost_rom05.json"),
            "family_expert_025": str(GEN / "grouped_concat_fraction025_family_expert.json"),
            "family_expert_05": str(GEN / "grouped_concat_fraction05_family_expert.json"),
            "family_expert_10": str(GEN / "grouped_concat_family_expert.json"),
        },
        "rows": rows,
        "takeaways": [
            "Grouped-concat + family expert remains zero-harm across 0.25, 0.5, and 1.0 fractions.",
            "The correction effect strengthens as the symbolic base gets weaker; the largest jump appears at fraction 0.25.",
            "Threshold stability is broad: 0.3 and 0.4 are zero-harm across all three fractions, and 0.5 remains zero-harm at 0.5 and 1.0.",
            "Under the audited current-code pipeline, grouped-concat + family expert is now a robust frontier rather than a single-regime accident.",
        ],
    }

    out_json = GEN / "grouped_concat_family_expert_robustness.json"
    out_md = SUM / "grouped_concat_family_expert_robustness.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Grouped Concat + Family Expert Robustness",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "| fraction | base seq | corrected seq | base win | best thr | improved | harmed | family old | family new | stable zero-harm thresholds |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['fraction']} | {fmt(row['base_pretrained_sequence_accuracy'])} | {fmt(row['new_overall_sequence_accuracy'])} | {fmt(row['base_pretrained_window_accuracy'])} | {row['best_zero_harm_threshold']} | {row['num_improved']} | {row['num_harmed']} | {fmt(row['family_accuracy_old'])} | {fmt(row['family_accuracy_new'])} | {row['stable_zero_harm_thresholds']} |"
        )

    lines.extend(["", "## Takeaways", ""])
    for item in payload["takeaways"]:
        lines.append(f"- {item}")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
