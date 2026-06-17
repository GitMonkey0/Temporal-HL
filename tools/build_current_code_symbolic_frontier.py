#!/usr/bin/env python3
"""Build a consolidated current-code symbolic frontier report."""

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


def fmt(x):
    if x is None:
        return "n/a"
    return f"{x:.4f}"


def corrected_stats(base_seq_acc: float, improved: int, harmed: int, total_rows: int = TOTAL_SEQ_ROWS):
    base_correct = round(base_seq_acc * total_rows)
    new_correct = base_correct + improved - harmed
    return {
        "base_num_correct": base_correct,
        "new_num_correct": new_correct,
        "new_seq_acc": new_correct / total_rows,
    }


def best_zero_harm(payload: dict[str, object]):
    return payload["best_zero_harm"][0]


def main():
    flat = load_json(GEN / "symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_curve_rerun_20260609.json")
    grouped_main = load_json(GEN / "symbolic_pretrain_grouped_concat_pre_only_168_84_boost_rom05.json")
    grouped_025 = load_json(GEN / "symbolic_pretrain_grouped_concat_fraction025_pre_only_168_84_boost_rom05.json")
    fx_025 = load_json(GEN / "grouped_concat_fraction025_family_expert.json")
    fx_05 = load_json(GEN / "grouped_concat_fraction05_family_expert.json")
    fx_10 = load_json(GEN / "grouped_concat_family_expert.json")

    rows = []
    for fraction, grouped_summary, fx_payload in [
        ("0.25", grouped_025["summary"]["0.25"], fx_025),
        ("0.5", grouped_main["summary"]["0.5"], fx_05),
        ("1.0", grouped_main["summary"]["1.0"], fx_10),
    ]:
        flat_summary = flat["summary"].get(fraction)
        fx_best = best_zero_harm(fx_payload)
        corr = corrected_stats(
            grouped_summary["pretrained_sequence_accuracy_mean"],
            fx_best["counts"]["num_improved"],
            fx_best["counts"]["num_harmed"],
        )
        rows.append(
            {
                "fraction": fraction,
                "flat_seq": None if flat_summary is None else flat_summary["pretrained_sequence_accuracy_mean"],
                "flat_win": None if flat_summary is None else flat_summary["pretrained_window_accuracy_mean"],
                "grouped_seq": grouped_summary["pretrained_sequence_accuracy_mean"],
                "grouped_win": grouped_summary["pretrained_window_accuracy_mean"],
                "family_threshold": fx_best["threshold"],
                "family_improved": fx_best["counts"]["num_improved"],
                "family_harmed": fx_best["counts"]["num_harmed"],
                "family_seq": corr["new_seq_acc"],
                "family_correct": corr["new_num_correct"],
                "delta_grouped_minus_flat": None
                if flat_summary is None
                else grouped_summary["pretrained_sequence_accuracy_mean"] - flat_summary["pretrained_sequence_accuracy_mean"],
                "delta_family_minus_grouped": corr["new_seq_acc"] - grouped_summary["pretrained_sequence_accuracy_mean"],
            }
        )

    payload = {
        "date": "2026-06-09",
        "artifacts": {
            "flat_rerun": str(GEN / "symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_curve_rerun_20260609.json"),
            "grouped_concat_025": str(GEN / "symbolic_pretrain_grouped_concat_fraction025_pre_only_168_84_boost_rom05.json"),
            "grouped_concat_05_10": str(GEN / "symbolic_pretrain_grouped_concat_pre_only_168_84_boost_rom05.json"),
            "family_expert_025": str(GEN / "grouped_concat_fraction025_family_expert.json"),
            "family_expert_05": str(GEN / "grouped_concat_fraction05_family_expert.json"),
            "family_expert_10": str(GEN / "grouped_concat_family_expert.json"),
        },
        "rows": rows,
        "takeaways": [
            "Under the current-code audit, grouped-concat is the strongest structural symbolic baseline.",
            "Where a direct flat rerun exists, grouped-concat improves sequence accuracy by +0.2051 at both 0.5 and 1.0 fractions.",
            "Family-expert correction adds a second gain layer on top of grouped-concat: +0.3590 at 0.25, +0.1026 at 0.5, and +0.1026 at 1.0.",
            "The correction gain grows as the symbolic base weakens, while remaining zero-harm on the tested thresholds.",
        ],
    }

    out_json = GEN / "current_code_symbolic_frontier.json"
    out_md = SUM / "current_code_symbolic_frontier.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Current-Code Symbolic Frontier",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "| fraction | flat rerun seq | flat rerun win | grouped seq | grouped win | grouped-flat | family thr | family improved | family harmed | family seq | family-grouped |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['fraction']} | {fmt(row['flat_seq'])} | {fmt(row['flat_win'])} | "
            f"{fmt(row['grouped_seq'])} | {fmt(row['grouped_win'])} | {fmt(row['delta_grouped_minus_flat'])} | "
            f"{row['family_threshold']:.1f} | {row['family_improved']} | {row['family_harmed']} | "
            f"{fmt(row['family_seq'])} | {fmt(row['delta_family_minus_grouped'])} |"
        )

    lines.extend(["", "## Takeaways", ""])
    for item in payload["takeaways"]:
        lines.append(f"- {item}")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
