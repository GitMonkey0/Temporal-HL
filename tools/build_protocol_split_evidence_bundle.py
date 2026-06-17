#!/usr/bin/env python3
"""Build a protocol-split evidence bundle for flat-merge vs grouped-concat symbolic variants."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from tools.compare_symbolic_slices import slice_membership


ROOT = Path("/opt/tiger/hand")
GEN = ROOT / "experiments/generated"
SUM = GEN / "summary_tables"


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def fmt(x: float) -> str:
    return f"{x:.4f}"


def load_rows(paths: list[Path], mode: str = "pretrained"):
    rows = []
    for path in paths:
        payload = load_json(path)
        seed = payload["seed"]
        for row in payload["details"][mode]["sequence_rows"]:
            rows.append({"seed": seed, **row})
    return rows


def aggregate_slice_acc_from_rows(rows: list[dict[str, object]]):
    totals = defaultdict(int)
    correct = defaultdict(int)
    for row in rows:
        for group in slice_membership(row["seq_name"]):
            totals[group] += 1
            if row["correct"]:
                correct[group] += 1
    out = {}
    for group in sorted(totals):
        out[group] = {
            "total": totals[group],
            "correct": correct[group],
            "accuracy": correct[group] / max(totals[group], 1),
        }
    return out


def diff_slices(old_slices, new_slices):
    rows = []
    for key in sorted(set(old_slices) | set(new_slices)):
        old_acc = old_slices.get(key, {}).get("accuracy", 0.0)
        new_acc = new_slices.get(key, {}).get("accuracy", 0.0)
        rows.append(
            {
                "slice": key,
                "old_accuracy": old_acc,
                "new_accuracy": new_acc,
                "delta": new_acc - old_acc,
                "old_total": old_slices.get(key, {}).get("total", 0),
                "new_total": new_slices.get(key, {}).get("total", 0),
            }
        )
    rows.sort(key=lambda row: (-row["delta"], row["slice"]))
    return rows


def error_map_from_rows(rows: list[dict[str, object]]):
    counts = Counter()
    for row in rows:
        if not row["correct"]:
            counts[(row["target"], row["prediction"])] += 1
    return dict(counts)


def compare_error_maps(flat_map, concat_map):
    rows = []
    for key in sorted(set(flat_map) | set(concat_map)):
        flat = flat_map.get(key, 0)
        concat = concat_map.get(key, 0)
        rows.append(
            {
                "target": key[0],
                "prediction": key[1],
                "flat_count": flat,
                "concat_count": concat,
                "delta_concat_minus_flat": concat - flat,
            }
        )
    rows.sort(key=lambda row: (-abs(row["delta_concat_minus_flat"]), row["target"], row["prediction"]))
    return rows


def classify_regime_winner(flat_summary, concat_summary):
    out = {}
    for frac in ("0.5", "1.0"):
        flat_seq = flat_summary[frac]["pretrained_sequence_accuracy_mean"]
        concat_seq = concat_summary[frac]["pretrained_sequence_accuracy_mean"]
        if concat_seq > flat_seq:
            winner = "grouped_concat"
        elif concat_seq < flat_seq:
            winner = "flat_merge"
        else:
            winner = "tie"
        out[frac] = {
            "flat_pretrained_sequence_accuracy": flat_seq,
            "concat_pretrained_sequence_accuracy": concat_seq,
            "winner": winner,
            "delta_concat_minus_flat": concat_seq - flat_seq,
            "flat_pretrained_window_accuracy": flat_summary[frac]["pretrained_window_accuracy_mean"],
            "concat_pretrained_window_accuracy": concat_summary[frac]["pretrained_window_accuracy_mean"],
            "window_delta_concat_minus_flat": concat_summary[frac]["pretrained_window_accuracy_mean"] - flat_summary[frac]["pretrained_window_accuracy_mean"],
        }
    return out


def main():
    flat_curve = load_json(GEN / "symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_curve_rerun_20260609.json")
    concat_curve = load_json(GEN / "symbolic_pretrain_grouped_concat_pre_only_168_84_boost_rom05.json")
    flat_full_paths = [
        GEN / "symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_seed0_analysis.json",
        GEN / "symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_seed1_analysis.json",
        GEN / "symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_seed2_analysis.json",
    ]
    flat_low_paths = [
        GEN / "symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_fraction05_seed0_analysis.json",
        GEN / "symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_fraction05_seed1_analysis.json",
        GEN / "symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_fraction05_seed2_analysis.json",
    ]
    concat_full_paths = [
        GEN / "symbolic_pretrain_grouped_concat_seed0_analysis.json",
        GEN / "symbolic_pretrain_grouped_concat_seed1_analysis.json",
        GEN / "symbolic_pretrain_grouped_concat_seed2_analysis.json",
    ]
    concat_low_paths = [
        GEN / "symbolic_pretrain_grouped_concat_fraction05_seed0_analysis.json",
        GEN / "symbolic_pretrain_grouped_concat_fraction05_seed1_analysis.json",
        GEN / "symbolic_pretrain_grouped_concat_fraction05_seed2_analysis.json",
    ]
    flat_seed1 = load_json(GEN / "symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_seed1_analysis.json")
    concat_seed1 = load_json(GEN / "symbolic_pretrain_grouped_concat_seed1_analysis.json")
    event_alignment = load_json(GEN / "hardcase_event_alignment_report.json")

    flat_full_rows = load_rows(flat_full_paths)
    flat_low_rows = load_rows(flat_low_paths)
    concat_full_rows = load_rows(concat_full_paths)
    concat_low_rows = load_rows(concat_low_paths)

    flat_slices_full = aggregate_slice_acc_from_rows(flat_full_rows)
    concat_slices_full = aggregate_slice_acc_from_rows(concat_full_rows)
    flat_slices_low = aggregate_slice_acc_from_rows(flat_low_rows)
    concat_slices_low = aggregate_slice_acc_from_rows(concat_low_rows)

    seed1_shift = []
    flat_seed1_rows = {row["seq_name"]: row for row in flat_seed1["details"]["pretrained"]["sequence_rows"]}
    concat_seed1_rows = {row["seq_name"]: row for row in concat_seed1["details"]["pretrained"]["sequence_rows"]}
    for seq_name in sorted(flat_seed1_rows):
        flat_row = flat_seed1_rows[seq_name]
        concat_row = concat_seed1_rows[seq_name]
        if flat_row["prediction"] != concat_row["prediction"] or flat_row["correct"] != concat_row["correct"]:
            seed1_shift.append(
                {
                    "seq_name": seq_name,
                    "target": flat_row["target"],
                    "flat_prediction": flat_row["prediction"],
                    "flat_correct": flat_row["correct"],
                    "concat_prediction": concat_row["prediction"],
                    "concat_correct": concat_row["correct"],
                    "slices": sorted(slice_membership(seq_name)),
                }
            )

    payload = {
        "artifacts": {
            "flat_curve": str(GEN / "symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_curve_rerun_20260609.json"),
            "concat_curve": str(GEN / "symbolic_pretrain_grouped_concat_pre_only_168_84_boost_rom05.json"),
            "flat_full_seed_analyses": [str(path) for path in flat_full_paths],
            "flat_low_seed_analyses": [str(path) for path in flat_low_paths],
            "concat_full_seed_analyses": [str(path) for path in concat_full_paths],
            "concat_low_seed_analyses": [str(path) for path in concat_low_paths],
            "flat_seed1_analysis": str(GEN / "symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_seed1_analysis.json"),
            "concat_seed1_analysis": str(GEN / "symbolic_pretrain_grouped_concat_seed1_analysis.json"),
            "event_alignment": str(GEN / "hardcase_event_alignment_report.json"),
        },
        "regime_summary": classify_regime_winner(flat_curve["summary"], concat_curve["summary"]),
        "slice_comparison": {
            "full_data": diff_slices(flat_slices_full, concat_slices_full),
            "low_data": diff_slices(flat_slices_low, concat_slices_low),
        },
        "error_shift_comparison": {
            "full_data_multiseed": compare_error_maps(error_map_from_rows(flat_full_rows), error_map_from_rows(concat_full_rows)),
            "low_data_multiseed": compare_error_maps(error_map_from_rows(flat_low_rows), error_map_from_rows(concat_low_rows)),
        },
        "seed1_prediction_shift": seed1_shift,
        "event_alignment_context": event_alignment["summary"],
        "takeaways": [
            "Under the current code audit, grouped concat beats the rerun flat-merge baseline in both low-data and full-data sequence accuracy.",
            "Grouped concat is also the stronger window-level discriminator under the current code audit.",
            "The old flat-merge mainline artifacts are stale relative to the current worktree and should not be used as authoritative evidence.",
            "Grouped concat still shows a specific full-data weakness on hard right-hand occlusion families at seed 1, but its aggregate performance remains better than the current rerun flat baseline.",
            "The event-alignment evidence still supports the broader thesis that temporal symbolic structure is especially useful on hard occlusion and neighbor-confusion cases.",
        ],
    }

    out_json = GEN / "protocol_split_evidence_bundle.json"
    out_md = SUM / "protocol_split_evidence_bundle.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Protocol-Split Evidence Bundle",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "## Regime Summary",
        "",
        "| regime | flat seq | concat seq | winner | seq delta | flat win | concat win | win delta |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for frac in ("0.5", "1.0"):
        row = payload["regime_summary"][frac]
        lines.append(
            f"| {frac} | {fmt(row['flat_pretrained_sequence_accuracy'])} | {fmt(row['concat_pretrained_sequence_accuracy'])} | {row['winner']} | {fmt(row['delta_concat_minus_flat'])} | {fmt(row['flat_pretrained_window_accuracy'])} | {fmt(row['concat_pretrained_window_accuracy'])} | {fmt(row['window_delta_concat_minus_flat'])} |"
        )

    lines.extend(["", "## Slice Deltas: Concat vs Flat", "", "### Full Data", "", "| slice | flat | concat | delta |", "| --- | ---: | ---: | ---: |"])
    for row in payload["slice_comparison"]["full_data"]:
        lines.append(f"| {row['slice']} | {fmt(row['old_accuracy'])} | {fmt(row['new_accuracy'])} | {fmt(row['delta'])} |")
    lines.extend(["", "### Low Data", "", "| slice | flat | concat | delta |", "| --- | ---: | ---: | ---: |"])
    for row in payload["slice_comparison"]["low_data"]:
        lines.append(f"| {row['slice']} | {fmt(row['old_accuracy'])} | {fmt(row['new_accuracy'])} | {fmt(row['delta'])} |")

    lines.extend(["", "## Seed-1 Prediction Shifts", ""])
    for row in payload["seed1_prediction_shift"]:
        lines.append(
            f"- `{row['seq_name']}`: flat=`{row['flat_prediction']}` ({row['flat_correct']}), concat=`{row['concat_prediction']}` ({row['concat_correct']}), slices={row['slices']}"
        )

    lines.extend(["", "## Event Alignment Context", ""])
    for key, value in payload["event_alignment_context"].items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(["", "## Takeaways", ""])
    for item in payload["takeaways"]:
        lines.append(f"- {item}")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
