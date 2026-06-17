#!/usr/bin/env python3
"""Build a matched old-HL vs temporal-HL comparison report."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from tools.train_symbolic_classifier import canonical_label


ROOT = Path("/opt/tiger/hand")
GEN = ROOT / "experiments/generated"
SUM = GEN / "summary_tables"


def load(name: str):
    return json.loads((GEN / name).read_text())


def fmt(x: float) -> str:
    return f"{x:.4f}"


def mean_acc(obj, key):
    runs = obj["runs"]
    return sum(run[key] for run in runs) / len(runs)


def aggregate_seq(obj):
    out = defaultdict(lambda: {"pretrained_correct_runs": 0, "scratch_correct_runs": 0})
    for seq, row in obj["sequence_consistency"].items():
        key = canonical_label(seq)
        out[key]["pretrained_correct_runs"] += row["pretrained_correct_runs"]
        out[key]["scratch_correct_runs"] += row["scratch_correct_runs"]
    return out


def main():
    state = load("symbolic_pretrain_state_pre_only_168_84_boost_rom05_multiseed_analysis.json")
    temporal = load("symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_multiseed_analysis.json")

    state_seq = aggregate_seq(state)
    temporal_seq = aggregate_seq(temporal)
    seq_rows = []
    for seq in sorted(set(state_seq) | set(temporal_seq)):
        s_pre = state_seq[seq]["pretrained_correct_runs"]
        t_pre = temporal_seq[seq]["pretrained_correct_runs"]
        s_scratch = state_seq[seq]["scratch_correct_runs"]
        t_scratch = temporal_seq[seq]["scratch_correct_runs"]
        seq_rows.append(
            {
                "seq_name": seq,
                "state_pretrained_correct_runs": s_pre,
                "temporal_pretrained_correct_runs": t_pre,
                "delta_pretrained": t_pre - s_pre,
                "state_scratch_correct_runs": s_scratch,
                "temporal_scratch_correct_runs": t_scratch,
            }
        )

    summary = {
        "matched_protocol": {
            "pretrain_window_span_units": state["pretrain_window_span_units"],
            "pretrain_window_step_units": state["pretrain_window_step_units"],
            "fraction": state["fraction"],
            "seeds": state["seeds"],
            "boost_map": state["boost_map"],
            "hidden_dim": 128,
            "pretrain_epochs": 120,
            "finetune_epochs": 120,
        },
        "state": {
            "scratch_mean_sequence_accuracy": mean_acc(state, "scratch_sequence_accuracy"),
            "pretrained_mean_sequence_accuracy": mean_acc(state, "pretrained_sequence_accuracy"),
        },
        "temporal": {
            "scratch_mean_sequence_accuracy": mean_acc(temporal, "scratch_sequence_accuracy"),
            "pretrained_mean_sequence_accuracy": mean_acc(temporal, "pretrained_sequence_accuracy"),
        },
        "delta": {
            "scratch_temporal_minus_state": mean_acc(temporal, "scratch_sequence_accuracy") - mean_acc(state, "scratch_sequence_accuracy"),
            "pretrained_temporal_minus_state": mean_acc(temporal, "pretrained_sequence_accuracy") - mean_acc(state, "pretrained_sequence_accuracy"),
        },
        "sequence_comparison": {
            "temporal_beats_state": sum(row["delta_pretrained"] > 0 for row in seq_rows),
            "state_beats_temporal": sum(row["delta_pretrained"] < 0 for row in seq_rows),
            "tie": sum(row["delta_pretrained"] == 0 for row in seq_rows),
        },
    }

    payload = {
        "focus": {
            "goal": "matched old-HL vs temporal-HL comparison under the same strong encoder/training protocol",
            "claim_boundary": "plain sequence classification alone does not establish the temporal-HL advantage",
        },
        "summary": summary,
        "sequence_rows": seq_rows,
        "source_artifacts": {
            "state_multiseed": str(GEN / "symbolic_pretrain_state_pre_only_168_84_boost_rom05_multiseed_analysis.json"),
            "temporal_multiseed": str(GEN / "symbolic_pretrain_temporal_pre_only_168_84_boost_rom05_multiseed_analysis.json"),
        },
    }

    out_json = GEN / "oldhl_temporal_matched_report.json"
    out_md = SUM / "oldhl_temporal_matched_report.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Old-HL vs Temporal-HL Matched Report",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "Same protocol, different representation channels:",
        "- same pretrain -> finetune split",
        "- same `168/84` pretrain window schedule",
        "- same hidden size / epochs / aggregation / class boost",
        "- only `mode=state` vs `mode=temporal` changes",
        "",
        "## Summary",
        "",
        "| model | scratch seq | pretrained seq |",
        "| --- | ---: | ---: |",
        f"| old HL (`state`) | {fmt(summary['state']['scratch_mean_sequence_accuracy'])} | {fmt(summary['state']['pretrained_mean_sequence_accuracy'])} |",
        f"| temporal HL (`temporal`) | {fmt(summary['temporal']['scratch_mean_sequence_accuracy'])} | {fmt(summary['temporal']['pretrained_mean_sequence_accuracy'])} |",
        "",
        "| delta | value |",
        "| --- | ---: |",
        f"| scratch temporal - state | {fmt(summary['delta']['scratch_temporal_minus_state'])} |",
        f"| pretrained temporal - state | {fmt(summary['delta']['pretrained_temporal_minus_state'])} |",
        "",
        "## Sequence Comparison",
        "",
        "| outcome | count |",
        "| --- | ---: |",
        f"| temporal beats state | {summary['sequence_comparison']['temporal_beats_state']} |",
        f"| state beats temporal | {summary['sequence_comparison']['state_beats_temporal']} |",
        f"| tie | {summary['sequence_comparison']['tie']} |",
        "",
        "## Interpretation",
        "",
        "- This matched comparison does **not** show a simple classification win for temporal HL.",
        "- Therefore the current mainline should not argue that temporal HL wins merely because the same encoder classifies it better.",
        "- The stronger temporal story must continue to rest on:",
        "  - transition-conditioned control",
        "  - hard-slice compact search",
        "  - interaction-aware editing advantages",
        "",
        "## Per Sequence",
        "",
        "| sequence | state pre | temporal pre | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in seq_rows:
        lines.append(
            f"| {row['seq_name']} | {row['state_pretrained_correct_runs']} | "
            f"{row['temporal_pretrained_correct_runs']} | {row['delta_pretrained']} |"
        )

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
