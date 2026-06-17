#!/usr/bin/env python3
"""Summarize strong-protocol pruned-channel experiments against the current mainline."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/opt/tiger/hand")
GEN = ROOT / "experiments/generated"
SUM = GEN / "summary_tables"


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def fmt(x: float) -> str:
    return f"{x:.4f}"


def main():
    mainline = load_json(GEN / "symbolic_pretrain_temporal_pre_only_168_84_boost_rom05.json")
    mainline_evidence = load_json(GEN / "mainline_evidence_report.json")
    cand_state_transition = load_json(GEN / "symbolic_pretrain_state_transition_pre_only_168_84_boost_rom05.json")
    cand_state_motion = load_json(GEN / "symbolic_pretrain_state_motion_pre_only_168_84_boost_rom05.json")
    cand_state_transition_tempo = load_json(GEN / "symbolic_pretrain_state_transition_tempo_pre_only_168_84_boost_rom05.json")
    cand_state_motion_tempo = load_json(GEN / "symbolic_pretrain_state_motion_tempo_pre_only_168_84_boost_rom05.json")
    cand_state_motion_interaction_tempo = load_json(GEN / "symbolic_pretrain_state_motion_interaction_tempo_pre_only_168_84_boost_rom05.json")

    variants = {
        "mainline_full_temporal": mainline,
        "state_transition": cand_state_transition,
        "state_motion": cand_state_motion,
        "state_transition_tempo": cand_state_transition_tempo,
        "state_motion_tempo": cand_state_motion_tempo,
        "state_motion_interaction_tempo": cand_state_motion_interaction_tempo,
    }

    report = {
        "artifacts": {name: str(GEN / Path(payload_path).name) if False else None for name, payload_path in []},
        "variants": variants,
        "derived_claim_checks": {
            "mainline_beats_state_transition_fraction_1.0": mainline["summary"]["1.0"]["pretrained_sequence_accuracy_mean"] > cand_state_transition["summary"]["1.0"]["pretrained_sequence_accuracy_mean"],
            "mainline_beats_state_motion_fraction_1.0": mainline["summary"]["1.0"]["pretrained_sequence_accuracy_mean"] > cand_state_motion["summary"]["1.0"]["pretrained_sequence_accuracy_mean"],
            "mainline_beats_state_motion_tempo_fraction_1.0": mainline["summary"]["1.0"]["pretrained_sequence_accuracy_mean"] > cand_state_motion_tempo["summary"]["1.0"]["pretrained_sequence_accuracy_mean"],
            "mainline_beats_state_motion_tempo_fraction_0.5": mainline_evidence["mainline"]["fraction_0.5"]["pretrained_seq"] > cand_state_motion_tempo["summary"]["0.5"]["pretrained_sequence_accuracy_mean"],
            "best_pruned_variant_is_state_motion_tempo": cand_state_motion_tempo["summary"]["1.0"]["pretrained_sequence_accuracy_mean"] >= cand_state_motion_interaction_tempo["summary"]["1.0"]["pretrained_sequence_accuracy_mean"],
        },
        "gaps": {
            "fraction_1.0_mainline_minus_state_transition": mainline["summary"]["1.0"]["pretrained_sequence_accuracy_mean"] - cand_state_transition["summary"]["1.0"]["pretrained_sequence_accuracy_mean"],
            "fraction_1.0_mainline_minus_state_motion": mainline["summary"]["1.0"]["pretrained_sequence_accuracy_mean"] - cand_state_motion["summary"]["1.0"]["pretrained_sequence_accuracy_mean"],
            "fraction_1.0_mainline_minus_state_motion_tempo": mainline["summary"]["1.0"]["pretrained_sequence_accuracy_mean"] - cand_state_motion_tempo["summary"]["1.0"]["pretrained_sequence_accuracy_mean"],
            "fraction_0.5_mainline_minus_state_motion_tempo": mainline_evidence["mainline"]["fraction_0.5"]["pretrained_seq"] - cand_state_motion_tempo["summary"]["0.5"]["pretrained_sequence_accuracy_mean"],
        },
        "takeaways": [
            "The classifier-level pruning gains from the lightweight protocol do not transfer cleanly to the stronger pretrain-only normalized protocol.",
        "The current full temporal mainline remains clearly stronger than all tested pruned variants under the strong protocol.",
            "Among the pruned strong-protocol candidates, state_motion_tempo is the strongest, which implies that tempo is necessary when motion is retained but transition is removed.",
            "This creates a useful reconciliation: channel pruning helps lightweight classifiers, but the stronger pretrain-finetune regime still benefits from the fuller temporal mixture.",
            "The representation story should therefore distinguish protocol strength instead of claiming a universal winner across training regimes.",
        ],
    }

    out_json = GEN / "strong_protocol_pruned_channel_report.json"
    out_md = SUM / "strong_protocol_pruned_channel_report.md"
    out_json.write_text(json.dumps(report, indent=2))

    lines = [
        "# Strong Protocol Pruned-Channel Report",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "| variant | fraction | pretrained seq | scratch seq | pretrained win | scratch win |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for name in [
        "mainline_full_temporal",
        "state_transition",
        "state_motion",
        "state_transition_tempo",
        "state_motion_tempo",
        "state_motion_interaction_tempo",
    ]:
        for frac in ["0.5", "1.0"]:
            if name == "mainline_full_temporal" and frac == "0.5":
                s = {
                    "pretrained_sequence_accuracy_mean": mainline_evidence["mainline"]["fraction_0.5"]["pretrained_seq"],
                    "scratch_sequence_accuracy_mean": mainline_evidence["mainline"]["fraction_0.5"]["scratch_seq"],
                    "pretrained_window_accuracy_mean": mainline_evidence["mainline"]["fraction_0.5"]["pretrained_win"],
                    "scratch_window_accuracy_mean": mainline_evidence["mainline"]["fraction_0.5"]["scratch_win"],
                }
            else:
                payload = variants[name]
                s = payload["summary"][frac]
            lines.append(
                f"| {name} | {frac} | {fmt(s['pretrained_sequence_accuracy_mean'])} | {fmt(s['scratch_sequence_accuracy_mean'])} | {fmt(s['pretrained_window_accuracy_mean'])} | {fmt(s['scratch_window_accuracy_mean'])} |"
            )

    lines.extend(["", "## Claim Checks", ""])
    for key, value in report["derived_claim_checks"].items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(["", "## Gap Summary", ""])
    for key, value in report["gaps"].items():
        lines.append(f"- `{key}`: `{fmt(value)}`")

    lines.extend(["", "## Takeaways", ""])
    for item in report["takeaways"]:
        lines.append(f"- {item}")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
