#!/usr/bin/env python3
"""Summarize constrained strong-protocol weight search around the current mainline."""

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
    variants = {
        "motion_up": load_json(GEN / "symbolic_pretrain_weight_motion_up_full.json"),
        "motion_up_trans_down": load_json(GEN / "symbolic_pretrain_weight_motion_up_trans_down_full.json"),
        "motion_up_it_down": load_json(GEN / "symbolic_pretrain_weight_motion_up_it_down_full.json"),
        "trans_down_it_down": load_json(GEN / "symbolic_pretrain_weight_trans_down_it_down_full.json"),
        "balanced_it_down": load_json(GEN / "symbolic_pretrain_weight_balanced_it_down_full.json"),
        "motion_up_trans_down_it_down": load_json(GEN / "symbolic_pretrain_weight_motion_up_trans_down_it_down_full.json"),
    }
    motion_up_it_down_05 = load_json(GEN / "symbolic_pretrain_weight_motion_up_it_down_fraction05.json")

    full_baseline = mainline["summary"]["1.0"]["pretrained_sequence_accuracy_mean"]
    low_baseline = mainline_evidence["mainline"]["fraction_0.5"]["pretrained_seq"]

    best_full_name, best_full_payload = max(
        variants.items(),
        key=lambda kv: kv[1]["summary"]["1.0"]["pretrained_sequence_accuracy_mean"],
    )

    report = {
        "variants": variants,
        "lowdata_probe": motion_up_it_down_05,
        "derived_claim_checks": {
            "no_tested_reweight_beats_mainline_full": all(
                payload["summary"]["1.0"]["pretrained_sequence_accuracy_mean"] < full_baseline
                for payload in variants.values()
            ),
            "best_full_reweight_is_motion_up_it_down": best_full_name == "motion_up_it_down",
            "best_full_reweight_still_far_below_mainline": full_baseline - best_full_payload["summary"]["1.0"]["pretrained_sequence_accuracy_mean"] > 0.09,
            "motion_up_it_down_lowdata_still_below_mainline": low_baseline > motion_up_it_down_05["summary"]["0.5"]["pretrained_sequence_accuracy_mean"],
        },
        "gaps": {
            "best_full_reweight_gap_to_mainline": full_baseline - best_full_payload["summary"]["1.0"]["pretrained_sequence_accuracy_mean"],
            "motion_up_it_down_lowdata_gap_to_mainline": low_baseline - motion_up_it_down_05["summary"]["0.5"]["pretrained_sequence_accuracy_mean"],
        },
        "takeaways": [
            "A constrained weight search around the current strong-protocol mainline does not find a better temporal mixture.",
            "The best full-data local reweight is motion_up_it_down, but it still stays well below the current mainline.",
            "This suggests the strong protocol is using the fuller temporal mixture in a way that cannot be improved by simple scalar reweighting alone.",
            "The next strong-protocol upgrade should therefore target representation structure or branch interaction rather than only channel weights.",
        ],
    }

    out_json = GEN / "strong_protocol_weight_search_report.json"
    out_md = SUM / "strong_protocol_weight_search_report.md"
    out_json.write_text(json.dumps(report, indent=2))

    lines = [
        "# Strong Protocol Weight Search Report",
        "",
        "This is an experiment memo, not paper text.",
        "",
        f"- `mainline_full_pretrained_seq`: `{fmt(full_baseline)}`",
        f"- `mainline_lowdata_pretrained_seq`: `{fmt(low_baseline)}`",
        "",
        "| variant | full pretrained seq | full scratch seq | full pretrained win |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, payload in variants.items():
        s = payload["summary"]["1.0"]
        lines.append(
            f"| {name} | {fmt(s['pretrained_sequence_accuracy_mean'])} | {fmt(s['scratch_sequence_accuracy_mean'])} | {fmt(s['pretrained_window_accuracy_mean'])} |"
        )

    lines.extend(
        [
            "",
            "## Low-Data Probe",
            "",
            f"- `motion_up_it_down` low-data pretrained seq: `{fmt(motion_up_it_down_05['summary']['0.5']['pretrained_sequence_accuracy_mean'])}`",
            "",
            "## Claim Checks",
            "",
        ]
    )
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
