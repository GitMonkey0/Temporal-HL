#!/usr/bin/env python3
"""Build a unified control-facing evidence bundle."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


ROOT = Path("/opt/tiger/hand")
GEN = ROOT / "experiments/generated"
SUM = GEN / "summary_tables"


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def fmt(x: float) -> str:
    return f"{x:.4f}"


def stability_section(stability: dict[str, object]):
    shared = stability["shared_zero_harm_thresholds"]
    recurring = stability["recurring_cases_across_fractions"][:4]
    top_slices = {}
    for frac, rows in stability["per_fraction_slice_summaries"].items():
        top_slices[frac] = {row["slice"]: row for row in rows}
    return {
        "shared_zero_harm_thresholds": shared,
        "recurring_cases": recurring,
        "slice_support": {
            frac: {
                "all_mean_delta": top_slices[frac]["all"]["mean_delta"],
                "occlusion_mean_delta": top_slices[frac]["occlusion"]["mean_delta"],
                "finger_occlusion_mean_delta": top_slices[frac]["finger_occlusion"]["mean_delta"],
            }
            for frac in ("0.25", "0.5", "1.0")
        },
    }


def gallery_shift_section(gallery: dict[str, object]):
    rows = gallery["rows"]
    by_fraction = defaultdict(list)
    for row in rows:
        by_fraction[row["fraction"]].append(row)
    out = {}
    for frac, items in by_fraction.items():
        out[frac] = {
            "all_variants_zero_harm": all(item["has_zero_harm"] for item in items),
            "variants": {
                item["variant"]: {
                    "family_old": item["family_old"],
                    "family_new": item["family_new"],
                    "num_improved": item["num_improved"],
                    "num_harmed": item["num_harmed"],
                }
                for item in items
            },
        }
    return out


def local_edit_section(audit: dict[str, object]):
    out = {}
    for source in audit["sources"]:
        by_task = {}
        for row in source["task_summary"]:
            task = f"{row['task_field']}->{row['task_target']}"
            by_task[task] = {
                "symbolic_clean_edit_rate": row["symbolic_clean_edit_rate"],
                "symbolic_mean_collateral_fields": row["symbolic_mean_collateral_fields"],
                "proxy_clean_edit_rate": row["proxy_clean_edit_rate"],
                "proxy_mean_collateral_fields": row["proxy_mean_collateral_fields"],
                "proxy_target_success_rate": row["proxy_target_success_rate"],
            }
        out[source["source"]] = by_task
    return out


def casebook_section(casebook: dict[str, object]):
    out = {}
    for source in casebook["sources"]:
        cases = []
        for case in source["selected_cases"][:6]:
            first = case["frame_rows"][0]
            cases.append(
                {
                    "seq_name": case["seq_name"],
                    "task": f"{case['task_field']}->{case['task_target']}",
                    "run_length": case["run_length"],
                    "proxy_mean_collateral_fields": case["proxy_mean_collateral_fields"],
                    "proxy_target_success_rate": case["proxy_target_success_rate"],
                    "first_frame_proxy_changed_fields": first["proxy_changed_fields"],
                    "first_frame_original": first["original"],
                    "first_frame_proxy": first["proxy"],
                }
            )
        out[source["source"]] = cases
    return out


def main():
    stability = load_json(GEN / "grouped_concat_family_expert_stability_audit.json")
    gallery = load_json(GEN / "grouped_concat_family_expert_gallery_shift_audit.json")
    local_edit = load_json(GEN / "local_edit_audit.json")
    casebook = load_json(GEN / "local_edit_casebook.json")
    geometry = load_json(GEN / "geometry_locality_audit.json")

    payload = {
        "artifacts": {
            "stability_audit": str(GEN / "grouped_concat_family_expert_stability_audit.json"),
            "gallery_shift_audit": str(GEN / "grouped_concat_family_expert_gallery_shift_audit.json"),
            "local_edit_audit": str(GEN / "local_edit_audit.json"),
            "local_edit_casebook": str(GEN / "local_edit_casebook.json"),
            "geometry_locality_audit": str(GEN / "geometry_locality_audit.json"),
        },
        "repair_stability": stability_section(stability),
        "gallery_shift": gallery_shift_section(gallery),
        "local_editability": local_edit_section(local_edit),
        "casebook_examples": casebook_section(casebook),
        "geometry_locality": {
            source["source"]: source["summary"]
            for source in geometry["sources"]
        },
        "takeaways": [
            "Family-expert repair is threshold-stable and repeatedly fixes the same hard right-hand occlusion and finger-occlusion confusions.",
            "The repair remains zero-harm under degraded gallery templates, so it is not tied to having the full template sequence.",
            "Direct symbolic edits are perfectly local under the tracked-field audit, while opaque token proxies usually achieve the target only with multiple collateral semantic changes.",
            "Frame-level casebook examples show that opaque proxy edits often rewrite the other hand, interaction state, or state signatures together with the requested target field.",
            "A first geometry-aware audit on interaction edits shows stronger symbolic locality than either opaque proxy family.",
        ],
    }

    out_json = GEN / "control_evidence_bundle.json"
    out_md = SUM / "control_evidence_bundle.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Control Evidence Bundle",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "## Repair Stability",
        "",
        f"- Shared zero-harm thresholds across `0.25 / 0.5 / 1.0`: `{payload['repair_stability']['shared_zero_harm_thresholds']}`",
        "- Recurring repaired cases:",
    ]
    for row in payload["repair_stability"]["recurring_cases"]:
        lines.append(
            f"  - `{row['seq_name']}`: `{row['original_prediction']} -> {row['new_prediction']}` across fractions `{row['fractions']}`"
        )

    lines.extend(["", "## Gallery Shift", ""])
    for frac in ("0.25", "0.5", "1.0"):
        section = payload["gallery_shift"][frac]
        lines.append(f"- Fraction `{frac}` all tested gallery variants zero-harm: `{section['all_variants_zero_harm']}`")
        full = section["variants"]["full"]
        stride2 = section["variants"]["stride2"]
        lines.append(
            f"  - full: `{fmt(full['family_old'])} -> {fmt(full['family_new'])}`, improved `{full['num_improved']}`, harmed `{full['num_harmed']}`"
        )
        lines.append(
            f"  - stride2: `{fmt(stride2['family_old'])} -> {fmt(stride2['family_new'])}`, improved `{stride2['num_improved']}`, harmed `{stride2['num_harmed']}`"
        )

    lines.extend(["", "## Local Editability", ""])
    for source_name, task_map in payload["local_editability"].items():
        lines.append(f"- Source `{source_name}`")
        for task in [
            "right_hand_motion->opening",
            "left_hand_motion->closing",
            "interaction_motion->approach",
            "interaction_motion->separate",
        ]:
            row = task_map[task]
            lines.append(
                f"  - `{task}`: symbolic clean `{fmt(row['symbolic_clean_edit_rate'])}`, proxy clean `{fmt(row['proxy_clean_edit_rate'])}`, proxy collateral `{fmt(row['proxy_mean_collateral_fields'])}`, proxy success `{fmt(row['proxy_target_success_rate'])}`"
            )

    lines.extend(["", "## Representative Casebook Examples", ""])
    for source_name, cases in payload["casebook_examples"].items():
        lines.append(f"- Source `{source_name}`")
        for case in cases[:3]:
            lines.append(
                f"  - `{case['seq_name']}` `{case['task']}`: proxy collateral `{fmt(case['proxy_mean_collateral_fields'])}`, changed fields `{case['first_frame_proxy_changed_fields']}`"
            )

    lines.extend(["", "## Geometry Locality", ""])
    for source_name, rows in payload["geometry_locality"].items():
        lines.append(f"- Source `{source_name}`")
        for row in rows:
            lines.append(
                f"  - `{row['task']}`: symbolic locality `{fmt(row['symbolic_locality_ratio'])}`, proxy locality `{fmt(row['proxy_locality_ratio'])}`, symbolic collateral `{fmt(row['symbolic_collateral_delta'])}`, proxy collateral `{fmt(row['proxy_collateral_delta'])}`"
            )

    lines.extend(["", "## Takeaways", ""])
    for item in payload["takeaways"]:
        lines.append(f"- {item}")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
