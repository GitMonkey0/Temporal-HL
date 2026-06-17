#!/usr/bin/env python3
"""Build a slice-level frontier report for the symbolic mainline."""

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


def to_map(rows):
    return {row["slice"]: row for row in rows}


def merged_slice_rows(symbolic_rows, joint_rows):
    sym = to_map(symbolic_rows)
    jnt = to_map(joint_rows)
    keys = sorted(set(sym) | set(jnt))
    out = []
    for key in keys:
        s = sym.get(key)
        j = jnt.get(key)
        out.append(
            {
                "slice": key,
                "symbolic_old_accuracy": None if s is None else s["old_accuracy"],
                "symbolic_new_accuracy": None if s is None else s["new_accuracy"],
                "symbolic_delta": None if s is None else s["delta"],
                "symbolic_total": None if s is None else s["new_total"],
                "joint_old_accuracy": None if j is None else j["old_accuracy"],
                "joint_new_accuracy": None if j is None else j["new_accuracy"],
                "joint_delta": None if j is None else j["delta"],
                "joint_total": None if j is None else j["new_total"],
                "symbolic_advantage_new": None
                if s is None or j is None
                else s["new_accuracy"] - j["new_accuracy"],
                "symbolic_advantage_delta": None
                if s is None or j is None
                else s["delta"] - j["delta"],
            }
        )
    out.sort(
        key=lambda row: (
            -(row["symbolic_advantage_new"] if row["symbolic_advantage_new"] is not None else -999),
            -(row["symbolic_delta"] if row["symbolic_delta"] is not None else -999),
            row["slice"],
        )
    )
    return out


def top(rows, key, n=5, reverse=True, require_nonnull=True):
    filt = [r for r in rows if (r[key] is not None or not require_nonnull)]
    filt.sort(key=lambda r: r[key], reverse=reverse)
    return filt[:n]


def main():
    sym_full = load_json(GEN / "symbolic_slice_compare_old_vs_new.json")
    sym_low = load_json(GEN / "symbolic_slice_compare_old_vs_new_fraction05.json")
    joint_full = load_json(GEN / "joint_sequence_slice_compare_v1.json")
    joint_low = load_json(GEN / "joint_sequence_slice_compare_fraction05_v1.json")
    frontier = load_json(GEN / "representation_frontier_full_lowdata.json")

    full_rows = merged_slice_rows(sym_full["pretrained_diff"], joint_full["pretrained_diff"])
    low_rows = merged_slice_rows(sym_low["pretrained_diff"], joint_low["pretrained_diff"])

    report = {
        "artifacts": {
            "symbolic_slice_compare_fraction_1.0": str(GEN / "symbolic_slice_compare_old_vs_new.json"),
            "symbolic_slice_compare_fraction_0.5": str(GEN / "symbolic_slice_compare_old_vs_new_fraction05.json"),
            "joint_slice_compare_fraction_1.0": str(GEN / "joint_sequence_slice_compare_v1.json"),
            "joint_slice_compare_fraction_0.5": str(GEN / "joint_sequence_slice_compare_fraction05_v1.json"),
            "representation_frontier_full_lowdata": str(GEN / "representation_frontier_full_lowdata.json"),
        },
        "fraction_1.0": {
            "rows": full_rows,
            "top_symbolic_delta": top(full_rows, "symbolic_delta", n=6, reverse=True),
            "worst_symbolic_delta": top(full_rows, "symbolic_delta", n=3, reverse=False),
            "top_symbolic_advantage_new": top(full_rows, "symbolic_advantage_new", n=6, reverse=True),
        },
        "fraction_0.5": {
            "rows": low_rows,
            "top_symbolic_delta": top(low_rows, "symbolic_delta", n=6, reverse=True),
            "worst_symbolic_delta": top(low_rows, "symbolic_delta", n=3, reverse=False),
            "top_symbolic_advantage_new": top(low_rows, "symbolic_advantage_new", n=6, reverse=True),
        },
        "cross_regime_patterns": {
            "positive_both_regimes": [
                row["slice"]
                for row in full_rows
                if row["slice"] in {r["slice"] for r in low_rows}
                and row["symbolic_delta"] is not None
                and row["symbolic_delta"] > 0
                and next(r for r in low_rows if r["slice"] == row["slice"])["symbolic_delta"] > 0
            ],
            "flat_or_saturated_both_regimes": [
                row["slice"]
                for row in full_rows
                if row["slice"] in {r["slice"] for r in low_rows}
                and abs(row["symbolic_delta"] or 0.0) < 1e-12
                and abs(next(r for r in low_rows if r["slice"] == row["slice"])["symbolic_delta"] or 0.0) < 1e-12
            ],
            "negative_any_regime": [
                row["slice"]
                for row in full_rows
                if row["slice"] in {r["slice"] for r in low_rows}
                and (
                    (row["symbolic_delta"] is not None and row["symbolic_delta"] < 0)
                    or (next(r for r in low_rows if r["slice"] == row["slice"])["symbolic_delta"] is not None and next(r for r in low_rows if r["slice"] == row["slice"])["symbolic_delta"] < 0)
                )
            ],
        },
        "derived_claim_checks": {
            "lowdata_wrist_rom_is_major_symbolic_gain": next(r for r in low_rows if r["slice"] == "wrist_rom")["symbolic_delta"] >= 0.5,
            "full_right_gain_exceeds_full_overall_gain": next(r for r in full_rows if r["slice"] == "right")["symbolic_delta"] > next(r for r in full_rows if r["slice"] == "all")["symbolic_delta"],
            "low_wrist_rom_gain_exceeds_joint_control": next(r for r in low_rows if r["slice"] == "wrist_rom")["symbolic_delta"] > next(r for r in low_rows if r["slice"] == "wrist_rom")["joint_delta"],
            "full_finger_occlusion_delta_exceeds_joint": next(r for r in full_rows if r["slice"] == "finger_occlusion")["symbolic_delta"] > next(r for r in full_rows if r["slice"] == "finger_occlusion")["joint_delta"],
            "saturated_slices_remain_flat": all(
                abs(next(r for r in full_rows if r["slice"] == s)["symbolic_delta"] or 0.0) < 1e-12
                and abs(next(r for r in low_rows if r["slice"] == s)["symbolic_delta"] or 0.0) < 1e-12
                for s in ["no_occlusion", "touching"]
            ),
            "full_left_has_regression": next(r for r in full_rows if r["slice"] == "left")["symbolic_delta"] < 0,
        },
        "context": {
            "sequence_frontier": frontier["frontier"],
            "sequence_gaps": frontier["gaps"],
        },
        "takeaways": [
            "The symbolic mainline does not improve uniformly; gains concentrate on a small set of difficult slices.",
            "In low data, the strongest symbolic gain is wrist-ROM, followed by interaction-heavy and no-interaction identity slices that were previously underfit.",
            "In full data, the largest symbolic gains are right-hand, finger-occlusion, and overall occlusion slices rather than already-saturated easy slices.",
            "No-occlusion and touching slices are saturated and stay flat, which is consistent with the idea that transition helps hard cases rather than easy cases.",
            "The only visible regression is the full-data left slice, so any final story must acknowledge asymmetry instead of claiming universal gains.",
            "Against the best joint-sequence control, symbolic gains are especially distinctive on low-data wrist-ROM; for full-data finger-occlusion, symbolic shows a positive delta while joint-sequence is already saturated at ceiling.",
        ],
    }

    out_json = GEN / "slice_frontier_report.json"
    out_md = SUM / "slice_frontier_report.md"
    out_json.write_text(json.dumps(report, indent=2))

    lines = [
        "# Slice Frontier Report",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "## Fraction 1.0",
        "",
        "| slice | sym old | sym new | sym delta | joint old | joint new | joint delta | sym-new minus joint-new |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in full_rows:
        lines.append(
            f"| {row['slice']} | {fmt(row['symbolic_old_accuracy'])} | {fmt(row['symbolic_new_accuracy'])} | {fmt(row['symbolic_delta'])} | {fmt(row['joint_old_accuracy'])} | {fmt(row['joint_new_accuracy'])} | {fmt(row['joint_delta'])} | {fmt(row['symbolic_advantage_new'])} |"
        )

    lines.extend(
        [
            "",
            "## Fraction 0.5",
            "",
            "| slice | sym old | sym new | sym delta | joint old | joint new | joint delta | sym-new minus joint-new |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in low_rows:
        lines.append(
            f"| {row['slice']} | {fmt(row['symbolic_old_accuracy'])} | {fmt(row['symbolic_new_accuracy'])} | {fmt(row['symbolic_delta'])} | {fmt(row['joint_old_accuracy'])} | {fmt(row['joint_new_accuracy'])} | {fmt(row['joint_delta'])} | {fmt(row['symbolic_advantage_new'])} |"
        )

    lines.extend(["", "## Cross-Regime Patterns", ""])
    for key, value in report["cross_regime_patterns"].items():
        lines.append(f"- `{key}`: `{value}`")

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
