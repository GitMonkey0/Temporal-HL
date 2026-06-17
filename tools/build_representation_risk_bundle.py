#!/usr/bin/env python3
"""Build a unified risk/advantage bundle for the temporal-HL representation."""

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


def best_proxy_rows(proxy_report: dict[str, object]):
    out = {}
    for row in proxy_report["codebook_results"]:
        source = row["source"]
        retrieval_best = max(row["retrieval"], key=lambda item: item["summary"]["top1_accuracy"])
        classifier_summary = row["classifier"]["summary"]
        current = out.get(source)
        candidate = {
            "source": source,
            "num_clusters": row["num_clusters"],
            "retrieval_name": retrieval_best["name"],
            "retrieval_top1": retrieval_best["summary"]["top1_accuracy"],
            "retrieval_margin": retrieval_best["summary"]["mean_positive_margin"],
            "retrieval_rank": retrieval_best["summary"]["mean_correct_rank"],
            "classifier_025": classifier_summary["0.25"]["sequence_accuracy_mean"],
            "classifier_05": classifier_summary["0.5"]["sequence_accuracy_mean"],
            "classifier_10": classifier_summary["1.0"]["sequence_accuracy_mean"],
        }
        if current is None or candidate["retrieval_top1"] > current["retrieval_top1"] or (
            candidate["retrieval_top1"] == current["retrieval_top1"]
            and candidate["retrieval_margin"] > current["retrieval_margin"]
        ):
            out[source] = candidate
    return out


def frontier_section(frontier: dict[str, object]):
    rows = {row["fraction"]: row for row in frontier["rows"]}
    return {
        frac: {
            "grouped_seq": rows[frac]["grouped_seq"],
            "family_seq": rows[frac]["family_seq"],
            "flat_seq": rows[frac]["flat_seq"],
            "delta_grouped_minus_flat": rows[frac]["delta_grouped_minus_flat"],
            "delta_family_minus_grouped": rows[frac]["delta_family_minus_grouped"],
            "family_improved": rows[frac]["family_improved"],
            "family_harmed": rows[frac]["family_harmed"],
        }
        for frac in ("0.25", "0.5", "1.0")
    }


def accuracy_risk_section(frontier: dict[str, object], proxy_report: dict[str, object]):
    frontier_rows = {row["fraction"]: row for row in frontier["rows"]}
    best_proxies = best_proxy_rows(proxy_report)
    best_cont = best_proxies["continuous_frame"]
    best_sem = best_proxies["semantic_frame"]
    symbolic_family_10 = frontier_rows["1.0"]["family_seq"]
    grouped_10 = frontier_rows["1.0"]["grouped_seq"]
    return {
        "best_continuous_proxy": best_cont,
        "best_semantic_proxy": best_sem,
        "current_symbolic_family_10": symbolic_family_10,
        "current_symbolic_grouped_10": grouped_10,
        "why_accuracy_is_risky": [
            "Opaque learned-token proxies are already very strong on retrieval and sequence classification.",
            "Best continuous proxy reaches retrieval top1 1.0000, which means pure discriminability cannot be the main symbolic novelty wedge.",
            "Best semantic proxy reaches 0.7692 sequence accuracy at fraction 1.0, matching or exceeding several weaker symbolic baselines.",
            "The current-code symbolic frontier is strongest after grouped structure plus family repair, so the defensible symbolic story is structural and post-hoc stable, not raw token-separability alone.",
        ],
    }


def control_section(control_bundle: dict[str, object]):
    local = control_bundle["local_editability"]
    geom = control_bundle["geometry_locality"]
    return {
        "shared_zero_harm_thresholds": control_bundle["repair_stability"]["shared_zero_harm_thresholds"],
        "local_editability": {
            source: {
                task: {
                    "symbolic_clean": row["symbolic_clean_edit_rate"],
                    "proxy_clean": row["proxy_clean_edit_rate"],
                    "proxy_collateral": row["proxy_mean_collateral_fields"],
                    "proxy_success": row["proxy_target_success_rate"],
                }
                for task, row in task_map.items()
                if task in {
                    "right_hand_motion->opening",
                    "left_hand_motion->closing",
                    "interaction_motion->approach",
                    "interaction_motion->separate",
                }
            }
            for source, task_map in local.items()
        },
        "interaction_geometry_locality": geom,
        "why_control_is_strong": [
            "Family-expert repair is zero-harm over a broad threshold band and remains zero-harm under gallery degradation.",
            "Direct symbolic edits are perfectly local on the tracked-field audit, while opaque proxies usually need around two collateral field rewrites.",
            "Interaction-motion geometry locality is stronger for symbolic edits than for either proxy family.",
        ],
    }


def conditioned_geometry_section(conditioned: dict[str, object]):
    return {
        "conditioning_rule": conditioned["conditioning_rule"],
        "sources": {
            source["source"]: {row["task"]: row for row in source["summary"]}
            for source in conditioned["sources"]
        },
        "why_not_mainline": [
            "The conditioned protocol fixes the exact-neighbor sparsity problem, so it is a valid diagnostic.",
            "But geometry-only locality can still look artificially strong for proxies when a nearest real frame is geometrically close even though the decoded semantic edit remains entangled.",
            "Proxy semantic collateral remains around 1.3 to 2.0 extra fields on average, so this line is not yet clean enough for a main claim.",
        ],
    }


def counterfactual_section(counterfactual: dict[str, object]):
    return {
        "sources": {
            source["source"]: {row["task"]: row for row in source["summary"]}
            for source in counterfactual["sources"]
        },
        "what_it_changes": [
            "This audit replaces unstable unbounded locality ratios with a bounded target-share metric.",
            "It combines geometry concentration with semantic context preservation into a counterfactual score.",
            "It strengthens the interaction-motion control line but does not yet make hand-motion a universal win.",
        ],
    }


def transplant_section(transplant: dict[str, object]):
    return {
        "sources": {
            source["source"]: {row["task"]: row for row in source["summary"]}
            for source in transplant["sources"]
        },
        "what_it_changes": [
            "This audit removes the whole-frame retrieval confound by comparing hand-only donor quality.",
            "It shows a mostly stable symbolic advantage on hand-motion donor closeness while proxy context preservation remains imperfect.",
            "It strengthens hand-motion as supporting control evidence, but it is still not a full conditional reconstruction result.",
        ],
    }


def realization_oracle_section(realization: dict[str, object]):
    return {
        "sources": {
            source["source"]: {row["task"]: row for row in source["summary"]}
            for source in realization["sources"]
        },
        "what_it_changes": [
            "This oracle asks whether current-frame donors alone can realize hand-motion labels under the original previous-frame context.",
            "The answer is effectively no for both symbolic and proxy donor families under the conditioned search protocol.",
            "That negative result supports the core thesis that hand-motion control is transition-aware rather than reducible to current-state substitution.",
        ],
    }


def transition_conditioned_section(transition: dict[str, object]):
    return {
        "sources": {
            source["source"]: {row["task"]: row for row in source["summary"]}
            for source in transition["sources"]
        },
        "what_it_changes": [
            "This audit upgrades hand-motion evaluation from current-frame donors to donor pairs that already realize the requested motion.",
            "Under this transition-conditioned protocol, symbolic pair quality is better on most hand-motion tasks while proxy semantic collateral remains high.",
            "It provides a positive temporal-structure result that complements the negative donor-only oracle.",
        ],
    }


def transition_slice_section(slice_audit: dict[str, object]):
    rows = slice_audit["summary"]
    selected = {}
    for source in ("semantic_frame", "continuous_frame"):
        selected[source] = {}
        for task in (
            "left_hand_motion->closing",
            "left_hand_motion->opening",
            "right_hand_motion->closing",
            "right_hand_motion->opening",
        ):
            selected[source][task] = {}
            for slice_name in ("all", "occlusion", "finger_occlusion", "wrist_rom", "interaction"):
                match = [r for r in rows if r["source"] == source and r["task"] == task and r["slice"] == slice_name]
                if match:
                    selected[source][task][slice_name] = match[0]
    return {
        "sources": selected,
        "what_it_changes": [
            "The transition-conditioned hand-motion advantage is extremely stable on hard single-hand slices such as occlusion, finger-occlusion, and wrist-ROM.",
            "The main remaining weakness is concentrated in interacting-hand contexts rather than in generic hand-motion difficulty.",
            "This sharpens the agenda: interaction-aware motion editing is now the dominant remaining control challenge.",
        ],
    }


def decision_section():
    return {
        "mainline_claims_to_keep": [
            "sequence-native symbolic structure",
            "anatomy-aware grouped factorization",
            "zero-harm family-level repair",
            "local controllability and editability",
            "interaction-motion geometry locality",
            "interaction-motion counterfactual consistency",
            "hand-motion conditional transplant quality",
            "hand-motion requires explicit temporal transition channels",
            "hand-motion transition-conditioned motif quality",
            "single-hand hard slices are largely solved by transition-conditioned motifs",
        ],
        "claims_to_avoid": [
            "symbolic wins because pure retrieval/classification is higher than learned tokens",
            "hand-motion geometry locality is already a clean main result",
            "temporal HL is just a stronger temporal backbone over frame-wise HL",
        ],
        "next_experiment_targets": [
            "conditional reconstruction or conditional edit evaluation for hand-motion",
            "more manipulation-oriented interaction/control evaluation",
            "keep learned-token proxies as negative pressure baselines, not as the main battleground",
        ],
    }


def main():
    frontier = load_json(GEN / "current_code_symbolic_frontier.json")
    proxy_report = load_json(GEN / "learned_token_proxy_report.json")
    control_bundle = load_json(GEN / "control_evidence_bundle.json")
    conditioned = load_json(GEN / "hand_motion_conditioned_geometry_audit.json")
    counterfactual = load_json(GEN / "counterfactual_edit_audit.json")
    transplant = load_json(GEN / "conditional_hand_transplant_audit.json")
    realization = load_json(GEN / "conditional_motion_realization_oracle_audit.json")
    transition = load_json(GEN / "transition_conditioned_hand_motion_audit.json")
    transition_slice = load_json(GEN / "transition_conditioned_hand_motion_slice_audit.json")

    payload = {
        "artifacts": {
            "current_code_symbolic_frontier": str(GEN / "current_code_symbolic_frontier.json"),
            "learned_token_proxy_report": str(GEN / "learned_token_proxy_report.json"),
            "control_evidence_bundle": str(GEN / "control_evidence_bundle.json"),
            "hand_motion_conditioned_geometry_audit": str(GEN / "hand_motion_conditioned_geometry_audit.json"),
            "counterfactual_edit_audit": str(GEN / "counterfactual_edit_audit.json"),
            "conditional_hand_transplant_audit": str(GEN / "conditional_hand_transplant_audit.json"),
            "conditional_motion_realization_oracle_audit": str(GEN / "conditional_motion_realization_oracle_audit.json"),
            "transition_conditioned_hand_motion_audit": str(GEN / "transition_conditioned_hand_motion_audit.json"),
            "transition_conditioned_hand_motion_slice_audit": str(GEN / "transition_conditioned_hand_motion_slice_audit.json"),
        },
        "symbolic_frontier": frontier_section(frontier),
        "accuracy_risk": accuracy_risk_section(frontier, proxy_report),
        "control_advantage": control_section(control_bundle),
        "conditioned_geometry_diagnostic": conditioned_geometry_section(conditioned),
        "counterfactual_consistency": counterfactual_section(counterfactual),
        "conditional_hand_transplant": transplant_section(transplant),
        "realization_oracle": realization_oracle_section(realization),
        "transition_conditioned_hand_motion": transition_conditioned_section(transition),
        "transition_conditioned_slices": transition_slice_section(transition_slice),
        "decision": decision_section(),
    }

    out_json = GEN / "representation_risk_bundle.json"
    out_md = SUM / "representation_risk_bundle.md"
    out_json.write_text(json.dumps(payload, indent=2))

    best_cont = payload["accuracy_risk"]["best_continuous_proxy"]
    best_sem = payload["accuracy_risk"]["best_semantic_proxy"]

    lines = [
        "# Representation Risk Bundle",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "## Accuracy Risk",
        "",
        f"- Best `continuous_frame` proxy: `{best_cont['num_clusters']}` clusters, retrieval `{best_cont['retrieval_name']}` top1 `{fmt(best_cont['retrieval_top1'])}`, margin `{fmt(best_cont['retrieval_margin'])}`, classifier seq accs `{fmt(best_cont['classifier_025'])} / {fmt(best_cont['classifier_05'])} / {fmt(best_cont['classifier_10'])}`.",
        f"- Best `semantic_frame` proxy: `{best_sem['num_clusters']}` clusters, retrieval `{best_sem['retrieval_name']}` top1 `{fmt(best_sem['retrieval_top1'])}`, margin `{fmt(best_sem['retrieval_margin'])}`, classifier seq accs `{fmt(best_sem['classifier_025'])} / {fmt(best_sem['classifier_05'])} / {fmt(best_sem['classifier_10'])}`.",
        f"- Current symbolic frontier at fraction `1.0`: grouped `{fmt(payload['symbolic_frontier']['1.0']['grouped_seq'])}`, grouped+family `{fmt(payload['symbolic_frontier']['1.0']['family_seq'])}`.",
    ]
    for item in payload["accuracy_risk"]["why_accuracy_is_risky"]:
        lines.append(f"- {item}")

    lines.extend([
        "",
        "## Structural Frontier",
        "",
        "| frac | flat seq | grouped seq | family seq | grouped-flat | family-grouped | improved | harmed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for frac in ("0.25", "0.5", "1.0"):
        row = payload["symbolic_frontier"][frac]
        flat = "n/a" if row["flat_seq"] is None else fmt(row["flat_seq"])
        delta_g = "n/a" if row["delta_grouped_minus_flat"] is None else fmt(row["delta_grouped_minus_flat"])
        lines.append(
            f"| {frac} | {flat} | {fmt(row['grouped_seq'])} | {fmt(row['family_seq'])} | {delta_g} | {fmt(row['delta_family_minus_grouped'])} | {row['family_improved']} | {row['family_harmed']} |"
        )

    lines.extend([
        "",
        "## Control Advantage",
        "",
        f"- Shared zero-harm thresholds: `{payload['control_advantage']['shared_zero_harm_thresholds']}`.",
    ])
    for source, task_map in payload["control_advantage"]["local_editability"].items():
        lines.append(f"- Source `{source}`")
        for task, row in task_map.items():
            lines.append(
                f"  - `{task}`: symbolic clean `{fmt(row['symbolic_clean'])}`, proxy clean `{fmt(row['proxy_clean'])}`, proxy collateral `{fmt(row['proxy_collateral'])}`, proxy success `{fmt(row['proxy_success'])}`"
            )
    for item in payload["control_advantage"]["why_control_is_strong"]:
        lines.append(f"- {item}")

    lines.extend(["", "## Conditioned Geometry Diagnostic", ""])
    for source, task_map in payload["conditioned_geometry_diagnostic"]["sources"].items():
        lines.append(f"- Source `{source}`")
        for task in ("left_hand_motion->opening", "left_hand_motion->closing", "right_hand_motion->opening", "right_hand_motion->closing"):
            row = task_map[task]
            lines.append(
                f"  - `{task}`: proxy conditioned clean `{fmt(row['proxy_conditioned_clean_rate'])}`, proxy semantic collateral `{fmt(row['proxy_semantic_collateral_fields'])}`, symbolic locality `{fmt(row['symbolic_locality_ratio'])}`, proxy locality `{fmt(row['proxy_locality_ratio'])}`"
            )
    for item in payload["conditioned_geometry_diagnostic"]["why_not_mainline"]:
        lines.append(f"- {item}")

    lines.extend(["", "## Counterfactual Consistency", ""])
    for source, task_map in payload["counterfactual_consistency"]["sources"].items():
        lines.append(f"- Source `{source}`")
        for task in ("interaction_motion->approach", "interaction_motion->separate", "right_hand_motion->closing", "right_hand_motion->opening"):
            if task not in task_map:
                continue
            row = task_map[task]
            lines.append(
                f"  - `{task}`: symbolic cf `{fmt(row['symbolic_counterfactual_score'])}`, proxy cf `{fmt(row['proxy_counterfactual_score'])}`, proxy preserved clean `{fmt(row['proxy_preserved_clean_rate'])}`, proxy semantic collateral `{fmt(row['proxy_total_semantic_collateral'])}`"
            )
    for item in payload["counterfactual_consistency"]["what_it_changes"]:
        lines.append(f"- {item}")

    lines.extend(["", "## Conditional Hand Transplant", ""])
    for source, task_map in payload["conditional_hand_transplant"]["sources"].items():
        lines.append(f"- Source `{source}`")
        for task in ("left_hand_motion->closing", "left_hand_motion->opening", "right_hand_motion->closing", "right_hand_motion->opening"):
            row = task_map[task]
            lines.append(
                f"  - `{task}`: symbolic donor delta `{fmt(row['symbolic_target_hand_delta'])}`, proxy donor delta `{fmt(row['proxy_target_hand_delta'])}`, symbolic beats proxy `{fmt(row['symbolic_beats_proxy_rate'])}`, proxy preserved clean `{fmt(row['proxy_preserved_clean_rate'])}`"
            )
    for item in payload["conditional_hand_transplant"]["what_it_changes"]:
        lines.append(f"- {item}")

    lines.extend(["", "## Realization Oracle", ""])
    for source, task_map in payload["realization_oracle"]["sources"].items():
        lines.append(f"- Source `{source}`")
        for task in ("left_hand_motion->closing", "left_hand_motion->opening", "right_hand_motion->closing", "right_hand_motion->opening"):
            row = task_map[task]
            lines.append(
                f"  - `{task}`: symbolic realizing cand `{fmt(row['symbolic_realizing_candidate_rate'])}`, proxy realizing cand `{fmt(row['proxy_realizing_candidate_rate'])}`, symbolic oracle success `{fmt(row['symbolic_oracle_success'])}`, proxy oracle success `{fmt(row['proxy_oracle_success'])}`"
            )
    for item in payload["realization_oracle"]["what_it_changes"]:
        lines.append(f"- {item}")

    lines.extend(["", "## Transition-Conditioned Hand Motion", ""])
    for source, task_map in payload["transition_conditioned_hand_motion"]["sources"].items():
        lines.append(f"- Source `{source}`")
        for task in ("left_hand_motion->closing", "left_hand_motion->opening", "right_hand_motion->closing", "right_hand_motion->opening"):
            row = task_map[task]
            lines.append(
                f"  - `{task}`: symbolic pair avail `{fmt(row['symbolic_pair_available_rate'])}`, proxy pair avail `{fmt(row['proxy_pair_available_rate'])}`, symbolic pair dist `{fmt(row['symbolic_pair_distance'])}`, proxy pair dist `{fmt(row['proxy_pair_distance'])}`, symbolic beats proxy `{fmt(row['symbolic_beats_proxy_rate'])}`"
            )
    for item in payload["transition_conditioned_hand_motion"]["what_it_changes"]:
        lines.append(f"- {item}")

    lines.extend(["", "## Transition Slices", ""])
    for source, task_map in payload["transition_conditioned_slices"]["sources"].items():
        lines.append(f"- Source `{source}`")
        for task, slice_map in task_map.items():
            if "interaction" in slice_map and "occlusion" in slice_map:
                lines.append(
                    f"  - `{task}`: interaction beats proxy `{fmt(slice_map['interaction']['symbolic_beats_proxy_rate'])}`, occlusion beats proxy `{fmt(slice_map['occlusion']['symbolic_beats_proxy_rate'])}`, finger-occlusion beats proxy `{fmt(slice_map.get('finger_occlusion', slice_map['occlusion'])['symbolic_beats_proxy_rate'])}`"
                )
    for item in payload["transition_conditioned_slices"]["what_it_changes"]:
        lines.append(f"- {item}")

    lines.extend(["", "## Decision Rules", ""])
    lines.append("- Keep:")
    for item in payload["decision"]["mainline_claims_to_keep"]:
        lines.append(f"  - `{item}`")
    lines.append("- Avoid:")
    for item in payload["decision"]["claims_to_avoid"]:
        lines.append(f"  - `{item}`")
    lines.append("- Next:")
    for item in payload["decision"]["next_experiment_targets"]:
        lines.append(f"  - `{item}`")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
