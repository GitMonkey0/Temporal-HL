#!/usr/bin/env python3
"""Evaluate a transition-conditioned symbolic hand-motion editor.

This script goes beyond donor-pair retrieval audits. For each test transition,
it selects a donor pair that already realizes a target hand-motion label, then
composes the donor target-hand transition onto the original previous-frame
context and measures whether the edited transition still realizes the target.

The current implementation is intentionally restricted to left/right hand-motion
tasks. Inter-hand interaction edits require editable global cross-hand geometry,
which is not exposed by the current local label export.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from tools.build_conditional_hand_transplant_audit import (
    TASKS,
    canonical,
    preserved_fields,
    target_hand_name,
)
from tools.build_geometry_locality_audit import frame_geom, hand_delta
from tools.build_learned_token_proxy_report import (
    build_semantic_frame_vocab,
    frame_token_set,
    frame_vector,
    overlap_labels,
)
from tools.build_local_edit_audit import (
    TRACKED_FIELDS,
    cluster_majority_attrs,
    contiguous_runs,
    eligible_value,
    frame_attrs,
)
from tools.build_temporal_hl import (
    EDGE_ORDER,
    FINGER_NAMES,
    normalize,
    quantize_direction,
    summarize_flexion_transition,
    transition_label,
)


ROOT = Path("/opt/tiger/hand")
GEN = ROOT / "experiments/generated"
SUM = GEN / "summary_tables"
FLEX_THRESHOLD = 0.05
STAY_DEG = 10.0
MINOR_DEG = 35.0


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def fmt(x: float) -> str:
    return f"{x:.4f}"


def build_pair_bank(train_data, labels, semantic_vocab):
    token_to_idx = {tok: idx for idx, tok in enumerate(semantic_vocab)}
    bank = []
    for sequence in train_data["sequences"]:
        if canonical(sequence["seq_name"]) not in labels and sequence["seq_name"] not in labels:
            continue
        frames = sequence["frames"]
        for idx in range(1, len(frames)):
            prev_frame = frames[idx - 1]
            curr_frame = frames[idx]
            semantic_vec = np.zeros(len(semantic_vocab), dtype=np.float32)
            for tok in frame_token_set(curr_frame, mode="temporal", include_persistence=True):
                j = token_to_idx.get(tok)
                if j is not None:
                    semantic_vec[j] = 1.0
            bank.append(
                {
                    "seq_name": sequence["seq_name"],
                    "prev_frame_idx": prev_frame["frame_idx"],
                    "curr_frame_idx": curr_frame["frame_idx"],
                    "prev_frame": prev_frame,
                    "curr_frame": curr_frame,
                    "prev_attrs": frame_attrs(prev_frame),
                    "curr_attrs": frame_attrs(curr_frame),
                    "prev_geom": frame_geom(prev_frame),
                    "curr_geom": frame_geom(curr_frame),
                    "semantic_vec": semantic_vec,
                    "continuous_vec": np.asarray(frame_vector(curr_frame), dtype=np.float32),
                }
            )
    return bank


def source_cluster_decoded(source_name, pair_bank):
    if source_name == "semantic_frame":
        matrix = np.asarray([row["semantic_vec"] for row in pair_bank], dtype=np.float32)
    else:
        matrix = np.asarray([row["continuous_vec"] for row in pair_bank], dtype=np.float32)
    kmeans = KMeans(n_clusters=32, random_state=0, n_init=10)
    kmeans.fit(matrix)
    assignments = list(map(int, kmeans.labels_))
    decoded = cluster_majority_attrs(assignments, [row["curr_attrs"] for row in pair_bank], 32)
    for row, cid in zip(pair_bank, assignments):
        row[f"{source_name}_cluster"] = cid
    return decoded


def realized_motion(prev_hand: dict[str, object] | None, curr_hand: dict[str, object] | None) -> str:
    if prev_hand is None or curr_hand is None:
        return "unknown"
    return summarize_flexion_transition(
        prev_hand.get("flexion_scores", {}),
        curr_hand.get("flexion_scores", {}),
        FLEX_THRESHOLD,
    )


def geom_motion(prev_geom_hand: dict[str, object] | None, curr_geom_hand: dict[str, object] | None) -> str:
    if prev_geom_hand is None or curr_geom_hand is None:
        return "unknown"
    prev_arr = np.asarray(prev_geom_hand["flexion"], dtype=np.float32)
    curr_arr = np.asarray(curr_geom_hand["flexion"], dtype=np.float32)
    prev_scores = {finger: float(prev_arr[idx]) for idx, finger in enumerate(FINGER_NAMES)}
    curr_scores = {finger: float(curr_arr[idx]) for idx, finger in enumerate(FINGER_NAMES)}
    return summarize_flexion_transition(prev_scores, curr_scores, FLEX_THRESHOLD)


def raw_prev_to_scores(prev_hand: dict[str, object] | None) -> dict[str, float] | None:
    if prev_hand is None:
        return None
    scores = prev_hand.get("flexion_scores", {})
    return {finger: float(scores.get(finger, 0.0)) for finger in FINGER_NAMES}


def mixed_motion(prev_hand_raw: dict[str, object] | None, curr_geom_hand: dict[str, object] | None) -> str:
    if prev_hand_raw is None or curr_geom_hand is None:
        return "unknown"
    prev_scores = raw_prev_to_scores(prev_hand_raw)
    if prev_scores is None:
        return "unknown"
    curr_arr = np.asarray(curr_geom_hand["flexion"], dtype=np.float32)
    curr_scores = {finger: float(curr_arr[idx]) for idx, finger in enumerate(FINGER_NAMES)}
    return summarize_flexion_transition(prev_scores, curr_scores, FLEX_THRESHOLD)


def pair_realizes_target(row: dict[str, object], task_field: str, target_value: str) -> bool:
    hand_name = target_hand_name(task_field)
    return geom_motion(
        row["prev_geom"][hand_name],  # type: ignore[index]
        row["curr_geom"][hand_name],  # type: ignore[index]
    ) == target_value


def pair_distance(test_prev_geom, test_curr_geom, donor_row, task_field: str) -> float:
    hand_name = target_hand_name(task_field)
    return (
        hand_delta(test_prev_geom[hand_name], donor_row["prev_geom"][hand_name])
        + hand_delta(test_curr_geom[hand_name], donor_row["curr_geom"][hand_name])
    )


def symbolic_pair_candidates(pair_bank, curr_attrs, task_field, target_value):
    keep = preserved_fields(task_field)
    out = []
    for row in pair_bank:
        attrs = row["curr_attrs"]
        if attrs[task_field] != target_value:
            continue
        if any(attrs[field] != curr_attrs[field] for field in keep):
            continue
        out.append(row)
    return out


def pick_best_symbolic_pair(pair_bank, curr_attrs, test_prev_geom, test_curr_geom, task_field, target_value):
    cands = [
        row
        for row in symbolic_pair_candidates(pair_bank, curr_attrs, task_field, target_value)
        if pair_realizes_target(row, task_field, target_value)
    ]
    if not cands:
        return None
    cands.sort(key=lambda row: (pair_distance(test_prev_geom, test_curr_geom, row, task_field), row["seq_name"], row["curr_frame_idx"]))
    return cands[0]


def best_proxy_attrs(curr_attrs, task_field, target_value, cluster_decoded):
    keep = preserved_fields(task_field)
    candidates = []
    for cid, attrs in cluster_decoded.items():
        if attrs[task_field] != target_value:
            continue
        preserved_mismatch = sum(1 for field in keep if attrs[field] != curr_attrs[field])
        total_collateral = sum(1 for field in TRACKED_FIELDS if field != task_field and attrs[field] != curr_attrs[field])
        candidates.append((preserved_mismatch, total_collateral, cid, attrs))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    return candidates[0][3]


def pick_best_proxy_pair(pair_bank, source_name, cluster_decoded, proxy_attrs, test_prev_geom, test_curr_geom, task_field, target_value):
    matched_clusters = [cid for cid, attrs in cluster_decoded.items() if attrs == proxy_attrs]
    cands = [
        row
        for row in pair_bank
        if row[f"{source_name}_cluster"] in matched_clusters
        and row["curr_attrs"][task_field] == target_value
        and pair_realizes_target(row, task_field, target_value)
    ]
    if not cands:
        return None
    cands.sort(key=lambda row: (pair_distance(test_prev_geom, test_curr_geom, row, task_field), row["seq_name"], row["curr_frame_idx"]))
    return cands[0]


def compose_target_hand_transition(test_prev_geom, donor_row, task_field: str):
    hand_name = target_hand_name(task_field)
    prev_hand = test_prev_geom[hand_name]
    donor_prev = donor_row["prev_geom"][hand_name]
    donor_curr = donor_row["curr_geom"][hand_name]
    if prev_hand is None or donor_prev is None or donor_curr is None:
        return None

    new_local = (
        np.asarray(prev_hand["local_vectors"], dtype=np.float32)
        + (np.asarray(donor_curr["local_vectors"], dtype=np.float32) - np.asarray(donor_prev["local_vectors"], dtype=np.float32))
    )
    new_flex = (
        np.asarray(prev_hand["flexion"], dtype=np.float32)
        + (np.asarray(donor_curr["flexion"], dtype=np.float32) - np.asarray(donor_prev["flexion"], dtype=np.float32))
    )
    return {
        "local_vectors": new_local,
        "flexion": new_flex,
    }


def summarize_composed_motif(prev_frame, edited_hand_geom, donor_row, task_field: str):
    hand_name = target_hand_name(task_field)
    prev_hand = prev_frame.get(hand_name)
    donor_curr_hand = donor_row["curr_frame"].get(hand_name)
    if prev_hand is None or donor_curr_hand is None or edited_hand_geom is None:
        return None

    prev_vectors = [normalize(list(map(float, vec))) for vec in prev_hand.get("local_vectors", [])]
    prev_tokens = list(map(int, prev_hand.get("token_ids", [])))
    edited_vectors = [normalize(list(map(float, vec))) for vec in np.asarray(edited_hand_geom["local_vectors"], dtype=np.float32).tolist()]
    edited_tokens = [quantize_direction(vec)[0] for vec in edited_vectors]
    edited_labels = [quantize_direction(vec)[1] for vec in edited_vectors]
    edited_transition_labels, _ = transition_label(prev_vectors, edited_vectors, prev_tokens, edited_tokens, STAY_DEG, MINOR_DEG)

    donor_token_labels = donor_curr_hand.get("token_labels", [])
    donor_transition_labels = donor_curr_hand.get("transition_labels", [])
    state_agreement = 0.0
    transition_agreement = 0.0
    if donor_token_labels:
        state_agreement = sum(int(a == b) for a, b in zip(edited_labels, donor_token_labels)) / max(len(donor_token_labels), 1)
    if donor_transition_labels:
        transition_agreement = sum(int(a == b) for a, b in zip(edited_transition_labels, donor_transition_labels)) / max(len(donor_transition_labels), 1)
    return {
        "state_agreement": state_agreement,
        "transition_agreement": transition_agreement,
        "edited_transition_labels": edited_transition_labels,
        "donor_transition_labels": donor_transition_labels,
        "edited_hand_motion": mixed_motion(prev_frame.get(hand_name), edited_hand_geom),
        "donor_hand_motion": str(donor_curr_hand.get("hand_motion", "unknown")),
    }


def finger_activity_profile(transition_labels: list[str]) -> list[str]:
    by_finger: dict[str, int] = {finger: 0 for finger in FINGER_NAMES}
    for (finger_name, _), label in zip(EDGE_ORDER, transition_labels):
        if label != "stay":
            by_finger[finger_name] += 1
    profile = []
    for finger in FINGER_NAMES:
        count = by_finger[finger]
        if count == 0:
            profile.append("still")
        elif count == 1:
            profile.append("local")
        else:
            profile.append("active")
    return profile


def grouped_motif_signature(hand_motion: str, transition_labels: list[str]) -> str:
    profile = finger_activity_profile(transition_labels)
    return hand_motion + "|" + "|".join(profile)


def active_finger_count(transition_labels: list[str]) -> int:
    return sum(level != "still" for level in finger_activity_profile(transition_labels))


def coordination_class(right_motion: str, left_motion: str) -> str:
    if right_motion == left_motion:
        return "sync_same"
    if "steady" in (right_motion, left_motion) and right_motion != left_motion:
        return "one_active"
    if {right_motion, left_motion} == {"opening", "closing"}:
        return "opposed"
    return "other"


def summarize(rows):
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    out = []
    for task, items in sorted(by_task.items()):
        n = len(items)
        sym_avail = sum(row["symbolic_pair_available"] for row in items)
        prox_avail = sum(row["proxy_pair_available"] for row in items)
        both_avail = sum(row["both_available"] for row in items)
        out.append(
            {
                "task": task,
                "num_frames": n,
                "symbolic_pair_available_rate": sym_avail / n,
                "proxy_pair_available_rate": prox_avail / n,
                "symbolic_composed_success_rate": sum(row["symbolic_composed_success"] for row in items) / n,
                "proxy_composed_success_rate": sum(row["proxy_composed_success"] for row in items) / n,
                "symbolic_beats_proxy_rate": (sum(row["symbolic_beats_proxy"] for row in items) / both_avail) if both_avail else 0.0,
                "symbolic_to_donor_residual": (sum(row["symbolic_to_donor_residual"] for row in items if row["symbolic_pair_available"]) / max(sym_avail, 1)),
                "proxy_to_donor_residual": (sum(row["proxy_to_donor_residual"] for row in items if row["proxy_pair_available"]) / max(prox_avail, 1)),
                "symbolic_state_agreement": (sum(row["symbolic_state_agreement"] for row in items if row["symbolic_pair_available"]) / max(sym_avail, 1)),
                "proxy_state_agreement": (sum(row["proxy_state_agreement"] for row in items if row["proxy_pair_available"]) / max(prox_avail, 1)),
                "symbolic_transition_agreement": (sum(row["symbolic_transition_agreement"] for row in items if row["symbolic_pair_available"]) / max(sym_avail, 1)),
                "proxy_transition_agreement": (sum(row["proxy_transition_agreement"] for row in items if row["proxy_pair_available"]) / max(prox_avail, 1)),
                "symbolic_grouped_motif_match": (sum(row["symbolic_grouped_motif_match"] for row in items if row["symbolic_pair_available"]) / max(sym_avail, 1)),
                "proxy_grouped_motif_match": (sum(row["proxy_grouped_motif_match"] for row in items if row["proxy_pair_available"]) / max(prox_avail, 1)),
                "donor_active_slice_rate": (sum(row["donor_active_slice"] for row in items) / n),
                "symbolic_grouped_motif_match_active": (
                    sum(row["symbolic_grouped_motif_match"] for row in items if row["symbolic_pair_available"] and row["donor_active_slice"])
                    / max(sum(row["symbolic_pair_available"] and row["donor_active_slice"] for row in items), 1)
                ),
                "proxy_grouped_motif_match_active": (
                    sum(row["proxy_grouped_motif_match"] for row in items if row["proxy_pair_available"] and row["donor_active_slice"])
                    / max(sum(row["proxy_pair_available"] and row["donor_active_slice"] for row in items), 1)
                ),
                "donor_dense_slice_rate": (sum(row["donor_dense_slice"] for row in items) / n),
                "symbolic_grouped_motif_match_dense": (
                    sum(row["symbolic_grouped_motif_match"] for row in items if row["symbolic_pair_available"] and row["donor_dense_slice"])
                    / max(sum(row["symbolic_pair_available"] and row["donor_dense_slice"] for row in items), 1)
                ),
                "proxy_grouped_motif_match_dense": (
                    sum(row["proxy_grouped_motif_match"] for row in items if row["proxy_pair_available"] and row["donor_dense_slice"])
                    / max(sum(row["proxy_pair_available"] and row["donor_dense_slice"] for row in items), 1)
                ),
                "interaction_slice_rate": (sum(row["interaction_slice"] for row in items) / n),
                "symbolic_grouped_motif_match_interaction": (
                    sum(row["symbolic_grouped_motif_match"] for row in items if row["symbolic_pair_available"] and row["interaction_slice"])
                    / max(sum(row["symbolic_pair_available"] and row["interaction_slice"] for row in items), 1)
                ),
                "proxy_grouped_motif_match_interaction": (
                    sum(row["proxy_grouped_motif_match"] for row in items if row["proxy_pair_available"] and row["interaction_slice"])
                    / max(sum(row["proxy_pair_available"] and row["interaction_slice"] for row in items), 1)
                ),
                "symbolic_grouped_motif_match_noninteraction": (
                    sum(row["symbolic_grouped_motif_match"] for row in items if row["symbolic_pair_available"] and not row["interaction_slice"])
                    / max(sum(row["symbolic_pair_available"] and not row["interaction_slice"] for row in items), 1)
                ),
                "proxy_grouped_motif_match_noninteraction": (
                    sum(row["proxy_grouped_motif_match"] for row in items if row["proxy_pair_available"] and not row["interaction_slice"])
                    / max(sum(row["proxy_pair_available"] and not row["interaction_slice"] for row in items), 1)
                ),
                "proxy_preserved_clean_rate": sum(row["proxy_preserved_clean"] for row in items) / n,
                "proxy_total_semantic_collateral": sum(row["proxy_total_semantic_collateral"] for row in items) / n,
            }
        )
    return out


def summarize_weak_slice(rows):
    filtered = [
        row for row in rows
        if row["task"] == "right_hand_motion->opening" and row["interaction_slice"]
    ]
    by_key = defaultdict(list)
    for row in filtered:
        by_key[(row["other_hand_motion"], row["interaction_motion_value"], row["coordination_class"])].append(row)
    out = []
    for key, items in sorted(by_key.items()):
        other_hand_motion, interaction_motion_value, coord_class = key
        n = len(items)
        sym_avail = sum(row["symbolic_pair_available"] for row in items)
        prox_avail = sum(row["proxy_pair_available"] for row in items)
        out.append(
            {
                "other_hand_motion": other_hand_motion,
                "interaction_motion_value": interaction_motion_value,
                "coordination_class": coord_class,
                "num_frames": n,
                "symbolic_grouped_motif_match": (
                    sum(row["symbolic_grouped_motif_match"] for row in items if row["symbolic_pair_available"]) / max(sym_avail, 1)
                ),
                "proxy_grouped_motif_match": (
                    sum(row["proxy_grouped_motif_match"] for row in items if row["proxy_pair_available"]) / max(prox_avail, 1)
                ),
                "symbolic_state_agreement": (
                    sum(row["symbolic_state_agreement"] for row in items if row["symbolic_pair_available"]) / max(sym_avail, 1)
                ),
                "proxy_state_agreement": (
                    sum(row["proxy_state_agreement"] for row in items if row["proxy_pair_available"]) / max(prox_avail, 1)
                ),
                "symbolic_transition_agreement": (
                    sum(row["symbolic_transition_agreement"] for row in items if row["symbolic_pair_available"]) / max(sym_avail, 1)
                ),
                "proxy_transition_agreement": (
                    sum(row["proxy_transition_agreement"] for row in items if row["proxy_pair_available"]) / max(prox_avail, 1)
                ),
                "proxy_preserved_clean_rate": sum(row["proxy_preserved_clean"] for row in items) / n,
            }
        )
    return out


def main():
    train_data = load_json(GEN / "temporal_hl_val.json")
    test_data = load_json(GEN / "temporal_hl_test.json")
    labels = overlap_labels(train_data, test_data)
    semantic_vocab = build_semantic_frame_vocab(train_data, labels)
    pair_bank = build_pair_bank(train_data, labels, semantic_vocab)

    semantic_decoded = source_cluster_decoded("semantic_frame", pair_bank)
    continuous_decoded = source_cluster_decoded("continuous_frame", pair_bank)

    payload = {
        "artifacts": {
            "train_json": str(GEN / "temporal_hl_val.json"),
            "test_json": str(GEN / "temporal_hl_test.json"),
        },
        "editor": {
            "type": "target_hand_delta_composition",
            "description": "compose donor target-hand transition onto the original previous-frame hand geometry",
            "tasks": [f"{field}->{value}" for field, value in TASKS],
            "interaction_edits_supported": False,
        },
        "sources": [],
    }

    for source_name, decoded in [("semantic_frame", semantic_decoded), ("continuous_frame", continuous_decoded)]:
        rows = []
        for sequence in test_data["sequences"]:
            if canonical(sequence["seq_name"]) not in labels and sequence["seq_name"] not in labels:
                continue
            frames = sequence["frames"]
            for task_field, task_target in TASKS:
                for start, end, value in contiguous_runs(frames, task_field):
                    if end - start < 3 or not eligible_value(task_field, value, task_target):
                        continue
                    for idx in range(max(start, 1), end):
                        prev_frame = frames[idx - 1]
                        curr_frame = frames[idx]
                        curr_attrs = frame_attrs(curr_frame)
                        test_prev_geom = frame_geom(prev_frame)
                        test_curr_geom = frame_geom(curr_frame)

                        sym = pick_best_symbolic_pair(pair_bank, curr_attrs, test_prev_geom, test_curr_geom, task_field, task_target)
                        proxy_attrs = best_proxy_attrs(curr_attrs, task_field, task_target, decoded)
                        prox = None if proxy_attrs is None else pick_best_proxy_pair(
                            pair_bank, source_name, decoded, proxy_attrs, test_prev_geom, test_curr_geom, task_field, task_target
                        )

                        proxy_preserved_mismatch = 0 if proxy_attrs is None else sum(
                            1 for field in preserved_fields(task_field) if proxy_attrs[field] != curr_attrs[field]
                        )
                        proxy_total_semantic_collateral = 0 if proxy_attrs is None else sum(
                            1 for field in TRACKED_FIELDS if field != task_field and proxy_attrs[field] != curr_attrs[field]
                        )

                        sym_edited = None if sym is None else compose_target_hand_transition(test_prev_geom, sym, task_field)
                        prox_edited = None if prox is None else compose_target_hand_transition(test_prev_geom, prox, task_field)

                        sym_success = int(sym_edited is not None and geom_motion(test_prev_geom[target_hand_name(task_field)], sym_edited) == task_target)
                        prox_success = int(prox_edited is not None and geom_motion(test_prev_geom[target_hand_name(task_field)], prox_edited) == task_target)
                        sym_motif = None if sym_edited is None else summarize_composed_motif(prev_frame, sym_edited, sym, task_field)
                        prox_motif = None if prox_edited is None else summarize_composed_motif(prev_frame, prox_edited, prox, task_field)
                        donor_transition_labels = [] if sym_motif is None else list(sym_motif["donor_transition_labels"])
                        donor_hand_motion = "unknown" if sym_motif is None else str(sym_motif["donor_hand_motion"])
                        donor_grouped = grouped_motif_signature(donor_hand_motion, donor_transition_labels) if donor_transition_labels else "unknown"
                        sym_grouped = "unknown" if sym_motif is None else grouped_motif_signature(str(sym_motif["edited_hand_motion"]), list(sym_motif["edited_transition_labels"]))
                        prox_grouped = "unknown" if prox_motif is None else grouped_motif_signature(str(prox_motif["edited_hand_motion"]), list(prox_motif["edited_transition_labels"]))
                        donor_active = int(bool(donor_transition_labels) and active_finger_count(donor_transition_labels) >= 1)
                        donor_dense = int(bool(donor_transition_labels) and active_finger_count(donor_transition_labels) >= 2)
                        other_hand_name = "left" if target_hand_name(task_field) == "right" else "right"
                        other_hand_motion = "none" if curr_frame.get(other_hand_name) is None else str(curr_frame.get(other_hand_name).get("hand_motion", "unknown"))
                        interaction_motion_value = str(curr_frame.get("interaction_motion", "unknown"))
                        coord_class = coordination_class(
                            "none" if curr_frame.get("right") is None else str(curr_frame.get("right").get("hand_motion", "unknown")),
                            "none" if curr_frame.get("left") is None else str(curr_frame.get("left").get("hand_motion", "unknown")),
                        )

                        sym_residual = 0.0
                        if sym_edited is not None:
                            sym_residual = hand_delta(sym_edited, sym["curr_geom"][target_hand_name(task_field)])
                        prox_residual = 0.0
                        if prox_edited is not None:
                            prox_residual = hand_delta(prox_edited, prox["curr_geom"][target_hand_name(task_field)])

                        rows.append(
                            {
                                "task": f"{task_field}->{task_target}",
                                "symbolic_pair_available": int(sym is not None),
                                "proxy_pair_available": int(prox is not None),
                                "both_available": int(sym is not None and prox is not None),
                                "symbolic_composed_success": sym_success,
                                "proxy_composed_success": prox_success,
                                "symbolic_beats_proxy": int(sym is not None and prox is not None and sym_success > prox_success),
                                "symbolic_to_donor_residual": sym_residual,
                                "proxy_to_donor_residual": prox_residual,
                                "symbolic_state_agreement": 0.0 if sym_motif is None else sym_motif["state_agreement"],
                                "proxy_state_agreement": 0.0 if prox_motif is None else prox_motif["state_agreement"],
                                "symbolic_transition_agreement": 0.0 if sym_motif is None else sym_motif["transition_agreement"],
                                "proxy_transition_agreement": 0.0 if prox_motif is None else prox_motif["transition_agreement"],
                                "symbolic_grouped_motif_match": int(sym_motif is not None and sym_grouped == donor_grouped),
                                "proxy_grouped_motif_match": int(prox_motif is not None and prox_grouped == donor_grouped),
                                "donor_active_slice": donor_active,
                                "donor_dense_slice": donor_dense,
                                "interaction_slice": int(curr_frame.get("hand_type") == "interacting"),
                                "other_hand_motion": other_hand_motion,
                                "interaction_motion_value": interaction_motion_value,
                                "coordination_class": coord_class,
                                "proxy_preserved_clean": int(proxy_preserved_mismatch == 0),
                                "proxy_total_semantic_collateral": proxy_total_semantic_collateral,
                            }
                        )
        payload["sources"].append(
            {
                "source": source_name,
                "summary": summarize(rows),
                "weak_slice_summary": summarize_weak_slice(rows),
                "rows": rows,
            }
        )

    out_json = GEN / "transition_conditioned_symbolic_editor.json"
    out_md = SUM / "transition_conditioned_symbolic_editor.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Transition-Conditioned Symbolic Editor",
        "",
        "This is an experiment memo, not paper text.",
        "",
        "This evaluator composes donor target-hand transitions onto the original previous-frame context and checks whether the edited transition still realizes the requested hand-motion label.",
        "",
        "Interaction edits are not included here because the current exported representation does not expose editable global cross-hand geometry.",
    ]
    for source in payload["sources"]:
        lines.extend(
            [
                "",
                f"## Source: {source['source']}",
                "",
                "| task | frames | symbolic pair avail | proxy pair avail | symbolic composed success | proxy composed success | symbolic donor residual | proxy donor residual | symbolic state agree | proxy state agree | symbolic transition agree | proxy transition agree | symbolic grouped motif | proxy grouped motif | active donor slice | symbolic grouped active | proxy grouped active | dense donor slice | symbolic grouped dense | proxy grouped dense | interaction slice | symbolic grouped interaction | proxy grouped interaction | symbolic grouped noninteraction | proxy grouped noninteraction | proxy preserved clean | proxy semantic collateral |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in source["summary"]:
            lines.append(
                f"| {row['task']} | {row['num_frames']} | {fmt(row['symbolic_pair_available_rate'])} | {fmt(row['proxy_pair_available_rate'])} | "
                f"{fmt(row['symbolic_composed_success_rate'])} | {fmt(row['proxy_composed_success_rate'])} | "
                f"{fmt(row['symbolic_to_donor_residual'])} | {fmt(row['proxy_to_donor_residual'])} | {fmt(row['symbolic_state_agreement'])} | "
                f"{fmt(row['proxy_state_agreement'])} | {fmt(row['symbolic_transition_agreement'])} | {fmt(row['proxy_transition_agreement'])} | "
                f"{fmt(row['symbolic_grouped_motif_match'])} | {fmt(row['proxy_grouped_motif_match'])} | {fmt(row['donor_active_slice_rate'])} | "
                f"{fmt(row['symbolic_grouped_motif_match_active'])} | {fmt(row['proxy_grouped_motif_match_active'])} | {fmt(row['donor_dense_slice_rate'])} | "
                f"{fmt(row['symbolic_grouped_motif_match_dense'])} | {fmt(row['proxy_grouped_motif_match_dense'])} | {fmt(row['interaction_slice_rate'])} | "
                f"{fmt(row['symbolic_grouped_motif_match_interaction'])} | {fmt(row['proxy_grouped_motif_match_interaction'])} | "
                f"{fmt(row['symbolic_grouped_motif_match_noninteraction'])} | {fmt(row['proxy_grouped_motif_match_noninteraction'])} | "
                f"{fmt(row['proxy_preserved_clean_rate'])} | {fmt(row['proxy_total_semantic_collateral'])} |"
            )
        if source["weak_slice_summary"]:
            lines.extend(
                [
                    "",
                    "### Weak Slice Breakdown: `right_hand_motion->opening` within interaction",
                    "",
                    "| other hand motion | interaction motion | coordination | frames | symbolic grouped motif | proxy grouped motif | symbolic state agree | proxy state agree | symbolic transition agree | proxy transition agree | proxy preserved clean |",
                    "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for row in source["weak_slice_summary"]:
                lines.append(
                    f"| {row['other_hand_motion']} | {row['interaction_motion_value']} | {row['coordination_class']} | {row['num_frames']} | "
                    f"{fmt(row['symbolic_grouped_motif_match'])} | {fmt(row['proxy_grouped_motif_match'])} | {fmt(row['symbolic_state_agreement'])} | "
                    f"{fmt(row['proxy_state_agreement'])} | {fmt(row['symbolic_transition_agreement'])} | {fmt(row['proxy_transition_agreement'])} | "
                    f"{fmt(row['proxy_preserved_clean_rate'])} |"
                )
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
