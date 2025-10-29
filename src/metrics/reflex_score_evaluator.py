# src/metrics/reflex_score_evaluator.py
# 📊 Reflex Score Evaluator — audits both summary logs and snapshot trace integrity
# 📌 Evaluates solver trace fidelity and mutation causality.
# Supports reflex scoring, suppression diagnostics, and projection depth analysis.
# Does NOT exclude cells based on adjacency or boundary proximity — only explicit fluid_mask=False cells are excluded upstream.

import os
import statistics
import json

# ✅ Centralized debug flag for GitHub Actions logging
debug = False


def evaluate_reflex_score(summary_file_path: str) -> dict:
    if not os.path.isfile(summary_file_path):
        raise FileNotFoundError(f"🔍 Summary file not found → {summary_file_path}")

    with open(summary_file_path, "r") as f:
        lines = f.readlines()

    step_scores = {}
    current_step = None
    score_components = {}

    for line in lines:
        line = line.strip()
        if line.startswith("[🔄 Step"):
            if current_step is not None and score_components:
                score = compute_score(score_components)
                step_scores[current_step] = score
            current_step = int(line.split("Step")[1].split("Summary")[0].strip())
            score_components = {}
        elif "Influence applied" in line:
            score_components["influence"] = int(line.split(":")[1].strip())
        elif "Fluid–ghost adjacents" in line:
            raw = line.split(":")[1].strip()
            score_components["adjacency"] = int(raw) if raw.isdigit() else 0
        elif "Pressure mutated" in line:
            score_components["mutation"] = "True" in line
        elif "Suppression zones" in line:
            raw = line.split(":")[1].strip()
            score_components["suppression"] = int(raw) if raw.isdigit() else 0

    if current_step is not None and score_components:
        step_scores[current_step] = compute_score(score_components)

    return {
        "step_scores": step_scores,
        "average_score": statistics.mean(step_scores.values()) if step_scores else 0.0,
        "max_score": max(step_scores.values(), default=0.0),
        "min_score": min(step_scores.values(), default=0.0),
        "step_count": len(step_scores),
    }


def compute_score(inputs: dict) -> float:
    mutation = inputs.get("mutation", False)
    adjacency = inputs.get("adjacency", 0)
    influence = inputs.get("influence", 0)
    suppression = inputs.get("suppression", 0)
    mutation_density = inputs.get("mutation_density", 0.0)
    projection_passes = inputs.get("projection_passes", 1)
    triggered_by = inputs.get("triggered_by", [])
    boundary_mutation_ratio = inputs.get("boundary_mutation_ratio", 0.0)

    score = 0.0
    if mutation:
        if influence > 0:
            score += 2.0
        elif adjacency > 0:
            score += 0.2
        else:
            score += 0.1

        if mutation_density > 0.2:
            score += 0.5
        elif mutation_density > 0.1:
            score += 0.2

        if projection_passes > 1:
            score += 0.2 * min(projection_passes, 5)

        if "ghost_influence" in triggered_by:
            score += 0.3

        if boundary_mutation_ratio > 0.5 and suppression > 0:
            score -= 0.3
        elif boundary_mutation_ratio > 0.25 and suppression > 0:
            score -= 0.1

    if suppression > 0 and not mutation:
        score -= 0.1 * suppression

    return round(max(score, 0.0), 3)


# 🧩 STUB — pressure mutation volume scoring
def score_pressure_mutation_volume(delta_map: dict) -> int:
    return sum(1 for cell in delta_map.values() if abs(cell.get("delta", 0.0)) > 0.0)


# 🧩 STUB — mutation pathway presence detection
def score_mutation_pathway_presence(trace: list, step_index: int) -> bool:
    return any(entry.get("step_index") == step_index for entry in trace)


# 🧩 STUB — reflex metadata field extraction
def score_reflex_metadata_fields(reflex: dict) -> dict:
    return {
        "has_projection": reflex.get("pressure_solver_invoked", False),
        "divergence_logged": "post_projection_divergence" in reflex,
        "reflex_score": reflex.get("reflex_score", 0.0),
        "suppression_zone_count": len(reflex.get("suppression_zones", [])),
        "mutation_density": reflex.get("mutation_density", 0.0),
        "projection_passes": reflex.get("projection_passes", 1),
        "adjacency_count": len(reflex.get("adjacency_zones", [])),
        "triggered_by": reflex.get("triggered_by", []),
        "boundary_mutation_ratio": reflex.get("boundary_mutation_ratio", 0.0),
    }


def evaluate_snapshot_health(step_index: int, delta_map_path: str, pathway_log_path: str, reflex_metadata: dict) -> dict:
    with open(delta_map_path) as f:
        delta_map = json.load(f)
    with open(pathway_log_path) as f:
        trace = json.load(f)

    mutated_cells = score_pressure_mutation_volume(delta_map)
    pathway_recorded = score_mutation_pathway_presence(trace, step_index)
    metadata = score_reflex_metadata_fields(reflex_metadata)
    score = compute_score(reflex_metadata)

    return {
        "step_index": step_index,
        "mutated_cells": mutated_cells,
        "pathway_recorded": pathway_recorded,
        "reflex_score": score,
        **metadata,
    }


# 🧩 STUB — batch audit scoring interface (CLI-compatible)
def batch_evaluate_trace(snapshot_dir: str, pathway_log: str, snapshot_list: list) -> list:
    if debug:
        print(f"[AUDIT] Evaluating {len(snapshot_list)} snapshots from {snapshot_dir}")
    return [compute_score(snapshot.get("reflex_components", {})) for snapshot in snapshot_list]


__all__ = [
    "evaluate_reflex_score",
    "compute_score",
    "evaluate_snapshot_health",
    "score_pressure_mutation_volume",
    "score_mutation_pathway_presence",
    "score_reflex_metadata_fields",
    "batch_evaluate_trace",
]
