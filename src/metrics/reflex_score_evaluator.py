# src/metrics/reflex_score_evaluator.py
# 📊 Reflex Score Evaluator — audits both summary logs and snapshot trace integrity

import os
import json
import statistics
from typing import List

# 🔍 Line-based scoring from step_summary.txt
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
        "step_count": len(step_scores)
    }

# ✅ Reflex scoring logic — maps mutation causality to physical enforcement
def compute_score(inputs: dict) -> float:
    mutation = inputs.get("mutation", False)
    adjacency = inputs.get("adjacency", 0)
    influence = inputs.get("influence", 0)
    suppression = inputs.get("suppression", 0)
    mutation_density = inputs.get("mutation_density", 0.0)
    projection_passes = inputs.get("projection_passes", 1)

    print(f"[DEBUG] [score] Inputs → mutation={mutation}, adjacency={adjacency}, influence={influence}, suppression={suppression}, density={mutation_density}, passes={projection_passes}")

    score = 0.0
    if mutation:
        if influence > 0:
            score += 2.0
        elif adjacency > 0:
            score += 0.2
        else:
            score += 0.1

        # Reward higher mutation density
        if mutation_density > 0.2:
            score += 0.5
        elif mutation_density > 0.1:
            score += 0.2

        # Reward deeper projection
        if projection_passes > 1:
            score += 0.2 * min(projection_passes, 5)

    # 🧭 Suppression penalty
    if suppression > 0 and not mutation:
        score -= 0.1 * suppression

    score = max(score, 0.0)
    print(f"[DEBUG] [score] Final score={score}")
    return round(score, 3)

# 🧠 Snapshot-based scoring — evaluates trace integrity and solver metadata
def load_json_safe(path: str):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def score_pressure_mutation_volume(delta_map: dict) -> int:
    return sum(1 for cell in delta_map.values() if abs(cell.get("delta", 0.0)) > 0.0)

def score_mutation_pathway_presence(trace: list, step_index: int) -> bool:
    return any(entry.get("step_index") == step_index for entry in trace)

def score_reflex_metadata_fields(reflex: dict) -> dict:
    return {
        "has_projection": reflex.get("pressure_solver_invoked", False),
        "divergence_logged": "post_projection_divergence" in reflex,
        "reflex_score": reflex.get("reflex_score", 0),
        "suppression_zone_count": len(reflex.get("suppression_zones", [])),
        "mutation_density": reflex.get("mutation_density", 0.0),
        "projection_passes": reflex.get("projection_passes", 1),
        "adjacency_count": len(reflex.get("adjacency_zones", []))  # ✅ Added for trace visibility
    }

def evaluate_snapshot_health(
    step_index: int,
    delta_map_path: str,
    pathway_log_path: str,
    reflex_metadata: dict
) -> dict:
    delta_map = load_json_safe(delta_map_path) or {}
    pathway_trace = load_json_safe(pathway_log_path) or []

    mutation_count = score_pressure_mutation_volume(delta_map)
    pathway_exists = score_mutation_pathway_presence(pathway_trace, step_index)
    field_checks = score_reflex_metadata_fields(reflex_metadata)

    score_inputs = {
        "mutation": reflex_metadata.get("pressure_mutated", False),
        "adjacency": field_checks["adjacency_count"],
        "influence": reflex_metadata.get("ghost_influence_count", 0),
        "suppression": field_checks["suppression_zone_count"],
        "mutation_density": field_checks["mutation_density"],
        "projection_passes": field_checks["projection_passes"]
    }

    reflex_score = compute_score(score_inputs)

    return {
        "step_index": step_index,
        "mutated_cells": mutation_count,
        "pathway_recorded": pathway_exists,
        "has_projection": field_checks["has_projection"],
        "divergence_logged": field_checks["divergence_logged"],
        "reflex_score": reflex_score,
        "suppression_zone_count": field_checks["suppression_zone_count"],
        "adjacency_count": field_checks["adjacency_count"]  # ✅ Added for overlay and scoring trace
    }

def batch_evaluate_trace(
    trace_folder: str,
    pathway_log_path: str,
    reflex_snapshots: List[dict]
) -> List[dict]:
    evaluations = []
    for snapshot in reflex_snapshots:
        step = snapshot.get("step_index")
        delta_path = os.path.join(trace_folder, f"pressure_delta_map_step_{step:04d}.json")
        report = evaluate_snapshot_health(
            step_index=step,
            delta_map_path=delta_path,
            pathway_log_path=pathway_log_path,
            reflex_metadata=snapshot
        )
        evaluations.append(report)
    return evaluations



