# src/metrics/reflex_score_evaluator.py
# 📊 Reflex Score Evaluator — audits both summary logs and snapshot trace integrity

import os
import json
import statistics
from typing import List

# 🔍 Existing step_summary.txt scoring (line-based)
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

    if current_step is not None and score_components:
        step_scores[current_step] = compute_score(score_components)

    return {
        "step_scores": step_scores,
        "average_score": statistics.mean(step_scores.values()) if step_scores else 0.0,
        "max_score": max(step_scores.values(), default=0.0),
        "min_score": min(step_scores.values(), default=0.0),
        "step_count": len(step_scores)
    }

# ✅ Patch: fallback scoring logic for suppressed influence
def compute_score(inputs: dict) -> float:
    mutation = inputs.get("mutation", False)
    adjacency = inputs.get("adjacency", 0)
    influence = inputs.get("influence", 0)

    print(f"[DEBUG] [score] Inputs → mutation={mutation}, adjacency={adjacency}, influence={influence}")

    score = 0.0
    if mutation:
        if influence > 0:
            score += 2.0
        elif adjacency > 0:
            print("[DEBUG] [score] Mutation near ghost but influence was suppressed")
            score += 0.2
        elif adjacency == 0 and influence == 0:
            print("[DEBUG] [score] Mutation near ghost but tagging suppressed → soft fallback applied")
            score += 0.2
        else:
            print("[DEBUG] [score] Mutation occurred without ghost relation")

    print(f"[DEBUG] [score] Final score={score}")
    return score

# 🆕 Patch: Structured trace scoring (snapshot-based)
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
        "reflex_score": reflex.get("reflex_score", 0)
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

    return {
        "step_index": step_index,
        "mutated_cells": mutation_count,
        "pathway_recorded": pathway_exists,
        "has_projection": field_checks["has_projection"],
        "divergence_logged": field_checks["divergence_logged"],
        "reflex_score": field_checks["reflex_score"]
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



