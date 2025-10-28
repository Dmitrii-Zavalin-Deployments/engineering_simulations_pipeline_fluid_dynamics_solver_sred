# src/metrics/reflex_score_evaluator.py
# 📊 Reflex Score Evaluator — audits both summary logs and snapshot trace
# integrity
# 📌 This module evaluates solver trace fidelity and mutation causality.
# It supports reflex scoring, suppression diagnostics, and projection depth
# analysis.
# It does NOT exclude cells based on adjacency or boundary proximity — only
# explicit fluid_mask=False cells are excluded upstream.

import os
import statistics
from src.visualization.reflex_score_visualizer import (
    plot_reflex_score_evolution  # ✅ Added
)

# ✅ Centralized debug flag for GitHub Actions logging
debug = True


# 🔍 Line-based scoring from step_summary.txt
def evaluate_reflex_score(summary_file_path: str) -> dict:
    if not os.path.isfile(summary_file_path):
        raise FileNotFoundError(
            f"🔍 Summary file not found → {summary_file_path}"
        )

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
            current_step = int(
                line.split("Step")[1].split("Summary")[0].strip()
            )
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
        "average_score": statistics.mean(step_scores.values())
        if step_scores
        else 0.0,
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

    if debug:
        print(
            f"[SCORE] Inputs → mutation={mutation}, adjacency={adjacency}, "
            f"influence={influence}, suppression={suppression}, "
            f"density={mutation_density}, passes={projection_passes}, "
            f"triggered_by={triggered_by}, boundary_mutation_ratio={boundary_mutation_ratio}"
        )

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

    score = max(score, 0.0)

    if debug:
        print(f"[SCORE] Final reflex score = {score:.3f}")

    return round(score, 3)
