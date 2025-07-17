# src/metrics/reflex_score_evaluator.py

import os
import json
import statistics

ADJACENCY_BONUS = 0.15  # Partial credit when ghost adjacency occurs but mutation is suppressed

def evaluate_reflex_score(summary_file_path: str) -> dict:
    """
    Computes reflex-related metrics across simulation steps.

    Parameters:
    - summary_file_path: Path to step_summary.txt file containing simulation logs.

    Returns:
    - Dictionary with step-wise reflex scores and key statistics.
    """
    if not os.path.isfile(summary_file_path):
        raise FileNotFoundError(f"🔍 Summary file not found → {summary_file_path}")

    with open(summary_file_path, "r") as f:
        lines = f.readlines()

    step_scores = {}
    score_tags = {}  # Optional: capture reasoning behind scores
    current_step = None
    score_components = {}

    for line in lines:
        line = line.strip()
        if line.startswith("[🔄 Step"):
            if current_step is not None and score_components:
                score, tags = compute_score(score_components)
                step_scores[current_step] = score
                score_tags[current_step] = tags
            current_step = int(line.split("Step")[1].split("Summary")[0].strip())
            score_components = {}
        elif "Influence applied" in line:
            score_components["influence"] = int(line.split(":")[1].strip())
        elif "Fluid–ghost adjacents" in line:
            raw = line.split(":")[1].strip()
            score_components["adjacency"] = int(raw) if raw.isdigit() else 0
        elif "Pressure mutated" in line:
            score_components["mutation"] = "True" in line
        elif "Influence suppressed" in line:
            raw = line.split(":")[1].strip()
            score_components["suppressed"] = int(raw) if raw.isdigit() else 0

    if current_step is not None and score_components:
        score, tags = compute_score(score_components)
        step_scores[current_step] = score
        score_tags[current_step] = tags

    return {
        "step_scores": step_scores,
        "score_tags": score_tags,
        "average_score": statistics.mean(step_scores.values()) if step_scores else 0.0,
        "max_score": max(step_scores.values(), default=0.0),
        "min_score": min(step_scores.values(), default=0.0),
        "step_count": len(step_scores)
    }

def compute_score(components: dict) -> tuple:
    """
    Calculates a composite reflex score from extracted metrics.

    Weights:
    - influence: 0.5
    - adjacency: 0.3
    - mutation: 0.2

    Bonus:
    - Partial credit if ghost adjacency occurs but mutation is suppressed
    """
    tags = []

    influence_score = min(1.0, components.get("influence", 0) / 10.0)
    adjacency_score = min(1.0, components.get("adjacency", 0) / 10.0)
    mutation_score = 1.0 if components.get("mutation", False) else 0.0

    reflex_score = round(
        0.5 * influence_score +
        0.3 * adjacency_score +
        0.2 * mutation_score,
        4
    )

    # Apply bonus if ghost adjacency occurred but pressure mutation was suppressed
    if not components.get("mutation", False):
        if components.get("adjacency", 0) > 0 or components.get("suppressed", 0) > 0:
            reflex_score = round(reflex_score + ADJACENCY_BONUS, 4)
            tags.append("ghost_adjacency_no_mutation")

    if components.get("mutation", False):
        tags.append("mutation_detected")

    return reflex_score, tags



