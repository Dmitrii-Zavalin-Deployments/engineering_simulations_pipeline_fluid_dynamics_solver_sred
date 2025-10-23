# src/utils/snapshot_summary_writer.py
# 📊 Snapshot Summary Writer — exports per-step diagnostics to CSV
# 📌 This module logs reflex score, mutation density, divergence, ghost adjacency, and suppression fallback.
# It excludes only cells explicitly marked fluid_mask=False.
# It does NOT skip based on adjacency or ghost proximity — all logic is geometry-mask-driven.

import os
import csv

# ✅ Centralized debug flag for GitHub Actions logging
debug = True

def write_step_summary(
    step_index: int,
    reflex_metadata: dict,
    output_folder: str = "data/summaries"
):
    """
    Appends per-step simulation summary fields to snapshot_summary.csv

    Roadmap Alignment:
    Reflex Scoring:
    - Logs reflex score, mutation density, divergence, ghost influence
    - Tracks suppression fallback, damping triggers, overflow flags, CFL diagnostics
    - Logs ghost adjacency count and boundary mutation ratio

    Args:
        step_index (int): Simulation step index
        reflex_metadata (dict): Snapshot metadata dictionary containing reflex metrics
        output_folder (str): Directory to store summary file
    """
    os.makedirs(output_folder, exist_ok=True)
    summary_path = os.path.join(output_folder, "snapshot_summary.csv")

    fields = [
        "step_index",
        "reflex_score",
        "adaptive_timestep",
        "mutation_density",
        "pressure_mutated",
        "projection_passes",
        "ghost_influence_count",
        "ghost_adjacency_count",
        "boundary_cell_count",
        "boundary_mutation_ratio",
        "max_divergence",
        "mean_divergence",
        "ghost_adjacent_but_influence_suppressed",
        "mutation_count",
        "damping_triggered_count",
        "overflow_triggered_count",
        "cfl_exceeded_count"
    ]

    suppression_flag = (
        reflex_metadata.get("fluid_cells_modified_by_ghost", 0) == 0 and
        reflex_metadata.get("ghost_influence_count", 0) > 0
    )

    reflex_score = reflex_metadata.get("reflex_score", 0.0)
    if not isinstance(reflex_score, (int, float)):
        try:
            reflex_score = float(reflex_score)
        except:
            reflex_score = 0.0

    if debug:
        print(f"[SUMMARY] Reflex score for step {step_index}: {reflex_score}")

    mutation_count = len(reflex_metadata.get("mutated_cells", []))
    damping_count = reflex_metadata.get("damping_triggered_count", 0)
    overflow_count = reflex_metadata.get("overflow_triggered_count", 0)
    cfl_count = reflex_metadata.get("cfl_exceeded_count", 0)
    adjacency_count = len(reflex_metadata.get("adjacency_zones", []))
    boundary_count = reflex_metadata.get("boundary_cell_count", 0)
    boundary_ratio = reflex_metadata.get("boundary_mutation_ratio", 0.0)

    row = {
        "step_index": step_index,
        "reflex_score": reflex_score,
        "adaptive_timestep": reflex_metadata.get("adaptive_timestep", ""),
        "mutation_density": reflex_metadata.get("mutation_density", ""),
        "pressure_mutated": reflex_metadata.get("pressure_mutated", ""),
        "projection_passes": reflex_metadata.get("projection_passes", ""),
        "ghost_influence_count": reflex_metadata.get("ghost_influence_count", ""),
        "ghost_adjacency_count": adjacency_count,
        "boundary_cell_count": boundary_count,
        "boundary_mutation_ratio": round(boundary_ratio, 4),
        "max_divergence": reflex_metadata.get("post_projection_divergence", ""),
        "mean_divergence": reflex_metadata.get("mean_divergence", ""),
        "ghost_adjacent_but_influence_suppressed": suppression_flag,
        "mutation_count": mutation_count,
        "damping_triggered_count": damping_count,
        "overflow_triggered_count": overflow_count,
        "cfl_exceeded_count": cfl_count
    }

    file_exists = os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    if debug:
        print(f"[SUMMARY] 📊 Step {step_index} summary written → {summary_path}")



