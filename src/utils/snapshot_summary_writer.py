# src/utils/snapshot_summary_writer.py
# 📊 Snapshot Summary Writer — exports per-step diagnostics to CSV

import os
import csv

def write_step_summary(
    step_index: int,
    reflex_metadata: dict,
    output_folder: str = "data/summaries"
):
    """
    Appends per-step simulation summary fields to snapshot_summary.csv

    Args:
        step_index (int): Simulation step index
        reflex_metadata (dict): Reflex metadata dictionary from evolve_step
        output_folder (str): Directory to store summary file
    """
    os.makedirs(output_folder, exist_ok=True)
    summary_path = os.path.join(output_folder, "snapshot_summary.csv")

    # Fields to log
    fields = [
        "step_index",
        "reflex_score",
        "adaptive_timestep",
        "mutation_density",
        "pressure_mutated",
        "projection_passes",
        "ghost_influence_count",
        "max_divergence",
        "mean_divergence",
        "ghost_adjacent_but_influence_suppressed"  # ✅ Audit flag via tagging
    ]

    # Diagnostic suppression check based on influence tagging only
    suppression_flag = (
        reflex_metadata.get("fluid_cells_modified_by_ghost", 0) == 0 and
        reflex_metadata.get("ghost_influence_count", 0) > 0
    )

    # Construct row
    row = {
        "step_index": step_index,
        "reflex_score": reflex_metadata.get("reflex_score", ""),
        "adaptive_timestep": reflex_metadata.get("adaptive_timestep", ""),
        "mutation_density": reflex_metadata.get("mutation_density", ""),
        "pressure_mutated": reflex_metadata.get("pressure_mutated", ""),
        "projection_passes": reflex_metadata.get("projection_passes", ""),
        "ghost_influence_count": reflex_metadata.get("ghost_influence_count", ""),
        "max_divergence": reflex_metadata.get("post_projection_divergence", ""),
        "mean_divergence": reflex_metadata.get("mean_divergence", ""),
        "ghost_adjacent_but_influence_suppressed": suppression_flag
    }

    # Write header if new file
    file_exists = os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"[SUMMARY] 📊 Step {step_index} summary written → {summary_path}")



