# src/physics/divergence_tracker.py
# 📊 Divergence Tracker — computes and logs ∇ · u for continuity enforcement and reflex diagnostics
# 📌 This module tracks divergence across fluid cells and exports reflex metadata.
# It excludes ghost cells and cells explicitly marked fluid_mask=False.
# It does NOT skip based on adjacency or boundary proximity.

import os
import json
from typing import List, Dict, Set
from src.grid_modules.cell import Cell
from src.physics.divergence import compute_divergence

# ✅ Centralized debug flag for GitHub Actions logging
debug = True

def compute_divergence_stats(
    grid: List[Cell],
    spacing: tuple,
    label: str,
    step_index: int,
    output_folder: str,
    config: Dict,
    ghost_registry: Set[int] = set()
) -> Dict:
    """
    Computes divergence across the fluid grid and logs diagnostic stats.
    """
    dx, dy, dz = spacing

    # 🧮 Compute divergence across valid fluid cells
    divergence = compute_divergence(grid, config=config, ghost_registry=ghost_registry)

    max_div = max(abs(d) for d in divergence) if divergence else 0.0
    mean_div = sum(abs(d) for d in divergence) / len(divergence) if divergence else 0.0

    if debug:
        print(f"[TRACKER] Step {step_index} — {label}: max={max_div:.3e}, mean={mean_div:.3e}")
        for i, d in enumerate(divergence):
            print(f"[TRACKER] Divergence[{i}] = {d:.6e}")

    # 🧠 Annotate cells with divergence for reflex metadata
    fluid_index = 0
    for cell in grid:
        if getattr(cell, "fluid_mask", False) and id(cell) not in ghost_registry:
            cell.divergence = round(divergence[fluid_index], 6)
            fluid_index += 1

    # 🗂️ Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # 🗂️ Export summary log
    log_path = os.path.join(output_folder, "divergence_log.txt")
    with open(log_path, "a") as f:
        f.write(f"Step {step_index:04d} | Stage: {label} | Max: {max_div:.6e} | Mean: {mean_div:.6e}\n")

    # 🗂️ Export per-cell divergence map
    map_path = os.path.join(output_folder, f"divergence_map_step_{step_index:04d}.json")
    divergence_map = {
        f"{cell.x:.2f},{cell.y:.2f},{cell.z:.2f}": cell.divergence
        for cell in grid if hasattr(cell, "divergence")
    }
    with open(map_path, "w") as f:
        json.dump(divergence_map, f, indent=2)

    if debug:
        print(f"[TRACKER] Divergence log written → {log_path}")
        print(f"[TRACKER] Divergence map exported → {map_path}")

    return {
        "max": max_div,
        "mean": mean_div,
        "divergence": divergence
    }



