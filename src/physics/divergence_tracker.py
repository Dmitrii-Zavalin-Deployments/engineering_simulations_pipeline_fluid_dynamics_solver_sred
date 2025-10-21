# src/physics/divergence_tracker.py
# 📊 Divergence Tracker — computes and logs ∇ · u for continuity enforcement and reflex diagnostics

import os
import json
from typing import List, Dict, Set
from src.grid_modules.cell import Cell
from src.physics.divergence import compute_divergence

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

    Roadmap Alignment:
    Governing Equation:
        Continuity: ∇ · u = ∂u/∂x + ∂v/∂y + ∂w/∂z

    Modular Enforcement:
    - Velocity field → advection.py, viscosity.py
    - Divergence diagnostics → divergence.py
    - Pressure solve input → pressure_projection.py
    - Reflex scoring → reflex_controller.py

    Purpose:
    - Validate incompressibility before and after pressure correction
    - Support pressure Poisson solve: ∇²P = ∇ · u
    - Track solver performance and continuity enforcement
    - Provide diagnostic visibility for reflex scoring and mutation tracing

    Strategy:
    - Compute divergence at each fluid cell
    - Log max and mean divergence for audit and reflex overlays
    - Export diagnostic trace for scoring and mutation pathway tracking

    Args:
        grid (List[Cell]): Simulation grid with velocity fields
        spacing (tuple): Grid spacing (dx, dy, dz)
        label (str): Diagnostic label (e.g. "before projection")
        step_index (int): Current timestep index
        output_folder (str): Folder to write divergence logs
        config (dict): Full simulation config
        ghost_registry (Set[int]): Set of ghost cell IDs to exclude

    Returns:
        dict: Summary statistics including max, mean, and divergence array
    """
    dx, dy, dz = spacing
    divergence = compute_divergence(grid, config=config, ghost_registry=ghost_registry, debug=False)

    max_div = max(abs(d) for d in divergence) if divergence else 0.0
    mean_div = sum(abs(d) for d in divergence) / len(divergence) if divergence else 0.0

    print(f"[DIV] Step {step_index} — {label}: max={max_div:.3e}, mean={mean_div:.3e}")

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

    return {
        "max": max_div,
        "mean": mean_div,
        "divergence": divergence
    }



