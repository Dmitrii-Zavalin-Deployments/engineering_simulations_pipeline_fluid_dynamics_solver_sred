# src/physics/divergence_tracker.py
# 📊 Divergence Tracker — computes and logs ∇ · u for continuity enforcement

import os
import json
from typing import List, Dict
from src.grid_modules.cell import Cell
from src.physics.divergence import compute_divergence

def compute_divergence_stats(
    grid: List[Cell],
    spacing: tuple,
    label: str,
    step_index: int,
    output_folder: str,
    config: Dict
) -> Dict:
    """
    Computes divergence across the fluid grid and logs diagnostic stats.

    Governing Equation:
        ∇ · u = ∂u/∂x + ∂v/∂y + ∂w/∂z

    Purpose:
    - Validate incompressibility
    - Support pressure correction
    - Track solver performance

    Args:
        grid (List[Cell]): Simulation grid with velocity fields
        spacing (tuple): Grid spacing (dx, dy, dz)
        label (str): Diagnostic label (e.g. "before projection")
        step_index (int): Current timestep index
        output_folder (str): Folder to write divergence logs
        config (dict): Full simulation config

    Returns:
        dict: Summary statistics including max, mean, and divergence array
    """
    dx, dy, dz = spacing
    divergence = compute_divergence(grid, dx, dy, dz)

    max_div = max(abs(d) for d in divergence) if divergence else 0.0
    mean_div = sum(abs(d) for d in divergence) / len(divergence) if divergence else 0.0

    print(f"[DIV] Step {step_index} — {label}: max={max_div:.3e}, mean={mean_div:.3e}")

    # Optional export
    log_path = os.path.join(output_folder, "divergence_log.txt")
    with open(log_path, "a") as f:
        f.write(f"Step {step_index:04d} | Stage: {label} | Max: {max_div:.6e} | Mean: {mean_div:.6e}\n")

    return {
        "max": max_div,
        "mean": mean_div,
        "divergence": divergence
    }



