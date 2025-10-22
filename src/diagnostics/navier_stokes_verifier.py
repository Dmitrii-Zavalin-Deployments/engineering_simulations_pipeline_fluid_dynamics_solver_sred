# src/diagnostics/navier_stokes_verifier.py
# 🧠 Navier-Stokes Verifier — validates continuity and pressure consistency when diagnostic warnings are triggered

import os
import json
from typing import List, Tuple
from src.grid_modules.cell import Cell
from src.physics.divergence import compute_divergence

# 🛠️ Toggle debug logging
DEBUG = True

def verify_continuity(grid: List[Cell], spacing: Tuple[float, float, float], step_index: int, output_folder: str):
    """
    Verifies ∇ · u = 0 across fluid cells to ensure mass conservation.
    """
    divergence = compute_divergence(grid, config={}, ghost_registry=set())
    max_div = max(abs(d) for d in divergence) if divergence else 0.0
    mean_div = sum(abs(d) for d in divergence) / len(divergence) if divergence else 0.0

    result = {
        "step_index": step_index,
        "max_divergence": max_div,
        "mean_divergence": mean_div,
        "status": "PASS" if max_div < 1e-6 else "FAIL"
    }

    os.makedirs(output_folder, exist_ok=True)
    log_path = os.path.join(output_folder, f"continuity_verification_step_{step_index:04d}.json")
    with open(log_path, "w") as f:
        json.dump(result, f, indent=2)

    if DEBUG:
        print(f"[VERIFIER] Continuity check → max ∇·u = {max_div:.3e}, mean = {mean_div:.3e}")
        print(f"[VERIFIER] Status: {result['status']} → saved to {log_path}")

def verify_pressure_consistency(grid: List[Cell], step_index: int, output_folder: str):
    """
    Placeholder for pressure gradient verification logic.
    Will compare pressure gradients to expected momentum balance in future iterations.
    """
    flagged = []
    for cell in grid:
        if cell.fluid_mask and isinstance(cell.pressure, float):
            if abs(cell.pressure) > 1e5:  # Arbitrary threshold for now
                flagged.append({
                    "x": cell.x,
                    "y": cell.y,
                    "z": cell.z,
                    "pressure": cell.pressure
                })

    result = {
        "step_index": step_index,
        "flagged_cells": flagged,
        "status": "PASS" if not flagged else "WARN"
    }

    os.makedirs(output_folder, exist_ok=True)
    log_path = os.path.join(output_folder, f"pressure_verification_step_{step_index:04d}.json")
    with open(log_path, "w") as f:
        json.dump(result, f, indent=2)

    if DEBUG:
        print(f"[VERIFIER] Pressure consistency check → {len(flagged)} cells flagged")
        print(f"[VERIFIER] Status: {result['status']} → saved to {log_path}")

def run_verification_if_triggered(
    grid: List[Cell],
    spacing: Tuple[float, float, float],
    step_index: int,
    output_folder: str,
    triggered_flags: List[str]
):
    """
    Runs verification routines if any diagnostic flags are triggered.
    """
    if not triggered_flags:
        return

    if DEBUG:
        print(f"[VERIFIER] Triggered flags: {triggered_flags}")
        print(f"[VERIFIER] Running physics-based verification for step {step_index}")

    verify_continuity(grid, spacing, step_index, output_folder)
    verify_pressure_consistency(grid, step_index, output_folder)



