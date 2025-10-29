# src/diagnostics/navier_stokes_verifier.py
# 🧠 Navier-Stokes Verifier — validates continuity, pressure consistency,
# and downgrade diagnostics when triggered
# 📌 This module operates on post-simulation grid data.
# It verifies divergence, pressure anomalies, and downgrade reasons.
# It enforces that only fluid_mask=False cells are excluded from solver
# routines.

import json
import pathlib
from typing import List, Tuple
from src.grid_modules.cell import Cell
from src.physics.divergence import compute_divergence

# ✅ Centralized debug flag for GitHub Actions logging
debug = False


def verify_continuity(
    grid: List[Cell],
    spacing: Tuple[float, float, float],
    step_index: int,
    output_folder: str
):
    """
    Verifies ∇ · u = 0 across fluid cells to ensure mass conservation.
    """
    divergence = compute_divergence(grid, config={}, ghost_registry=set())
    max_div = max(abs(d) for d in divergence) if divergence else 0.0
    mean_div = (
        sum(abs(d) for d in divergence) / len(divergence)
        if divergence else 0.0
    )

    result = {
        "step_index": step_index,
        "max_divergence": max_div,
        "mean_divergence": mean_div,
        "status": "PASS" if max_div < 1e-6 else "FAIL"
    }

    log_path = (
        pathlib.Path(output_folder) /
        f"continuity_verification_step_{step_index:04d}.json"
    )
    with open(log_path, "w") as f:
        json.dump(result, f, indent=2)

    if debug:
        print(
            f"[VERIFIER] Continuity check → max ∇·u = {max_div:.3e}, "
            f"mean = {mean_div:.3e}"
        )
        print(
            f"[VERIFIER] Status: {result['status']} → saved to {log_path}"
        )


def verify_pressure_consistency(
    grid: List[Cell],
    step_index: int,
    output_folder: str
):
    """
    Flags fluid cells with extreme pressure values.
    Placeholder for future momentum-balance verification.
    """
    flagged = []
    for cell in grid:
        if cell.fluid_mask and isinstance(cell.pressure, float):
            if abs(cell.pressure) > 1e5:
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

    log_path = (
        pathlib.Path(output_folder) /
        f"pressure_verification_step_{step_index:04d}.json"
    )
    with open(log_path, "w") as f:
        json.dump(result, f, indent=2)

    if debug:
        print(
            f"[VERIFIER] Pressure consistency check → "
            f"{len(flagged)} cells flagged"
        )
        print(
            f"[VERIFIER] Status: {result['status']} → saved to {log_path}"
        )


def verify_downgraded_cells(
    grid: List[Cell],
    step_index: int,
    output_folder: str
):
    """
    Logs downgraded cells and reasons for exclusion from pressure correction.
    Only flags cells with malformed velocity or fluid_mask=False.
    """
    downgraded = []
    for i, cell in enumerate(grid):
        reasons = []
        if not isinstance(cell.velocity, list):
            reasons.append("missing or malformed velocity")
        if not cell.fluid_mask:
            reasons.append("fluid_mask=False")

        downgraded.append({
            "index": i,
            "x": cell.x,
            "y": cell.y,
            "z": cell.z,
            "reasons": reasons if reasons else ["no downgrade reason detected"]
        })

    result = {
        "step_index": step_index,
        "downgraded_cell_count": len(downgraded),
        "downgraded_cells": downgraded
    }

    log_path = (
        pathlib.Path(output_folder) /
        f"downgrade_verification_step_{step_index:04d}.json"
    )
    with open(log_path, "w") as f:
        json.dump(result, f, indent=2)

    if debug:
        print(
            f"[VERIFIER] Downgrade check → {len(downgraded)} cells processed"
        )
        print(
            f"[VERIFIER] Downgrade log saved to {log_path}"
        )


def run_verification_if_triggered(
    grid: List[Cell],
    spacing: Tuple[float, float, float],
    step_index: int,
    output_folder: str,
    triggered_flags: List[str]
):
    """
    Runs verification routines if any diagnostic flags are triggered.

    FIX: Ensure the output directory exists before any logging or file writes.
    This resolves the FileNotFoundError by handling directory creation centrally.
    """
    if not triggered_flags:
        return

    output_path = pathlib.Path(output_folder)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(
            f"[ERROR] Could not create verification output directory "
            f"{output_folder}: {e}"
        )
        return

    if debug:
        print(f"[VERIFIER] Triggered flags: {triggered_flags}")
        print(
            f"[VERIFIER] Running physics-based verification for step "
            f"{step_index}"
        )

    if (
        "empty_divergence" in triggered_flags or
        "no_pressure_mutation" in triggered_flags
    ):
        verify_continuity(grid, spacing, step_index, output_folder)

    if "no_pressure_mutation" in triggered_flags:
        verify_pressure_consistency(grid, step_index, output_folder)

    if "downgraded_cells" in triggered_flags:
        verify_downgraded_cells(grid, step_index, output_folder)
