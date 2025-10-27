# src/solvers/navier_stokes_solver.py
# 🧠 Navier-Stokes Solver — centralized logic for momentum and continuity enforcement
# 📌 This module sequences momentum evolution, pressure correction, and velocity projection.
# It excludes only cells explicitly marked fluid_mask=False.
# It does NOT skip based on adjacency or ghost proximity — all logic is geometry-mask-driven.

from typing import List, Tuple, Dict
from src.grid_modules.cell import Cell
from src.solvers.momentum_solver import apply_momentum_update
from src.solvers.pressure_solver import apply_pressure_correction
from src.physics.velocity_projection import apply_pressure_velocity_projection
# Removed: from src.diagnostics.navier_stokes_verifier import run_verification_if_triggered
# FIX: The verifier is now called exclusively within pressure_solver.py

# ✅ Centralized debug flag for GitHub Actions logging
debug = True

def solve_navier_stokes_step(
    grid: List[Cell],
    input_data: dict,
    step_index: int,
    output_folder: str = "data/testing-input-output/navier_stokes_output"
) -> Tuple[List[Cell], Dict]:
    """
    Executes one full Navier-Stokes update step.

    Roadmap Alignment:
    Governing Equation:
        ρ(∂u/∂t + u · ∇u) = -∇P + μ∇²u + F
        ∇ · u = 0

    Modular Enforcement:
    - Momentum update → advection + viscosity
    - Pressure solve → ∇²P = ∇ · u
    - Velocity projection → u ← u - ∇P
    - Reflex diagnostics → mutation traceability and suppression detection
    - NOTE: Verification is now handled exclusively by the pressure_solver.py module.

    Returns:
        Tuple[List[Cell], Dict]: Updated grid and reflex metadata
    """
    # 💨 Step 1: Momentum update — applies advection and viscosity
    grid_after_momentum = apply_momentum_update(grid, input_data, step_index)

    # 💧 Step 2: Pressure correction — solves ∇²P = ∇ · u to enforce ∇ · u = 0
    # The pressure_solver executes the verification run after its corrections.
    grid_after_pressure, pressure_mutated, projection_passes, pressure_metadata = apply_pressure_correction(
        grid_after_momentum, input_data, step_index
    )

    # 🔁 Step 3: Velocity projection — updates u ← u - ∇P to complete continuity enforcement
    grid_after_projection = apply_pressure_velocity_projection(grid_after_pressure, input_data)

    # 📦 Metadata packaging for reflex and diagnostics
    metadata = {
        "pressure_mutated": pressure_mutated,
        "projection_passes": projection_passes
    }
    if isinstance(pressure_metadata, dict):
        metadata.update(pressure_metadata)

    # ❌ REMOVED: Redundant verification flag detection and call to run_verification_if_triggered.
    # This diagnostic is owned by apply_pressure_correction.
    
    return grid_after_projection, metadata


