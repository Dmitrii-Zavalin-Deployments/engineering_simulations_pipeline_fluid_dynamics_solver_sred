# src/solvers/navier_stokes_solver.py
# 🧠 Navier-Stokes Solver — centralized logic for momentum and continuity enforcement

from typing import List, Tuple, Dict
from src.grid_modules.cell import Cell
from src.solvers.momentum_solver import apply_momentum_update
from src.solvers.pressure_solver import apply_pressure_correction
from src.physics.velocity_projection import apply_pressure_velocity_projection
from src.diagnostics.navier_stokes_verifier import run_verification_if_triggered  # ✅ Verifier integration

def solve_navier_stokes_step(
    grid: List[Cell],
    input_data: dict,
    step_index: int
) -> Tuple[List[Cell], Dict]:
    """
    Executes one full Navier-Stokes update step.
    """
    # 💨 Step 1: Momentum update — applies advection and viscosity
    grid_after_momentum = apply_momentum_update(grid, input_data, step_index)

    # 💧 Step 2: Pressure correction — solves ∇²P = ∇ · u to enforce ∇ · u = 0
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

    # ✅ Trigger verifier if diagnostic flags are present
    triggered_flags = []
    if metadata.get("pressure_mutation_count", 0) == 0:
        triggered_flags.append("no_pressure_mutation")
    if not metadata.get("divergence", []):
        triggered_flags.append("empty_divergence")
    if any(not isinstance(c.velocity, list) or not c.fluid_mask for c in grid):
        triggered_flags.append("downgraded_cells")

    run_verification_if_triggered(
        grid=grid,  # ✅ Pass original grid for downgrade detection
        spacing=(
            (input_data["domain_definition"]["max_x"] - input_data["domain_definition"]["min_x"]) / input_data["domain_definition"]["nx"],
            (input_data["domain_definition"]["max_y"] - input_data["domain_definition"]["min_y"]) / input_data["domain_definition"]["ny"],
            (input_data["domain_definition"]["max_z"] - input_data["domain_definition"]["min_z"]) / input_data["domain_definition"]["nz"]
        ),
        step_index=step_index,
        output_folder="data/testing-input-output/navier_stokes_output",
        triggered_flags=triggered_flags
    )

    return grid_after_projection, metadata



