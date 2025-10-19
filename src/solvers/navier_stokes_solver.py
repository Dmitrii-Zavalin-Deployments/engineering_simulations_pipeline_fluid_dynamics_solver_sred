# src/solvers/navier_stokes_solver.py
# 🧠 Navier-Stokes Solver — centralized logic for momentum and continuity enforcement

from typing import List, Tuple, Dict
from src.grid_modules.cell import Cell
from src.solvers.momentum_solver import apply_momentum_update
from src.solvers.pressure_solver import apply_pressure_correction
from src.physics.velocity_projection import apply_pressure_velocity_projection

def solve_navier_stokes_step(
    grid: List[Cell],
    input_data: dict,
    step_index: int
) -> Tuple[List[Cell], Dict]:
    """
    Executes one full Navier-Stokes update step.

    Governing Equations:
    - Momentum: ρ(∂u/∂t + u · ∇u) = -∇P + μ∇²u
    - Continuity: ∇ · u = 0

    Numerical Strategy:
    1. Explicit Euler update for momentum (advection + viscosity)
    2. Pressure Poisson solve for incompressibility
    3. Velocity projection using pressure gradient

    Args:
        grid (List[Cell]): Current simulation grid
        input_data (dict): Full simulation configuration
        step_index (int): Current timestep index

    Returns:
        Tuple:
            - Updated grid with velocity and pressure fields
            - Metadata dict with pressure mutation info and projection passes
    """
    # 💨 Step 1: Momentum update
    grid_after_momentum = apply_momentum_update(grid, input_data, step_index)

    # 💧 Step 2: Pressure correction (Poisson solve for ∇ · u = 0)
    grid_after_pressure, pressure_mutated, projection_passes, pressure_metadata = apply_pressure_correction(
        grid_after_momentum, input_data, step_index
    )

    # 🔁 Step 3: Velocity projection (u ← u - ∇P)
    grid_after_projection = apply_pressure_velocity_projection(grid_after_pressure, input_data)

    # 📦 Metadata packaging
    metadata = {
        "pressure_mutated": pressure_mutated,
        "projection_passes": projection_passes
    }
    if isinstance(pressure_metadata, dict):
        metadata.update(pressure_metadata)

    return grid_after_projection, metadata



