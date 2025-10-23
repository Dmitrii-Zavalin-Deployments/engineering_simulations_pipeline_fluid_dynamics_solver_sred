# src/solvers/momentum_solver.py
# 🔧 Momentum Solver — evolves velocity using advection and viscosity
# 📌 This module enforces directional momentum transport and viscous damping.
# It excludes only cells explicitly marked fluid_mask=False.
# It does NOT skip based on adjacency or ghost proximity — all logic is geometry-mask-driven.

from typing import List
from src.grid_modules.cell import Cell
from src.physics.advection import compute_advection
from src.physics.viscosity import apply_viscous_terms

# ✅ Centralized debug flag for GitHub Actions logging
debug = True

def apply_momentum_update(grid: List[Cell], input_data: dict, step: int) -> List[Cell]:
    """
    Evolves velocity field using the momentum equation.

    Roadmap Alignment:
    Governing Equation:
        ρ(∂u/∂t + u · ∇u) = -∇P + μ∇²u + F

    Modular Enforcement:
    - ∂u/∂t: Euler time stepping (implicit via update)
    - u · ∇u: nonlinear advection → advection.py
    - μ∇²u: viscous diffusion → viscosity.py
    - -∇P: pressure gradient handled in pressure_solver.py
    - F: external forces handled separately if present

    Purpose:
    - Enforces directional momentum transport and viscous damping
    - Anchors velocity evolution prior to pressure correction
    - Supports reflex diagnostics and mutation traceability

    Args:
        grid (List[Cell]): Current simulation grid
        input_data (dict): Full simulation input config
        step (int): Current step index

    Returns:
        List[Cell]: Updated grid with evolved velocity fields
    """
    dt = input_data["simulation_parameters"]["time_step"]

    # 🌀 Step 1: Advection — nonlinear transport (u · ∇u)
    grid_advected = compute_advection(grid, dt, input_data)

    # 💧 Step 2: Viscosity — Laplacian diffusion (μ∇²u)
    grid_viscous = apply_viscous_terms(grid_advected, dt, input_data)

    # ✅ Step 3: Finalize velocity update per fluid cell
    updated_grid = []
    for original, evolved in zip(grid, grid_viscous):
        if evolved.fluid_mask:
            updated_cell = Cell(
                x=evolved.x,
                y=evolved.y,
                z=evolved.z,
                velocity=evolved.velocity[:],  # updated velocity
                pressure=original.pressure,   # keep pressure unchanged here
                fluid_mask=True
            )
            if debug:
                print(f"[MOMENTUM] Fluid cell @ ({updated_cell.x:.2f}, {updated_cell.y:.2f}, {updated_cell.z:.2f}) → velocity updated")
        else:
            updated_cell = Cell(
                x=original.x,
                y=original.y,
                z=original.z,
                velocity=None,
                pressure=None,
                fluid_mask=False
            )
        updated_grid.append(updated_cell)

    return updated_grid



