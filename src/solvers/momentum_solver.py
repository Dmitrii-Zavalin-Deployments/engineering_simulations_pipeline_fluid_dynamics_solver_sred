# src/solvers/momentum_solver.py
# 🔧 Momentum solver — evolves velocity using advection and viscosity

from typing import List
from src.grid_modules.cell import Cell
from src.physics.advection import compute_advection
from src.physics.viscosity import apply_viscous_terms

def apply_momentum_update(grid: List[Cell], input_data: dict, step: int) -> List[Cell]:
    """
    Applies momentum equation logic to evolve velocity fields.

    Args:
        grid (List[Cell]): Current simulation grid
        input_data (dict): Full simulation input config
        step (int): Current step index

    Returns:
        List[Cell]: Updated grid with evolved velocity fields
    """
    dt = input_data["simulation_parameters"]["time_step"]

    # 🌀 Step 1: Apply advection (currently stubbed)
    grid_advected = compute_advection(grid, dt, input_data)

    # 💧 Step 2: Apply viscous diffusion (currently stubbed)
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



