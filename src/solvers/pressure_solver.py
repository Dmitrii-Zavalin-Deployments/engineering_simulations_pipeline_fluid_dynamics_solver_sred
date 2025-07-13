# src/solvers/pressure_solver.py
# 🔧 Pressure solver — enforces incompressibility via divergence correction

from typing import List
from src.grid_modules.cell import Cell
from src.physics.divergence import compute_divergence
from src.physics.pressure_projection import solve_pressure_poisson

def apply_pressure_correction(grid: List[Cell], input_data: dict, step: int) -> List[Cell]:
    """
    Applies pressure correction to enforce incompressible flow.

    Args:
        grid (List[Cell]): Grid of simulation cells
        input_data (dict): Full simulation config
        step (int): Current simulation step index

    Returns:
        List[Cell]: Grid with updated pressure values (fluid cells only)
    """
    # 🧼 Step 0: Filter out malformed velocity cells before computing divergence
    safe_grid = []
    for cell in grid:
        if cell.fluid_mask and not isinstance(cell.velocity, list):
            # Downgrade malformed fluid cell to solid to skip divergence
            cell = Cell(
                x=cell.x, y=cell.y, z=cell.z,
                velocity=None,
                pressure=None,
                fluid_mask=False
            )
        safe_grid.append(cell)

    # 🔍 Step 1: Compute divergence of velocity field
    divergence = compute_divergence(safe_grid)  # returns List[float] for valid fluid cells

    # ⚡ Step 2: Solve pressure Poisson equation for correction
    grid_with_pressure = solve_pressure_poisson(safe_grid, divergence, input_data)

    # 📤 Step 3: Return pressure-corrected grid (velocity projection to be added later)
    return grid_with_pressure



