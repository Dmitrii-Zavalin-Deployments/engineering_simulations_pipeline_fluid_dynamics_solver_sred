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
    # 🔍 Step 1: Compute divergence of velocity field
    divergence = compute_divergence(grid)  # returns List[float] for fluid cells

    # ⚡ Step 2: Solve pressure Poisson equation for correction
    grid_with_pressure = solve_pressure_poisson(grid, divergence, input_data)

    # 📤 Step 3: Return pressure-corrected grid (velocity projection to be added later)
    return grid_with_pressure



