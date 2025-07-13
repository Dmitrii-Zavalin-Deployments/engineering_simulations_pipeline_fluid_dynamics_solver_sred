# src/physics/divergence.py
# 🔍 Central-difference divergence calculation for fluid incompressibility checks

from src.grid_modules.cell import Cell
from typing import List, Dict, Tuple
from src.physics.divergence_methods.central import compute_central_divergence

def compute_divergence(grid: List[Cell], config: dict = {}) -> List[float]:
    """
    Computes divergence values for fluid cells using central-difference approximation.

    Args:
        grid (List[Cell]): Grid of Cell objects
        config (dict): Domain configuration including spacing and resolution

    Returns:
        List[float]: Divergence values for fluid cells (order matches input)
    """
    # 🧼 Step 1: Downgrade malformed fluid cells (non-list velocity) to solid
    safe_grid = [
        Cell(
            x=cell.x,
            y=cell.y,
            z=cell.z,
            velocity=cell.velocity if cell.fluid_mask and isinstance(cell.velocity, list) else None,
            pressure=cell.pressure if cell.fluid_mask and isinstance(cell.velocity, list) else None,
            fluid_mask=cell.fluid_mask if cell.fluid_mask and isinstance(cell.velocity, list) else False
        )
        for cell in grid
    ]

    # 🧪 Step 2: Call divergence engine on sanitized grid
    return compute_central_divergence(safe_grid, config)



