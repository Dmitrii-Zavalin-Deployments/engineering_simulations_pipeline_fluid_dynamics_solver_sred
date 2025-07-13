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
    return compute_central_divergence(grid, config)



