# src/physics/divergence.py
# 🔍 Stub: Divergence computation for incompressibility checks

from src.grid_modules.cell import Cell
from typing import List

def compute_divergence(grid: List[Cell]) -> List[float]:
    """
    Placeholder for divergence calculation on fluid cells.

    Args:
        grid (List[Cell]): List of Cell objects with velocity data

    Returns:
        List[float]: Divergence value per fluid cell (0.0 for stub)

    Notes:
        This stub returns zero divergence for all fluid cells.
        Future implementation will compute central-difference-based divergence
        using spatial velocity gradients and neighbor topology.
    """
    divergence = []
    for cell in grid:
        if cell.fluid_mask:
            divergence.append(0.0)  # stubbed zero divergence
    return divergence



