# src/physics/pressure_methods/utils.py
# 🛠️ Utilities for fluid cell indexing and pressure field mapping

from src.grid_modules.cell import Cell
from typing import List, Dict, Tuple

def index_fluid_cells(grid: List[Cell]) -> List[Tuple[float, float, float]]:
    """
    Identify fluid cell coordinates.

    Args:
        grid (List[Cell]): Simulation grid

    Returns:
        List of (x, y, z) coordinates for fluid cells
    """
    return [(cell.x, cell.y, cell.z) for cell in grid if cell.fluid_mask]


def build_pressure_map(grid: List[Cell]) -> Dict[Tuple[float, float, float], float]:
    """
    Build pressure lookup map from grid.

    Args:
        grid (List[Cell]): Simulation grid

    Returns:
        Dict of coordinates to pressure values
    """
    return {
        (cell.x, cell.y, cell.z): cell.pressure
        for cell in grid
        if cell.fluid_mask and isinstance(cell.pressure, (int, float))
    }


def flatten_pressure_field(pressure_map: Dict[Tuple[float, float, float], float],
                           fluid_coords: List[Tuple[float, float, float]]) -> List[float]:
    """
    Convert pressure map to ordered list matching fluid_coords.

    Args:
        pressure_map (Dict): Map of coordinates to pressure
        fluid_coords (List): Ordered list of fluid cell coordinates

    Returns:
        List[float]: Flattened pressure field
    """
    return [pressure_map.get(coord, 0.0) for coord in fluid_coords]



