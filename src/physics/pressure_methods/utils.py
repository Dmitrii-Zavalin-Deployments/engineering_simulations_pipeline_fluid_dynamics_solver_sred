# src/physics/pressure_methods/utils.py
# 🛠️ Pressure Utilities — fluid cell indexing and pressure field mapping for ∇²P enforcement
# 📌 This module supports pressure solve routines by indexing fluid cells and building pressure maps.
# It excludes only cells explicitly marked fluid_mask=False.
# It does NOT skip based on adjacency, boundary proximity, or pressure anomalies.

from src.grid_modules.cell import Cell
from typing import List, Dict, Tuple

# ✅ Centralized debug flag for GitHub Actions logging
debug = True

def index_fluid_cells(grid: List[Cell]) -> List[Tuple[float, float, float]]:
    """
    Identify fluid cell coordinates.

    Args:
        grid (List[Cell]): Simulation grid

    Returns:
        List of (x, y, z) coordinates for fluid cells
    """
    coords = [(cell.x, cell.y, cell.z) for cell in grid if cell.fluid_mask]
    if debug:
        print(f"[UTILS] Indexed {len(coords)} fluid cells")
    return coords


def build_pressure_map(grid: List[Cell]) -> Dict[Tuple[float, float, float], float]:
    """
    Build pressure lookup map from grid.

    Args:
        grid (List[Cell]): Simulation grid

    Returns:
        Dict of coordinates to pressure values
    """
    pressure_map = {
        (cell.x, cell.y, cell.z): cell.pressure
        for cell in grid
        if isinstance(cell.pressure, (int, float))
    }
    if debug:
        print(f"[UTILS] Built pressure map with {len(pressure_map)} entries")
    return pressure_map



