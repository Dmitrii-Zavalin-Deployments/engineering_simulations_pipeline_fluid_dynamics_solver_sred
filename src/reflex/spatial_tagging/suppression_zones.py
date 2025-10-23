# src/reflex/spatial_tagging/suppression_zones.py
# 🚫 Suppression Zones — tags fluid cells near ghost boundaries that failed to mutate or register influence
# 📌 This module identifies ghost-adjacent fluid cells that were not mutated or tagged as influenced.
# It excludes only cells explicitly marked fluid_mask=False.
# It does NOT skip based on adjacency or proximity — all logic is geometry-mask-driven.

from typing import List, Tuple, Set
from src.grid_modules.cell import Cell

# ✅ Centralized debug flag for GitHub Actions logging
debug = True

def detect_suppression_zones(
    grid: List[Cell],
    ghost_coords: Set[Tuple[float, float, float]],
    mutated_coords: Set[Tuple[float, float, float]],
    spacing: Tuple[float, float, float],
    tolerance: float = 1e-3
) -> List[Tuple[float, float]]:
    """
    Identifies fluid cells adjacent to ghost cells that were not mutated or tagged as influenced.

    Roadmap Alignment:
    Reflex Suppression:
    - Missed adjacency → ghost proximity without mutation
    - Suppression zones → fallback scoring and overlay diagnostics

    Args:
        grid (List[Cell]): Full simulation grid
        ghost_coords (Set[Tuple]): Coordinates of ghost cells
        mutated_coords (Set[Tuple]): Coordinates of mutated fluid cells
        spacing (Tuple): Grid spacing (dx, dy, dz)
        tolerance (float): Numerical tolerance for adjacency detection

    Returns:
        List[Tuple]: (x, y) coordinates of suppressed fluid cells
    """
    dx, dy, dz = spacing
    suppressed = []

    def is_adjacent(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> bool:
        return (
            abs(a[0] - b[0]) <= dx + tolerance and
            abs(a[1] - b[1]) <= dy + tolerance and
            abs(a[2] - b[2]) <= dz + tolerance
        )

    for cell in grid:
        if not getattr(cell, "fluid_mask", False):
            continue
        coord = (cell.x, cell.y, cell.z)
        if coord in mutated_coords or getattr(cell, "influenced_by_ghost", False):
            continue
        for ghost in ghost_coords:
            if is_adjacent(coord, ghost):
                suppressed.append((cell.x, cell.y))
                if debug:
                    print(f"[SUPPRESSION] Fluid cell @ {coord} adjacent to ghost @ {ghost} but not mutated/influenced")
                break

    return suppressed


def extract_mutated_coordinates(mutated_cells: List[object]) -> Set[Tuple[float, float, float]]:
    """
    Extracts (x, y, z) coordinates from mutated cell objects or tuples.

    Args:
        mutated_cells (List): List of mutated cell objects or coordinate tuples

    Returns:
        Set[Tuple]: Coordinates of mutated fluid cells
    """
    coords = set()
    for cell in mutated_cells:
        if isinstance(cell, tuple) and len(cell) == 3:
            coords.add(cell)
        elif hasattr(cell, "x") and hasattr(cell, "y") and hasattr(cell, "z"):
            coord = (cell.x, cell.y, cell.z)
            coords.add(coord)
            if debug:
                print(f"[SUPPRESSION] Extracted mutated coord → {coord}")
    return coords



