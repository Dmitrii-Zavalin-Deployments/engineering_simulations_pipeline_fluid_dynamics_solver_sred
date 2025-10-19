# src/reflex/spatial_tagging/adjacency_zones.py
# 🧭 Adjacency Zones — tags fluid cells near ghost cells for reflex overlays and mutation proximity scoring

from typing import List, Tuple, Set
from src.grid_modules.cell import Cell

def detect_adjacency_zones(
    grid: List[Cell],
    ghost_coords: Set[Tuple[float, float, float]],
    spacing: Tuple[float, float, float],
    tolerance: float = 1e-3
) -> List[Tuple[float, float]]:
    """
    Identifies fluid cells adjacent to ghost cells using spatial proximity.

    Roadmap Alignment:
    Reflex Overlays:
    - Adjacency zones → mutation proximity scoring
    - Ghost proximity → boundary enforcement visibility

    Args:
        grid (List[Cell]): Full simulation grid
        ghost_coords (Set[Tuple]): Coordinates of ghost cells
        spacing (Tuple): Grid spacing (dx, dy, dz)
        tolerance (float): Numerical tolerance for adjacency detection

    Returns:
        List[Tuple]: (x, y) coordinates of adjacent fluid cells
    """
    dx, dy, dz = spacing
    adjacent_cells = []

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
        for ghost in ghost_coords:
            if is_adjacent(coord, ghost):
                adjacent_cells.append((cell.x, cell.y))
                break  # Avoid duplicate tagging

    return adjacent_cells


def extract_ghost_coordinates(ghost_registry) -> Set[Tuple[float, float, float]]:
    """
    Extracts ghost coordinates from registry.

    Args:
        ghost_registry (dict or set): Ghost metadata or ghost cell set

    Returns:
        Set[Tuple]: Coordinates of ghost cells
    """
    coords = set()
    if isinstance(ghost_registry, dict):
        for meta in ghost_registry.values():
            coord = meta.get("coordinate")
            if isinstance(coord, tuple):
                coords.add(coord)
    elif isinstance(ghost_registry, set):
        for cell in ghost_registry:
            coords.add((cell.x, cell.y, cell.z))
    return coords



