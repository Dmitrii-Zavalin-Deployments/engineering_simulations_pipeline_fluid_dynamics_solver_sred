# src/reflex/spatial_tagging/ghost_face_mapper.py
# 🧭 Ghost Face Mapper — tags fluid cells adjacent to ghost cells for reflex overlays and scoring

from src.grid_modules.cell import Cell
from typing import List, Tuple, Set

def tag_ghost_adjacency(
    grid: List[Cell],
    ghost_coords: Set[Tuple[float, float, float]],
    spacing: Tuple[float, float, float]
) -> List[Tuple[float, float]]:
    """
    Tags fluid cells that are adjacent to ghost cells and returns their coordinates for overlay rendering.

    Roadmap Alignment:
    Reflex Scoring:
    - Adjacency → proximity of fluid cells to ghost boundaries
    - Supports reflex overlays and mutation causality tracing

    Args:
        grid (List[Cell]): Simulation grid
        ghost_coords (Set[Tuple]): Set of ghost cell coordinates
        spacing (Tuple): Grid spacing (dx, dy, dz)

    Returns:
        List[Tuple]: Coordinates of ghost-adjacent fluid cells (for overlay rendering)
    """
    dx, dy, dz = spacing
    adjacency_offsets = [
        (dx, 0.0, 0.0), (-dx, 0.0, 0.0),
        (0.0, dy, 0.0), (0.0, -dy, 0.0),
        (0.0, 0.0, dz), (0.0, 0.0, -dz),
    ]

    adjacency_coords = []

    for cell in grid:
        if not getattr(cell, "fluid_mask", False):
            continue

        cx, cy, cz = cell.x, cell.y, cell.z
        for offset in adjacency_offsets:
            neighbor = (cx + offset[0], cy + offset[1], cz + offset[2])
            if neighbor in ghost_coords:
                cell.ghost_adjacent = True
                adjacency_coords.append((cx, cy))
                break  # Tag once per cell

    return adjacency_coords



