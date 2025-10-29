# src/reflex/spatial_tagging/ghost_face_mapper.py
# 🧭 Ghost Face Mapper — tags fluid cells adjacent to ghost cells for reflex overlays and scoring
# 📌 This module identifies ghost-adjacent fluid cells for reflex overlays and mutation causality.
# It excludes only cells explicitly marked fluid_mask=False.
# It does NOT skip based on adjacency or proximity — all logic is geometry-mask-driven.

from src.grid_modules.cell import Cell
from typing import List, Tuple, Set

# ✅ Centralized debug flag for GitHub Actions logging
debug = False

def tag_ghost_adjacency(
    grid: List[Cell],
    ghost_coords: Set[Tuple[float, float, float]],
    spacing: Tuple[float, float, float],
    verbose: bool = False  # ✅ Optional diagnostics
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
        verbose (bool): If True, prints debug info

    Returns:
        List[Tuple]: Coordinates of ghost-adjacent fluid cells (for overlay rendering)
    """
    dx, dy, dz = spacing
    adjacency_coords = []

    def is_adjacent(a: Tuple[float, float, float], b: Tuple[float, float, float], tol=1e-3) -> bool:
        return (
            abs(a[0] - b[0]) <= dx + tol and
            abs(a[1] - b[1]) <= dy + tol and
            abs(a[2] - b[2]) <= dz + tol
        )

    for cell in grid:
        if not getattr(cell, "fluid_mask", False):
            continue

        coord = (cell.x, cell.y, cell.z)
        if getattr(cell, "ghost_adjacent", False):
            continue  # ✅ Skip if already tagged

        for ghost in ghost_coords:
            if is_adjacent(coord, ghost):
                cell.ghost_adjacent = True
                cell.mutation_triggered_by = "ghost_adjacency"
                adjacency_coords.append((cell.x, cell.y))
                if debug and verbose:
                    print(f"[ADJACENCY] Fluid cell @ {coord} adjacent to ghost @ {ghost}")
                break  # Tag once per cell

    return adjacency_coords



