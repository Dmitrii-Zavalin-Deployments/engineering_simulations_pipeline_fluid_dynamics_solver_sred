# src/grid_modules/boundary_manager.py

import logging
from src.grid_modules.cell import Cell

def apply_boundaries(cells: list[Cell], domain: dict) -> list[Cell]:
    """
    Applies simplified boundary conditions by tagging edge cells using resolution values.
    Marks cells on the outermost x/y/z faces with 'wall' type; others as 'interior'.

    🧱 Strategy-aligned fallback:
    If any resolution axis is zero, the domain is degenerate and all cells are tagged as 'wall'.
    This avoids undefined edge detection logic and ensures consistent state.

    Args:
        cells (list[Cell]): Structured grid of cells
        domain (dict): Must contain "nx", "ny", "nz"

    Returns:
        list[Cell]: Tagged cells with 'boundary_type' set
    """
    nx = domain.get("nx", 0)
    ny = domain.get("ny", 0)
    nz = domain.get("nz", 0)

    # 🧱 Fallback: handle degenerate domain dimensions gracefully
    if nx == 0 or ny == 0 or nz == 0:
        logging.warning("⚠️ Domain resolution contains zero dimensions — tagging all cells as 'wall'")
        for cell in cells:
            cell.boundary_type = "wall"
        return cells

    # Edge positions based on resolution
    edge_x = {0, nx - 1}
    edge_y = {0, ny - 1}
    edge_z = {0, nz - 1}

    for cell in cells:
        if cell.x in edge_x or cell.y in edge_y or cell.z in edge_z:
            cell.boundary_type = "wall"
        else:
            cell.boundary_type = "interior"

    return cells



