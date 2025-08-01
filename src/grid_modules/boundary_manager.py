# src/grid_modules/boundary_manager.py

from src.grid_modules.cell import Cell

def apply_boundaries(cells: list[Cell], domain: dict) -> list[Cell]:
    """
    Applies simplified boundary conditions by tagging edge cells using resolution values.
    Marks cells on the outermost x/y/z faces with 'wall' type; others as 'interior'.
    Does not require physical min/max domain coordinates.

    Args:
        cells (list[Cell]): Structured grid of cells
        domain (dict): Must contain "nx", "ny", "nz"

    Returns:
        list[Cell]: Tagged cells with 'boundary_type' set
    """
    # Extract resolution values; assume valid integers
    nx = domain.get("nx", 0)
    ny = domain.get("ny", 0)
    nz = domain.get("nz", 0)

    # Determine edge positions; empty sets for zero-sized grids
    edge_x = {0, nx - 1} if nx > 0 else set()
    edge_y = {0, ny - 1} if ny > 0 else set()
    edge_z = {0, nz - 1} if nz > 0 else set()

    for cell in cells:
        if cell.x in edge_x or cell.y in edge_y or cell.z in edge_z:
            cell.boundary_type = "wall"
        else:
            cell.boundary_type = "interior"

    return cells



