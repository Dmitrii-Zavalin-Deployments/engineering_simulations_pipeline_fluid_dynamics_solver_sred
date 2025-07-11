# src/grid_modules/boundary_manager.py

from src.grid_modules.cell import Cell

def apply_boundaries(cells: list[Cell], domain: dict) -> list[Cell]:
    """
    Applies simplified boundary conditions by tagging edge cells.
    Marks cells on the outermost faces (x/y/z boundaries) for future constraint logic.
    Currently supports 'wall' tagging only; ghost cell creation planned.

    Args:
        cells (list[Cell]): Structured grid of cells
        domain (dict): Domain definition with min/max and resolution keys

    Returns:
        list[Cell]: Modified cells with basic boundary flags applied
    """
    min_x, max_x = domain["min_x"], domain["max_x"]
    min_y, max_y = domain["min_y"], domain["max_y"]
    min_z, max_z = domain["min_z"], domain["max_z"]

    nx, ny, nz = domain["nx"], domain["ny"], domain["nz"]

    # Compute outer bounds for grid indexing
    edge_x = {0, nx - 1}
    edge_y = {0, ny - 1}
    edge_z = {0, nz - 1}

    for cell in cells:
        # Simple tag: mark if cell is on any boundary
        if (
            cell.x in edge_x or
            cell.y in edge_y or
            cell.z in edge_z
        ):
            cell.boundary_type = "wall"
        else:
            cell.boundary_type = "interior"

    return cells



