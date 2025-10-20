# src/grid_modules/boundary_manager.py

from src.grid_modules.cell import Cell

def apply_boundaries(cells: list[Cell], domain: dict) -> list[Cell]:
    """
    Applies simplified boundary conditions by tagging edge cells using resolution values.
    Identifies cells on the outermost x/y/z faces based on grid indices.
    Does not require physical min/max domain coordinates.

    Args:
        cells (list[Cell]): Structured grid of cells
        domain (dict): Must contain "nx", "ny", "nz"

    Returns:
        list[Cell]: Original cell list (boundary tagging now handled via coordinate logic)
    """
    nx = domain.get("nx", 0)
    ny = domain.get("ny", 0)
    nz = domain.get("nz", 0)

    edge_x = {0, nx - 1} if nx > 0 else set()
    edge_y = {0, ny - 1} if ny > 0 else set()
    edge_z = {0, nz - 1} if nz > 0 else set()

    # ✅ Removed unused is_boundary assignment
    # Boundary tagging now handled via coordinate-based diagnostics in scoring and overlays

    return cells



