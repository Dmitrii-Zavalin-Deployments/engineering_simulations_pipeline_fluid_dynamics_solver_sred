# src/grid_modules/boundary_manager.py
# 🧱 Applies boundary condition tagging based on grid resolution

from src.grid_modules.cell import Cell

def apply_boundaries(cells: list[Cell], domain: dict) -> list[Cell]:
    """
    Applies simplified boundary conditions by tagging edge cells using grid resolution.
    Marks cells on outermost x/y/z layers with 'wall'; all others as 'interior'.
    Assumes physical coordinates are centered on indices and do not infer boundaries.

    Args:
        cells (list[Cell]): Grid cells with spatial data
        domain (dict): Contains "nx", "ny", "nz" resolution values

    Returns:
        list[Cell]: Updated cells with 'boundary_type' field
    """
    nx = domain.get("nx", 0)
    ny = domain.get("ny", 0)
    nz = domain.get("nz", 0)

    # Precompute physical edges using coordinate resolution
    dx = (domain["max_x"] - domain["min_x"]) / nx if nx else 0
    dy = (domain["max_y"] - domain["min_y"]) / ny if ny else 0
    dz = (domain["max_z"] - domain["min_z"]) / nz if nz else 0

    edge_x = {domain["min_x"] + i * dx for i in (0, nx - 1)} if nx > 0 else set()
    edge_y = {domain["min_y"] + j * dy for j in (0, ny - 1)} if ny > 0 else set()
    edge_z = {domain["min_z"] + k * dz for k in (0, nz - 1)} if nz > 0 else set()

    for cell in cells:
        is_edge = (
            any(abs(cell.x - ex) < 1e-6 for ex in edge_x) or
            any(abs(cell.y - ey) < 1e-6 for ey in edge_y) or
            any(abs(cell.z - ez) < 1e-6 for ez in edge_z)
        )
        cell.boundary_type = "wall" if is_edge else "interior"

    return cells



