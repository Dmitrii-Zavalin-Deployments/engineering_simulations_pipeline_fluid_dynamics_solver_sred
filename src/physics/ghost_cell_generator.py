# src/physics/ghost_cell_generator.py
# 🧱 Ghost Cell Generator — domain-edge padding with ghost registry tagging

from typing import List, Tuple, Set
from src.grid_modules.cell import Cell

def generate_ghost_cells(grid: List[Cell], config: dict) -> Tuple[List[Cell], Set[int]]:
    """
    Generates ghost cells at domain boundaries based on tagged faces.

    Args:
        grid (List[Cell]): Physical simulation grid.
        config (dict): Full simulation input with domain_definition and boundary_conditions.

    Returns:
        Tuple[List[Cell], Set[int]]: Augmented grid including ghost cells, and ghost registry by ID
    """
    domain = config.get("domain_definition", {})
    boundaries = config.get("boundary_conditions", {})

    nx = domain.get("nx", 1)
    ny = domain.get("ny", 1)
    nz = domain.get("nz", 1)

    dx = (domain.get("max_x", 1.0) - domain.get("min_x", 0.0)) / nx
    dy = (domain.get("max_y", 1.0) - domain.get("min_y", 0.0)) / ny
    dz = (domain.get("max_z", 1.0) - domain.get("min_z", 0.0)) / nz

    x_min = domain.get("min_x", 0.0)
    x_max = domain.get("max_x", 1.0)
    y_min = domain.get("min_y", 0.0)
    y_max = domain.get("max_y", 1.0)
    z_min = domain.get("min_z", 0.0)
    z_max = domain.get("max_z", 1.0)

    ghost_cells = []
    ghost_registry = set()

    def add_ghost(x, y, z):
        ghost = Cell(x=x, y=y, z=z, velocity=None, pressure=None, fluid_mask=False)
        ghost_cells.append(ghost)
        ghost_registry.add(id(ghost))

    for cell in grid:
        x, y, z = cell.x, cell.y, cell.z

        if boundaries.get("x_min") and abs(x - x_min) < 0.5 * dx:
            add_ghost(x - dx, y, z)
        if boundaries.get("x_max") and abs(x - x_max) < 0.5 * dx:
            add_ghost(x + dx, y, z)
        if boundaries.get("y_min") and abs(y - y_min) < 0.5 * dy:
            add_ghost(x, y - dy, z)
        if boundaries.get("y_max") and abs(y - y_max) < 0.5 * dy:
            add_ghost(x, y + dy, z)
        if boundaries.get("z_min") and abs(z - z_min) < 0.5 * dz:
            add_ghost(x, y, z - dz)
        if boundaries.get("z_max") and abs(z - z_max) < 0.5 * dz:
            add_ghost(x, y, z + dz)

    padded_grid = grid + ghost_cells
    return padded_grid, ghost_registry



