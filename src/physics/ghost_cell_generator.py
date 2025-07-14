# src/physics/ghost_cell_generator.py
# 🧱 Ghost Cell Generator — domain-edge padding with ghost registry tagging

from typing import List, Tuple, Dict
from src.grid_modules.cell import Cell

def generate_ghost_cells(grid: List[Cell], config: dict) -> Tuple[List[Cell], Dict[int, dict]]:
    """
    Generates ghost cells at domain boundaries based on tagged faces and no-slip enforcement.

    Args:
        grid (List[Cell]): Physical simulation grid.
        config (dict): Full simulation input with domain_definition and boundary_conditions.

    Returns:
        Tuple[List[Cell], Dict[int, dict]]: Augmented grid including ghost cells, and ghost registry with metadata
    """
    domain = config.get("domain_definition", {})
    boundaries = config.get("boundary_conditions", {})
    no_slip = boundaries.get("no_slip", False)
    tagged_faces = boundaries.get("faces", [])

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
    ghost_registry = {}

    def add_ghost(x, y, z, face, origin):
        ghost = Cell(x=x, y=y, z=z, velocity=None, pressure=None, fluid_mask=False)
        ghost.ghost_face = face  # ✅ Face metadata for diagnostics
        ghost_cells.append(ghost)
        ghost_registry[id(ghost)] = {
            "face": face,
            "origin": origin
        }

    for cell in grid:
        x, y, z = cell.x, cell.y, cell.z

        # Only pad domain faces if tagged or no_slip is active
        if "x_min" in boundaries and (1 in tagged_faces or no_slip):
            if abs(x - x_min) < 0.5 * dx:
                add_ghost(x - dx, y, z, "x_min", (x, y, z))
        if "x_max" in boundaries and (2 in tagged_faces or no_slip):
            if abs(x - x_max) < 0.5 * dx:
                add_ghost(x + dx, y, z, "x_max", (x, y, z))
        if "y_min" in boundaries and (3 in tagged_faces or no_slip):
            if abs(y - y_min) < 0.5 * dy:
                add_ghost(x, y - dy, z, "y_min", (x, y, z))
        if "y_max" in boundaries and (4 in tagged_faces or no_slip):
            if abs(y - y_max) < 0.5 * dy:
                add_ghost(x, y + dy, z, "y_max", (x, y, z))
        if "z_min" in boundaries and (5 in tagged_faces or no_slip):
            if abs(z - z_min) < 0.5 * dz:
                add_ghost(x, y, z - dz, "z_min", (x, y, z))
        if "z_max" in boundaries and (6 in tagged_faces or no_slip):
            if abs(z - z_max) < 0.5 * dz:
                add_ghost(x, y, z + dz, "z_max", (x, y, z))

    padded_grid = grid + ghost_cells
    return padded_grid, ghost_registry



