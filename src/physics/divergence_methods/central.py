# src/physics/divergence_methods/central.py
# 📏 Central-difference divergence scheme for structured fluid grids

from src.grid_modules.cell import Cell
from typing import List, Dict, Tuple
from src.physics.divergence_methods.divergence_helpers import (
    get_neighbor_velocity,
    central_difference
)

def compute_central_divergence(grid: List[Cell], config: dict) -> List[float]:
    """
    Computes divergence using central difference for structured fluid grid.

    Args:
        grid (List[Cell]): Grid with velocity and position data
        config (dict): Simulation configuration including domain spacing

    Returns:
        List[float]: Divergence values for fluid cells only (matching input order)
    """
    domain = config.get("domain_definition", {})
    nx = domain.get("nx", 1)
    ny = domain.get("ny", 1)
    nz = domain.get("nz", 1)

    dx = (domain.get("max_x", 1.0) - domain.get("min_x", 0.0)) / nx if nx else 1.0
    dy = (domain.get("max_y", 1.0) - domain.get("min_y", 0.0)) / ny if ny else 1.0
    dz = (domain.get("max_z", 1.0) - domain.get("min_z", 0.0)) / nz if nz else 1.0

    spacing = {'x': dx, 'y': dy, 'z': dz}

    # Index grid by (x, y, z)
    lookup: Dict[Tuple[float, float, float], Cell] = {
        (cell.x, cell.y, cell.z): cell for cell in grid
    }

    divergence = []
    for cell in grid:
        if not cell.fluid_mask or not isinstance(cell.velocity, list):
            continue

        grad_x = central_difference(
            get_neighbor_velocity(lookup, cell.x, cell.y, cell.z, 'x', +1, spacing['x']),
            get_neighbor_velocity(lookup, cell.x, cell.y, cell.z, 'x', -1, spacing['x']),
            spacing['x'],
            component=0
        )

        grad_y = central_difference(
            get_neighbor_velocity(lookup, cell.x, cell.y, cell.z, 'y', +1, spacing['y']),
            get_neighbor_velocity(lookup, cell.x, cell.y, cell.z, 'y', -1, spacing['y']),
            spacing['y'],
            component=1
        )

        grad_z = central_difference(
            get_neighbor_velocity(lookup, cell.x, cell.y, cell.z, 'z', +1, spacing['z']),
            get_neighbor_velocity(lookup, cell.x, cell.y, cell.z, 'z', -1, spacing['z']),
            spacing['z'],
            component=2
        )

        divergence.append(grad_x + grad_y + grad_z)

    return divergence



