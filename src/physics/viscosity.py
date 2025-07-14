# src/physics/viscosity.py
# 💧 Viscous diffusion module — Laplacian smoothing of velocity field

from src.grid_modules.cell import Cell
from typing import List, Dict, Tuple
from src.physics.advection_methods.helpers import vector_add, vector_scale, copy_cell

def apply_viscous_terms(grid: List[Cell], dt: float, config: dict) -> List[Cell]:
    """
    Applies Laplacian-based smoothing to velocity field using 6-point stencil.
    Only fluid cells are updated; solid and ghost cells are passed unchanged.

    Args:
        grid (List[Cell]): Current grid with velocity and pressure
        dt (float): Time step duration
        config (dict): Fluid properties and simulation config

    Returns:
        List[Cell]: Grid with viscosity-adjusted velocities (fluid cells only)
    """
    viscosity = config.get("fluid_properties", {}).get("viscosity", 0.0)
    domain = config.get("domain_definition", {})

    dx = (domain.get("max_x", 1.0) - domain.get("min_x", 0.0)) / domain.get("nx", 1)
    dy = (domain.get("max_y", 1.0) - domain.get("min_y", 0.0)) / domain.get("ny", 1)
    dz = (domain.get("max_z", 1.0) - domain.get("min_z", 0.0)) / domain.get("nz", 1)

    grid_lookup: Dict[Tuple[float, float, float], Cell] = {
        (cell.x, cell.y, cell.z): cell for cell in grid
    }

    updated = []

    for cell in grid:
        if not cell.fluid_mask or not isinstance(cell.velocity, list):
            updated.append(copy_cell(cell))
            continue

        neighbor_offsets = [
            (dx, 0.0, 0.0), (-dx, 0.0, 0.0),
            (0.0, dy, 0.0), (0.0, -dy, 0.0),
            (0.0, 0.0, dz), (0.0, 0.0, -dz),
        ]

        neighbors = []
        for offset in neighbor_offsets:
            pos = (cell.x + offset[0], cell.y + offset[1], cell.z + offset[2])
            neighbor = grid_lookup.get(pos)
            if neighbor and neighbor.fluid_mask and isinstance(neighbor.velocity, list):
                neighbors.append(neighbor.velocity)

        if not neighbors:
            updated.append(copy_cell(cell))
            continue

        avg_velocity = [0.0, 0.0, 0.0]
        for v in neighbors:
            for i in range(3):
                avg_velocity[i] += v[i]
        for i in range(3):
            avg_velocity[i] /= len(neighbors)

        laplacian_term = [avg_velocity[i] - cell.velocity[i] for i in range(3)]
        viscosity_update = vector_scale(laplacian_term, viscosity * dt)
        new_velocity = vector_add(cell.velocity, viscosity_update)

        updated.append(copy_cell(cell, velocity=new_velocity))

    return updated



