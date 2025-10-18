# src/physics/advection_methods/euler.py
# 🔄 Forward Euler advection method for velocity evolution

from src.grid_modules.cell import Cell
from typing import List
from src.physics.advection_methods.helpers import copy_cell, vector_add, vector_scale

def compute_euler_velocity(grid: List[Cell], dt: float, config: dict) -> List[Cell]:
    """
    Applies a Forward Euler update to fluid cell velocities using a basic upwind approximation.

    Args:
        grid (List[Cell]): Simulation grid
        dt (float): Time step
        config (dict): Simulation configuration

    Returns:
        List[Cell]: Grid with updated fluid velocities
    """
    # Extract domain resolution and spacing
    domain = config.get("domain_definition", {})
    nx = domain.get("nx", 1)
    domain.get("ny", 1)
    domain.get("nz", 1)
    min_x = domain.get("min_x", 0.0)
    max_x = domain.get("max_x", 1.0)
    dx = (max_x - min_x) / nx if nx > 0 else 1.0

    # Utility: Index grid cells by (x, y, z) for neighbor lookup
    grid_lookup = {(cell.x, cell.y, cell.z): cell for cell in grid}
    updated = []

    for cell in grid:
        if not cell.fluid_mask or not isinstance(cell.velocity, list):
            updated.append(copy_cell(cell))
            continue

        # Get upstream neighbor cell for upwind approximation (simple x-direction)
        neighbor_coords = (cell.x - dx, cell.y, cell.z)
        neighbor = grid_lookup.get(neighbor_coords)

        if neighbor and neighbor.fluid_mask and isinstance(neighbor.velocity, list):
            neighbor_velocity = neighbor.velocity
            # Euler update: v_new = v + dt * (neighbor_velocity - v) / dx
            velocity_diff = [neighbor_velocity[i] - cell.velocity[i] for i in range(3)]
            advection_term = vector_scale(velocity_diff, dt / dx)
            new_velocity = vector_add(cell.velocity, advection_term)
        else:
            # No valid neighbor → fallback to unchanged velocity
            new_velocity = cell.velocity

        updated.append(copy_cell(cell, velocity=new_velocity))

    return updated



