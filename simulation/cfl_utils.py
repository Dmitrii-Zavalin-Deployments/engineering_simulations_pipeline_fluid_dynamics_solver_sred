# simulation/cfl_utils.py

import numpy as np

def calculate_max_cfl(sim):
    """
    Calculates the maximum CFL number in the domain.
    CFL = max(|u|*dt/dx, |v|*dt/dy, |w|*dt/dz)

    Args:
        sim: Simulation object containing current velocity field and mesh info.

    Returns:
        float: Maximum CFL number observed across all components.
    """
    field = sim.velocity_field
    dt, dx, dy, dz = sim.time_step, sim.dx, sim.dy, sim.dz

    # Interior slice to avoid ghost zones
    u = field[1:-1, 1:-1, 1:-1, 0]
    v = field[1:-1, 1:-1, 1:-1, 1]
    w = field[1:-1, 1:-1, 1:-1, 2]

    # Clamp invalid values
    u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

    cfl_x = np.abs(u) * dt / dx if dx > 0 else 0.0
    cfl_y = np.abs(v) * dt / dy if dy > 0 else 0.0
    cfl_z = np.abs(w) * dt / dz if dz > 0 else 0.0

    max_cfl = np.max(np.maximum(np.maximum(cfl_x, cfl_y), cfl_z))
    return max_cfl



