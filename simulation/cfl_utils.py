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
    dt = sim.time_step
    dx = sim.dx if hasattr(sim, "dx") else sim.mesh_info.get("dx", 1.0)
    dy = sim.dy if hasattr(sim, "dy") else sim.mesh_info.get("dy", 1.0)
    dz = sim.dz if hasattr(sim, "dz") else sim.mesh_info.get("dz", 1.0)

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
    print(f"🔄 Max CFL: {max_cfl:.4e} (dx={dx:.4e}, dy={dy:.4e}, dz={dz:.4e}, dt={dt:.4e})")
    return max_cfl



