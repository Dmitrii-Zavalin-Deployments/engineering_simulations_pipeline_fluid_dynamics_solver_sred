# simulation/cfl_utils.py

import numpy as np

def calculate_max_cfl(sim):
    """
    Calculates the maximum CFL number in the domain and prints diagnostics.

    CFL = max(|u| * dt / dx, |v| * dt / dy, |w| * dt / dz)

    Args:
        sim: Simulation object containing velocity field and mesh parameters.

    Returns:
        float: Maximum CFL number across all components.
    """
    field = sim.velocity_field
    dt = sim.time_step
    dx = getattr(sim, "dx", sim.mesh_info.get("dx", 1.0))
    dy = getattr(sim, "dy", sim.mesh_info.get("dy", 1.0))
    dz = getattr(sim, "dz", sim.mesh_info.get("dz", 1.0))

    if field.shape[-1] != 3:
        raise ValueError(f"Invalid velocity field shape: {field.shape}. Expected last dimension to be 3.")

    # Interior slice only (excludes ghost zones)
    u = field[1:-1, 1:-1, 1:-1, 0]
    v = field[1:-1, 1:-1, 1:-1, 1]
    w = field[1:-1, 1:-1, 1:-1, 2]

    # Sanitize bad values
    u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

    cfl_x = np.abs(u) * dt / dx if dx > 0 else 0.0
    cfl_y = np.abs(v) * dt / dy if dy > 0 else 0.0
    cfl_z = np.abs(w) * dt / dz if dz > 0 else 0.0

    max_cfl_x = float(np.max(cfl_x))
    max_cfl_y = float(np.max(cfl_y))
    max_cfl_z = float(np.max(cfl_z))
    max_cfl = max(max_cfl_x, max_cfl_y, max_cfl_z)

    print(f"🔄 CFL Diagnostics @ dt={dt:.4e}")
    print(f"    • Max CFL x: {max_cfl_x:.4e}")
    print(f"    • Max CFL y: {max_cfl_y:.4e}")
    print(f"    • Max CFL z: {max_cfl_z:.4e}")
    print(f"    • Global Max CFL: {max_cfl:.4e}")
    print(f"    • Grid spacing — dx={dx:.4e}, dy={dy:.4e}, dz={dz:.4e}")

    return max_cfl



