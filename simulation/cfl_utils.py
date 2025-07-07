# simulation/cfl_utils.py

import numpy as np

def calculate_max_cfl(sim):
    """
    Calculates maximum CFL number across the domain and logs diagnostics.

    CFL = max(|u| * dt / dx, |v| * dt / dy, |w| * dt / dz)

    Args:
        sim: Simulation object containing velocity_field and mesh_info

    Returns:
        float: Maximum CFL number observed across any direction
    """
    field = sim.velocity_field
    dt = sim.time_step
    dx = sim.mesh_info.get("dx", 1.0)
    dy = sim.mesh_info.get("dy", 1.0)
    dz = sim.mesh_info.get("dz", 1.0)

    if field.ndim != 4 or field.shape[-1] != 3:
        raise ValueError(f"Expected velocity field shape [..., 3]; got {field.shape}")

    # Interior slice (excluding ghost zones)
    u = field[1:-1, 1:-1, 1:-1, 0]
    v = field[1:-1, 1:-1, 1:-1, 1]
    w = field[1:-1, 1:-1, 1:-1, 2]

    # Sanitize invalid values
    u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

    cfl_x = np.abs(u) * dt / dx if dx > 0 else np.zeros_like(u)
    cfl_y = np.abs(v) * dt / dy if dy > 0 else np.zeros_like(v)
    cfl_z = np.abs(w) * dt / dz if dz > 0 else np.zeros_like(w)

    max_cfl_x = float(np.max(cfl_x))
    max_cfl_y = float(np.max(cfl_y))
    max_cfl_z = float(np.max(cfl_z))
    max_cfl = max(max_cfl_x, max_cfl_y, max_cfl_z)

    print(f"🔄 CFL Diagnostics @ Step {sim.step_count}")
    print(f"    • Time Step (dt): {dt:.4e}")
    print(f"    • Max CFL x     : {max_cfl_x:.4e}")
    print(f"    • Max CFL y     : {max_cfl_y:.4e}")
    print(f"    • Max CFL z     : {max_cfl_z:.4e}")
    print(f"    • Global Max CFL: {max_cfl:.4e}")
    print(f"    • Grid Spacing  : dx={dx:.4e}, dy={dy:.4e}, dz={dz:.4e}")

    return max_cfl



