# src/scheduler/adaptive_timestep.py

import numpy as np

def compute_max_cfl(
    velocity_field: np.ndarray,
    mesh_info: dict,
    dt: float
) -> float:
    """
    Computes the maximum CFL number across the entire domain for stability analysis.

    CFL = |u| * dt / dx

    Args:
        velocity_field (np.ndarray): Full velocity field [nx+2, ny+2, nz+2, 3]
        mesh_info (dict): Includes 'dx', 'dy', 'dz' spacings
        dt (float): Current time step size

    Returns:
        float: Maximum CFL number observed across all components
    """
    dx, dy, dz = mesh_info["dx"], mesh_info["dy"], mesh_info["dz"]

    # Extract and clamp components
    vel_x = np.nan_to_num(velocity_field[..., 0], nan=0.0, posinf=0.0, neginf=0.0)
    vel_y = np.nan_to_num(velocity_field[..., 1], nan=0.0, posinf=0.0, neginf=0.0)
    vel_z = np.nan_to_num(velocity_field[..., 2], nan=0.0, posinf=0.0, neginf=0.0)

    cfl_x = np.abs(vel_x) * dt / dx
    cfl_y = np.abs(vel_y) * dt / dy
    cfl_z = np.abs(vel_z) * dt / dz

    # Maximum CFL across all components
    cfl_max = np.max(np.maximum(np.maximum(cfl_x, cfl_y), cfl_z))
    return cfl_max


def adjust_time_step(
    velocity_field: np.ndarray,
    mesh_info: dict,
    current_dt: float,
    dt_min: float = 1e-6,
    dt_max: float = 1e-1,
    target_cfl: float = 0.9,
    aggressive_scaling: bool = False
) -> float:
    """
    Adjusts the time step size based on CFL feedback.

    Args:
        velocity_field (np.ndarray): Full velocity field
        mesh_info (dict): Grid spacings
        current_dt (float): Current time step
        dt_min (float): Minimum time step allowed
        dt_max (float): Maximum time step allowed
        target_cfl (float): Desired stability CFL threshold
        aggressive_scaling (bool): If True, uses stronger dt adaptation

    Returns:
        float: New time step to be used in the next iteration
    """
    cfl = compute_max_cfl(velocity_field, mesh_info, current_dt)

    if cfl > target_cfl:
        scaling_factor = 0.25 if aggressive_scaling else 0.5
        new_dt = max(current_dt * scaling_factor, dt_min)
        print(f"⏱️ CFL={cfl:.3f} > {target_cfl:.2f} → reducing dt from {current_dt:.2e} to {new_dt:.2e}")
    elif cfl < 0.3 * target_cfl and current_dt < dt_max:
        scaling_factor = 2.0 if aggressive_scaling else 1.5
        new_dt = min(current_dt * scaling_factor, dt_max)
        print(f"⏩ CFL={cfl:.3f} < {0.3 * target_cfl:.2f} → increasing dt from {current_dt:.2e} to {new_dt:.2e}")
    else:
        new_dt = current_dt
        print(f"✅ CFL={cfl:.3f} within target → dt unchanged at {new_dt:.2e}")

    return new_dt



