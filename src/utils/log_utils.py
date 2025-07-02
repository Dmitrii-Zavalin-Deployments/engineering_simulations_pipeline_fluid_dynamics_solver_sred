# utils/log_utils.py

import numpy as np

def log_flow_metrics(
    velocity_field: np.ndarray,
    pressure_field: np.ndarray,
    divergence_field: np.ndarray,
    fluid_density: float,
    step_count: int,
    current_time: float
):
    """
    Prints diagnostic metrics to stdout for visibility in GitHub Actions logs.

    Args:
        velocity_field (np.ndarray): Full velocity field [nx, ny, nz, 3]
        pressure_field (np.ndarray): Full pressure field [nx, ny, nz]
        divergence_field (np.ndarray): Computed ∇·u field [nx, ny, nz]
        fluid_density (float): Fluid density value
        step_count (int): Current step number
        current_time (float): Current simulation time
    """
    interior_v = velocity_field[1:-1, 1:-1, 1:-1, :]
    interior_p = pressure_field[1:-1, 1:-1, 1:-1]
    interior_div = divergence_field[1:-1, 1:-1, 1:-1]

    kinetic_energy = 0.5 * fluid_density * np.sum(np.linalg.norm(interior_v, axis=-1)**2)
    max_velocity = np.max(np.linalg.norm(interior_v, axis=-1))
    min_p = np.min(interior_p)
    max_p = np.max(interior_p)
    max_div = np.max(np.abs(interior_div))
    mean_div = np.mean(np.abs(interior_div))

    print(f"📊 Step {step_count} @ t = {current_time:.4f}s")
    print(f"   • Total Kinetic Energy: {kinetic_energy:.4e}")
    print(f"   • Max Velocity Magnitude: {max_velocity:.4e}")
    print(f"   • Pressure Range (interior): [{min_p:.4e}, {max_p:.4e}]")
    print(f"   • Divergence ∇·u: Max = {max_div:.4e}, Mean = {mean_div:.4e}")



