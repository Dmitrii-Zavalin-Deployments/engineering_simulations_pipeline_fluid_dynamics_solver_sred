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
        velocity_field (np.ndarray): Full velocity field [nx+2, ny+2, nz+2, 3]
        pressure_field (np.ndarray): Full pressure field [nx+2, ny+2, nz+2]
        divergence_field (np.ndarray): Computed ∇·u field [nx+2, ny+2, nz+2]
        fluid_density (float): Fluid density value
        step_count (int): Current step number
        current_time (float): Current simulation time
    """
    # Trim ghost cells
    interior_v = velocity_field[1:-1, 1:-1, 1:-1, :]
    interior_p = pressure_field[1:-1, 1:-1, 1:-1]
    interior_div = divergence_field[1:-1, 1:-1, 1:-1]

    if interior_v.size == 0 or interior_p.size == 0:
        print(f"⚠️ Empty interior domain at step {step_count} — skipping diagnostics.")
        return

    # Clamp invalid values
    interior_v = np.nan_to_num(interior_v, nan=0.0, posinf=0.0, neginf=0.0)
    interior_p = np.nan_to_num(interior_p, nan=0.0, posinf=0.0, neginf=0.0)
    interior_div = np.nan_to_num(interior_div, nan=0.0, posinf=0.0, neginf=0.0)

    # Kinetic energy: ½ρ‖u‖² over all interior cells
    velocity_mag = np.linalg.norm(interior_v, axis=-1)
    kinetic_energy = 0.5 * fluid_density * np.sum(velocity_mag**2)
    max_velocity = np.max(velocity_mag)

    # Pressure stats
    min_p = np.min(interior_p)
    max_p = np.max(interior_p)
    mean_p = np.mean(interior_p)
    std_p = np.std(interior_p)

    # Divergence stats
    max_div = np.max(np.abs(interior_div)) if interior_div.size > 0 else 0.0
    mean_div = np.mean(np.abs(interior_div)) if interior_div.size > 0 else 0.0

    # Log to stdout
    print(f"📊 Step {step_count} @ t = {current_time:.4f}s")
    print(f"   • Total Kinetic Energy     : {kinetic_energy:.4e}")
    print(f"   • Max Velocity Magnitude   : {max_velocity:.4e}")
    print(f"   • Pressure Range (interior): [{min_p:.4e}, {max_p:.4e}]")
    print(f"   • Mean Pressure (interior) : {mean_p:.4e}")
    print(f"   • Std Dev Pressure         : {std_p:.4e}")
    print(f"   • Divergence ∇·u           : Max = {max_div:.4e}, Mean = {mean_div:.4e}")



