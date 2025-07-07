# src/numerical_methods/pressure_correction.py

import numpy as np

def apply_pressure_correction(
    tentative_velocity_field: np.ndarray,
    current_pressure_field: np.ndarray,
    phi: np.ndarray,
    mesh_info: dict,
    dt: float,
    density: float,
    return_residual: bool = False
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, float]:
    """
    Applies pressure correction to velocity field and updates pressure.
    Optionally returns post-correction ∇·u residual magnitude.

    Args:
        tentative_velocity_field (np.ndarray): Velocity after advection/diffusion [nx+2, ny+2, nz+2, 3]
        current_pressure_field (np.ndarray): Pressure before correction [nx+2, ny+2, nz+2]
        phi (np.ndarray): Pressure correction potential [nx+2, ny+2, nz+2]
        mesh_info (dict): Grid spacing and topology
        dt (float): Time step
        density (float): Fluid density
        return_residual (bool): Flag to compute divergence residual after correction

    Returns:
        Tuple: corrected_velocity, corrected_pressure [, max_divergence]
    """
    dx, dy, dz = mesh_info["dx"], mesh_info["dy"], mesh_info["dz"]
    updated_velocity = tentative_velocity_field.copy()
    updated_pressure = current_pressure_field.copy()
    interior = (slice(1, -1), slice(1, -1), slice(1, -1))

    # 🛡️ Clamp φ if invalid values detected
    if np.isnan(phi).any() or np.isinf(phi).any():
        print("⚠️ Invalid φ values detected — clamping.")
        max_safe = np.finfo(np.float64).max / 10
        phi = np.nan_to_num(phi, nan=0.0, posinf=max_safe, neginf=-max_safe)
        phi = np.clip(phi, -max_safe, max_safe)

    # 🎯 Compute gradients of φ
    grad_phi_x = (phi[2:, 1:-1, 1:-1] - phi[1:-1, 1:-1, 1:-1]) / dx
    grad_phi_y = (phi[1:-1, 2:, 1:-1] - phi[1:-1, 1:-1, 1:-1]) / dy
    grad_phi_z = (phi[1:-1, 1:-1, 2:] - phi[1:-1, 1:-1, 1:-1]) / dz

    # 🔄 Correct velocity
    updated_velocity[interior + (0,)] -= (dt / density) * grad_phi_x
    updated_velocity[interior + (1,)] -= (dt / density) * grad_phi_y
    updated_velocity[interior + (2,)] -= (dt / density) * grad_phi_z

    # ⛓️ Update pressure
    updated_pressure[interior] += density * phi[interior]

    if not return_residual:
        return updated_velocity, updated_pressure

    # 📉 Post-correction ∇·u calculation
    dudx = (updated_velocity[2:, 1:-1, 1:-1, 0] - updated_velocity[1:-1, 1:-1, 1:-1, 0]) / dx
    dvdy = (updated_velocity[1:-1, 2:, 1:-1, 1] - updated_velocity[1:-1, 1:-1, 1:-1, 1]) / dy
    dwdz = (updated_velocity[1:-1, 1:-1, 2:, 2] - updated_velocity[1:-1, 1:-1, 1:-1, 2]) / dz

    div_u = dudx + dvdy + dwdz
    max_div_residual = float(np.max(np.abs(div_u)))

    return updated_velocity, updated_pressure, max_div_residual



