# src/numerical_methods/pressure_correction.py

import numpy as np
import sys

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
    Applies pressure correction to tentative velocity and updates pressure.
    Optionally returns the residual divergence magnitude after correction.

    Args:
        tentative_velocity_field (np.ndarray): Velocity after advection/diffusion.
        current_pressure_field (np.ndarray): Pressure field before correction.
        phi (np.ndarray): Pressure correction potential.
        mesh_info (dict): Contains grid spacing dx, dy, dz.
        dt (float): Time step size.
        density (float): Fluid density.
        return_residual (bool): Whether to return post-correction ∇·u magnitude.

    Returns:
        Tuple: (corrected_velocity, corrected_pressure [, max_divergence])
    """
    dx, dy, dz = mesh_info["dx"], mesh_info["dy"], mesh_info["dz"]
    updated_velocity = tentative_velocity_field.copy()
    updated_pressure = current_pressure_field.copy()
    interior = (slice(1, -1), slice(1, -1), slice(1, -1))

    # 🛡️ Clamp correction potential for safety
    if np.any(np.isnan(phi)) or np.any(np.isinf(phi)):
        print("⚠️ Detected NaN/Inf in φ — clamping.")
        max_val = np.finfo(np.float64).max / 10
        phi = np.nan_to_num(phi, nan=0.0, posinf=max_val, neginf=-max_val)
        phi = np.clip(phi, -max_val, max_val)

    # 🎯 Compute gradient of φ via central differences
    grad_phi_x = (phi[2:, 1:-1, 1:-1] - phi[1:-1, 1:-1, 1:-1]) / dx
    grad_phi_y = (phi[1:-1, 2:, 1:-1] - phi[1:-1, 1:-1, 1:-1]) / dy
    grad_phi_z = (phi[1:-1, 1:-1, 2:] - phi[1:-1, 1:-1, 1:-1]) / dz

    # 🔄 Correct velocity field
    updated_velocity[interior + (0,)] -= (dt / density) * grad_phi_x
    updated_velocity[interior + (1,)] -= (dt / density) * grad_phi_y
    updated_velocity[interior + (2,)] -= (dt / density) * grad_phi_z

    # ⛓️ Update pressure field
    updated_pressure[interior] += density * phi[interior]

    if not return_residual:
        return updated_velocity, updated_pressure

    # 📉 Compute residual divergence ∇·u
    dudx = (updated_velocity[2:, 1:-1, 1:-1, 0] - updated_velocity[1:-1, 1:-1, 1:-1, 0]) / dx
    dvdy = (updated_velocity[1:-1, 2:, 1:-1, 1] - updated_velocity[1:-1, 1:-1, 1:-1, 1]) / dy
    dwdz = (updated_velocity[1:-1, 1:-1, 2:, 2] - updated_velocity[1:-1, 1:-1, 1:-1, 2]) / dz
    div_u = dudx + dvdy + dwdz
    max_residual = float(np.max(np.abs(div_u)))

    return updated_velocity, updated_pressure, max_residual



