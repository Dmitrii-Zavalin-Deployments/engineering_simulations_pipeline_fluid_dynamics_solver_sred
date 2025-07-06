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
    Applies the pressure correction to the tentative velocity field and updates the pressure.
    Optionally returns an estimate of the residual divergence after correction.

    Args:
        tentative_velocity_field (np.ndarray): Velocity field after advection and diffusion.
        current_pressure_field (np.ndarray): Previous pressure field.
        phi (np.ndarray): Pressure correction potential from Poisson solve.
        mesh_info (dict): Mesh parameters including dx, dy, dz.
        dt (float): Time step size.
        density (float): Fluid density.
        return_residual (bool): Whether to compute and return residual divergence.

    Returns:
        tuple: (updated_velocity_field, updated_pressure_field [, divergence_residual])
    """
    dx, dy, dz = mesh_info["dx"], mesh_info["dy"], mesh_info["dz"]

    updated_velocity = tentative_velocity_field.copy()
    updated_pressure = current_pressure_field.copy()

    # Interior slice (exclude ghost cells)
    interior = (slice(1, -1), slice(1, -1), slice(1, -1)

    )

    # Compute gradient of phi using central differences
    grad_phi_x = (phi[2:, 1:-1, 1:-1] - phi[1:-1, 1:-1, 1:-1]) / dx
    grad_phi_y = (phi[1:-1, 2:, 1:-1] - phi[1:-1, 1:-1, 1:-1]) / dy
    grad_phi_z = (phi[1:-1, 1:-1, 2:] - phi[1:-1, 1:-1, 1:-1]) / dz

    # Apply velocity corrections
    updated_velocity[1:-1, 1:-1, 1:-1, 0] -= (dt / density) * grad_phi_x
    updated_velocity[1:-1, 1:-1, 1:-1, 1] -= (dt / density) * grad_phi_y
    updated_velocity[1:-1, 1:-1, 1:-1, 2] -= (dt / density) * grad_phi_z

    # Apply pressure correction
    updated_pressure[1:-1, 1:-1, 1:-1] += density * phi[1:-1, 1:-1, 1:-1]

    if not return_residual:
        return updated_velocity, updated_pressure

    # Estimate residual divergence ∇·u (optional)
    dudx = (updated_velocity[2:, 1:-1, 1:-1, 0] - updated_velocity[1:-1, 1:-1, 1:-1, 0]) / dx
    dvdy = (updated_velocity[1:-1, 2:, 1:-1, 1] - updated_velocity[1:-1, 1:-1, 1:-1, 1]) / dy
    dwdz = (updated_velocity[1:-1, 1:-1, 2:, 2] - updated_velocity[1:-1, 1:-1, 1:-1, 2]) / dz
    div_u = dudx + dvdy + dwdz
    max_residual = np.max(np.abs(div_u))

    return updated_velocity, updated_pressure, max_residual



