# src/numerical_methods/crank_nicolson.py

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def crank_nicolson_1d_step(u_old, dx, viscosity, dt):
    """
    Performs Crank–Nicolson update for 1D diffusion:
        u_new = u_old + 0.5 * dt * ν * (L · u_old + L · u_new)
    Where L is the finite-difference Laplacian.

    Args:
        u_old (np.ndarray): 1D interior field values.
        dx (float): Grid spacing.
        viscosity (float): Fluid viscosity.
        dt (float): Time step.

    Returns:
        np.ndarray: Updated 1D solution u_new.
    """
    n = len(u_old)
    alpha = viscosity * dt / (dx ** 2)

    diagonals = [
        -2.0 * np.ones(n),
        np.ones(n - 1),
        np.ones(n - 1),
    ]
    L = diags(diagonals, [0, 1, -1], format="csr")

    I = diags([1.0] * n, 0, format="csr")
    A = I - 0.5 * alpha * L
    B = I + 0.5 * alpha * L
    b = B.dot(u_old)

    u_new = spsolve(A, b)
    return u_new


def apply_crank_nicolson_3d_scalar(field, viscosity, mesh_info, dt):
    """
    Applies Crank–Nicolson diffusion update to a 3D scalar field.

    Args:
        field (np.ndarray): Full field with ghost cells.
        viscosity (float): Fluid viscosity.
        mesh_info (dict): Grid metadata including 'dx', 'dy', 'dz'.
        dt (float): Time step.

    Returns:
        np.ndarray: Updated full field with ghost cells.
    """
    field = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)
    updated = field.copy()

    dx, dy, dz = mesh_info["dx"], mesh_info["dy"], mesh_info["dz"]
    nx, ny, nz = field.shape[0] - 2, field.shape[1] - 2, field.shape[2] - 2

    # Sweep in x-direction
    for j in range(1, ny + 1):
        for k in range(1, nz + 1):
            u_old = updated[1:-1, j, k]
            updated[1:-1, j, k] = crank_nicolson_1d_step(u_old, dx, viscosity, dt)

    # Sweep in y-direction
    for i in range(1, nx + 1):
        for k in range(1, nz + 1):
            u_old = updated[i, 1:-1, k]
            updated[i, 1:-1, k] = crank_nicolson_1d_step(u_old, dy, viscosity, dt)

    # Sweep in z-direction
    for i in range(1, nx + 1):
        for j in range(1, ny + 1):
            u_old = updated[i, j, 1:-1]
            updated[i, j, 1:-1] = crank_nicolson_1d_step(u_old, dz, viscosity, dt)

    # Final clamp and log
    updated = np.nan_to_num(updated, nan=0.0, posinf=0.0, neginf=0.0)
    delta = np.abs(updated - field)
    print(f"📉 Crank–Nicolson update: max change = {np.max(delta):.4e}")

    return updated



