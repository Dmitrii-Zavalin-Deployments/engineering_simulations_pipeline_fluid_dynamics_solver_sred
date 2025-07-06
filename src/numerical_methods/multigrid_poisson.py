# src/numerical_methods/multigrid_poisson.py

import numpy as np

def red_black_gauss_seidel(phi, rhs, dx, dy, dz, iterations=3):
    """
    Applies Red-Black Gauss–Seidel smoothing to phi.
    """
    nx, ny, nz = phi.shape

    for _ in range(iterations):
        for color in [0, 1]:
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    for k in range(1, nz - 1):
                        if (i + j + k) % 2 == color:
                            phi[i, j, k] = (1 / (2/dx**2 + 2/dy**2 + 2/dz**2)) * (
                                rhs[i, j, k]
                                + (phi[i+1, j, k] + phi[i-1, j, k]) / dx**2
                                + (phi[i, j+1, k] + phi[i, j-1, k]) / dy**2
                                + (phi[i, j, k+1] + phi[i, j, k-1]) / dz**2
                            )
    return phi


def restrict(field):
    """Restrict to coarser grid via injection"""
    return field[::2, ::2, ::2]


def prolong(coarse, target_shape=None):
    """
    Prolong to finer grid via trilinear interpolation.

    Args:
        coarse (np.ndarray): Coarse grid correction.
        target_shape (tuple): Desired output shape to match fine grid interior.
    
    Returns:
        np.ndarray: Prolonged correction matching fine grid interior shape.
    """
    nx, ny, nz = coarse.shape
    fine = np.zeros((2 * nx, 2 * ny, 2 * nz))

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                base = coarse[i, j, k]
                fi, fj, fk = 2 * i, 2 * j, 2 * k
                for di in [0, 1]:
                    for dj in [0, 1]:
                        for dk in [0, 1]:
                            ii, jj, kk = fi + di, fj + dj, fk + dk
                            if ii < fine.shape[0] and jj < fine.shape[1] and kk < fine.shape[2]:
                                fine[ii, jj, kk] += base
    fine /= 8

    if target_shape is not None:
        sx, sy, sz = target_shape
        fine = fine[:sx, :sy, :sz]

    return fine


def compute_residual(phi, rhs, dx):
    """
    Computes the residual ∇²phi - rhs for diagnostic purposes.
    """
    residual = np.zeros_like(rhs)
    residual[1:-1, 1:-1, 1:-1] = rhs[1:-1, 1:-1, 1:-1] - (
        -6 * phi[1:-1, 1:-1, 1:-1]
        + phi[2:, 1:-1, 1:-1] + phi[:-2, 1:-1, 1:-1]
        + phi[1:-1, 2:, 1:-1] + phi[1:-1, :-2, 1:-1]
        + phi[1:-1, 1:-1, 2:] + phi[1:-1, 1:-1, :-2]
    ) / dx**2
    return residual


def v_cycle(phi, rhs, dx, dy, dz, levels):
    if levels == 1:
        phi = red_black_gauss_seidel(phi, rhs, dx, dy, dz, iterations=20)
        return phi

    # Pre-smoothing
    phi = red_black_gauss_seidel(phi, rhs, dx, dy, dz, iterations=3)

    # Compute residual and log diagnostics
    residual = compute_residual(phi, rhs, dx)
    max_residual = np.max(np.abs(residual))
    mean_residual = np.mean(np.abs(residual))
    print(f"📉 V-cycle residual: max={max_residual:.4e}, mean={mean_residual:.4e}, shape={residual.shape}")

    # Restrict residual
    coarse_rhs = restrict(residual)
    coarse_phi = np.zeros_like(coarse_rhs)

    # Recursive V-cycle
    coarse_phi = v_cycle(coarse_phi, coarse_rhs, 2*dx, 2*dy, 2*dz, levels - 1)

    # Prolongate and correct
    correction = prolong(coarse_phi, target_shape=phi.shape)
    phi += correction

    # Post-smoothing
    phi = red_black_gauss_seidel(phi, rhs, dx, dy, dz, iterations=3)

    return phi


def solve_poisson_multigrid(divergence, mesh_info, dt, levels=3):
    """
    Solves ∇²φ = ∇·u/dt using multigrid V-cycle.

    Args:
        divergence (np.ndarray): Divergence field (ghost-padded).
        mesh_info (dict): Contains dx, dy, dz.
        dt (float): Time step.
        levels (int): Number of multigrid levels (powers of 2)

    Returns:
        np.ndarray: Full phi field (with ghost cells)
    """
    phi = np.zeros_like(divergence)
    rhs = divergence / dt

    dx, dy, dz = mesh_info["dx"], mesh_info["dy"], mesh_info["dz"]

    interior = (slice(1, -1), slice(1, -1), slice(1, -1))
    phi[interior] = v_cycle(phi[interior], rhs[interior], dx, dy, dz, levels)

    # Apply zero Neumann ghost padding
    phi[0, :, :] = phi[1, :, :]
    phi[-1, :, :] = phi[-2, :, :]
    phi[:, 0, :] = phi[:, 1, :]
    phi[:, -1, :] = phi[:, -2, :]
    phi[:, :, 0] = phi[:, :, 1]
    phi[:, :, -1] = phi[:, :, -2]

    return phi



