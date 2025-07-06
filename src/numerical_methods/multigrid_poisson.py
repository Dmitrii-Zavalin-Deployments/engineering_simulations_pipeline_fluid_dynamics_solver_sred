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


def prolong(coarse):
    """Prolong to finer grid via trilinear interpolation"""
    nx, ny, nz = coarse.shape
    fine = np.zeros((2*nx, 2*ny, 2*nz))

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                base = coarse[i, j, k]
                fine[2*i  , 2*j  , 2*k  ] += base
                fine[2*i+1, 2*j  , 2*k  ] += base
                fine[2*i  , 2*j+1, 2*k  ] += base
                fine[2*i  , 2*j  , 2*k+1] += base
                fine[2*i+1, 2*j+1, 2*k  ] += base
                fine[2*i+1, 2*j  , 2*k+1] += base
                fine[2*i  , 2*j+1, 2*k+1] += base
                fine[2*i+1, 2*j+1, 2*k+1] += base
    fine /= 8
    return fine


def v_cycle(phi, rhs, dx, dy, dz, levels):
    if levels == 1:
        return red_black_gauss_seidel(phi, rhs, dx, dy, dz, iterations=20)

    # Pre-smoothing
    phi = red_black_gauss_seidel(phi, rhs, dx, dy, dz, iterations=3)

    # Compute residual
    residual = np.zeros_like(rhs)
    residual[1:-1, 1:-1, 1:-1] = rhs[1:-1, 1:-1, 1:-1] - (
        -6 * phi[1:-1, 1:-1, 1:-1]
        + phi[2:, 1:-1, 1:-1] + phi[:-2, 1:-1, 1:-1]
        + phi[1:-1, 2:, 1:-1] + phi[1:-1, :-2, 1:-1]
        + phi[1:-1, 1:-1, 2:] + phi[1:-1, 1:-1, :-2]
    ) / dx**2

    # Restrict residual
    coarse_rhs = restrict(residual)
    coarse_phi = np.zeros_like(coarse_rhs)

    # Recursive V-cycle
    coarse_phi = v_cycle(coarse_phi, coarse_rhs, 2*dx, 2*dy, 2*dz, levels - 1)

    # Prolongate and correct
    correction = prolong(coarse_phi)
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


