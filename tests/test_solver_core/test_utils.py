# tests/test_solver_core/test_utils.py

import numpy as np


def mesh_metadata(shape, dx=1.0, dy=1.0, dz=1.0):
    """
    Constructs mesh metadata for velocity and pressure correction tests.
    
    Args:
        shape (tuple): Physical grid shape (nx, ny, nz) without ghost cells.
        dx (float): Spacing in x-direction.
        dy (float): Spacing in y-direction.
        dz (float): Spacing in z-direction.
    
    Returns:
        dict: Metadata including padded grid shape and spacings.
    """
    nx, ny, nz = shape
    return {
        "grid_shape": (nx + 2, ny + 2, nz + 2),
        "dx": dx,
        "dy": dy,
        "dz": dz,
    }


def create_mesh_info(nx, ny, nz, dx=1.0, dy=1.0, dz=1.0):
    """
    Alias of mesh_metadata with explicit (nx, ny, nz) signature.

    Args:
        nx (int): Grid points in x-direction.
        ny (int): Grid points in y-direction.
        nz (int): Grid points in z-direction.
        dx (float): Grid spacing in x.
        dy (float): Grid spacing in y.
        dz (float): Grid spacing in z.

    Returns:
        dict: Mesh info dictionary for solver input.
    """
    return mesh_metadata((nx, ny, nz), dx, dy, dz)


def add_zero_padding(core_field):
    """
    Adds one layer of zero-padding around a 3D core field.

    Args:
        core_field (np.ndarray): Array of shape (nx, ny, nz)

    Returns:
        np.ndarray: Array of shape (nx+2, ny+2, nz+2) with zeros on all sides
    """
    padded = np.zeros(tuple(s + 2 for s in core_field.shape), dtype=core_field.dtype)
    padded[1:-1, 1:-1, 1:-1] = core_field
    return padded


def compute_scaled_laplacian(phi_padded, dx):
    """
    Computes the 7-point finite-difference Laplacian scaled by 1/dx².

    Args:
        phi_padded (np.ndarray): Padded scalar field of shape (nx+2, ny+2, nz+2)
        dx (float): Grid spacing (assumed uniform)

    Returns:
        np.ndarray: Laplacian approximation over the interior (nx, ny, nz)
    """
    return (
        -6.0 * phi_padded[1:-1, 1:-1, 1:-1] +
        phi_padded[2:, 1:-1, 1:-1] + phi_padded[:-2, 1:-1, 1:-1] +
        phi_padded[1:-1, 2:, 1:-1] + phi_padded[1:-1, :-2, 1:-1] +
        phi_padded[1:-1, 1:-1, 2:] + phi_padded[1:-1, 1:-1, :-2]
    ) / dx**2


def compute_relative_residual(laplacian, rhs_core):
    """
    Computes the relative L2 residual norm ||∇²φ - f|| / ||f||.

    Args:
        laplacian (np.ndarray): Interior Laplacian estimate
        rhs_core (np.ndarray): Unpadded right-hand side

    Returns:
        float: Relative residual norm
    """
    residual = laplacian - rhs_core
    norm_rhs = np.linalg.norm(rhs_core)
    return np.linalg.norm(residual) / norm_rhs if norm_rhs > 1e-12 else np.linalg.norm(residual)



