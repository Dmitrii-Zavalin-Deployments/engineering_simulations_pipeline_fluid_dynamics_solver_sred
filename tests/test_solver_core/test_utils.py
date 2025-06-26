# tests/test_solver_core/test_utils.py

import numpy as np
from src.numerical_methods.pressure_divergence import compute_pressure_divergence


def mesh_metadata(shape, dx=1.0, dy=1.0, dz=1.0):
    nx, ny, nz = shape
    return {
        "grid_shape": (nx + 2, ny + 2, nz + 2),
        "dx": dx,
        "dy": dy,
        "dz": dz,
    }


def create_mesh_info(nx, ny, nz, dx=1.0, dy=1.0, dz=1.0):
    return mesh_metadata((nx, ny, nz), dx, dy, dz)


def add_zero_padding(core_field):
    padded = np.zeros(tuple(s + 2 for s in core_field.shape), dtype=core_field.dtype)
    padded[1:-1, 1:-1, 1:-1] = core_field
    return padded


def compute_scaled_laplacian(phi_padded, dx):
    return (
        -6.0 * phi_padded[1:-1, 1:-1, 1:-1] +
        phi_padded[2:, 1:-1, 1:-1] + phi_padded[:-2, 1:-1, 1:-1] +
        phi_padded[1:-1, 2:, 1:-1] + phi_padded[1:-1, :-2, 1:-1] +
        phi_padded[1:-1, 1:-1, 2:] + phi_padded[1:-1, 1:-1, :-2]
    ) / dx**2


def compute_relative_residual(laplacian, rhs_core):
    residual = laplacian - rhs_core
    norm_rhs = np.linalg.norm(rhs_core)
    return np.linalg.norm(residual) / norm_rhs if norm_rhs > 1e-12 else np.linalg.norm(residual)


def compute_mean_divergence(velocity, mesh):
    divergence = compute_pressure_divergence(velocity, mesh)
    return np.mean(np.abs(divergence))


def divergence_reduction_ratio(before, after):
    norm_before = np.mean(np.abs(before))
    norm_after = np.mean(np.abs(after))
    return norm_after / norm_before if norm_before > 1e-12 else norm_after


def log_divergence_progression(divergence_list):
    print("\nDivergence convergence history:")
    for i, div in enumerate(divergence_list):
        tag = "initial" if i == 0 else f"step {i}"
        print(f"  {tag:<8} → mean |div| = {div:.5e}")
    if len(divergence_list) > 1:
        final = divergence_list[-1]
        initial = divergence_list[0]
        reduction = final / initial if initial > 1e-12 else float("inf")
        print(f"  Reduction factor: {reduction:.3e}")


def create_sinusoidal_velocity(shape):
    """
    Creates a smooth, structured 2D velocity field with predictable divergence.
    Useful for stable and repeatable projection testing.
    """
    x = np.linspace(0, 2 * np.pi, shape[0] + 2)
    y = np.linspace(0, 2 * np.pi, shape[1] + 2)
    z = np.linspace(0, 2 * np.pi, shape[2] + 2)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    velocity = np.zeros((shape[0]+2, shape[1]+2, shape[2]+2, 3))
    velocity[..., 0] = np.sin(X) * np.cos(Y) * np.cos(Z)
    velocity[..., 1] = -np.cos(X) * np.sin(Y) * np.cos(Z)
    velocity[..., 2] = 0.0
    return velocity



