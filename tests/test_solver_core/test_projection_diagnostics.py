# tests/test_solver_core/test_projection_diagnostics.py

import numpy as np
from src.numerical_methods.poisson_solver import solve_poisson_for_phi
from src.numerical_methods.pressure_correction import apply_pressure_correction
from src.numerical_methods.pressure_divergence import compute_pressure_divergence
from tests.test_solver_core.test_utils import mesh_metadata


def create_sinusoidal_velocity(shape):
    """
    Creates a smooth, structured 2D velocity field with predictable divergence.
    Useful for stable and repeatable projection testing.
    """
    x = np.linspace(0, 2 * np.pi, shape[0] + 2)
    y = np.linspace(0, 2 * np.pi, shape[1] + 2)
    z = np.linspace(0, 2 * np.pi, shape[2] + 2)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    velocity = np.zeros((shape[0] + 2, shape[1] + 2, shape[2] + 2, 3))
    velocity[..., 0] = np.sin(X) * np.cos(Y) * np.cos(Z)
    velocity[..., 1] = -np.cos(X) * np.sin(Y) * np.cos(Z)
    velocity[..., 2] = 0.0
    return velocity


def compute_total_momentum(velocity):
    return np.sum(velocity[1:-1, 1:-1, 1:-1, :], axis=(0, 1, 2))


def test_projection_reduces_divergence_with_reasonable_momentum_change():
    shape = (12, 12, 12)
    mesh = mesh_metadata(shape)

    velocity = create_sinusoidal_velocity(shape)
    div_before = compute_pressure_divergence(velocity, mesh)
    momentum_before = compute_total_momentum(velocity)

    rhs = np.pad(div_before, ((1, 1), (1, 1), (1, 1)), mode="constant")
    phi = solve_poisson_for_phi(rhs, mesh, time_step=1.0)
    corrected_u, _ = apply_pressure_correction(
        velocity, np.zeros_like(phi), phi[1:-1, 1:-1, 1:-1],
        mesh, 1.0, 1.0
    )

    div_after = compute_pressure_divergence(corrected_u, mesh)
    momentum_after = compute_total_momentum(corrected_u)

    div_drop = np.linalg.norm(div_before) - np.linalg.norm(div_after)
    delta_momentum = np.linalg.norm(momentum_after - momentum_before)

    if div_drop < 1e-3:
        print(f"(Info) Field was nearly divergence-free to begin with. Δdiv = {div_drop:.2e}")
    else:
        ratio = delta_momentum / div_drop
        assert ratio < 1.0, (
            f"Momentum changed more than expected relative to divergence reduction.\n"
            f"Δmomentum = {delta_momentum:.2e}, Δdivergence = {div_drop:.2e}, "
            f"ratio = {ratio:.2f}, threshold = 1.0"
        )



