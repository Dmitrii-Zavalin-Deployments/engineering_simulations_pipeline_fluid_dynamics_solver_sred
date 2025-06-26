# tests/test_solver_core/test_solver_extended_consistency.py

import numpy as np
from src.numerical_methods.poisson_solver import solve_poisson_for_phi
from src.numerical_methods.pressure_divergence import compute_pressure_divergence, compute_pressure_gradient
from src.numerical_methods.pressure_correction import apply_pressure_correction
from tests.test_solver_core.test_utils import mesh_metadata


def test_poisson_residual_drops_to_expected_threshold():
    shape = (20, 20, 20)
    mesh = mesh_metadata(shape)
    rng = np.random.default_rng(42)
    rhs_core = rng.normal(0.0, 1.0, shape)
    rhs = np.pad(rhs_core, ((1, 1), (1, 1), (1, 1)), mode="constant")

    _, residual = solve_poisson_for_phi(
        rhs, mesh, time_step=1.0,
        max_iterations=1500, tolerance=1e-6,
        return_residual=True
    )
    assert residual < 1e-4, f"Poisson solver residual too high: {residual:.2e}"


def test_anisotropic_gradient_and_divergence_behavior():
    shape = (12, 12, 12)
    mesh = mesh_metadata(shape, dx=1.0, dy=2.0, dz=0.5)

    pressure = np.zeros((shape[0]+2, shape[1]+2, shape[2]+2))
    grid_y = np.linspace(0, 1, shape[1]+2)
    pressure[:, :, :] = grid_y[None, :, None]

    grad = compute_pressure_gradient(pressure, mesh)
    core = grad[1:-1, 1:-1, 1:-1]

    assert np.allclose(core[..., 0], 0.0, atol=1e-10)
    assert np.allclose(core[..., 2], 0.0, atol=1e-10)
    assert np.all(core[..., 1] > 0.0), "Expected gradient in y-direction"


def test_divergence_decreases_monotonically_over_steps():
    shape = (16, 16, 16)
    mesh = mesh_metadata(shape)
    velocity = np.random.randn(shape[0]+2, shape[1]+2, shape[2]+2, 3) * 0.1

    history = []
    for _ in range(5):
        div = compute_pressure_divergence(velocity, mesh)
        history.append(np.mean(np.abs(div)))

        rhs = np.pad(div, ((1, 1), (1, 1), (1, 1)), mode="constant")
        phi = solve_poisson_for_phi(rhs, mesh, time_step=1.0)
        velocity, _ = apply_pressure_correction(
            velocity, np.zeros_like(phi),
            phi[1:-1, 1:-1, 1:-1],
            mesh, 1.0, 1.0
        )

    for i in range(1, len(history)):
        assert history[i] <= history[i-1] + 1e-8, (
            f"Divergence increased at step {i}: {history[i]:.3e} > {history[i-1]:.3e}"
        )


def test_zero_divergence_velocity_remains_unchanged():
    shape = (10, 10, 10)
    mesh = mesh_metadata(shape)

    x = np.linspace(-1, 1, shape[0]+2)
    y = np.linspace(-1, 1, shape[1]+2)
    z = np.linspace(-1, 1, shape[2]+2)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    velocity = np.zeros((shape[0]+2, shape[1]+2, shape[2]+2, 3))
    velocity[..., 0] = -Y
    velocity[..., 1] = X

    initial_div = compute_pressure_divergence(velocity, mesh)
    rhs = np.pad(initial_div, ((1, 1), (1, 1), (1, 1)), mode="constant")
    phi = solve_poisson_for_phi(rhs, mesh, time_step=1.0)

    corrected_u, _ = apply_pressure_correction(
        velocity, np.zeros_like(phi), phi[1:-1, 1:-1, 1:-1],
        mesh, 1.0, 1.0
    )

    diff = np.linalg.norm(corrected_u - velocity)
    assert diff < 1e-8, f"Projection altered divergence-free field: Δ = {diff:.2e}"



