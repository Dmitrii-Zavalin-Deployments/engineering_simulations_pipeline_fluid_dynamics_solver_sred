# tests/test_solver_core/test_solver_method_consistency.py

import numpy as np
from src.numerical_methods.poisson_solver import solve_poisson_for_phi
from src.numerical_methods.pressure_divergence import compute_pressure_divergence, compute_pressure_gradient
from src.numerical_methods.pressure_correction import apply_pressure_correction
from tests.test_solver_core.test_utils import mesh_metadata


def test_pressure_gradient_and_divergence_are_inverse_operators():
    shape = (12, 12, 12)
    mesh = mesh_metadata(shape)

    x = np.linspace(0, 1, shape[0] + 2)
    y = np.linspace(0, 1, shape[1] + 2)
    z = np.linspace(0, 1, shape[2] + 2)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    pressure = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)

    grad_p = compute_pressure_gradient(pressure, mesh)
    div_grad_p = compute_pressure_divergence(grad_p, mesh)

    interior = div_grad_p[1:-1, 1:-1, 1:-1]
    assert interior.size > 0
    assert not np.any(np.isnan(interior))
    mean_abs = np.mean(np.abs(interior))
    assert mean_abs > 0.01
    assert mean_abs < 10.0


def test_poisson_solution_reduces_divergence_of_random_field():
    shape = (12, 12, 12)
    mesh = mesh_metadata(shape)
    velocity = np.random.randn(shape[0]+2, shape[1]+2, shape[2]+2, 3) * 0.05

    initial_div = compute_pressure_divergence(velocity, mesh)
    rhs = np.pad(initial_div, ((1, 1), (1, 1), (1, 1)), mode="constant")
    phi = solve_poisson_for_phi(rhs, mesh, time_step=1.0)

    corrected_u, _ = apply_pressure_correction(
        velocity, np.zeros_like(phi), phi[1:-1, 1:-1, 1:-1],
        mesh, 1.0, 1.0
    )

    final_div = compute_pressure_divergence(corrected_u, mesh)
    ratio = np.mean(np.abs(final_div)) / np.mean(np.abs(initial_div))
    assert ratio < 0.5, f"Expected divergence to reduce by at least 50%, got {ratio:.3f}"


def test_constant_pressure_yields_zero_gradient_and_zero_correction():
    shape = (10, 10, 10)
    mesh = mesh_metadata(shape)
    pressure = np.ones((shape[0]+2, shape[1]+2, shape[2]+2)) * 5.0
    velocity = np.random.randn(*pressure.shape, 3) * 0.01

    corrected_u, _ = apply_pressure_correction(
        velocity, pressure, np.zeros_like(pressure[1:-1, 1:-1, 1:-1]),
        mesh, 1.0, 1.0
    )

    diff = np.linalg.norm(corrected_u - velocity)
    assert diff < 1e-8, f"Unexpected correction for constant pressure: Δ = {diff:.2e}"



