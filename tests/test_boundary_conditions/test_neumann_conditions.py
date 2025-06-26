# tests/test_solver_core/test_neumann_conditions.py

import numpy as np
from src.numerical_methods.poisson_solver import solve_poisson_for_phi
from src.numerical_methods.pressure_correction import apply_pressure_correction
from src.numerical_methods.pressure_divergence import compute_pressure_divergence
from tests.test_solver_core.test_utils import mesh_metadata


def create_uniform_divergence_field(grid_shape, magnitude=1.0):
    """
    Returns a synthetic velocity field with uniform divergence, which would normally
    induce a pressure gradient. Used to test response under Neumann boundaries.
    """
    velocity = np.zeros((*grid_shape, 3))
    velocity[1:-1, 1:-1, 1:-1, 0] = magnitude  # constant x-component
    return velocity


def check_pressure_gradient_near_zero(phi, axis=0, tolerance=1e-6):
    """
    Verifies that the gradient of the pressure correction field is small near Neumann boundaries.
    """
    grad = np.gradient(phi, axis=axis)
    edge_slice = tuple(slice(0, 2) if i == axis else slice(None) for i in range(3))
    boundary_grad = grad[edge_slice]
    return np.allclose(boundary_grad, 0.0, atol=tolerance)


def test_poisson_solution_respects_neumann_boundary_at_outflow():
    shape = (16, 16, 16)
    mesh = mesh_metadata(shape)
    velocity = create_uniform_divergence_field(mesh["grid_shape"], magnitude=0.5)

    div = compute_pressure_divergence(velocity, mesh)
    rhs = np.pad(div, ((1, 1), (1, 1), (1, 1)), mode="constant")

    phi = solve_poisson_for_phi(rhs, mesh, time_step=1.0)

    # Optional: validate gradient near outlet (e.g. x = -1 side) is near zero
    is_zero_gradient = check_pressure_gradient_near_zero(phi, axis=0)
    assert is_zero_gradient, (
        "Poisson solution shows non-zero gradient near boundary — may violate Neumann condition"
    )


def test_neumann_boundary_preserves_flow_direction():
    shape = (16, 16, 16)
    mesh = mesh_metadata(shape)

    velocity = create_uniform_divergence_field(mesh["grid_shape"], magnitude=0.25)
    momentum_before = np.sum(velocity[1:-1, 1:-1, 1:-1, :], axis=(0, 1, 2))

    div = compute_pressure_divergence(velocity, mesh)
    rhs = np.pad(div, ((1, 1), (1, 1), (1, 1)), mode="constant")
    phi = solve_poisson_for_phi(rhs, mesh, time_step=1.0)

    corrected_u, _ = apply_pressure_correction(
        velocity, np.zeros_like(phi), phi[1:-1, 1:-1, 1:-1],
        mesh, 1.0, 1.0
    )
    momentum_after = np.sum(corrected_u[1:-1, 1:-1, 1:-1, :], axis=(0, 1, 2))

    delta = np.linalg.norm(momentum_after - momentum_before)
    print(f"ℹ️ Momentum shift with Neumann BC: Δ = {delta:.3e}")
    assert delta < 0.5, "Projection under Neumann BC altered total momentum more than expected"



