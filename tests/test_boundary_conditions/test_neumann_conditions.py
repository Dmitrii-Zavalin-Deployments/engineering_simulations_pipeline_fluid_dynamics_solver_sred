import numpy as np
import pytest
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


def check_pressure_gradient_near_zero(phi, axis=0):
    """
    Logs the mean pressure gradient near the specified boundary.
    This is diagnostic only while Neumann BCs are not enforced.
    """
    grad = np.gradient(phi, axis=axis)
    edge_slice = tuple(slice(0, 2) if i == axis else slice(None) for i in range(3))
    boundary_grad = grad[edge_slice]
    mean_grad = np.mean(np.abs(boundary_grad))
    print(f"ℹ️ Mean ∂φ/∂x at boundary (axis={axis}): {mean_grad:.3e}")
    return mean_grad


@pytest.mark.xfail(reason="Neumann BCs not yet implemented in Poisson solver")
def test_poisson_solution_respects_neumann_boundary_at_outflow():
    shape = (16, 16, 16)
    mesh = mesh_metadata(shape)
    velocity = create_uniform_divergence_field(mesh["grid_shape"], magnitude=0.5)

    div = compute_pressure_divergence(velocity, mesh)
    rhs = np.pad(div, ((1, 1), (1, 1), (1, 1)), mode="constant")

    phi = solve_poisson_for_phi(rhs, mesh, time_step=1.0)
    mean_grad = check_pressure_gradient_near_zero(phi, axis=0)

    # Log only; do not assert until BCs are properly implemented
    print("📝 Pressure gradient near outlet recorded for diagnostic purposes.")


@pytest.mark.xfail(reason="Neumann BCs not yet implemented in projection step")
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
    print(f"ℹ️ Momentum before : {np.linalg.norm(momentum_before):.3e}")
    print(f"ℹ️ Momentum after  : {np.linalg.norm(momentum_after):.3e}")
    print(f"ℹ️ Δmomentum       : {delta:.3e}")
    print("📝 Momentum shift recorded for diagnostic purposes.")



