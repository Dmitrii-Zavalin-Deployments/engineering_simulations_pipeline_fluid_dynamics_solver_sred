# tests/test_solver_core/test_dirichlet_conditions.py

import numpy as np
from src.numerical_methods.poisson_solver import solve_poisson_for_phi
from src.numerical_methods.pressure_correction import apply_pressure_correction
from src.numerical_methods.pressure_divergence import compute_pressure_divergence
from tests.test_solver_core.test_utils import mesh_metadata


def create_velocity_with_dirichlet_bc(grid_shape, inlet_velocity=1.0):
    """
    Creates a velocity field with fixed Dirichlet boundary conditions:
    - u = inlet_velocity at x=0 (inlet)
    - u = 0 elsewhere
    """
    velocity = np.zeros((*grid_shape, 3))
    velocity[0, :, :, 0] = inlet_velocity  # x-component at inlet
    return velocity


def test_dirichlet_bc_inlet_flow_remains_consistent():
    shape = (16, 16, 16)
    mesh = mesh_metadata(shape, dx=1.0)
    inlet_velocity = 0.5

    velocity = create_velocity_with_dirichlet_bc(mesh["grid_shape"], inlet_velocity=inlet_velocity)
    div_before = compute_pressure_divergence(velocity, mesh)

    # Sanity check: field is initially divergent at the inlet
    mean_div = np.mean(np.abs(div_before))
    assert mean_div > 1e-3, f"Unexpectedly low initial divergence: {mean_div:.3e}"

    # Apply projection
    rhs = np.pad(div_before, ((1, 1), (1, 1), (1, 1)), mode="constant")
    phi = solve_poisson_for_phi(rhs, mesh, time_step=1.0)
    corrected_u, _ = apply_pressure_correction(
        velocity, np.zeros_like(phi), phi[1:-1, 1:-1, 1:-1],
        mesh, 1.0, 1.0
    )

    # Validate divergence dropped
    div_after = compute_pressure_divergence(corrected_u, mesh)
    drop = np.linalg.norm(div_before) - np.linalg.norm(div_after)
    assert drop > 0.1, f"Divergence did not significantly drop after projection (Δ = {drop:.3e})"

    # Check inlet velocity remains near prescribed value
    inlet_u_corrected = corrected_u[0, :, :, 0]
    avg_inlet_u = np.mean(inlet_u_corrected)
    assert np.isclose(avg_inlet_u, inlet_velocity, rtol=0.2), (
        f"Inlet velocity changed too much: expected ~{inlet_velocity:.2f}, got {avg_inlet_u:.3f}"
    )



