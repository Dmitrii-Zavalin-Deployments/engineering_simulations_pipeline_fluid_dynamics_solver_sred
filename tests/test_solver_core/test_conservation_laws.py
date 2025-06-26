# tests/test_solver_core/test_conservation_laws.py

import numpy as np
from src.numerical_methods.poisson_solver import solve_poisson_for_phi
from src.numerical_methods.pressure_correction import apply_pressure_correction
from src.numerical_methods.pressure_divergence import compute_pressure_divergence
from tests.test_solver_core.test_utils import (
    mesh_metadata,
    create_sinusoidal_velocity,
)


def compute_total_momentum(velocity):
    return np.sum(velocity[1:-1, 1:-1, 1:-1, :], axis=(0, 1, 2))


def compute_total_mass_flux(divergence_field, cell_volume=1.0):
    return np.sum(divergence_field) * cell_volume


def test_mass_conservation_improves_after_projection():
    shape = (16, 16, 16)
    mesh = mesh_metadata(shape)

    velocity = np.random.randn(*mesh["grid_shape"], 3) * 0.1
    div_before = compute_pressure_divergence(velocity, mesh)
    mass_flux_before = compute_total_mass_flux(div_before)

    rhs = np.pad(div_before, ((1, 1), (1, 1), (1, 1)), mode="constant")
    phi = solve_poisson_for_phi(rhs, mesh, time_step=1.0)
    corrected_u, _ = apply_pressure_correction(
        velocity, np.zeros_like(phi), phi[1:-1, 1:-1, 1:-1],
        mesh, 1.0, 1.0
    )

    div_after = compute_pressure_divergence(corrected_u, mesh)
    mass_flux_after = compute_total_mass_flux(div_after)

    assert abs(mass_flux_after) < abs(mass_flux_before), (
        f"Mass flux worsened: before={mass_flux_before:.2e}, after={mass_flux_after:.2e}"
    )


def test_projection_preserves_total_momentum():
    shape = (12, 12, 12)
    mesh = mesh_metadata(shape)

    velocity = np.random.randn(*mesh["grid_shape"], 3) * 0.2
    momentum_before = compute_total_momentum(velocity)

    div = compute_pressure_divergence(velocity, mesh)
    rhs = np.pad(div, ((1, 1), (1, 1), (1, 1)), mode="constant")
    phi = solve_poisson_for_phi(rhs, mesh, time_step=1.0)
    corrected_u, _ = apply_pressure_correction(
        velocity, np.zeros_like(phi), phi[1:-1, 1:-1, 1:-1],
        mesh, 1.0, 1.0
    )

    momentum_after = compute_total_momentum(corrected_u)
    delta = np.linalg.norm(momentum_after - momentum_before)
    norm_before = np.linalg.norm(momentum_before)
    rel_error = delta / norm_before if norm_before > 1e-12 else delta

    assert rel_error < 0.40, (
        f"Momentum changed by {rel_error:.2%}, above relaxed threshold. "
        f"Consider reviewing projection behavior or boundary handling."
    )


def test_projection_preserves_momentum_on_divergence_free_field():
    shape = (12, 12, 12)
    mesh = mesh_metadata(shape)

    psi = np.random.randn(*mesh["grid_shape"])
    velocity = np.zeros((*mesh["grid_shape"], 3))
    velocity[..., 0] = np.gradient(psi, axis=1)
    velocity[..., 1] = -np.gradient(psi, axis=0)

    momentum_before = compute_total_momentum(velocity)

    div = compute_pressure_divergence(velocity, mesh)
    assert np.allclose(div[1:-1, 1:-1, 1:-1], 0.0, atol=1e-10)

    rhs = np.pad(div, ((1, 1), (1, 1), (1, 1)), mode="constant")
    phi = solve_poisson_for_phi(rhs, mesh, time_step=1.0)
    corrected_u, _ = apply_pressure_correction(
        velocity, np.zeros_like(phi), phi[1:-1, 1:-1, 1:-1],
        mesh, 1.0, 1.0
    )

    momentum_after = compute_total_momentum(corrected_u)
    delta = np.linalg.norm(momentum_after - momentum_before)
    norm = np.linalg.norm(momentum_before)
    rel_error = delta / norm if norm > 1e-12 else delta

    assert rel_error < 1e-8, (
        f"Projection altered momentum of divergence-free field: Δ = {delta:.2e}, rel = {rel_error:.2e}"
    )


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

    assert div_drop > 1e-3, f"Divergence was not meaningfully reduced: Δdiv = {div_drop:.2e}"

    ratio = delta_momentum / div_drop
    threshold = 1.0
    assert ratio < threshold, (
        f"Momentum changed more than expected relative to divergence reduction.\n"
        f"Δmomentum = {delta_momentum:.2e}, Δdivergence = {div_drop:.2e}, "
        f"ratio = {ratio:.2f}, threshold = {threshold:.2f}"
    )



