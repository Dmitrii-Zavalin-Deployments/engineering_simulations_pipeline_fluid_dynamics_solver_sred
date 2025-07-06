# tests/test_solver_core/test_conservation_extended.py

import numpy as np
from src.numerical_methods.poisson_solver import solve_poisson_for_phi
from src.numerical_methods.pressure_divergence import compute_pressure_divergence
from src.numerical_methods.pressure_correction import apply_pressure_correction
from tests.test_solver_core.test_utils import mesh_metadata


def compute_kinetic_energy(velocity):
    core = velocity[1:-1, 1:-1, 1:-1, :]
    return 0.5 * np.sum(core ** 2)


def test_kinetic_energy_drops_after_projection():
    shape = (10, 10, 10)
    mesh = mesh_metadata(shape)
    velocity = np.random.randn(*mesh["grid_shape"], 3) * 0.2
    ke_before = compute_kinetic_energy(velocity)

    div = compute_pressure_divergence(velocity, mesh)
    rhs = np.pad(div, ((1, 1), (1, 1), (1, 1)), mode="constant")
    phi = solve_poisson_for_phi(rhs, mesh, time_step=1.0)
    corrected_u, _ = apply_pressure_correction(
        velocity, np.zeros_like(phi), phi[1:-1, 1:-1, 1:-1],
        mesh, 1.0, 1.0
    )
    ke_after = compute_kinetic_energy(corrected_u)
    assert ke_after < ke_before, f"Kinetic energy did not drop: {ke_after:.2e} ≥ {ke_before:.2e}"


def test_projection_on_divergence_free_field_is_passive():
    shape = (10, 10, 10)
    mesh = mesh_metadata(shape)
    x = np.linspace(-1, 1, mesh["grid_shape"][0])
    y = np.linspace(-1, 1, mesh["grid_shape"][1])
    z = np.linspace(-1, 1, mesh["grid_shape"][2])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    velocity = np.zeros((*mesh["grid_shape"], 3))
    velocity[..., 0] = -Y
    velocity[..., 1] = X

    div = compute_pressure_divergence(velocity, mesh)
    assert np.allclose(div[1:-1, 1:-1, 1:-1], 0.0, atol=1e-10)

    rhs = np.pad(div, ((1, 1), (1, 1), (1, 1)), mode="constant")
    phi = solve_poisson_for_phi(rhs, mesh, time_step=1.0)
    corrected_u, _ = apply_pressure_correction(
        velocity, np.zeros_like(phi), phi[1:-1, 1:-1, 1:-1],
        mesh, 1.0, 1.0
    )
    delta = np.linalg.norm(corrected_u - velocity)
    assert delta < 1e-8, f"Passive field changed during projection: Δ = {delta:.2e}"


def test_projection_preserves_velocity_symmetry():
    shape = (10, 10, 10)
    mesh = mesh_metadata(shape)
    velocity = np.zeros((*mesh["grid_shape"], 3))

    mid_y = mesh["grid_shape"][1] // 2
    velocity[:, :mid_y, :, 0] = -1.0
    velocity[:, mid_y:, :, 0] = 1.0

    div = compute_pressure_divergence(velocity, mesh)
    rhs = np.pad(div, ((1, 1), (1, 1), (1, 1)), mode="constant")
    phi = solve_poisson_for_phi(rhs, mesh, time_step=1.0)
    corrected_u, _ = apply_pressure_correction(
        velocity, np.zeros_like(phi), phi[1:-1, 1:-1, 1:-1],
        mesh, 1.0, 1.0
    )
    left = corrected_u[:, :mid_y, :, 0]
    right = corrected_u[:, mid_y:, :, 0]
    flipped_right = -right[:, ::-1, :]
    assert np.allclose(left, flipped_right, atol=1e-6), "Symmetry violated"


def test_projection_on_constant_pressure_does_nothing():
    shape = (10, 10, 10)
    mesh = mesh_metadata(shape)
    velocity = np.random.randn(*mesh["grid_shape"], 3) * 0.1
    pressure = np.ones(mesh["grid_shape"]) * 7.0

    corrected_u, _ = apply_pressure_correction(
        velocity, pressure, pressure[1:-1, 1:-1, 1:-1],
        mesh, 1.0, 1.0
    )
    delta = np.linalg.norm(corrected_u - velocity)
    assert delta < 1e-10, f"Velocity changed under constant pressure: Δ = {delta:.2e}"


def test_projection_is_independent_of_velocity_magnitude():
    shape = (12, 12, 12)
    mesh = mesh_metadata(shape)
    base_velocity = np.random.randn(*mesh["grid_shape"], 3)
    divergences = []

    for scale in [0.1, 1.0, 5.0]:
        velocity = base_velocity * scale
        div = compute_pressure_divergence(velocity, mesh)
        rhs = np.pad(div, ((1, 1), (1, 1), (1, 1)), mode="constant")
        phi = solve_poisson_for_phi(rhs, mesh, time_step=1.0)
        corrected_u, _ = apply_pressure_correction(
            velocity, np.zeros_like(phi), phi[1:-1, 1:-1, 1:-1],
            mesh, 1.0, 1.0
        )
        div_after = compute_pressure_divergence(corrected_u, mesh)
        divergences.append(np.mean(np.abs(div_after)) / scale)

    spread = max(divergences) - min(divergences)
    assert spread < 1e-3, f"Non-scale-invariant divergence suppression: spread = {spread:.2e}"



