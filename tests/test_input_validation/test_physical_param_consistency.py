# tests/test_input_validation/test_physical_param_consistency.py

import pytest

def test_positive_density(basic_solver_config):
    """Ensure fluid density is physically valid (positive scalar)."""
    density = basic_solver_config.get("fluid", {}).get("density", None)
    assert density is not None, "Density field missing in config."
    assert isinstance(density, (int, float)), "Density must be a number."
    assert density > 0, "Density must be positive."

def test_non_negative_viscosity(basic_solver_config):
    """Viscosity must be non-negative (zero allowed for inviscid models)."""
    viscosity = basic_solver_config.get("fluid", {}).get("viscosity", None)
    assert viscosity is not None, "Viscosity field missing in config."
    assert isinstance(viscosity, (int, float)), "Viscosity must be a number."
    assert viscosity >= 0, "Viscosity must be non-negative."

def test_valid_time_step_for_explicit_solver(basic_solver_config):
    """For explicit solvers, time step should be sufficiently small (mock CFL check)."""
    method = basic_solver_config.get("solver", {}).get("method", "").lower()
    dt = basic_solver_config.get("time", {}).get("time_step", None)
    dx = basic_solver_config.get("grid", {}).get("dx", 1.0)  # fallback default
    velocity = basic_solver_config.get("initial_conditions", {}).get("velocity_magnitude", 1.0)

    if method == "explicit":
        assert dt is not None, "Time step not provided."
        cfl_estimate = velocity * dt / dx
        assert cfl_estimate < 1.0, f"Time step too large for explicit solver: CFL={cfl_estimate:.2f}"

def test_grid_resolution_is_positive(basic_solver_config):
    """Grid must have positive number of cells in each direction."""
    grid = basic_solver_config.get("grid", {})
    nx = grid.get("nx", 0)
    ny = grid.get("ny", 0)
    nz = grid.get("nz", 0)
    assert all(isinstance(val, int) and val > 0 for val in [nx, ny, nz]), "Grid resolution must be positive in all dimensions."



