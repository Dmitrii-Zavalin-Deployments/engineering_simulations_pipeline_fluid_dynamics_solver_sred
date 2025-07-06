# tests/test_grid_initialization/test_grid_shape_and_spacing.py

import pytest

def test_grid_dimensions_are_positive(basic_solver_config):
    """Ensure grid resolution is positive in all directions."""
    grid = basic_solver_config.get("grid", {})
    nx = grid.get("nx", -1)
    ny = grid.get("ny", -1)
    nz = grid.get("nz", -1)

    assert isinstance(nx, int) and nx > 0, f"Invalid nx: {nx}"
    assert isinstance(ny, int) and ny > 0, f"Invalid ny: {ny}"
    assert isinstance(nz, int) and nz > 0, f"Invalid nz: {nz}"

def test_grid_spacing_is_positive(basic_solver_config):
    """Ensure that grid spacing dx (and optionally dy, dz) is physically valid."""
    grid = basic_solver_config.get("grid", {})
    dx = grid.get("dx", None)
    dy = grid.get("dy", dx)  # Default fallback to dx
    dz = grid.get("dz", dx)

    for name, spacing in zip(["dx", "dy", "dz"], [dx, dy, dz]):
        assert spacing is not None, f"{name} missing from config."
        assert isinstance(spacing, (int, float)) and spacing > 0, f"{name} must be positive."

def test_cell_count_consistent_with_physical_extent(basic_solver_config):
    """If domain extent is specified, verify dx matches number of cells."""
    grid = basic_solver_config.get("grid", {})
    extent_x = grid.get("extent_x")  # Optional
    nx = grid.get("nx")
    dx = grid.get("dx")

    if extent_x is not None and nx and dx:
        expected_dx = extent_x / nx
        assert abs(dx - expected_dx) < 1e-6, f"Inconsistent dx: expected {expected_dx}, got {dx}"



