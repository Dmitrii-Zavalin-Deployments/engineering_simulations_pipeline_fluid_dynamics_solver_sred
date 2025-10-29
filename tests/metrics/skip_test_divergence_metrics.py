# tests/metrics/test_divergence_metrics.py
# ✅ Validation suite for src/metrics/divergence_metrics.py

import pytest
from src.metrics.divergence_metrics import compute_max_divergence
from src.grid_modules.cell import Cell

def make_cell(x, y, z, velocity=None, fluid_mask=True):
    return Cell(
        x=x,
        y=y,
        z=z,
        velocity=velocity or [0.0, 0.0, 0.0],
        pressure=0.0,
        fluid_mask=fluid_mask
    )

def test_compute_max_divergence_returns_zero_for_empty_grid():
    domain = {"min_x": 0, "max_x": 1, "nx": 10}
    assert compute_max_divergence([], domain) == 0.0
    assert compute_max_divergence(None, domain) == 0.0
    assert compute_max_divergence([], {}) == 0.0

def test_compute_max_divergence_excludes_non_fluid_cells():
    domain = {
        "min_x": 0.0, "max_x": 1.0,
        "min_y": 0.0, "max_y": 1.0,
        "min_z": 0.0, "max_z": 1.0,
        "nx": 2, "ny": 2, "nz": 2
    }

    grid = [
        make_cell(0.5, 0.5, 0.5, velocity=[1.0, 0.0, 0.0], fluid_mask=False),  # solid
        make_cell(0.5, 0.5, 0.5, velocity=[1.0, 0.0, 0.0], fluid_mask=True)   # fluid
    ]

    result = compute_max_divergence(grid, domain)
    assert isinstance(result, float)
    assert result >= 0.0
    assert grid[0].__dict__.get("divergence") is None
    assert grid[1].__dict__.get("divergence") is not None

def test_compute_max_divergence_with_symmetric_neighbors():
    domain = {
        "min_x": 0.0, "max_x": 1.0,
        "min_y": 0.0, "max_y": 1.0,
        "min_z": 0.0, "max_z": 1.0,
        "nx": 2, "ny": 2, "nz": 2
    }

    dx = (domain["max_x"] - domain["min_x"]) / domain["nx"]
    dy = (domain["max_y"] - domain["min_y"]) / domain["ny"]
    dz = (domain["max_z"] - domain["min_z"]) / domain["nz"]

    center = make_cell(0.5, 0.5, 0.5, velocity=[0.0, 0.0, 0.0])
    x_plus = make_cell(0.5 + dx, 0.5, 0.5, velocity=[1.0, 0.0, 0.0])
    x_minus = make_cell(0.5 - dx, 0.5, 0.5, velocity=[1.0, 0.0, 0.0])
    y_plus = make_cell(0.5, 0.5 + dy, 0.5, velocity=[0.0, 2.0, 0.0])
    y_minus = make_cell(0.5, 0.5 - dy, 0.5, velocity=[0.0, 2.0, 0.0])
    z_plus = make_cell(0.5, 0.5, 0.5 + dz, velocity=[0.0, 0.0, 3.0])
    z_minus = make_cell(0.5, 0.5, 0.5 - dz, velocity=[0.0, 0.0, 3.0])

    grid = [center, x_plus, x_minus, y_plus, y_minus, z_plus, z_minus]
    result = compute_max_divergence(grid, domain)

    assert round(result, 5) == 0.0
    assert center.divergence == 0.0

def test_compute_max_divergence_detects_asymmetry():
    domain = {
        "min_x": 0.0, "max_x": 1.0,
        "min_y": 0.0, "max_y": 1.0,
        "min_z": 0.0, "max_z": 1.0,
        "nx": 2, "ny": 2, "nz": 2
    }

    dx = (domain["max_x"] - domain["min_x"]) / domain["nx"]
    center = make_cell(0.5, 0.5, 0.5, velocity=[0.0, 0.0, 0.0])
    x_plus = make_cell(0.5 + dx, 0.5, 0.5, velocity=[2.0, 0.0, 0.0])
    x_minus = make_cell(0.5 - dx, 0.5, 0.5, velocity=[0.0, 0.0, 0.0])

    grid = [center, x_plus, x_minus]
    result = compute_max_divergence(grid, domain)

    assert result > 0.0
    assert center.divergence > 0.0

def test_compute_max_divergence_handles_missing_neighbors_gracefully():
    domain = {
        "min_x": 0.0, "max_x": 1.0,
        "min_y": 0.0, "max_y": 1.0,
        "min_z": 0.0, "max_z": 1.0,
        "nx": 2, "ny": 2, "nz": 2
    }

    center = make_cell(0.5, 0.5, 0.5, velocity=[1.0, 2.0, 3.0])
    grid = [center]  # no neighbors

    result = compute_max_divergence(grid, domain)
    assert result == 0.0
    assert center.divergence == 0.0
