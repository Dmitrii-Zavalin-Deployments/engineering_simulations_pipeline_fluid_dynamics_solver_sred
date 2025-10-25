# tests/physics/divergence_methods/test_central.py
# ✅ Validation suite for src/physics/divergence_methods/central.py

import pytest
from src.physics.divergence_methods.central import compute_central_divergence
from src.grid_modules.cell import Cell

def make_cell(x, y, z, velocity=None, pressure=0.0, fluid_mask=True):
    return Cell(
        x=x,
        y=y,
        z=z,
        velocity=velocity or [0.0, 0.0, 0.0],
        pressure=pressure,
        fluid_mask=fluid_mask
    )

def test_compute_central_divergence_returns_empty_for_empty_grid():
    config = {"domain_definition": {"nx": 2, "ny": 2, "nz": 2}}
    assert compute_central_divergence([], config) == []

def test_compute_central_divergence_excludes_non_fluid_cells():
    config = {"domain_definition": {"nx": 2, "ny": 2, "nz": 2}}
    solid = make_cell(0.0, 0.0, 0.0, velocity=[1.0, 0.0, 0.0], fluid_mask=False)
    result = compute_central_divergence([solid], config)
    assert result == []

def test_compute_central_divergence_detects_uniform_flow():
    domain = {
        "min_x": 0.0, "max_x": 1.0,
        "min_y": 0.0, "max_y": 1.0,
        "min_z": 0.0, "max_z": 1.0,
        "nx": 2, "ny": 2, "nz": 2
    }

    dx = (domain["max_x"] - domain["min_x"]) / domain["nx"]
    dy = (domain["max_y"] - domain["min_y"]) / domain["ny"]
    dz = (domain["max_z"] - domain["min_z"]) / domain["nz"]

    config = {"domain_definition": domain}

    center = make_cell(0.5, 0.5, 0.5, velocity=[0.0, 0.0, 0.0])
    x_plus = make_cell(0.5 + dx, 0.5, 0.5, velocity=[1.0, 0.0, 0.0])
    x_minus = make_cell(0.5 - dx, 0.5, 0.5, velocity=[1.0, 0.0, 0.0])
    y_plus = make_cell(0.5, 0.5 + dy, 0.5, velocity=[0.0, 2.0, 0.0])
    y_minus = make_cell(0.5, 0.5 - dy, 0.5, velocity=[0.0, 2.0, 0.0])
    z_plus = make_cell(0.5, 0.5, 0.5 + dz, velocity=[0.0, 0.0, 3.0])
    z_minus = make_cell(0.5, 0.5, 0.5 - dz, velocity=[0.0, 0.0, 3.0])

    grid = [center, x_plus, x_minus, y_plus, y_minus, z_plus, z_minus]
    result = compute_central_divergence(grid, config)

    assert len(result) == 7
    assert all(round(val, 5) == 0.0 for val in result)

def test_compute_central_divergence_detects_asymmetric_flow():
    domain = {
        "min_x": 0.0, "max_x": 1.0,
        "min_y": 0.0, "max_y": 1.0,
        "min_z": 0.0, "max_z": 1.0,
        "nx": 2, "ny": 2, "nz": 2
    }

    dx = (domain["max_x"] - domain["min_x"]) / domain["nx"]
    config = {"domain_definition": domain}

    center = make_cell(0.5, 0.5, 0.5, velocity=[0.0, 0.0, 0.0])
    x_plus = make_cell(0.5 + dx, 0.5, 0.5, velocity=[2.0, 0.0, 0.0])
    x_minus = make_cell(0.5 - dx, 0.5, 0.5, velocity=[0.0, 0.0, 0.0])

    grid = [center, x_plus, x_minus]
    result = compute_central_divergence(grid, config)

    assert len(result) == 3
    assert any(val > 0.0 for val in result)

def test_compute_central_divergence_handles_missing_neighbors_gracefully():
    domain = {
        "min_x": 0.0, "max_x": 1.0,
        "min_y": 0.0, "max_y": 1.0,
        "min_z": 0.0, "max_z": 1.0,
        "nx": 2, "ny": 2, "nz": 2
    }

    config = {"domain_definition": domain}
    center = make_cell(0.5, 0.5, 0.5, velocity=[1.0, 2.0, 3.0])
    grid = [center]  # no neighbors

    result = compute_central_divergence(grid, config)
    assert len(result) == 1
    assert isinstance(result[0], float)



