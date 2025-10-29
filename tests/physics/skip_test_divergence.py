# tests/physics/test_divergence.py
# ✅ Validation suite for src/physics/divergence.py

import pytest
from src.physics.divergence import compute_divergence
from src.grid_modules.cell import Cell

def make_cell(x, y, z, velocity=None, fluid_mask=True):
    return Cell(
        x=x,
        y=y,
        z=z,
        velocity=velocity,
        pressure=0.0,
        fluid_mask=fluid_mask
    )

def test_divergence_excludes_ghost_cells():
    ghost = make_cell(0.0, 0.0, 0.0, velocity=[1.0, 0.0, 0.0])
    fluid = make_cell(1.0, 0.0, 0.0, velocity=[1.0, 0.0, 0.0])
    ghost_registry = {id(ghost)}
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 2.0,
            "min_y": 0.0, "max_y": 2.0,
            "min_z": 0.0, "max_z": 2.0,
            "nx": 2, "ny": 2, "nz": 2
        }
    }

    result = compute_divergence([ghost, fluid], config, ghost_registry)
    assert len(result) == 1

def test_divergence_excludes_nonfluid_cells():
    solid = make_cell(0.0, 0.0, 0.0, velocity=[1.0, 0.0, 0.0], fluid_mask=False)
    fluid = make_cell(1.0, 0.0, 0.0, velocity=[1.0, 0.0, 0.0], fluid_mask=True)
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 2.0,
            "min_y": 0.0, "max_y": 2.0,
            "min_z": 0.0, "max_z": 2.0,
            "nx": 2, "ny": 2, "nz": 2
        }
    }

    result = compute_divergence([solid, fluid], config)
    assert len(result) == 1

def test_divergence_excludes_cells_with_invalid_velocity():
    bad = make_cell(0.0, 0.0, 0.0, velocity=None, fluid_mask=True)
    good = make_cell(1.0, 0.0, 0.0, velocity=[1.0, 0.0, 0.0], fluid_mask=True)
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 2.0,
            "min_y": 0.0, "max_y": 2.0,
            "min_z": 0.0, "max_z": 2.0,
            "nx": 2, "ny": 2, "nz": 2
        }
    }

    result = compute_divergence([bad, good], config)
    assert len(result) == 1

def test_divergence_returns_empty_on_all_excluded():
    ghost = make_cell(0.0, 0.0, 0.0, velocity=[1.0, 0.0, 0.0])
    solid = make_cell(1.0, 0.0, 0.0, velocity=[1.0, 0.0, 0.0], fluid_mask=False)
    ghost_registry = {id(ghost)}
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 2.0,
            "min_y": 0.0, "max_y": 2.0,
            "min_z": 0.0, "max_z": 2.0,
            "nx": 2, "ny": 2, "nz": 2
        }
    }

    result = compute_divergence([ghost, solid], config, ghost_registry)
    assert result == []

def test_divergence_computes_valid_output_for_fluid_cells():
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 2.0,
            "min_y": 0.0, "max_y": 2.0,
            "min_z": 0.0, "max_z": 2.0,
            "nx": 2, "ny": 2, "nz": 2
        }
    }

    center = make_cell(1.0, 1.0, 1.0, velocity=[0.0, 0.0, 0.0])
    x_plus = make_cell(2.0, 1.0, 1.0, velocity=[1.0, 0.0, 0.0])
    x_minus = make_cell(0.0, 1.0, 1.0, velocity=[1.0, 0.0, 0.0])
    y_plus = make_cell(1.0, 2.0, 1.0, velocity=[0.0, 2.0, 0.0])
    y_minus = make_cell(1.0, 0.0, 1.0, velocity=[0.0, 2.0, 0.0])
    z_plus = make_cell(1.0, 1.0, 2.0, velocity=[0.0, 0.0, 3.0])
    z_minus = make_cell(1.0, 1.0, 0.0, velocity=[0.0, 0.0, 3.0])

    grid = [center, x_plus, x_minus, y_plus, y_minus, z_plus, z_minus]
    result = compute_divergence(grid, config)
    assert len(result) == 7
    assert all(round(val, 5) == 0.0 for val in result)



