# tests/physics/test_boundary_condition_solver.py
# 🧪 Unit tests for boundary_condition_solver.py — ghost field enforcement

from src.physics.boundary_condition_solver import apply_boundary_conditions
from src.grid_modules.cell import Cell

def make_ghost_cell(x, y, z):
    return Cell(x=x, y=y, z=z, velocity=None, pressure=None, fluid_mask=False)

def make_fluid_cell(x, y, z):
    return Cell(x=x, y=y, z=z, velocity=[1.0, 0.0, 0.0], pressure=10.0, fluid_mask=True)

def make_config(apply_to, velocity=None, pressure=None, no_slip=False):
    return {
        "boundary_conditions": {
            "apply_to": apply_to,
            "velocity": velocity,
            "pressure": pressure,
            "no_slip": no_slip
        }
    }

def test_velocity_and_pressure_applied_to_ghost():
    ghost = make_ghost_cell(0, 0, 0)
    registry = {id(ghost)}
    config = make_config(["velocity", "pressure"], [5.0, 0.0, -1.0], 99.0)

    updated = apply_boundary_conditions([ghost], registry, config)
    assert updated[0].velocity == [5.0, 0.0, -1.0]
    assert updated[0].pressure == 99.0

def test_no_slip_velocity_enforced():
    ghost = make_ghost_cell(0, 0, 0)
    registry = {id(ghost)}
    config = make_config(["velocity"], [3.0, 3.0, 3.0], pressure=None, no_slip=True)

    updated = apply_boundary_conditions([ghost], registry, config)
    assert updated[0].velocity == [0.0, 0.0, 0.0]
    assert updated[0].pressure is None

def test_only_pressure_applied():
    ghost = make_ghost_cell(0, 0, 0)
    registry = {id(ghost)}
    config = make_config(["pressure"], velocity=None, pressure=12.5)

    updated = apply_boundary_conditions([ghost], registry, config)
    assert updated[0].velocity is None
    assert updated[0].pressure == 12.5

def test_unconfigured_ghost_remains_null():
    ghost = make_ghost_cell(0, 0, 0)
    registry = {id(ghost)}
    config = {}  # No boundary_conditions

    updated = apply_boundary_conditions([ghost], registry, config)
    assert updated[0].velocity is None
    assert updated[0].pressure is None

def test_non_ghost_cells_not_touched():
    ghost = make_ghost_cell(0, 0, 0)
    fluid = make_fluid_cell(1, 0, 0)
    registry = {id(ghost)}
    config = make_config(["velocity", "pressure"], [9.0, 9.0, 9.0], 5.0)

    updated = apply_boundary_conditions([ghost, fluid], registry, config)
    assert updated[0].velocity == [9.0, 9.0, 9.0]
    assert updated[0].pressure == 5.0
    assert updated[1].velocity == [1.0, 0.0, 0.0]
    assert updated[1].pressure == 10.0

def test_missing_apply_to_clears_ghost_fields():
    ghost = make_ghost_cell(0, 0, 0)
    registry = {id(ghost)}
    config = {
        "boundary_conditions": {
            "velocity": [1.0, 0.0, 0.0],
            "pressure": 10.0
        }
    }

    updated = apply_boundary_conditions([ghost], registry, config)
    assert updated[0].velocity is None
    assert updated[0].pressure is None

def test_partial_fields_enforcement():
    ghost = make_ghost_cell(0, 0, 0)
    registry = {id(ghost)}
    config = {
        "boundary_conditions": {
            "apply_to": ["velocity"],
            "velocity": [0.5, 0.5, 0.5]
        }
    }

    updated = apply_boundary_conditions([ghost], registry, config)
    assert updated[0].velocity == [0.5, 0.5, 0.5]
    assert updated[0].pressure is None



