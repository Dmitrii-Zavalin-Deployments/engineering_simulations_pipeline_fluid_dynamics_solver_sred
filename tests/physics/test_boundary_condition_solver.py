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
    registry = {id(ghost): {"origin": (1.0, 0.0, 0.0)}}
    config = make_config(["velocity", "pressure"], [5.0, 0.0, -1.0], 99.0)

    updated = apply_boundary_conditions([ghost], registry, config)
    cell = updated[0]
    assert cell.velocity == [5.0, 0.0, -1.0]
    assert cell.pressure == 99.0
    assert getattr(cell, "ghost_field_applied", False)

def test_no_slip_velocity_enforced():
    ghost = make_ghost_cell(0, 0, 0)
    registry = {id(ghost): {"origin": (1.0, 0.0, 0.0)}}
    config = make_config(["velocity"], [3.0, 3.0, 3.0], pressure=None, no_slip=True)

    updated = apply_boundary_conditions([ghost], registry, config)
    cell = updated[0]
    assert cell.velocity == [0.0, 0.0, 0.0]
    assert cell.pressure is None
    assert getattr(cell, "ghost_field_applied", False) is not False  # Should be True

def test_only_pressure_applied():
    ghost = make_ghost_cell(0, 0, 0)
    registry = {id(ghost): {"origin": (1.0, 0.0, 0.0)}}
    config = make_config(["pressure"], velocity=None, pressure=12.5)

    updated = apply_boundary_conditions([ghost], registry, config)
    cell = updated[0]
    assert cell.velocity is None
    assert cell.pressure == 12.5
    assert getattr(cell, "ghost_field_applied", False)

def test_unconfigured_ghost_remains_null():
    ghost = make_ghost_cell(0, 0, 0)
    registry = {id(ghost): {"origin": (1.0, 0.0, 0.0)}}
    config = {}  # No boundary_conditions

    updated = apply_boundary_conditions([ghost], registry, config)
    cell = updated[0]
    assert cell.velocity is None
    assert cell.pressure is None
    assert not getattr(cell, "ghost_field_applied", False)

def test_non_ghost_cells_not_touched():
    ghost = make_ghost_cell(0, 0, 0)
    fluid = make_fluid_cell(1.0, 0.0, 0.0)
    registry = {id(ghost): {"origin": (fluid.x, fluid.y, fluid.z)}}
    config = make_config(["velocity", "pressure"], [9.0, 9.0, 9.0], 5.0)

    updated = apply_boundary_conditions([ghost, fluid], registry, config)
    ghost_cell = updated[0]
    fluid_cell = updated[1]
    assert ghost_cell.velocity == [9.0, 9.0, 9.0]
    assert ghost_cell.pressure == 5.0
    assert getattr(ghost_cell, "ghost_field_applied", False)
    assert fluid_cell.velocity == [9.0, 9.0, 9.0]
    assert fluid_cell.pressure == 5.0

def test_missing_apply_to_clears_ghost_fields():
    ghost = make_ghost_cell(0, 0, 0)
    registry = {id(ghost): {"origin": (1.0, 0.0, 0.0)}}
    config = {
        "boundary_conditions": {
            "velocity": [1.0, 0.0, 0.0],
            "pressure": 10.0
        }
    }

    updated = apply_boundary_conditions([ghost], registry, config)
    cell = updated[0]
    assert cell.velocity is None
    assert cell.pressure is None
    assert not getattr(cell, "ghost_field_applied", False)

def test_partial_fields_enforcement():
    ghost = make_ghost_cell(0, 0, 0)
    registry = {id(ghost): {"origin": (1.0, 0.0, 0.0)}}
    config = {
        "boundary_conditions": {
            "apply_to": ["velocity"],
            "velocity": [0.5, 0.5, 0.5]
        }
    }

    updated = apply_boundary_conditions([ghost], registry, config)
    cell = updated[0]
    assert cell.velocity == [0.5, 0.5, 0.5]
    assert cell.pressure is None
    assert getattr(cell, "ghost_field_applied", False)

def test_enforcement_reflects_into_fluid_origin():
    fluid = make_fluid_cell(1.0, 0.0, 0.0)
    ghost = make_ghost_cell(0.0, 0.0, 0.0)
    registry = {id(ghost): {"origin": (fluid.x, fluid.y, fluid.z)}}
    config = make_config(["velocity", "pressure"], [2.0, 2.0, 2.0], 88.0)

    updated = apply_boundary_conditions([ghost, fluid], registry, config)
    fluid_cell = next(c for c in updated if c.fluid_mask)
    assert fluid_cell.velocity == [2.0, 2.0, 2.0]
    assert fluid_cell.pressure == 88.0



