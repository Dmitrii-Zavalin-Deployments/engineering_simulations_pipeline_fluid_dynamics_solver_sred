# tests/physics/test_boundary_condition_solver.py
# 🧪 Validates ghost and fluid field enforcement per boundary config options

from src.grid_modules.cell import Cell
from src.physics.boundary_condition_solver import apply_boundary_conditions

def make_cell(x, y, z, fluid=True):
    return Cell(x=x, y=y, z=z, velocity=[0.0, 0.0, 0.0], pressure=0.0, fluid_mask=fluid)

def test_velocity_and_pressure_enforced_on_ghost_and_fluid():
    ghost = make_cell(0.0, 0.0, 0.0, fluid=False)
    fluid = make_cell(1.0, 0.0, 0.0, fluid=True)
    registry = {id(ghost): {"origin": (1.0, 0.0, 0.0), "face": "x_min"}}
    config = {
        "boundary_conditions": {
            "velocity": [1.0, 1.0, 1.0],
            "pressure": 99.0,
            "apply_to": ["velocity", "pressure"],
            "no_slip": False
        }
    }
    result = apply_boundary_conditions([ghost, fluid], registry, config)
    assert ghost.velocity == [1.0, 1.0, 1.0]
    assert ghost.pressure == 99.0
    assert ghost.ghost_field_applied is True
    assert fluid.velocity == [1.0, 1.0, 1.0]
    assert fluid.pressure == 99.0

def test_no_slip_overrides_velocity_to_zero():
    ghost = make_cell(0.0, 0.0, 0.0, fluid=False)
    fluid = make_cell(1.0, 0.0, 0.0, fluid=True)
    registry = {id(ghost): {"origin": (1.0, 0.0, 0.0), "face": "x_max"}}
    config = {
        "boundary_conditions": {
            "velocity": [9.9, 9.9, 9.9],
            "pressure": 55.0,
            "apply_to": ["velocity", "pressure"],
            "no_slip": True
        }
    }
    result = apply_boundary_conditions([ghost, fluid], registry, config)
    assert ghost.velocity == [0.0, 0.0, 0.0]
    assert fluid.velocity == [0.0, 0.0, 0.0]

def test_ghost_fields_cleared_if_not_applied():
    ghost = make_cell(0.0, 0.0, 0.0, fluid=False)
    config = {
        "boundary_conditions": {
            "apply_to": [],  # no enforcement
        }
    }
    result = apply_boundary_conditions([ghost], {id(ghost): {}}, config)
    assert ghost.velocity is None
    assert ghost.pressure is None
    assert not hasattr(ghost, "ghost_field_applied")

def test_missing_origin_skips_fluid_update():
    fluid = make_cell(0.0, 0.0, 0.0, fluid=True)
    ghost = make_cell(1.0, 0.0, 0.0, fluid=False)
    registry = {id(ghost): {"face": "x_min"}}  # no origin key
    config = {
        "boundary_conditions": {
            "velocity": [1.0, 2.0, 3.0],
            "pressure": 88.0,
            "apply_to": ["velocity", "pressure"]
        }
    }
    result = apply_boundary_conditions([ghost, fluid], registry, config)
    assert fluid.velocity == [0.0, 0.0, 0.0]
    assert fluid.pressure == 0.0  # untouched

def test_partial_field_enforcement_respects_apply_to():
    ghost = make_cell(0.0, 0.0, 0.0, fluid=False)
    fluid = make_cell(1.0, 0.0, 0.0, fluid=True)
    registry = {id(ghost): {"origin": (1.0, 0.0, 0.0)}}
    config = {
        "boundary_conditions": {
            "velocity": [9.9, 9.9, 9.9],
            "pressure": 77.0,
            "apply_to": ["velocity"],
            "no_slip": False
        }
    }
    result = apply_boundary_conditions([ghost, fluid], registry, config)
    assert ghost.velocity == [9.9, 9.9, 9.9]
    assert ghost.pressure is None
    assert fluid.velocity == [9.9, 9.9, 9.9]
    assert fluid.pressure == 0.0  # unchanged

def test_boundary_conditions_handles_missing_registry_keys_gracefully():
    ghost = make_cell(0.0, 0.0, 0.0, fluid=False)
    fluid = make_cell(1.0, 0.0, 0.0, fluid=True)
    registry = {id(ghost): {"origin": (1.0, 0.0)}}  # malformed origin
    config = {
        "boundary_conditions": {
            "velocity": [1.0, 1.0, 1.0],
            "pressure": 99.0,
            "apply_to": ["velocity", "pressure"]
        }
    }
    result = apply_boundary_conditions([ghost, fluid], registry, config)
    assert fluid.velocity == [0.0, 0.0, 0.0]  # untouched
    assert fluid.pressure == 0.0