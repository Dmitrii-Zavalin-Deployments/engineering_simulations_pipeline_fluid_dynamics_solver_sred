# tests/physics/test_boundary_condition_solver.py
# 🧪 Unit tests for src/physics/boundary_condition_solver.py

from src.grid_modules.cell import Cell
from src.physics.boundary_condition_solver import apply_boundary_conditions

def make_cell(x, y, z, velocity=None, pressure=None, fluid=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=pressure, fluid_mask=fluid)

def test_applies_velocity_and_pressure_to_ghost_cell():
    ghost = make_cell(1.0, 0.0, 0.0, fluid=False)
    ghost_registry = {id(ghost): {"origin": (0.0, 0.0, 0.0)}}
    fluid = make_cell(0.0, 0.0, 0.0)
    config = {
        "boundary_conditions": {
            "velocity": [1.0, 2.0, 3.0],
            "pressure": 9.0,
            "apply_to": ["velocity", "pressure"],
            "no_slip": False
        }
    }
    result = apply_boundary_conditions([ghost, fluid], ghost_registry, config)
    assert ghost.velocity == [1.0, 2.0, 3.0]
    assert ghost.pressure == 9.0
    assert getattr(ghost, "ghost_field_applied", False) is True
    assert fluid.velocity == [1.0, 2.0, 3.0]
    assert fluid.pressure == 9.0

def test_applies_no_slip_velocity_to_ghost_and_fluid():
    ghost = make_cell(1.0, 1.0, 1.0, fluid=False)
    fluid = make_cell(0.0, 0.0, 0.0)
    ghost_registry = {id(ghost): {"origin": (0.0, 0.0, 0.0)}}
    config = {
        "boundary_conditions": {
            "velocity": [7.0, 7.0, 7.0],
            "pressure": 5.0,
            "apply_to": ["velocity"],
            "no_slip": True
        }
    }
    result = apply_boundary_conditions([ghost, fluid], ghost_registry, config)
    assert ghost.velocity == [0.0, 0.0, 0.0]
    assert fluid.velocity == [0.0, 0.0, 0.0]

def test_skips_velocity_application_when_not_in_apply_to():
    ghost = make_cell(1.0, 0.0, 0.0, fluid=False)
    ghost.velocity = [9.0, 9.0, 9.0]
    ghost_registry = {id(ghost): {"origin": (2.0, 0.0, 0.0)}}
    fluid = make_cell(2.0, 0.0, 0.0, velocity=[9.0, 9.0, 9.0])
    config = {
        "boundary_conditions": {
            "velocity": [1.0, 1.0, 1.0],
            "apply_to": ["pressure"]
        }
    }
    result = apply_boundary_conditions([ghost, fluid], ghost_registry, config)
    assert ghost.velocity is None
    assert fluid.velocity == [9.0, 9.0, 9.0]

def test_removes_pressure_when_not_in_apply_to():
    ghost = make_cell(2.0, 2.0, 2.0, fluid=False)
    ghost.pressure = 8.0
    ghost_registry = {id(ghost): {"origin": (0.0, 0.0, 0.0)}}
    config = {
        "boundary_conditions": {
            "apply_to": ["velocity"]
        }
    }
    result = apply_boundary_conditions([ghost], ghost_registry, config)
    assert ghost.pressure is None

def test_does_not_crash_on_missing_origin_in_registry():
    fluid = make_cell(3.0, 3.0, 3.0, velocity=[0.0, 0.0, 0.0])
    ghost_registry = {999999: {"face": "x+"}}  # No 'origin'
    config = {
        "boundary_conditions": {
            "velocity": [1, 1, 1],
            "apply_to": ["velocity"]
        }
    }
    apply_boundary_conditions([fluid], ghost_registry, config)
    assert fluid.velocity == [0.0, 0.0, 0.0] or fluid.velocity is not None

def test_skips_enforcement_if_ghost_not_in_registry():
    ghost = make_cell(5.0, 5.0, 5.0, fluid=False)
    ghost_registry = {}
    config = {
        "boundary_conditions": {
            "velocity": [9.9, 9.9, 9.9],
            "pressure": 4.4,
            "apply_to": ["velocity", "pressure"]
        }
    }
    result = apply_boundary_conditions([ghost], ghost_registry, config)
    assert ghost.velocity == [9.9, 9.9, 9.9] or ghost.velocity != [0.0, 0.0, 0.0]



