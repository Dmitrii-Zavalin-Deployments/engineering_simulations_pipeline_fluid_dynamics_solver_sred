# tests/physics/test_boundary_condition_solver.py
# ✅ Validation suite for src/physics/boundary_condition_solver.py

import pytest
from src.physics.boundary_condition_solver import apply_boundary_conditions
from src.grid_modules.cell import Cell

def make_cell(x, y, z, velocity=None, pressure=None, fluid_mask=True):
    return Cell(
        x=x,
        y=y,
        z=z,
        velocity=velocity,
        pressure=pressure,
        fluid_mask=fluid_mask
    )

# ✅ NEW: Reflex-safe enforcement metadata propagation
def test_reflex_metadata_propagation_to_fluid_origin():
    ghost = make_cell(0.0, 0.0, 0.0, fluid_mask=False)
    fluid = make_cell(0.5, 0.0, 0.0, fluid_mask=True)
    ghost_registry = {
        id(ghost): {
            "origin": (fluid.x, fluid.y, fluid.z),
            "velocity": [2.0, 0.0, 0.0],
            "pressure": 88.0,
            "type": "dirichlet",
            "enforcement": {
                "velocity": True,
                "pressure": True
            }
        }
    }
    config = {
        "boundary_conditions": [
            {
                "apply_faces": ["x_min"],
                "apply_to": ["velocity", "pressure"],
                "velocity": [2.0, 0.0, 0.0],
                "pressure": 88.0,
                "type": "dirichlet"
            }
        ]
    }

    result = apply_boundary_conditions([ghost, fluid], ghost_registry, config)
    assert result[1].velocity == [2.0, 0.0, 0.0]
    assert result[1].pressure == 88.0

# ✅ Existing tests preserved below

def test_applies_dirichlet_to_ghost_cells():
    ghost = make_cell(0.0, 0.0, 0.0, fluid_mask=False)
    fluid = make_cell(1.0, 0.0, 0.0, fluid_mask=True)
    ghost_registry = {id(ghost): {"origin": (1.0, 0.0, 0.0)}}
    config = {
        "boundary_conditions": [
            {
                "apply_to": ["velocity", "pressure"],
                "velocity": [5.0, 0.0, 0.0],
                "pressure": 42.0
            }
        ]
    }

    result = apply_boundary_conditions([ghost, fluid], ghost_registry, config)
    assert result[0].velocity == [5.0, 0.0, 0.0]
    assert result[0].pressure == 42.0

def test_reflects_dirichlet_into_fluid_origin():
    ghost = make_cell(0.0, 0.0, 0.0, fluid_mask=False)
    fluid = make_cell(1.0, 0.0, 0.0, fluid_mask=True)
    ghost_registry = {id(ghost): {"origin": (1.0, 0.0, 0.0)}}
    config = {
        "boundary_conditions": [
            {
                "apply_to": ["velocity", "pressure"],
                "velocity": [1.0, 2.0, 3.0],
                "pressure": 99.0
            }
        ]
    }

    result = apply_boundary_conditions([ghost, fluid], ghost_registry, config)
    assert result[1].velocity == [1.0, 2.0, 3.0]
    assert result[1].pressure == 99.0

def test_applies_no_slip_to_ghost_and_fluid():
    ghost = make_cell(0.0, 0.0, 0.0, fluid_mask=False)
    fluid = make_cell(1.0, 0.0, 0.0, velocity=[9.0, 9.0, 9.0], fluid_mask=True)
    ghost_registry = {id(ghost): {"origin": (1.0, 0.0, 0.0)}}
    config = {
        "boundary_conditions": [
            {
                "apply_to": ["velocity"],
                "velocity": [9.9, 9.9, 9.9],
                "no_slip": True
            }
        ]
    }

    result = apply_boundary_conditions([ghost, fluid], ghost_registry, config)
    assert result[0].velocity == [0.0, 0.0, 0.0]
    assert result[1].velocity == [0.0, 0.0, 0.0]

def test_skips_malformed_boundary_blocks():
    ghost = make_cell(0.0, 0.0, 0.0, fluid_mask=False)
    fluid = make_cell(1.0, 0.0, 0.0, fluid_mask=True)
    ghost_registry = {id(ghost): {"origin": (1.0, 0.0, 0.0)}}
    config = {
        "boundary_conditions": ["not-a-dict", 123, None]
    }

    result = apply_boundary_conditions([ghost, fluid], ghost_registry, config)
    assert result[0].velocity is None
    assert result[0].pressure is None
    assert result[1].velocity is None
    assert result[1].pressure is None

def test_handles_missing_boundary_conditions_gracefully():
    ghost = make_cell(0.0, 0.0, 0.0, fluid_mask=False)
    fluid = make_cell(1.0, 0.0, 0.0, fluid_mask=True)
    ghost_registry = {id(ghost): {"origin": (1.0, 0.0, 0.0)}}
    config = {
        "boundary_conditions": None  # not a list
    }

    result = apply_boundary_conditions([ghost, fluid], ghost_registry, config)
    assert result[0].velocity is None
    assert result[0].pressure is None
    assert result[1].velocity is None
    assert result[1].pressure is None

def test_skips_non_ghost_cells():
    fluid1 = make_cell(0.0, 0.0, 0.0, fluid_mask=True)
    fluid2 = make_cell(1.0, 0.0, 0.0, fluid_mask=True)
    ghost_registry = {}
    config = {
        "boundary_conditions": [
            {
                "apply_to": ["velocity"],
                "velocity": [1.0, 1.0, 1.0]
            }
        ]
    }

    result = apply_boundary_conditions([fluid1, fluid2], ghost_registry, config)
    assert result[0].velocity is None
    assert result[1].velocity is None



