# tests/physics/pressure_methods/test_boundary.py
# ✅ Validation suite for src/physics/pressure_methods/boundary.py

import pytest
from src.physics.pressure_methods.boundary import apply_neumann_conditions, handle_solid_or_ghost_neighbors

# 🔧 apply_neumann_conditions
def test_apply_neumann_conditions_returns_self_pressure():
    coord = (1.0, 1.0, 1.0)
    neighbor = (2.0, 1.0, 1.0)
    pressure_map = {coord: 5.0}
    result = apply_neumann_conditions(coord, neighbor, pressure_map)
    assert result == 5.0

def test_apply_neumann_conditions_defaults_to_zero_if_missing():
    coord = (0.0, 0.0, 0.0)
    neighbor = (1.0, 0.0, 0.0)
    pressure_map = {}
    result = apply_neumann_conditions(coord, neighbor, pressure_map)
    assert result == 0.0

# 🔧 handle_solid_or_ghost_neighbors
def test_handle_neighbors_with_all_fluid_cells():
    coord = (0.0, 0.0, 0.0)
    neighbors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    pressure_map = {
        coord: 10.0,
        (1.0, 0.0, 0.0): 20.0,
        (0.0, 1.0, 0.0): 30.0
    }
    fluid_mask_map = {
        (1.0, 0.0, 0.0): True,
        (0.0, 1.0, 0.0): True
    }
    ghost_coords = set()
    ghost_pressure_map = {}
    diagnostics = {}

    result = handle_solid_or_ghost_neighbors(coord, neighbors, pressure_map, fluid_mask_map, ghost_coords, ghost_pressure_map, diagnostics)
    assert result == 50.0
    assert diagnostics["fluid_neighbor"] == 2

def test_handle_neighbors_with_solid_cells():
    coord = (0.0, 0.0, 0.0)
    neighbors = [(1.0, 0.0, 0.0)]
    pressure_map = {coord: 10.0}
    fluid_mask_map = {(1.0, 0.0, 0.0): False}
    ghost_coords = set()
    ghost_pressure_map = {}
    diagnostics = {}

    result = handle_solid_or_ghost_neighbors(coord, neighbors, pressure_map, fluid_mask_map, ghost_coords, ghost_pressure_map, diagnostics)
    assert result == 10.0
    assert diagnostics["solid_neumann"] == 1

def test_handle_neighbors_with_missing_cells():
    coord = (0.0, 0.0, 0.0)
    neighbors = [(2.0, 2.0, 2.0)]
    pressure_map = {coord: 7.0}
    fluid_mask_map = {}
    ghost_coords = set()
    ghost_pressure_map = {}
    diagnostics = {}

    result = handle_solid_or_ghost_neighbors(coord, neighbors, pressure_map, fluid_mask_map, ghost_coords, ghost_pressure_map, diagnostics)
    assert result == 7.0
    assert diagnostics["missing_neumann"] == 1

def test_handle_neighbors_with_ghost_dirichlet():
    coord = (0.0, 0.0, 0.0)
    ghost = (1.0, 0.0, 0.0)
    neighbors = [ghost]
    pressure_map = {coord: 5.0}
    fluid_mask_map = {}
    ghost_coords = {ghost}
    ghost_pressure_map = {ghost: 42.0}
    diagnostics = {}

    result = handle_solid_or_ghost_neighbors(coord, neighbors, pressure_map, fluid_mask_map, ghost_coords, ghost_pressure_map, diagnostics)
    assert result == 42.0
    assert diagnostics["ghost_dirichlet"] == 1

def test_handle_neighbors_with_ghost_neumann_fallback():
    coord = (0.0, 0.0, 0.0)
    ghost = (1.0, 0.0, 0.0)
    neighbors = [ghost]
    pressure_map = {coord: 8.0}
    fluid_mask_map = {}
    ghost_coords = {ghost}
    ghost_pressure_map = {}  # no explicit pressure
    diagnostics = {}

    result = handle_solid_or_ghost_neighbors(coord, neighbors, pressure_map, fluid_mask_map, ghost_coords, ghost_pressure_map, diagnostics)
    assert result == 8.0
    assert diagnostics["ghost_neumann"] == 1

def test_handle_neighbors_mixed_types():
    coord = (0.0, 0.0, 0.0)
    neighbors = [
        (1.0, 0.0, 0.0),  # fluid
        (2.0, 0.0, 0.0),  # solid
        (3.0, 0.0, 0.0),  # ghost with pressure
        (4.0, 0.0, 0.0),  # ghost without pressure
        (5.0, 0.0, 0.0),  # missing
    ]
    pressure_map = {
        coord: 1.0,
        (1.0, 0.0, 0.0): 2.0
    }
    fluid_mask_map = {
        (1.0, 0.0, 0.0): True,
        (2.0, 0.0, 0.0): False
    }
    ghost_coords = {(3.0, 0.0, 0.0), (4.0, 0.0, 0.0)}
    ghost_pressure_map = {(3.0, 0.0, 0.0): 9.0}
    diagnostics = {}

    result = handle_solid_or_ghost_neighbors(coord, neighbors, pressure_map, fluid_mask_map, ghost_coords, ghost_pressure_map, diagnostics)
    expected = 2.0 + 1.0 + 9.0 + 1.0 + 1.0  # fluid + solid + ghost_dirichlet + ghost_neumann + missing
    assert result == expected
    assert diagnostics == {
        "fluid_neighbor": 1,
        "solid_neumann": 1,
        "ghost_dirichlet": 1,
        "ghost_neumann": 1,
        "missing_neumann": 1
    }



