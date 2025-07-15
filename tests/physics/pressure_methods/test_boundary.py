# tests/test_pressure_boundary.py
# 🧪 Unit tests for boundary.py — validates ghost, solid, and fallback pressure logic

import pytest
from src.physics.pressure_methods.boundary import (
    apply_neumann_conditions,
    handle_solid_or_ghost_neighbors
)

def test_apply_neumann_returns_self_pressure():
    pressure_map = {
        (1.0, 1.0, 1.0): 42.0,
        (2.0, 2.0, 2.0): 99.9
    }
    result = apply_neumann_conditions((1.0, 1.0, 1.0), (2.0, 2.0, 2.0), pressure_map)
    assert result == 42.0

def test_apply_neumann_fallback_zero():
    result = apply_neumann_conditions((10.0, 10.0, 10.0), (20.0, 20.0, 20.0), {})
    assert result == 0.0

def test_handle_fluid_neighbors_returns_sum():
    pressure_map = {
        (1.0, 1.0, 1.0): 100.0,
        (2.0, 1.0, 1.0): 80.0,
        (0.0, 1.0, 1.0): 90.0
    }
    fluid_mask = {
        (2.0, 1.0, 1.0): True,
        (0.0, 1.0, 1.0): True
    }
    result = handle_solid_or_ghost_neighbors(
        coord=(1.0, 1.0, 1.0),
        neighbors=[(2.0, 1.0, 1.0), (0.0, 1.0, 1.0)],
        pressure_map=pressure_map,
        fluid_mask_map=fluid_mask,
        ghost_coords=set(),
        ghost_pressure_map={}
    )
    assert result == 170.0

def test_handle_solid_neighbors_applies_neumann():
    pressure_map = {(1.0, 1.0, 1.0): 100.0}
    fluid_mask = {
        (2.0, 1.0, 1.0): False,
        (0.0, 1.0, 1.0): False
    }
    result = handle_solid_or_ghost_neighbors(
        coord=(1.0, 1.0, 1.0),
        neighbors=[(2.0, 1.0, 1.0), (0.0, 1.0, 1.0)],
        pressure_map=pressure_map,
        fluid_mask_map=fluid_mask,
        ghost_coords=set(),
        ghost_pressure_map={}
    )
    assert result == 200.0  # 2 solids fallback to 100.0 each

def test_handle_ghost_neighbor_with_explicit_pressure():
    pressure_map = {(1.0, 1.0, 1.0): 50.0}
    ghost_coords = {(2.0, 1.0, 1.0)}
    ghost_pressure_map = {(2.0, 1.0, 1.0): 75.0}
    fluid_mask = {}
    result = handle_solid_or_ghost_neighbors(
        coord=(1.0, 1.0, 1.0),
        neighbors=[(2.0, 1.0, 1.0)],
        pressure_map=pressure_map,
        fluid_mask_map=fluid_mask,
        ghost_coords=ghost_coords,
        ghost_pressure_map=ghost_pressure_map
    )
    assert result == 75.0

def test_handle_ghost_neighbor_with_neumann_fallback():
    pressure_map = {(1.0, 1.0, 1.0): 50.0}
    ghost_coords = {(2.0, 1.0, 1.0)}
    ghost_pressure_map = {}  # no pressure for ghost
    fluid_mask = {}
    result = handle_solid_or_ghost_neighbors(
        coord=(1.0, 1.0, 1.0),
        neighbors=[(2.0, 1.0, 1.0)],
        pressure_map=pressure_map,
        fluid_mask_map=fluid_mask,
        ghost_coords=ghost_coords,
        ghost_pressure_map=ghost_pressure_map
    )
    assert result == 50.0  # Neumann fallback

def test_handle_unknown_neighbor_applies_neumann():
    pressure_map = {(1.0, 1.0, 1.0): 80.0}
    fluid_mask = {}
    ghost_coords = set()
    ghost_pressure_map = {}
    result = handle_solid_or_ghost_neighbors(
        coord=(1.0, 1.0, 1.0),
        neighbors=[(99.0, 99.0, 99.0)],
        pressure_map=pressure_map,
        fluid_mask_map=fluid_mask,
        ghost_coords=ghost_coords,
        ghost_pressure_map=ghost_pressure_map
    )
    assert result == 80.0  # Neumann fallback

def test_mixed_neighbor_types():
    pressure_map = {(1.0, 1.0, 1.0): 30.0, (0.0, 1.0, 1.0): 20.0}
    fluid_mask = {(0.0, 1.0, 1.0): True}
    ghost_coords = {(2.0, 1.0, 1.0)}
    ghost_pressure_map = {(2.0, 1.0, 1.0): 40.0}
    neighbors = [(0.0, 1.0, 1.0), (2.0, 1.0, 1.0), (99.0, 99.0, 99.0)]
    result = handle_solid_or_ghost_neighbors(
        coord=(1.0, 1.0, 1.0),
        neighbors=neighbors,
        pressure_map=pressure_map,
        fluid_mask_map=fluid_mask,
        ghost_coords=ghost_coords,
        ghost_pressure_map=ghost_pressure_map
    )
    # fluid: 20.0; ghost: 40.0; unknown: fallback to 30.0
    assert result == 90.0