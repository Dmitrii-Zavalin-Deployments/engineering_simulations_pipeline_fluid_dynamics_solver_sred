# tests/physics/pressure_methods/test_boundary.py
# 🧪 Unit tests for Neumann and ghost/solid neighbor pressure logic

import pytest
from src.physics.pressure_methods.boundary import apply_neumann_conditions, handle_solid_or_ghost_neighbors

def make_coord(x, y, z):
    return (float(x), float(y), float(z))

def test_neumann_condition_returns_own_pressure():
    coord = make_coord(1, 1, 1)
    neighbor = make_coord(2, 1, 1)
    pressure_map = {coord: 42.0}
    result = apply_neumann_conditions(coord, neighbor, pressure_map)
    assert result == pytest.approx(42.0)

def test_neumann_condition_handles_missing_pressure_key():
    coord = make_coord(1, 1, 1)
    neighbor = make_coord(9, 9, 9)
    pressure_map = {}  # coord key missing too
    result = apply_neumann_conditions(coord, neighbor, pressure_map)
    assert result == 0.0

def test_handle_all_fluid_neighbors():
    coord = make_coord(0, 0, 0)
    neighbors = [make_coord(1, 0, 0), make_coord(-1, 0, 0),
                 make_coord(0, 1, 0), make_coord(0, -1, 0),
                 make_coord(0, 0, 1), make_coord(0, 0, -1)]
    pressure_map = {n: i * 10.0 for i, n in enumerate(neighbors)}
    fluid_mask_map = {n: True for n in neighbors}
    ghost_coords = set()  # no ghosts

    result = handle_solid_or_ghost_neighbors(coord, neighbors, pressure_map, fluid_mask_map, ghost_coords)
    expected = sum(pressure_map[n] for n in neighbors)
    assert result == pytest.approx(expected)

def test_handle_mixed_fluid_solid_neighbors():
    coord = make_coord(0, 0, 0)
    neighbors = [make_coord(1, 0, 0), make_coord(-1, 0, 0),
                 make_coord(0, 1, 0), make_coord(0, -1, 0),
                 make_coord(0, 0, 1), make_coord(0, 0, -1)]
    pressure_map = {
        coord: 100.0,
        neighbors[0]: 10.0,
        neighbors[2]: 20.0,
        neighbors[4]: 30.0
    }
    fluid_mask_map = {
        neighbors[0]: True,
        neighbors[1]: False,
        neighbors[2]: True,
        neighbors[3]: False,
        neighbors[4]: True,
        neighbors[5]: False
    }
    ghost_coords = set()  # no ghosts

    result = handle_solid_or_ghost_neighbors(coord, neighbors, pressure_map, fluid_mask_map, ghost_coords)
    expected = (
        pressure_map[neighbors[0]] +  # fluid
        pressure_map[coord] +         # solid
        pressure_map[neighbors[2]] +  # fluid
        pressure_map[coord] +         # solid
        pressure_map[neighbors[4]] +  # fluid
        pressure_map[coord]           # solid
    )
    assert result == pytest.approx(expected)

def test_handle_missing_neighbors_gracefully():
    coord = make_coord(0, 0, 0)
    neighbors = [make_coord(9, 9, 9)]  # missing completely
    pressure_map = {coord: 123.0}
    fluid_mask_map = {}  # no entry
    ghost_coords = set()

    result = handle_solid_or_ghost_neighbors(coord, neighbors, pressure_map, fluid_mask_map, ghost_coords)
    assert result == pytest.approx(123.0)

def test_ghost_neighbors_trigger_neumann_fallback():
    coord = make_coord(0, 0, 0)
    ghost = make_coord(1, 0, 0)
    neighbors = [ghost]
    pressure_map = {coord: 88.0}
    fluid_mask_map = {ghost: True}  # will be ignored due to ghost
    ghost_coords = {ghost}

    result = handle_solid_or_ghost_neighbors(coord, neighbors, pressure_map, fluid_mask_map, ghost_coords)
    assert result == pytest.approx(88.0)



