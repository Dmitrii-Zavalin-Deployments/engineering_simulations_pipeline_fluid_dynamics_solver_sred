# tests/physics/pressure_methods/test_boundary.py
# 🧪 Unit tests for src/physics/pressure_methods/boundary.py

from src.physics.pressure_methods.boundary import apply_neumann_conditions, handle_solid_or_ghost_neighbors

def test_apply_neumann_conditions_returns_coord_pressure():
    coord = (1.0, 2.0, 3.0)
    neighbor = (2.0, 2.0, 3.0)
    pressure_map = {coord: 5.0}
    assert apply_neumann_conditions(coord, neighbor, pressure_map) == 5.0

def test_apply_neumann_conditions_returns_zero_if_coord_missing():
    coord = (0.0, 0.0, 0.0)
    neighbor = (1.0, 1.0, 1.0)
    pressure_map = {}
    assert apply_neumann_conditions(coord, neighbor, pressure_map) == 0.0

def test_handle_neighbors_all_fluid():
    coord = (0, 0, 0)
    neighbors = [(1, 0, 0), (0, 1, 0), (-1, 0, 0)]
    pressure_map = {
        coord: 2.0,
        (1, 0, 0): 3.0,
        (0, 1, 0): 4.0,
        (-1, 0, 0): 5.0
    }
    fluid_mask_map = {n: True for n in neighbors}
    ghost_coords = set()
    ghost_pressure_map = {}
    total = handle_solid_or_ghost_neighbors(coord, neighbors, pressure_map, fluid_mask_map, ghost_coords, ghost_pressure_map)
    assert total == 3.0 + 4.0 + 5.0

def test_handle_neighbors_with_solid_fallback():
    coord = (0, 0, 0)
    neighbors = [(1, 0, 0), (0, 1, 0), (-1, 0, 0)]
    pressure_map = {coord: 2.0, (1, 0, 0): 4.0}
    fluid_mask_map = {(1, 0, 0): True, (0, 1, 0): False, (-1, 0, 0): False}
    ghost_coords = set()
    ghost_pressure_map = {}
    total = handle_solid_or_ghost_neighbors(coord, neighbors, pressure_map, fluid_mask_map, ghost_coords, ghost_pressure_map)
    assert total == 4.0 + 2.0 + 2.0  # Neumann fallback uses coord pressure for solids

def test_handle_neighbors_with_ghost_pressure():
    coord = (0, 0, 0)
    ghost = (1, 1, 1)
    neighbors = [ghost]
    pressure_map = {coord: 7.0}
    fluid_mask_map = {}
    ghost_coords = {ghost}
    ghost_pressure_map = {ghost: 9.0}
    total = handle_solid_or_ghost_neighbors(coord, neighbors, pressure_map, fluid_mask_map, ghost_coords, ghost_pressure_map)
    assert total == 9.0

def test_handle_neighbors_with_ghost_neumann_fallback():
    coord = (0, 0, 0)
    ghost = (2, 2, 2)
    neighbors = [ghost]
    pressure_map = {coord: 8.0}
    fluid_mask_map = {}
    ghost_coords = {ghost}
    ghost_pressure_map = {}  # no Dirichlet value
    total = handle_solid_or_ghost_neighbors(coord, neighbors, pressure_map, fluid_mask_map, ghost_coords, ghost_pressure_map)
    assert total == 8.0

def test_handle_neighbors_with_missing_entries():
    coord = (0, 0, 0)
    neighbors = [(9, 9, 9)]
    pressure_map = {coord: 6.0}
    fluid_mask_map = {}
    ghost_coords = set()
    ghost_pressure_map = {}
    total = handle_solid_or_ghost_neighbors(coord, neighbors, pressure_map, fluid_mask_map, ghost_coords, ghost_pressure_map)
    assert total == 6.0  # Neumann fallback



