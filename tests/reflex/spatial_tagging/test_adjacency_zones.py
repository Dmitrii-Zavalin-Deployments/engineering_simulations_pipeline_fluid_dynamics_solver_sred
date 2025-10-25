import pytest
from src.reflex.spatial_tagging.adjacency_zones import extract_ghost_coordinates

class MockCell:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

def test_extract_from_registry_dict():
    registry = {
        101: {"coordinate": (1.0, 2.0, 3.0)},
        102: {"coordinate": (4.0, 5.0, 6.0)},
        103: {"coordinate": (7.0, 8.0, 9.0)}
    }
    coords = extract_ghost_coordinates(registry)
    expected = {(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)}
    assert coords == expected

def test_extract_from_registry_dict_with_missing_or_invalid_coords():
    registry = {
        201: {"coordinate": (1.0, 2.0, 3.0)},
        202: {"coordinate": None},
        203: {"coordinate": "invalid"},
        204: {}
    }
    coords = extract_ghost_coordinates(registry)
    assert coords == {(1.0, 2.0, 3.0)}

def test_extract_from_cell_set():
    cell1 = MockCell(1.0, 2.0, 3.0)
    cell2 = MockCell(4.0, 5.0, 6.0)
    cell3 = MockCell(7.0, 8.0, 9.0)
    ghost_set = {cell1, cell2, cell3}
    coords = extract_ghost_coordinates(ghost_set)
    expected = {(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)}
    assert coords == expected

def test_extract_from_empty_registry_dict():
    registry = {}
    coords = extract_ghost_coordinates(registry)
    assert coords == set()

def test_extract_from_empty_cell_set():
    ghost_set = set()
    coords = extract_ghost_coordinates(ghost_set)
    assert coords == set()

def test_extract_from_mixed_type_input():
    # Should handle gracefully and return empty set
    coords = extract_ghost_coordinates("invalid_type")
    assert coords == set()



