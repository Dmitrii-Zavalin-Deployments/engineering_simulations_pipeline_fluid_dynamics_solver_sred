import pytest
from src.reflex.spatial_tagging.suppression_zones import detect_suppression_zones, extract_mutated_coordinates

class MockCell:
    def __init__(self, x, y, z, fluid_mask=True, influenced_by_ghost=False):
        self.x = x
        self.y = y
        self.z = z
        self.fluid_mask = fluid_mask
        self.influenced_by_ghost = influenced_by_ghost

def test_detects_suppressed_fluid_cells():
    ghost_coords = {(1.0, 1.0, 1.0)}
    mutated_coords = set()
    spacing = (1.0, 1.0, 1.0)

    fluid = MockCell(2.0, 1.0, 1.0)  # adjacent to ghost
    grid = [fluid]

    result = detect_suppression_zones(grid, ghost_coords, mutated_coords, spacing)
    assert result == [(2.0, 1.0)]

def test_skips_mutated_cells():
    ghost_coords = {(1.0, 1.0, 1.0)}
    mutated_coords = {(2.0, 1.0, 1.0)}
    spacing = (1.0, 1.0, 1.0)

    fluid = MockCell(2.0, 1.0, 1.0)
    grid = [fluid]

    result = detect_suppression_zones(grid, ghost_coords, mutated_coords, spacing)
    assert result == []

def test_skips_influenced_cells():
    ghost_coords = {(1.0, 1.0, 1.0)}
    mutated_coords = set()
    spacing = (1.0, 1.0, 1.0)

    fluid = MockCell(2.0, 1.0, 1.0, influenced_by_ghost=True)
    grid = [fluid]

    result = detect_suppression_zones(grid, ghost_coords, mutated_coords, spacing)
    assert result == []

def test_skips_nonfluid_cells():
    ghost_coords = {(1.0, 1.0, 1.0)}
    mutated_coords = set()
    spacing = (1.0, 1.0, 1.0)

    solid = MockCell(2.0, 1.0, 1.0, fluid_mask=False)
    grid = [solid]

    result = detect_suppression_zones(grid, ghost_coords, mutated_coords, spacing)
    assert result == []

def test_returns_empty_if_no_adjacent_cells():
    ghost_coords = {(10.0, 10.0, 10.0)}
    mutated_coords = set()
    spacing = (1.0, 1.0, 1.0)

    fluid = MockCell(0.0, 0.0, 0.0)
    grid = [fluid]

    result = detect_suppression_zones(grid, ghost_coords, mutated_coords, spacing)
    assert result == []

def test_extracts_mutated_coordinates_from_cells():
    class Cell:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

    cells = [Cell(1.0, 2.0, 3.0), Cell(4.0, 5.0, 6.0)]
    result = extract_mutated_coordinates(cells)
    assert result == {(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)}

def test_extracts_mutated_coordinates_from_tuples():
    tuples = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
    result = extract_mutated_coordinates(tuples)
    assert result == {(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)}

def test_extracts_mutated_coordinates_from_mixed_input():
    class Cell:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

    mixed = [Cell(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
    result = extract_mutated_coordinates(mixed)
    assert result == {(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)}



