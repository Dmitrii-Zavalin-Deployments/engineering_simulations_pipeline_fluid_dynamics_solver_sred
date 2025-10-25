import pytest
from src.reflex.spatial_tagging.ghost_face_mapper import tag_ghost_adjacency

class MockCell:
    def __init__(self, x, y, z, fluid_mask=True):
        self.x = x
        self.y = y
        self.z = z
        self.fluid_mask = fluid_mask
        self.ghost_adjacent = False
        self.mutation_triggered_by = None

def test_tags_adjacent_fluid_cells():
    ghost_coords = {(1.0, 1.0, 1.0)}
    spacing = (1.0, 1.0, 1.0)

    fluid = MockCell(2.0, 1.0, 1.0)  # adjacent in x
    grid = [fluid]

    result = tag_ghost_adjacency(grid, ghost_coords, spacing)
    assert result == [(2.0, 1.0)]
    assert fluid.ghost_adjacent is True
    assert fluid.mutation_triggered_by == "ghost_adjacency"

def test_skips_nonfluid_cells():
    ghost_coords = {(1.0, 1.0, 1.0)}
    spacing = (1.0, 1.0, 1.0)

    solid = MockCell(2.0, 1.0, 1.0, fluid_mask=False)
    grid = [solid]

    result = tag_ghost_adjacency(grid, ghost_coords, spacing)
    assert result == []
    assert not getattr(solid, "ghost_adjacent", False)

def test_skips_already_tagged_cells():
    ghost_coords = {(1.0, 1.0, 1.0)}
    spacing = (1.0, 1.0, 1.0)

    fluid = MockCell(2.0, 1.0, 1.0)
    fluid.ghost_adjacent = True  # already tagged
    grid = [fluid]

    result = tag_ghost_adjacency(grid, ghost_coords, spacing)
    assert result == []

def test_tags_multiple_adjacent_cells():
    ghost_coords = {(1.0, 1.0, 1.0)}
    spacing = (1.0, 1.0, 1.0)

    fluid1 = MockCell(2.0, 1.0, 1.0)
    fluid2 = MockCell(1.0, 2.0, 1.0)
    fluid3 = MockCell(1.0, 1.0, 2.0)
    grid = [fluid1, fluid2, fluid3]

    result = tag_ghost_adjacency(grid, ghost_coords, spacing)
    assert set(result) == {(2.0, 1.0), (1.0, 2.0), (1.0, 1.0)}
    for cell in grid:
        assert cell.ghost_adjacent is True
        assert cell.mutation_triggered_by == "ghost_adjacency"

def test_returns_empty_if_no_adjacent_cells():
    ghost_coords = {(10.0, 10.0, 10.0)}
    spacing = (1.0, 1.0, 1.0)

    fluid = MockCell(0.0, 0.0, 0.0)
    grid = [fluid]

    result = tag_ghost_adjacency(grid, ghost_coords, spacing)
    assert result == []
    assert fluid.ghost_adjacent is False
    assert fluid.mutation_triggered_by is None



