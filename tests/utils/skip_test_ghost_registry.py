import pytest
from src.utils import ghost_registry

class MockCell:
    def __init__(
        self, x, y, z,
        fluid_mask=True,
        ghost_face=None,
        boundary_tag=None,
        ghost_type="generic",
        ghost_source_step=None,
        was_enforced=False,
        originated_from_boundary=False,
        velocity=None,
        pressure=None
    ):
        self.x = x
        self.y = y
        self.z = z
        self.fluid_mask = fluid_mask
        self.ghost_face = ghost_face
        self.boundary_tag = boundary_tag
        self.ghost_type = ghost_type
        self.ghost_source_step = ghost_source_step
        self.was_enforced = was_enforced
        self.originated_from_boundary = originated_from_boundary
        self.velocity = velocity or [1.0, 1.0, 1.0]
        self.pressure = pressure

def test_build_ghost_registry_includes_only_nonfluid():
    fluid = MockCell(0.0, 0.0, 0.0, fluid_mask=True)
    ghost = MockCell(1.0, 1.0, 1.0, fluid_mask=False, ghost_face="x+", pressure=1.0)
    grid = [fluid, ghost]
    registry = ghost_registry.build_ghost_registry(grid)
    assert len(registry) == 1
    entry = list(registry.values())[0]
    assert entry["coordinate"] == (1.0, 1.0, 1.0)
    assert entry["ghost_face"] == "x+"
    assert entry["pressure"] == 1.0
    assert entry["ghost_type"] == "generic"

def test_build_ghost_registry_verbose_output(capsys):
    ghost = MockCell(2.0, 2.0, 2.0, fluid_mask=False, ghost_face="y-", ghost_type="wall")
    registry = ghost_registry.build_ghost_registry([ghost], verbose=True)
    output = capsys.readouterr().out
    assert "[REGISTRY] Ghost cell @" in output
    assert "face=y-" in output
    assert "type=wall" in output

def test_extract_ghost_coordinates_returns_all():
    ghost1 = MockCell(0.0, 0.0, 0.0, fluid_mask=False)
    ghost2 = MockCell(1.0, 1.0, 1.0, fluid_mask=False)
    registry = ghost_registry.build_ghost_registry([ghost1, ghost2])
    coords = ghost_registry.extract_ghost_coordinates(registry)
    assert (0.0, 0.0, 0.0) in coords
    assert (1.0, 1.0, 1.0) in coords
    assert len(coords) == 2
