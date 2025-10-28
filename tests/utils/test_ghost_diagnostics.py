import pytest
from src.utils import ghost_diagnostics

class MockCell:
    def __init__(self, x, y, z, fluid_mask=True, **kwargs):
        self.x = x
        self.y = y
        self.z = z
        self.fluid_mask = fluid_mask
        self.influenced_by_ghost = False
        self.mutation_triggered_by = None
        self.__dict__.update(kwargs)  # ✅ Enables dynamic attribute access

def test_analyze_dict_registry_basic():
    ghost_registry = {
        "g1": {"coordinate": (0.0, 0.0, 0.0), "face": "x+", "pressure": 1.0, "velocity": [0.0, 0.0, 0.0]},
        "g2": {"coordinate": (1.0, 0.0, 0.0), "face": "x-", "velocity": [1.0, 1.0, 1.0]}
    }
    result = ghost_diagnostics.analyze_ghost_registry(ghost_registry)
    assert result["total"] == 2
    assert result["per_face"]["x+"] == 1
    assert result["per_face"]["x-"] == 1
    assert result["pressure_overrides"] == 1
    assert result["no_slip_enforced"] == 1

def test_analyze_set_registry_basic():
    ghost_cell_1 = MockCell(0.0, 0.0, 0.0, velocity=[0.0, 0.0, 0.0], pressure=1.0, ghost_face="y+")
    ghost_cell_2 = MockCell(1.0, 0.0, 0.0, ghost_face="y-")
    ghost_registry = {ghost_cell_1, ghost_cell_2}
    result = ghost_diagnostics.analyze_ghost_registry(ghost_registry)
    assert result["total"] == 2
    assert result["per_face"]["y+"] == 1
    assert result["per_face"]["y-"] == 1
    assert result["pressure_overrides"] == 1
    assert result["no_slip_enforced"] == 1

def test_fluid_ghost_adjacency_tags_influence():
    ghost_cell = MockCell(
        1.0, 1.0, 1.0,
        fluid_mask=False,
        ghost_face="z+",
        pressure=1.0,
        velocity=[0.0, 0.0, 0.0],
        was_enforced=True
    )
    ghost_registry = {ghost_cell}
    fluid_cell = MockCell(1.0, 1.0, 0.0, fluid_mask=True)
    grid = [fluid_cell]
    result = ghost_diagnostics.analyze_ghost_registry(ghost_registry, grid, spacing=(1.0, 1.0, 1.0))
    assert result["fluid_cells_adjacent_to_ghosts"] == 1
    assert fluid_cell.influenced_by_ghost is True
    assert fluid_cell.mutation_triggered_by == "ghost_influence"

def test_fluid_cell_not_adjacent_not_tagged():
    ghost_registry = {
        "g1": {
            "coordinate": (5.0, 5.0, 5.0),
            "face": "z+",
            "pressure": 1.0,
            "velocity": [0.0, 0.0, 0.0],
            "was_enforced": True
        }
    }
    fluid_cell = MockCell(0.0, 0.0, 0.0, fluid_mask=True)
    grid = [fluid_cell]
    result = ghost_diagnostics.analyze_ghost_registry(ghost_registry, grid, spacing=(1.0, 1.0, 1.0))
    assert result["fluid_cells_adjacent_to_ghosts"] == 0
    assert fluid_cell.influenced_by_ghost is False

def test_inject_diagnostics_embeds_summary():
    ghost_registry = {
        "g1": {"coordinate": (0.0, 0.0, 0.0), "face": "x+", "pressure": 1.0, "velocity": [0.0, 0.0, 0.0]}
    }
    snapshot = {}
    result = ghost_diagnostics.inject_diagnostics(snapshot, ghost_registry)
    assert "ghost_diagnostics" in result
    assert "ghost_registry" in result
    assert result["ghost_diagnostics"]["total"] == 1

def test_invalid_registry_type_raises():
    with pytest.raises(TypeError):
        ghost_diagnostics.analyze_ghost_registry(["invalid"])



