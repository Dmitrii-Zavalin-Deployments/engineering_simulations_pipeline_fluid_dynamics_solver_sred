import pytest
import types
from src.utils.snapshot_step_processor import process_snapshot_step

class MockCell:
    def __init__(self, x, y, z, fluid_mask=True, velocity=None, pressure=None):
        self.x = x
        self.y = y
        self.z = z
        self.fluid_mask = fluid_mask
        self.velocity = velocity or [0.0, 0.0, 0.0]
        self.pressure = pressure or 0.0

@pytest.fixture
def mock_grid():
    return [
        MockCell(0, 0, 0, fluid_mask=True),
        MockCell(1, 0, 0, fluid_mask=True),
        MockCell(0, 1, 0, fluid_mask=False)
    ]

@pytest.fixture
def mock_reflex():
    return {
        "reflex_score": 0.85,
        "ghost_influence_count": 2,
        "boundary_condition_applied": True,
        "pressure_mutated": True,
        "velocity_projected": True,
        "mutated_cells": [MockCell(0, 0, 0)],
        "ghost_trigger_chain": [(0, 0, 0)],
        "adaptive_timestep": 0.005,
        "max_divergence": 0.01
    }

@pytest.fixture
def mock_config():
    return {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 2, "ny": 2, "nz": 1
        },
        "geometry_mask_flat": [1, 1, 0, 1],
        "geometry_mask_shape": [2, 2, 1],
        "mask_encoding": {"fluid": 1, "solid": 0},
        "flattening_order": "x-major"
    }

@pytest.fixture
def spacing():
    return (1.0, 1.0, 1.0)

def test_process_snapshot_step_returns_grid_and_snapshot(tmp_path, mock_grid, mock_reflex, spacing, mock_config):
    output_folder = str(tmp_path)
    grid, snapshot = process_snapshot_step(
        step=1,
        grid=mock_grid,
        reflex=mock_reflex,
        spacing=spacing,
        config=mock_config,
        expected_size=3,
        output_folder=output_folder
    )
    assert isinstance(grid, list)
    assert isinstance(snapshot, dict)
    assert snapshot["step_index"] == 1
    assert "grid" in snapshot
    assert "reflex_score" in snapshot
    assert snapshot["reflex_score"] == 0.85
    assert snapshot["pressure_mutated"] is True
    assert snapshot["velocity_projected"] is True

def test_snapshot_serialization_respects_fluid_mask(tmp_path, mock_grid, mock_reflex, spacing, mock_config):
    grid, snapshot = process_snapshot_step(
        step=2,
        grid=mock_grid,
        reflex=mock_reflex,
        spacing=spacing,
        config=mock_config,
        expected_size=3,
        output_folder=str(tmp_path)
    )
    for cell in snapshot["grid"]:
        if cell["fluid_mask"]:
            assert cell["velocity"] is not None
            assert cell["pressure"] is not None
        else:
            assert cell["velocity"] is None
            assert cell["pressure"] is None

def test_missing_fields_are_handled_gracefully(tmp_path, spacing, mock_config):
    reflex = {}  # minimal reflex dict
    grid = [MockCell(0, 0, 0, fluid_mask=True)]
    grid_out, snapshot = process_snapshot_step(
        step=3,
        grid=grid,
        reflex=reflex,
        spacing=spacing,
        config=mock_config,
        expected_size=1,
        output_folder=str(tmp_path)
    )
    assert snapshot["reflex_score"] == 0.0
    assert snapshot["pressure_mutated"] in [False, True]  # fallback logic
    assert isinstance(snapshot["grid"], list)

def test_unexpected_fluid_cell_count_warns(tmp_path, mock_grid, mock_reflex, spacing, mock_config, capsys):
    process_snapshot_step(
        step=4,
        grid=mock_grid,
        reflex=mock_reflex,
        spacing=spacing,
        config=mock_config,
        expected_size=99,  # intentionally wrong
        output_folder=str(tmp_path)
    )
    output = capsys.readouterr().out
    assert "Unexpected fluid cell count" in output

def test_mutation_pathway_logging_runs(tmp_path, mock_grid, mock_reflex, spacing, mock_config, capsys):
    process_snapshot_step(
        step=5,
        grid=mock_grid,
        reflex=mock_reflex,
        spacing=spacing,
        config=mock_config,
        expected_size=3,
        output_folder=str(tmp_path)
    )
    output = capsys.readouterr().out
    assert "mutated_cells" in output



