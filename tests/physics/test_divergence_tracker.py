import os
import json
import shutil
import tempfile
import pytest
from src.physics.divergence_tracker import compute_divergence_stats
from src.grid_modules.cell import Cell

def create_test_cell(x, y, z, velocity, fluid_mask=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=None, fluid_mask=fluid_mask)

@pytest.fixture
def temp_output_dir():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)

def test_divergence_stats_basic(temp_output_dir):
    grid = [
        create_test_cell(0.0, 0.0, 0.0, [1.0, 0.0, 0.0]),
        create_test_cell(1.0, 0.0, 0.0, [-1.0, 0.0, 0.0]),
        create_test_cell(0.0, 1.0, 0.0, [0.0, 1.0, 0.0]),
        create_test_cell(0.0, 0.0, 1.0, [0.0, 0.0, -1.0]),
    ]
    spacing = (1.0, 1.0, 1.0)
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0, "nx": 1,
            "min_y": 0.0, "max_y": 1.0, "ny": 1,
            "min_z": 0.0, "max_z": 1.0, "nz": 1
        }
    }

    stats = compute_divergence_stats(
        grid=grid,
        spacing=spacing,
        label="test",
        step_index=0,
        output_folder=temp_output_dir,
        config=config,
        ghost_registry=set()
    )

    assert "max" in stats and "mean" in stats and "divergence" in stats
    assert isinstance(stats["divergence"], list)
    assert os.path.exists(os.path.join(temp_output_dir, "divergence_log.txt"))
    assert os.path.exists(os.path.join(temp_output_dir, "divergence_map_step_0000.json"))

def test_ghost_cells_are_excluded(temp_output_dir):
    cell1 = create_test_cell(0.0, 0.0, 0.0, [1.0, 0.0, 0.0])
    cell2 = create_test_cell(1.0, 0.0, 0.0, [-1.0, 0.0, 0.0])
    ghost_registry = {id(cell2)}
    grid = [cell1, cell2]
    spacing = (1.0, 1.0, 1.0)
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0, "nx": 1,
            "min_y": 0.0, "max_y": 1.0, "ny": 1,
            "min_z": 0.0, "max_z": 1.0, "nz": 1
        }
    }

    stats = compute_divergence_stats(
        grid=grid,
        spacing=spacing,
        label="ghost-test",
        step_index=1,
        output_folder=temp_output_dir,
        config=config,
        ghost_registry=ghost_registry
    )

    assert len(stats["divergence"]) == 1
    assert id(cell2) in ghost_registry
    assert not hasattr(cell2, "divergence")

def test_handles_empty_grid(temp_output_dir):
    grid = []
    spacing = (1.0, 1.0, 1.0)
    config = {"domain_definition": {}}

    stats = compute_divergence_stats(
        grid=grid,
        spacing=spacing,
        label="empty",
        step_index=2,
        output_folder=temp_output_dir,
        config=config,
        ghost_registry=set()
    )

    assert stats["max"] == 0.0
    assert stats["mean"] == 0.0
    assert stats["divergence"] == []

def test_divergence_map_file_format(temp_output_dir):
    grid = [
        create_test_cell(0.0, 0.0, 0.0, [0.0, 0.0, 0.0]),
        create_test_cell(1.0, 0.0, 0.0, [0.0, 0.0, 0.0])
    ]
    spacing = (1.0, 1.0, 1.0)
    config = {"domain_definition": {}}

    compute_divergence_stats(
        grid=grid,
        spacing=spacing,
        label="format-check",
        step_index=3,
        output_folder=temp_output_dir,
        config=config,
        ghost_registry=set()
    )

    map_path = os.path.join(temp_output_dir, "divergence_map_step_0003.json")
    with open(map_path, "r") as f:
        data = json.load(f)
        assert isinstance(data, dict)
        for key, value in data.items():
            assert isinstance(key, str)
            assert isinstance(value, float)



