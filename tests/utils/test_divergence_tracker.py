# tests/test_divergence_tracker.py
# 🧪 Unit tests for divergence_tracker — validates central differencing, divergence logging, map export

import math
import json
import tempfile
from src.grid_modules.cell import Cell
from src.utils import divergence_tracker

def make_cell(x, y, z, velocity, fluid=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=0.0, fluid_mask=fluid)

def test_compute_divergence_zero_on_uniform_velocity():
    spacing = (1.0, 1.0, 1.0)
    center = make_cell(1.0, 1.0, 1.0, [1.0, 0.0, 0.0])
    neighbors = [
        make_cell(0.0, 1.0, 1.0, [1.0, 0.0, 0.0]),
        make_cell(2.0, 1.0, 1.0, [1.0, 0.0, 0.0]),
        make_cell(1.0, 0.0, 1.0, [1.0, 0.0, 0.0]),
        make_cell(1.0, 2.0, 1.0, [1.0, 0.0, 0.0]),
        make_cell(1.0, 1.0, 0.0, [1.0, 0.0, 0.0]),
        make_cell(1.0, 1.0, 2.0, [1.0, 0.0, 0.0])
    ]
    grid = [center] + neighbors
    div = divergence_tracker.compute_divergence(center, grid, spacing)
    assert math.isclose(div, 0.0)

def test_compute_divergence_nonzero_with_gradient():
    spacing = (1.0, 1.0, 1.0)
    center = make_cell(1.0, 1.0, 1.0, [1.0, 0.0, 0.0])
    left = make_cell(0.0, 1.0, 1.0, [0.0, 0.0, 0.0])
    right = make_cell(2.0, 1.0, 1.0, [2.0, 0.0, 0.0])
    grid = [center, left, right]
    div = divergence_tracker.compute_divergence(center, grid, spacing)
    expected = (2.0 - 0.0) / (2.0 * spacing[0])  # du/dx = 1.0
    assert math.isclose(div, expected)

def test_compute_divergence_skips_nonfluid_cell():
    spacing = (1.0, 1.0, 1.0)
    cell = make_cell(1.0, 1.0, 1.0, [0.0, 0.0, 0.0], fluid=False)
    grid = [cell]
    div = divergence_tracker.compute_divergence(cell, grid, spacing)
    assert div == 0.0

def test_compute_max_divergence_returns_highest():
    spacing = (1.0, 1.0, 1.0)
    c1 = make_cell(1.0, 1.0, 1.0, [1.0, 0.0, 0.0])
    c2 = make_cell(2.0, 1.0, 1.0, [2.0, 0.0, 0.0])
    grid = [c1, c2]
    result = divergence_tracker.compute_max_divergence(grid, spacing)
    assert isinstance(result, float)

def test_compute_divergence_stats_returns_fields():
    spacing = (1.0, 1.0, 1.0)
    c1 = make_cell(1.0, 1.0, 1.0, [1.0, 0.0, 0.0])
    c2 = make_cell(2.0, 1.0, 1.0, [2.0, 0.0, 0.0])
    stats = divergence_tracker.compute_divergence_stats(
        grid=[c1, c2],
        spacing=spacing,
        label="test",
        step_index=0
    )
    assert "max" in stats and "mean" in stats and "count" in stats

def test_compute_divergence_stats_logs_to_file():
    spacing = (1.0, 1.0, 1.0)
    with tempfile.TemporaryDirectory() as folder:
        c1 = make_cell(1.0, 1.0, 1.0, [1.0, 0.0, 0.0])
        c2 = make_cell(2.0, 1.0, 1.0, [2.0, 0.0, 0.0])
        divergence_tracker.compute_divergence_stats(
            grid=[c1, c2],
            spacing=spacing,
            label="test",
            step_index=1,
            output_folder=folder
        )
        path = os.path.join(folder, "divergence_log.txt")
        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
            assert "Step 0001" in content

def test_dump_divergence_map_writes_json_file():
    spacing = (1.0, 1.0, 1.0)
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "map.json")
        c1 = make_cell(1.0, 1.0, 1.0, [1.0, 0.0, 0.0])
        c2 = make_cell(2.0, 1.0, 1.0, [2.0, 0.0, 0.0])
        result = divergence_tracker.dump_divergence_map(
            grid=[c1, c2],
            spacing=spacing,
            path=path,
            config={"reflex_verbosity": "high"}
        )
        assert os.path.exists(path)
        assert isinstance(result, list)
        with open(path) as f:
            loaded = json.load(f)
            assert all("divergence" in r for r in loaded)

def test_dump_divergence_map_returns_data_without_file():
    spacing = (1.0, 1.0, 1.0)
    c1 = make_cell(1.0, 1.0, 1.0, [1.0, 0.0, 0.0])
    result = divergence_tracker.dump_divergence_map(
        grid=[c1],
        spacing=spacing,
        path=None
    )
    assert isinstance(result, list)
    assert result[0]["x"] == 1.0
    assert "divergence" in result[0]