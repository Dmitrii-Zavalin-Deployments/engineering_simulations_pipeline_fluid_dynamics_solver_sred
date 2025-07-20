# ✅ Unit Test Suite — Divergence Tracker
# 📄 Full Path: tests/utils/test_divergence_tracker.py

import pytest
import os
import json
from tempfile import TemporaryDirectory
from src.utils import divergence_tracker as dt

class DummyCell:
    def __init__(self, x, y, z, velocity, fluid=True):
        self.x = x
        self.y = y
        self.z = z
        self.velocity = velocity
        self.fluid_mask = fluid

def build_grid():
    return [
        DummyCell(0.0, 0.0, 0.0, [1.0, 0.0, 0.0]),
        DummyCell(1.0, 0.0, 0.0, [3.0, 0.0, 0.0]),
        DummyCell(-1.0, 0.0, 0.0, [-1.0, 0.0, 0.0]),
        DummyCell(0.0, 1.0, 0.0, [1.0, 2.0, 0.0]),
        DummyCell(0.0, -1.0, 0.0, [1.0, -2.0, 0.0]),
        DummyCell(0.0, 0.0, 1.0, [1.0, 0.0, 2.0]),
        DummyCell(0.0, 0.0, -1.0, [1.0, 0.0, -2.0])
    ]

def test_compute_divergence_returns_float():
    grid = build_grid()
    spacing = (1.0, 1.0, 1.0)
    div = dt.compute_divergence(grid[0], grid, spacing)
    assert isinstance(div, float)

def test_compute_max_divergence_valid_range():
    grid = build_grid()
    spacing = (1.0, 1.0, 1.0)
    max_val = dt.compute_max_divergence(grid, spacing)
    assert max_val > 0.0
    assert isinstance(max_val, float)

def test_compute_divergence_stats_returns_keys():
    grid = build_grid()
    spacing = (1.0, 1.0, 1.0)
    stats = dt.compute_divergence_stats(grid, spacing)
    assert "max" in stats
    assert "mean" in stats
    assert "count" in stats

def test_compute_divergence_stats_saves_to_file(capsys):
    grid = build_grid()
    spacing = (1.0, 1.0, 1.0)
    with TemporaryDirectory() as tmp:
        stats = dt.compute_divergence_stats(
            grid,
            spacing,
            label="before projection",
            step_index=3,
            output_folder=tmp,
            config={"reflex_verbosity": "medium"},
            reference_divergence={
                (0.0, 0.0, 0.0): 0.1,
                (1.0, 0.0, 0.0): 0.0
            }
        )
        log_path = os.path.join(tmp, "divergence_log.txt")
        assert os.path.exists(log_path)
        with open(log_path) as f:
            contents = f.read()
        assert "Step 0003" in contents
        assert "before projection" in contents
        captured = capsys.readouterr()
        assert "Divergence stats" in captured.out

def test_compute_divergence_stats_handles_empty_grid(capsys):
    empty_grid = []
    spacing = (1.0, 1.0, 1.0)
    stats = dt.compute_divergence_stats(empty_grid, spacing, config={"reflex_verbosity": "medium"})
    assert stats["max"] == 0.0
    assert stats["mean"] == 0.0
    assert stats["count"] == 0
    captured = capsys.readouterr()
    assert "No fluid cells found" in captured.out

def test_dump_divergence_map_writes_json(capsys):
    grid = build_grid()
    spacing = (1.0, 1.0, 1.0)
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "map.json")
        result = dt.dump_divergence_map(grid, spacing, path, config={"reflex_verbosity": "medium"})
        assert os.path.exists(path)
        assert isinstance(result, list)
        assert result[0]["divergence"] != 0.0
        captured = capsys.readouterr()
        assert "Divergence map written" in captured.out



