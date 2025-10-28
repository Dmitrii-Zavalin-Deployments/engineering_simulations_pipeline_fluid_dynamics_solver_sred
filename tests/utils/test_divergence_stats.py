import pytest
import math
import os
from src.utils import divergence_tracker
from src.grid_modules.cell import Cell

def make_cell(x, y, z, velocity, fluid_mask=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=0.0, fluid_mask=fluid_mask)

@pytest.fixture
def spacing():
    return (1.0, 1.0, 1.0)

def test_compute_divergence_fluid_cell(spacing):
    grid = [
        make_cell(0.0, 0.0, 0.0, [1.0, 2.0, 3.0]),
        make_cell(1.0, 0.0, 0.0, [2.0, 2.0, 3.0]),
        make_cell(-1.0, 0.0, 0.0, [0.0, 2.0, 3.0]),
        make_cell(0.0, 1.0, 0.0, [1.0, 3.0, 3.0]),
        make_cell(0.0, -1.0, 0.0, [1.0, 1.0, 3.0]),
        make_cell(0.0, 0.0, 1.0, [1.0, 2.0, 4.0]),
        make_cell(0.0, 0.0, -1.0, [1.0, 2.0, 2.0])
    ]
    div = divergence_tracker.compute_divergence(grid[0], grid, spacing)
    assert math.isclose(div, 3.0, rel_tol=1e-6)  # ✅ Updated from 2.0 to 3.0

def test_compute_divergence_solid_cell(spacing):
    grid = [make_cell(0.0, 0.0, 0.0, [1.0, 2.0, 3.0], fluid_mask=False)]
    div = divergence_tracker.compute_divergence(grid[0], grid, spacing)
    assert div == 0.0

def test_compute_max_divergence(spacing):
    grid = [
        make_cell(0.0, 0.0, 0.0, [1.0, 2.0, 3.0]),
        make_cell(1.0, 0.0, 0.0, [2.0, 2.0, 3.0]),
        make_cell(-1.0, 0.0, 0.0, [0.0, 2.0, 3.0])
    ]
    max_div = divergence_tracker.compute_max_divergence(grid, spacing)
    assert max_div > 0.0

def test_compute_divergence_stats_basic(spacing, tmp_path):
    grid = [
        make_cell(0.0, 0.0, 0.0, [1.0, 2.0, 3.0]),
        make_cell(1.0, 0.0, 0.0, [2.0, 2.0, 3.0]),
        make_cell(-1.0, 0.0, 0.0, [0.0, 2.0, 3.0])
    ]
    result = divergence_tracker.compute_divergence_stats(
        grid, spacing, label="test", step_index=1, output_folder=str(tmp_path)
    )
    assert result["max"] > 0.0
    assert result["mean"] > 0.0
    assert result["count"] == 3

    log_path = tmp_path / "divergence_log.txt"
    assert log_path.exists()
    content = log_path.read_text()
    assert "Step 0001" in content
    assert "Stage: test" in content

def test_compute_divergence_stats_empty_grid(spacing, tmp_path):
    grid = [make_cell(0.0, 0.0, 0.0, [1.0, 2.0, 3.0], fluid_mask=False)]
    result = divergence_tracker.compute_divergence_stats(
        grid, spacing, label="empty", step_index=2, output_folder=str(tmp_path)
    )
    assert result == {"max": 0.0, "mean": 0.0, "count": 0}

def test_reference_divergence_export_called(monkeypatch, spacing, tmp_path):
    called = {}

    def mock_export(div_map, step_index, folder):
        called["step"] = step_index
        called["map"] = div_map

    monkeypatch.setattr("src.utils.divergence_tracker.export_divergence_map", mock_export)

    grid = [
        make_cell(0.0, 0.0, 0.0, [1.0, 2.0, 3.0]),
        make_cell(1.0, 0.0, 0.0, [2.0, 2.0, 3.0])
    ]
    ref = {(0.0, 0.0, 0.0): 0.1, (1.0, 0.0, 0.0): 0.2}
    divergence_tracker.compute_divergence_stats(
        grid, spacing, label="reflex", step_index=3, output_folder=str(tmp_path), reference_divergence=ref
    )

    assert called["step"] == 3
    assert all("pre" in v and "post" in v for v in called["map"].values())



