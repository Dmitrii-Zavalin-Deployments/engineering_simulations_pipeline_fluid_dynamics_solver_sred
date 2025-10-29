# tests/metrics/test_cfl_controller.py
# ✅ Validation suite for src/metrics/cfl_controller.py

import pytest
from src.metrics.cfl_controller import compute_global_cfl
from src.grid_modules.cell import Cell

def mock_cell(x, y, z, velocity, fluid_mask=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=0.0, fluid_mask=fluid_mask)

def test_cfl_computation_basic_case():
    grid = [
        mock_cell(0, 0, 0, [1.0, 0.0, 0.0]),
        mock_cell(1, 0, 0, [0.0, 2.0, 0.0]),
        mock_cell(2, 0, 0, [0.0, 0.0, 3.0])
    ]
    domain = {"min_x": 0.0, "max_x": 3.0, "nx": 3}
    time_step = 0.1

    result = compute_global_cfl(grid, time_step, domain)
    expected_max = round((3.0 * 0.1) / 1.0, 5)
    assert result == expected_max

def test_cfl_excludes_solid_cells():
    grid = [
        mock_cell(0, 0, 0, [10.0, 0.0, 0.0], fluid_mask=False),  # solid → excluded
        mock_cell(1, 0, 0, [1.0, 0.0, 0.0], fluid_mask=True)
    ]
    domain = {"min_x": 0.0, "max_x": 2.0, "nx": 2}
    time_step = 0.1

    result = compute_global_cfl(grid, time_step, domain)
    expected = round((1.0 * 0.1) / 1.0, 5)
    assert result == expected

def test_cfl_flags_exceeding_cells():
    grid = [
        mock_cell(0, 0, 0, [10.0, 0.0, 0.0]),  # exceeds threshold
        mock_cell(1, 0, 0, [0.5, 0.0, 0.0])    # below threshold
    ]
    domain = {"min_x": 0.0, "max_x": 2.0, "nx": 2}
    time_step = 0.1
    threshold = 0.5

    result = compute_global_cfl(grid, time_step, domain, cfl_threshold=threshold)
    assert result > threshold
    assert grid[0].cfl_exceeded is True
    assert grid[0].mutation_source == "cfl_violation"
    assert not hasattr(grid[1], "cfl_exceeded")

def test_cfl_handles_missing_velocity():
    grid = [
        mock_cell(0, 0, 0, None),
        mock_cell(1, 0, 0, [1.0, 1.0])  # malformed
    ]
    domain = {"min_x": 0.0, "max_x": 2.0, "nx": 2}
    time_step = 0.1

    result = compute_global_cfl(grid, time_step, domain)
    assert result == 0.0

def test_cfl_handles_empty_grid():
    result = compute_global_cfl([], 0.1, {"min_x": 0, "max_x": 1, "nx": 1})
    assert result == 0.0

def test_cfl_handles_missing_domain_keys():
    grid = [mock_cell(0, 0, 0, [1.0, 0.0, 0.0])]
    incomplete_domains = [
        {}, {"min_x": 0.0}, {"max_x": 1.0}, {"nx": 10}
    ]
    for domain in incomplete_domains:
        assert compute_global_cfl(grid, 0.1, domain) == 0.0

def test_cfl_rounding_precision():
    grid = [mock_cell(0, 0, 0, [1.0, 0.0, 0.0])]
    domain = {"min_x": 0.0, "max_x": 1.0, "nx": 1}
    time_step = 0.123456789

    result = compute_global_cfl(grid, time_step, domain)
    assert isinstance(result, float)
    assert len(str(result).split(".")[1]) <= 5



