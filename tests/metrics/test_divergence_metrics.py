# tests/metrics/test_divergence_metrics.py
# ✅ Unit tests for compute_max_divergence — central difference in 3D

import pytest
from src.grid_modules.cell import Cell
from src.metrics.divergence_metrics import compute_max_divergence

def make_fluid_cell(x, y, z, velocity):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=1.0, fluid_mask=True)

def make_solid_cell(x, y, z):
    return Cell(x=x, y=y, z=z, velocity=None, pressure=None, fluid_mask=False)

def domain_config(nx=3, ny=3, nz=3):
    return {
        "min_x": 0.0, "max_x": 3.0, "nx": nx,
        "min_y": 0.0, "max_y": 3.0, "ny": ny,
        "min_z": 0.0, "max_z": 3.0, "nz": nz,
    }

def test_uniform_velocity_gives_zero_divergence():
    cells = [
        make_fluid_cell(1.0, 1.0, 1.0, [1.0, 1.0, 1.0]),
        make_fluid_cell(2.0, 1.0, 1.0, [1.0, 1.0, 1.0]),
        make_fluid_cell(0.0, 1.0, 1.0, [1.0, 1.0, 1.0]),
        make_fluid_cell(1.0, 2.0, 1.0, [1.0, 1.0, 1.0]),
        make_fluid_cell(1.0, 0.0, 1.0, [1.0, 1.0, 1.0]),
        make_fluid_cell(1.0, 1.0, 2.0, [1.0, 1.0, 1.0]),
        make_fluid_cell(1.0, 1.0, 0.0, [1.0, 1.0, 1.0]),
    ]
    divergence = compute_max_divergence(cells, domain_config())
    assert divergence == 0.0

def test_high_x_velocity_gradient_detected():
    cells = [
        make_fluid_cell(0.0, 1.0, 1.0, [0.0, 0.0, 0.0]),
        make_fluid_cell(1.0, 1.0, 1.0, [1.0, 0.0, 0.0]),
        make_fluid_cell(2.0, 1.0, 1.0, [2.0, 0.0, 0.0]),
    ]
    divergence = compute_max_divergence(cells, domain_config())
    assert divergence > 0.0
    assert round(divergence, 5) == round((2.0 - 0.0) / (2.0 * 1.0), 5)

def test_divergence_ignores_solid_and_ghost_cells():
    cells = [
        make_fluid_cell(1.0, 1.0, 1.0, [3.0, 0.0, 0.0]),
        make_solid_cell(2.0, 1.0, 1.0),
        make_solid_cell(0.0, 1.0, 1.0),
    ]
    divergence = compute_max_divergence(cells, domain_config())
    assert divergence == 0.0

def test_partial_neighbors_produce_partial_contribution():
    cells = [
        make_fluid_cell(1.0, 1.0, 1.0, [5.0, 0.0, 0.0]),
        make_fluid_cell(0.0, 1.0, 1.0, [1.0, 0.0, 0.0]),
        # Missing +x neighbor at (2.0, 1.0, 1.0)
    ]
    divergence = compute_max_divergence(cells, domain_config())
    # Only -x side contributes; total central difference falls back to 0
    assert divergence == 0.0

def test_multiple_cells_compute_max_divergence():
    cells = [
        make_fluid_cell(1.0, 1.0, 1.0, [5.0, 0.0, 0.0]),
        make_fluid_cell(2.0, 1.0, 1.0, [0.0, 0.0, 0.0]),
        make_fluid_cell(0.0, 1.0, 1.0, [0.0, 0.0, 0.0]),
        make_fluid_cell(1.0, 2.0, 1.0, [0.0, 2.0, 0.0]),
        make_fluid_cell(1.0, 0.0, 1.0, [0.0, 0.0, 0.0]),
        make_fluid_cell(1.0, 1.0, 2.0, [0.0, 0.0, 3.0]),
        make_fluid_cell(1.0, 1.0, 0.0, [0.0, 0.0, 0.0]),
    ]
    divergence = compute_max_divergence(cells, domain_config())
    assert divergence > 0.0
    assert divergence == round((2.0 / (2.0 * 1.0)) + (3.0 / (2.0 * 1.0)), 5)

def test_empty_grid_returns_zero():
    assert compute_max_divergence([], domain_config()) == 0.0

def test_missing_domain_keys_returns_zero():
    cells = [make_fluid_cell(1.0, 1.0, 1.0, [1.0, 1.0, 1.0])]
    assert compute_max_divergence(cells, {}) == 0.0



