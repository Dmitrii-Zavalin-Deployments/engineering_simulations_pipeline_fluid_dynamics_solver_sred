# tests/physics/divergence_methods/test_central.py
# 🧪 Unit tests for src/physics/divergence_methods/central.py

from src.grid_modules.cell import Cell
from src.physics.divergence_methods.central import compute_central_divergence

def make_cell(x, y, z, velocity=None, fluid=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=0.0, fluid_mask=fluid)

def test_returns_empty_for_all_non_fluid_cells():
    grid = [
        make_cell(0.0, 0.0, 0.0, velocity=[0, 0, 0], fluid=False),
        make_cell(1.0, 0.0, 0.0, velocity=[1, 0, 0], fluid=False)
    ]
    config = {"domain_definition": {"nx": 1, "min_x": 0.0, "max_x": 1.0}}
    assert compute_central_divergence(grid, config) == []

def test_skips_cells_with_missing_velocity():
    grid = [
        make_cell(0.0, 0.0, 0.0, velocity=None),
        make_cell(1.0, 0.0, 0.0, velocity=[1.0, 0.0, 0.0])
    ]
    config = {"domain_definition": {"nx": 1, "min_x": 0.0, "max_x": 1.0}}
    result = compute_central_divergence(grid, config)
    assert len(result) == 1

def test_computes_zero_divergence_for_uniform_flow():
    grid = [
        make_cell(0.0, 0.0, 0.0, velocity=[1, 0, 0]),
        make_cell(1.0, 0.0, 0.0, velocity=[1, 0, 0]),
        make_cell(2.0, 0.0, 0.0, velocity=[1, 0, 0])
    ]
    config = {"domain_definition": {
        "nx": 2, "min_x": 0.0, "max_x": 2.0,
        "ny": 1, "min_y": 0.0, "max_y": 1.0,
        "nz": 1, "min_z": 0.0, "max_z": 1.0
    }}
    result = compute_central_divergence(grid, config)
    assert all(abs(d) < 1e-9 for d in result)

def test_computes_divergence_with_gradient():
    grid = [
        make_cell(0.0, 0.0, 0.0, velocity=[1.0, 0.0, 0.0]),
        make_cell(1.0, 0.0, 0.0, velocity=[3.0, 0.0, 0.0]),
        make_cell(2.0, 0.0, 0.0, velocity=[5.0, 0.0, 0.0])
    ]
    config = {"domain_definition": {
        "nx": 2, "min_x": 0.0, "max_x": 2.0,
        "ny": 1, "min_y": 0.0, "max_y": 1.0,
        "nz": 1, "min_z": 0.0, "max_z": 1.0
    }}
    result = compute_central_divergence(grid, config)
    assert len(result) == 3
    assert result[1] > 0.0  # Middle cell should show divergence



