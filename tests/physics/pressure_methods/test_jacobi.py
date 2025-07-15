# tests/test_jacobi_solver.py
# 🧪 Unit tests for solve_jacobi_pressure — validates ghost-aware Jacobi iteration with fluid/safe boundary logic

import pytest
from src.grid_modules.cell import Cell
from src.physics.pressure_methods.jacobi import solve_jacobi_pressure

def make_cell(x, y, z, pressure, fluid=True):
    return Cell(x=x, y=y, z=z, velocity=[0.0, 0.0, 0.0], pressure=pressure, fluid_mask=fluid)

@pytest.fixture
def config_1d_3cell():
    return {
        "domain_definition": {
            "min_x": 0.0, "max_x": 3.0, "nx": 3,
            "min_y": 0.0, "max_y": 1.0, "ny": 1,
            "min_z": 0.0, "max_z": 1.0, "nz": 1
        },
        "pressure_solver": {
            "max_iterations": 50,
            "tolerance": 1e-6
        }
    }

def test_solver_matches_divergence_count(config_1d_3cell):
    grid = [
        make_cell(0.0, 0.0, 0.0, 0.0),
        make_cell(1.0, 0.0, 0.0, 0.0),
        make_cell(2.0, 0.0, 0.0, 0.0)
    ]
    divergence = [0.1, 0.2, 0.3]
    pressure = solve_jacobi_pressure(grid, divergence, config_1d_3cell)
    assert isinstance(pressure, list)
    assert len(pressure) == 3

def test_solver_raises_if_divergence_mismatch(config_1d_3cell):
    grid = [make_cell(0.0, 0.0, 0.0, 0.0), make_cell(1.0, 0.0, 0.0, 0.0)]
    divergence = [0.1]  # wrong length
    with pytest.raises(ValueError, match="Mismatch between fluid cells"):
        solve_jacobi_pressure(grid, divergence, config_1d_3cell)

def test_solver_converges_with_uniform_divergence(config_1d_3cell):
    grid = [make_cell(x, 0.0, 0.0, 0.0) for x in [0.0, 1.0, 2.0]]
    divergence = [1.0, 1.0, 1.0]
    pressure = solve_jacobi_pressure(grid, divergence, config_1d_3cell)
    assert all(isinstance(p, float) for p in pressure)
    assert pressure[1] != 0.0  # center cell should be influenced

def test_solver_skips_solid_cells(config_1d_3cell):
    grid = [
        make_cell(0.0, 0.0, 0.0, 10.0, fluid=False),
        make_cell(1.0, 0.0, 0.0, 0.0, fluid=True),
        make_cell(2.0, 0.0, 0.0, 10.0, fluid=False)
    ]
    divergence = [0.2]
    pressure = solve_jacobi_pressure(grid, divergence, config_1d_3cell)
    assert len(pressure) == 1
    assert isinstance(pressure[0], float)

def test_solver_honors_dirichlet_ghost_pressure(config_1d_3cell):
    ghost = make_cell(0.0, 0.0, 0.0, pressure=100.0, fluid=False)
    center = make_cell(1.0, 0.0, 0.0, pressure=0.0)
    right = make_cell(2.0, 0.0, 0.0, pressure=100.0, fluid=False)
    grid = [ghost, center, right]
    divergence = [0.0]
    pressure = solve_jacobi_pressure(grid, divergence, config_1d_3cell, ghost_coords={(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)})
    assert pressure[0] > 0.0
    assert pressure[0] < 100.0

def test_solver_zero_spacing_defaults(config_1d_3cell):
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0, "nx": 0, "min_y": 0.0, "max_y": 1.0, "ny": 1, "min_z": 0.0, "max_z": 1.0, "nz": 1
        },
        "pressure_solver": {"max_iterations": 10, "tolerance": 1e-6}
    }
    grid = [make_cell(x, 0.0, 0.0, 0.0) for x in [0.0, 1.0, 2.0]]
    divergence = [0.1, 0.2, 0.1]
    pressure = solve_jacobi_pressure(grid, divergence, config)
    assert isinstance(pressure[1], float)

def test_solver_converges_under_custom_tolerance():
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 3.0, "nx": 3, "min_y": 0.0, "max_y": 1.0, "ny": 1, "min_z": 0.0, "max_z": 1.0, "nz": 1
        },
        "pressure_solver": {
            "max_iterations": 10,
            "tolerance": 1e-2
        }
    }
    grid = [make_cell(x, 0.0, 0.0, 0.0) for x in [0.0, 1.0, 2.0]]
    divergence = [0.1, 0.2, 0.3]
    pressure = solve_jacobi_pressure(grid, divergence, config)
    assert len(pressure) == 3
    assert any(p != 0.0 for p in pressure)