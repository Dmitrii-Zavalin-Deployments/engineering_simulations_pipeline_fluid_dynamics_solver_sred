# tests/physics/test_pressure_projection.py
# 🧪 Unit tests for pressure projection — validate incompressibility enforcement

import pytest
from src.physics.pressure_projection import solve_pressure_poisson
from src.grid_modules.cell import Cell

def make_cell(x, y, z, velocity, pressure=0.0, fluid_mask=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=pressure, fluid_mask=fluid_mask)

def make_config(method="jacobi", nx=3, ny=1, nz=1):
    return {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0 * nx,
            "min_y": 0.0, "max_y": 1.0 * ny,
            "min_z": 0.0, "max_z": 1.0 * nz,
            "nx": nx, "ny": ny, "nz": nz
        },
        "pressure_solver": {
            "method": method,
            "max_iterations": 20,
            "tolerance": 1e-4
        }
    }

# ------------------------------
# solve_pressure_poisson tests
# ------------------------------

def test_pressure_projection_preserves_grid_structure():
    grid = [
        make_cell(0.0, 0.0, 0.0, [1.0, 0.0, 0.0]),
        make_cell(1.0, 0.0, 0.0, [1.0, 0.0, 0.0], fluid_mask=False)
    ]
    divergence = [0.5]
    config = make_config()

    result = solve_pressure_poisson(grid, divergence, config)
    assert len(result) == len(grid)
    assert isinstance(result[0], Cell)
    assert result[0].fluid_mask is True
    assert result[1].fluid_mask is False
    assert result[1].pressure is None

def test_pressure_projection_updates_pressure_for_fluid_cells():
    grid = [
        make_cell(0.0, 0.0, 0.0, [0.0, 0.0, 0.0]),
        make_cell(1.0, 0.0, 0.0, [0.0, 0.0, 0.0])
    ]
    divergence = [1.0, -1.0]
    config = make_config()

    result = solve_pressure_poisson(grid, divergence, config)
    pressures = [cell.pressure for cell in result if cell.fluid_mask]
    assert len(pressures) == 2
    for p in pressures:
        assert isinstance(p, float)
        assert abs(p) > 0.0  # Should have nonzero response to divergence

def test_pressure_projection_ignores_solid_cells():
    grid = [
        make_cell(0.0, 0.0, 0.0, [0.0, 0.0, 0.0], fluid_mask=False),
        make_cell(1.0, 0.0, 0.0, [0.0, 0.0, 0.0], fluid_mask=True)
    ]
    divergence = [1.0]
    config = make_config()

    result = solve_pressure_poisson(grid, divergence, config)
    assert result[0].pressure is None
    assert result[1].pressure is not None

def test_pressure_projection_handles_empty_divergence_gracefully():
    grid = [
        make_cell(0.0, 0.0, 0.0, [0.0, 0.0, 0.0], fluid_mask=False),
        make_cell(1.0, 0.0, 0.0, [0.0, 0.0, 0.0], fluid_mask=False)
    ]
    divergence = []
    config = make_config()

    result = solve_pressure_poisson(grid, divergence, config)
    for cell in result:
        assert cell.pressure is None

def test_pressure_projection_rejects_unknown_solver_method():
    grid = [make_cell(0.0, 0.0, 0.0, [0.0, 0.0, 0.0])]
    divergence = [0.0]
    config = make_config(method="unsupported_method")

    with pytest.raises(ValueError):
        solve_pressure_poisson(grid, divergence, config)



