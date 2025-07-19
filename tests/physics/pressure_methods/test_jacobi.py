# tests/physics/pressure_methods/test_jacobi.py
# 🧪 Unit tests for src/physics/pressure_methods/jacobi.py

from src.grid_modules.cell import Cell
from src.physics.pressure_methods.jacobi import solve_jacobi_pressure

def make_cell(x, y, z, velocity=None, pressure=0.0, fluid=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=pressure, fluid_mask=fluid)

def test_solves_single_cell_with_zero_divergence():
    c = make_cell(0, 0, 0, pressure=0.0)
    config = {
        "domain_definition": {"nx": 1, "ny": 1, "nz": 1, "min_x": 0.0, "max_x": 1.0,
                              "min_y": 0.0, "max_y": 1.0, "min_z": 0.0, "max_z": 1.0},
        "pressure_solver": {"max_iterations": 10, "tolerance": 1e-6}
    }
    result = solve_jacobi_pressure([c], divergence=[0.0], config=config)
    assert isinstance(result, list)
    assert len(result) == 1
    assert abs(result[0]) < 1e-6

def test_raises_if_divergence_length_mismatch():
    c = make_cell(0, 0, 0)
    config = {
        "domain_definition": {"nx": 1, "ny": 1, "nz": 1,
                              "min_x": 0.0, "max_x": 1.0,
                              "min_y": 0.0, "max_y": 1.0,
                              "min_z": 0.0, "max_z": 1.0}
    }
    try:
        solve_jacobi_pressure([c], divergence=[], config=config)
        assert False  # Should not reach here
    except ValueError as e:
        assert "Mismatch" in str(e)

def test_solves_3_cell_line_with_known_divergence():
    grid = [
        make_cell(0.0, 0.0, 0.0, pressure=0.0),
        make_cell(1.0, 0.0, 0.0, pressure=0.0),
        make_cell(2.0, 0.0, 0.0, pressure=0.0)
    ]
    config = {
        "domain_definition": {"nx": 2, "ny": 1, "nz": 1,
                              "min_x": 0.0, "max_x": 2.0,
                              "min_y": 0.0, "max_y": 1.0,
                              "min_z": 0.0, "max_z": 1.0},
        "pressure_solver": {"max_iterations": 50, "tolerance": 1e-6}
    }
    divergence = [1.0, -2.0, 1.0]
    pressures = solve_jacobi_pressure(grid, divergence, config)
    assert len(pressures) == 3
    assert all(isinstance(p, float) for p in pressures)

def test_respects_ghost_cell_pressure_override():
    fluid = make_cell(1.0, 0.0, 0.0, pressure=0.0)
    ghost = make_cell(2.0, 0.0, 0.0, pressure=8.0, fluid=False)
    config = {
        "domain_definition": {"nx": 2, "ny": 1, "nz": 1,
                              "min_x": 0.0, "max_x": 2.0,
                              "min_y": 0.0, "max_y": 1.0,
                              "min_z": 0.0, "max_z": 1.0},
        "pressure_solver": {"max_iterations": 20, "tolerance": 1e-6}
    }
    pressures = solve_jacobi_pressure([fluid, ghost], divergence=[2.0], config=config, ghost_coords={(2.0, 0.0, 0.0)})
    assert len(pressures) == 1
    assert pressures[0] > 0.0  # Pressure influenced by ghost Dirichlet value



