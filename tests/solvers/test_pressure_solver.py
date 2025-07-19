# tests/solvers/test_pressure_solver.py
# 🧪 Unit tests for src/solvers/pressure_solver.py

from src.grid_modules.cell import Cell
from src.solvers.pressure_solver import apply_pressure_correction

def make_cell(x, y, z, velocity, pressure, fluid=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=pressure, fluid_mask=fluid)

def test_pressure_correction_returns_expected_structure():
    fluid = make_cell(0.0, 0.0, 0.0, velocity=[1.0, 0.0, 0.0], pressure=0.0)
    input_data = {
        "simulation_parameters": {"time_step": 0.1},
        "domain_definition": {
            "nx": 1, "ny": 1, "nz": 1,
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0
        },
        "pressure_solver": {"method": "jacobi"}
    }
    result = apply_pressure_correction([fluid], input_data, step=4)
    assert isinstance(result, tuple)
    grid, mutated, passes, meta = result
    assert isinstance(grid, list)
    assert isinstance(mutated, bool)
    assert isinstance(passes, int)
    assert isinstance(meta, dict)
    assert "pressure_mutation_count" in meta
    assert "mutated_cells" in meta

def test_mutated_cell_recorded_if_pressure_changes():
    fluid = make_cell(0.0, 0.0, 0.0, velocity=[2.0, 0.0, 0.0], pressure=0.0)
    input_data = {
        "simulation_parameters": {"time_step": 0.05},
        "domain_definition": {
            "nx": 1, "ny": 1, "nz": 1,
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0
        },
        "pressure_solver": {"method": "jacobi"}
    }
    _, mutated, _, meta = apply_pressure_correction([fluid], input_data, step=8)
    assert mutated is True
    assert meta["pressure_mutation_count"] > 0
    assert isinstance(meta["mutated_cells"], list)

def test_pressure_unchanged_when_no_divergence():
    fluid = make_cell(0.5, 0.5, 0.5, velocity=[0.0, 0.0, 0.0], pressure=1.0)
    input_data = {
        "simulation_parameters": {"time_step": 0.02},
        "domain_definition": {
            "nx": 1, "ny": 1, "nz": 1,
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0
        },
        "pressure_solver": {"method": "jacobi"}
    }
    result = apply_pressure_correction([fluid], input_data, step=10)
    _, mutated, _, meta = result
    assert mutated is False or meta["pressure_mutation_count"] == 0

def test_ghost_cell_excluded_from_pressure_mutation():
    fluid = make_cell(0.0, 0.0, 0.0, velocity=[1.0, 0.0, 0.0], pressure=0.0)
    ghost = make_cell(1.0, 0.0, 0.0, velocity=None, pressure=None, fluid=False)
    setattr(ghost, "ghost_face", "x_max")
    input_data = {
        "simulation_parameters": {"time_step": 0.1},
        "domain_definition": {
            "nx": 2, "ny": 1, "nz": 1,
            "min_x": 0.0, "max_x": 2.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0
        },
        "pressure_solver": {"method": "jacobi"}
    }
    result = apply_pressure_correction([fluid, ghost], input_data, step=12)
    updated_grid, _, _, _ = result
    assert updated_grid[1].fluid_mask is False
    assert updated_grid[1].pressure is None



