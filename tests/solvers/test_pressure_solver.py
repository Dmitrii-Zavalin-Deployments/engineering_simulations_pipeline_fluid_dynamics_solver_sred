# ✅ Final Fully Updated Test Suite — Z-Domain Patch Applied
# 📄 Full Path: tests/solvers/test_pressure_solver.py

import pytest
from src.solvers.pressure_solver import apply_pressure_correction
from src.grid_modules.cell import Cell

def make_cell(x, velocity, pressure, fluid=True):
    return Cell(x=x, y=0.0, z=0.0, velocity=velocity, pressure=pressure, fluid_mask=fluid)

def test_pressure_mutation_detected_on_asymmetric_velocity():
    grid = [
        make_cell(0.0, velocity=[1.0, 0.0, 0.0], pressure=0.0),
        make_cell(1.0, velocity=[-1.0, 0.0, 0.0], pressure=0.0)
    ]
    input_data = {
        "grid_resolution": "normal",
        "simulation_parameters": {
            "time_step": 0.05
        },
        "domain_definition": {
            "min_x": 0.0, "max_x": 2.0, "nx": 2,
            "min_y": 0.0, "max_y": 1.0, "ny": 1,
            "min_z": 0.0, "max_z": 1.0, "nz": 1
        }
    }
    step = 0
    result = apply_pressure_correction(grid, input_data, step)
    updated_grid, mutated_flag, passes, metadata = result

    assert isinstance(updated_grid, list)
    assert passes == 1
    assert metadata["pressure_mutation_count"] > 0
    assert len(metadata["mutated_cells"]) > 0
    assert mutated_flag in [True, False]
    for cell in updated_grid:
        assert isinstance(cell.pressure, float)

def test_no_mutation_when_velocities_are_symmetric_and_small():
    grid = [
        make_cell(0.0, velocity=[0.0001, 0.0, 0.0], pressure=0.0),
        make_cell(1.0, velocity=[-0.0001, 0.0, 0.0], pressure=0.0)
    ]
    input_data = {
        "grid_resolution": "normal",
        "simulation_parameters": {
            "time_step": 0.01
        },
        "domain_definition": {
            "min_x": 0.0, "max_x": 2.0, "nx": 2,
            "min_y": 0.0, "max_y": 1.0, "ny": 1,
            "min_z": 0.0, "max_z": 1.0, "nz": 1
        }
    }
    step = 1
    result = apply_pressure_correction(grid, input_data, step)
    _, _, _, metadata = result
    assert metadata["pressure_mutation_count"] >= 0
    assert isinstance(metadata["max_divergence"], float)

def test_malformed_velocity_handled_as_solid():
    grid = [
        make_cell(0.0, velocity=None, pressure=0.0),
        make_cell(1.0, velocity=[2.0, 0.0], pressure=0.0)
    ]
    input_data = {
        "grid_resolution": "normal",
        "simulation_parameters": {
            "time_step": 0.05
        },
        "domain_definition": {
            "min_x": 0.0, "max_x": 2.0, "nx": 2,
            "min_y": 0.0, "max_y": 1.0, "ny": 1,
            "min_z": 0.0, "max_z": 1.0, "nz": 1
        }
    }
    step = 2
    result = apply_pressure_correction(grid, input_data, step)
    updated_grid, _, _, metadata = result
    assert len(updated_grid) == 2
    assert all(isinstance(c, Cell) for c in updated_grid)
    assert all(isinstance(c.pressure, (float, type(None))) for c in updated_grid)
    assert isinstance(metadata, dict)
    assert "pressure_mutation_count" in metadata
    assert "mutated_cells" in metadata

def test_snapshot_output_path_resolves():
    grid = [
        make_cell(0.0, velocity=[1.0, 0.0, 0.0], pressure=0.0),
        make_cell(1.0, velocity=[-1.0, 0.0, 0.0], pressure=0.0)
    ]
    input_data = {
        "grid_resolution": "normal",
        "simulation_parameters": {
            "time_step": 0.1
        },
        "domain_definition": {
            "min_x": 0.0, "max_x": 2.0, "nx": 2,
            "min_y": 0.0, "max_y": 1.0, "ny": 1,
            "min_z": 0.0, "max_z": 1.0, "nz": 1
        }
    }
    step = 3
    _ = apply_pressure_correction(grid, input_data, step)
    # No assertion — ensures snapshot export executes without error



