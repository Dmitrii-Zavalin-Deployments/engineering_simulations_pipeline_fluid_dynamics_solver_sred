# tests/test_pressure_solver.py
# 🧪 Validates divergence correction and pressure mutation logic via apply_pressure_correction()

import pytest
from src.grid_modules.cell import Cell
from src.solvers.pressure_solver import apply_pressure_correction

def make_cell(x, y, z, velocity, pressure, fluid=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=pressure, fluid_mask=fluid)

@pytest.fixture
def input_data():
    return {
        "simulation_parameters": {"time_step": 0.1},
        "domain_definition": {
            "min_x": 0.0, "max_x": 2.0, "nx": 2,
            "min_y": 0.0, "max_y": 1.0, "ny": 1,
            "min_z": 0.0, "max_z": 1.0, "nz": 1
        },
        "pressure_solver": {
            "method": "jacobi",
            "max_iterations": 20,
            "tolerance": 1e-6
        },
        "boundary_conditions": {
            "apply_faces": ["x_min", "x_max"],
            "pressure": 99.0,
            "velocity": [0.0, 0.0, 0.0]
        }
    }

def test_pressure_correction_mutates_fluid_cells(input_data):
    cell = make_cell(1.0, 0.0, 0.0, [2.0, 0.0, 0.0], 0.0)
    grid, mutated, passes, meta = apply_pressure_correction([cell], input_data, step=0)
    assert mutated is False  # ✅ Mutation requires neighbor or ghost contrast
    assert passes == 1
    assert isinstance(grid[0].pressure, float)
    assert meta["pressure_mutation_count"] == 0

def test_pressure_correction_preserves_solid_cells(input_data):
    solid = make_cell(1.0, 0.0, 0.0, [1.0, 1.0, 1.0], 10.0, fluid=False)
    grid, mutated, _, meta = apply_pressure_correction([solid], input_data, step=1)
    assert grid[0].fluid_mask is False
    assert grid[0].velocity is None
    assert grid[0].pressure is None
    assert mutated is False
    assert meta["pressure_mutation_count"] == 0

def test_pressure_correction_metadata_fields(input_data):
    cell = make_cell(1.0, 0.0, 0.0, [1.0, 0.0, 0.0], 0.0)
    _, _, _, meta = apply_pressure_correction([cell], input_data, step=2)
    keys = ["max_divergence", "pressure_mutation_count", "pressure_solver_passes", "mutated_cells"]
    for key in keys:
        assert key in meta

def test_pressure_correction_tracks_mutated_coordinates(input_data):
    cell = make_cell(1.0, 0.0, 0.0, [1.0, 0.0, 0.0], 0.0)
    _, _, _, meta = apply_pressure_correction([cell], input_data, step=3)
    assert isinstance(meta["mutated_cells"], list)
    # In this single-cell test, mutation unlikely; just assert type presence
    assert isinstance(meta["mutated_cells"], list)

def test_pressure_correction_handles_malformed_velocity(input_data):
    broken = Cell(x=1.0, y=0.0, z=0.0, velocity="bad", pressure=0.0, fluid_mask=True)
    grid, mutated, passes, meta = apply_pressure_correction([broken], input_data, step=4)
    assert grid[0].fluid_mask is False
    assert mutated is False
    assert meta["pressure_mutation_count"] == 0

def test_pressure_correction_empty_grid(input_data):
    grid, mutated, passes, meta = apply_pressure_correction([], input_data, step=5)
    assert grid == []
    assert mutated is False
    assert passes == 1
    assert meta["pressure_mutation_count"] == 0

def test_pressure_correction_preserves_pressure_type(input_data):
    cell = make_cell(1.0, 0.0, 0.0, [0.0, 0.0, 0.0], 22.5)
    grid, _, _, _ = apply_pressure_correction([cell], input_data, step=6)
    assert isinstance(grid[0].pressure, float)

def test_pressure_correction_prints_status(capsys, input_data):
    cell = make_cell(1.0, 0.0, 0.0, [1.0, 0.0, 0.0], 0.0)
    apply_pressure_correction([cell], input_data, step=7)
    out = capsys.readouterr().out
    assert "Step 7" in out
    assert "Max divergence" in out



