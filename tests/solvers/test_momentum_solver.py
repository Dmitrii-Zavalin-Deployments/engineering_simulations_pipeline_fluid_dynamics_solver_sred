# tests/solvers/test_momentum_solver.py
# 🧪 Unit tests for src/solvers/momentum_solver.py

from src.grid_modules.cell import Cell
from src.solvers.momentum_solver import apply_momentum_update

def make_cell(x, y, z, velocity, pressure, fluid=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=pressure, fluid_mask=fluid)

def test_applies_momentum_to_fluid_cells_only():
    fluid = make_cell(0.0, 0.0, 0.0, velocity=[1.0, 0.0, 0.0], pressure=5.0)
    solid = make_cell(1.0, 0.0, 0.0, velocity=None, pressure=None, fluid=False)
    input_data = {
        "simulation_parameters": {"time_step": 0.05},
        "domain_definition": {
            "nx": 1, "min_x": 0.0, "max_x": 1.0,
            "ny": 1, "min_y": 0.0, "max_y": 1.0,
            "nz": 1, "min_z": 0.0, "max_z": 1.0
        },
        "fluid_properties": {"viscosity": 0.1}
    }
    result = apply_momentum_update([fluid, solid], input_data, step=3)
    assert result[0].fluid_mask is True
    assert isinstance(result[0].velocity, list)
    assert result[0].pressure == 5.0
    assert result[1].fluid_mask is False
    assert result[1].velocity is None
    assert result[1].pressure is None

def test_velocity_updated_but_pressure_preserved():
    fluid = make_cell(0.0, 0.0, 0.0, velocity=[2.0, 2.0, 2.0], pressure=9.9)
    input_data = {
        "simulation_parameters": {"time_step": 0.1},
        "domain_definition": {
            "nx": 2, "min_x": 0.0, "max_x": 1.0,
            "ny": 1, "min_y": 0.0, "max_y": 1.0,
            "nz": 1, "min_z": 0.0, "max_z": 1.0
        },
        "fluid_properties": {"viscosity": 0.1}
    }
    result = apply_momentum_update([fluid], input_data, step=5)
    updated = result[0]
    assert isinstance(updated.velocity, list)
    assert updated.pressure == 9.9

def test_handles_multiple_fluid_cells_consistently():
    c1 = make_cell(0.0, 0.0, 0.0, velocity=[1.0, 0.0, 0.0], pressure=1.0)
    c2 = make_cell(0.5, 0.0, 0.0, velocity=[2.0, 0.0, 0.0], pressure=2.0)
    input_data = {
        "simulation_parameters": {"time_step": 0.05},
        "domain_definition": {
            "nx": 2, "min_x": 0.0, "max_x": 1.0,
            "ny": 1, "min_y": 0.0, "max_y": 1.0,
            "nz": 1, "min_z": 0.0, "max_z": 1.0
        },
        "fluid_properties": {"viscosity": 0.2}
    }
    result = apply_momentum_update([c1, c2], input_data, step=1)
    assert len(result) == 2
    for original, updated in zip([c1, c2], result):
        assert updated.pressure == original.pressure
        assert isinstance(updated.velocity, list)



