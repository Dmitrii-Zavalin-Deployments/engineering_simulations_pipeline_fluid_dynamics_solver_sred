# tests/reflex/test_reflex_logic.py
# 🧪 Unit tests for src/reflex/reflex_logic.py

from src.grid_modules.cell import Cell
from src.reflex.reflex_logic import should_dampen, should_flag_overflow, adjust_time_step

def make_cell(x, y, z, velocity, fluid=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=None, fluid_mask=fluid)

def test_should_dampen_returns_true_on_spike():
    base = [make_cell(0.0, 0.0, i, [1.0, 0.0, 0.0]) for i in range(4)]
    spike = make_cell(0.0, 0.0, 4, [10.0, 0.0, 0.0])
    result = should_dampen(base + [spike], volatility_threshold=0.5)
    assert result is True

def test_should_dampen_returns_false_on_uniform_velocity():
    grid = [make_cell(0.0, 0.0, i, [2.0, 0.0, 0.0]) for i in range(5)]
    assert should_dampen(grid) is False

def test_should_dampen_handles_empty_or_invalid_cells():
    cell = make_cell(0.0, 0.0, 0.0, None)
    solid = make_cell(0.0, 1.0, 0.0, [3.0, 3.0, 3.0], fluid=False)
    assert should_dampen([cell, solid]) is False

def test_should_flag_overflow_triggers_when_velocity_exceeds_threshold():
    cell = make_cell(0.0, 0.0, 0.0, [20.0, 0.0, 0.0])
    assert should_flag_overflow([cell], threshold=10.0) is True

def test_should_flag_overflow_returns_false_for_safe_velocity():
    safe = make_cell(0.0, 0.0, 0.0, [2.0, 2.0, 2.0])
    assert should_flag_overflow([safe], threshold=10.0) is False

def test_adjust_time_step_reduces_dt_when_cfl_exceeds_limit():
    cell = make_cell(0.0, 0.0, 0.0, [5.0, 0.0, 0.0])
    config = {
        "simulation_parameters": {"time_step": 0.5},
        "domain_definition": {"nx": 1, "min_x": 0.0, "max_x": 1.0}
    }
    dt = adjust_time_step([cell], config, cfl_limit=1.0)
    assert dt < 0.5

def test_adjust_time_step_returns_original_dt_when_cfl_ok():
    cell = make_cell(0.0, 0.0, 0.0, [1.0, 0.0, 0.0])
    config = {
        "simulation_parameters": {"time_step": 0.05},
        "domain_definition": {"nx": 1, "min_x": 0.0, "max_x": 1.0}
    }
    dt = adjust_time_step([cell], config)
    assert abs(dt - 0.05) < 1e-8

def test_adjust_time_step_defaults_on_missing_domain():
    cell = make_cell(0.0, 0.0, 0.0, [2.0, 2.0, 2.0])
    config = {"simulation_parameters": {"time_step": 0.1}}
    result = adjust_time_step([cell], config)
    assert isinstance(result, float)



