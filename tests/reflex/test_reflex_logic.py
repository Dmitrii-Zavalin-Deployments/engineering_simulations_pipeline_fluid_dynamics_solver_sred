# tests/reflex/test_reflex_logic.py
# 🧪 Unit tests for reflex_logic.py — validate damping, overflow, and timestep logic

import pytest
from src.reflex.reflex_logic import should_dampen, should_flag_overflow, adjust_time_step
from src.grid_modules.cell import Cell

def make_cell(x, y, z, velocity, pressure=1.0, fluid_mask=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=pressure, fluid_mask=fluid_mask)

# ----------------------
# should_dampen() Tests
# ----------------------

def test_dampen_returns_false_for_uniform_velocity():
    grid = [make_cell(0, 0, 0, [1.0, 0.0, 0.0]),
            make_cell(1, 0, 0, [1.0, 0.0, 0.0]),
            make_cell(2, 0, 0, [1.0, 0.0, 0.0])]
    assert should_dampen(grid) is False

def test_dampen_triggers_on_spike_above_threshold():
    grid = [make_cell(0, 0, 0, [1.0, 0.0, 0.0]),
            make_cell(1, 0, 0, [1.0, 0.0, 0.0]),
            make_cell(2, 0, 0, [3.0, 0.0, 0.0])]  # 3x spike
    assert should_dampen(grid, volatility_threshold=0.5) is True

def test_dampen_returns_false_for_empty_grid():
    assert should_dampen([]) is False

def test_dampen_ignores_solid_cells_and_malformed_vectors():
    grid = [
        make_cell(0, 0, 0, [1.0, 0.0, 0.0], fluid_mask=True),
        make_cell(1, 0, 0, None, fluid_mask=False),
        make_cell(2, 0, 0, "bad", fluid_mask=True)
    ]
    assert should_dampen(grid) is False

# --------------------------
# should_flag_overflow() Tests
# --------------------------

def test_overflow_detects_high_velocity_magnitude():
    grid = [make_cell(0, 0, 0, [11.0, 0.0, 0.0]),
            make_cell(1, 0, 0, [1.0, 0.0, 0.0])]
    assert should_flag_overflow(grid) is True

def test_overflow_returns_false_when_all_below_threshold():
    grid = [make_cell(0, 0, 0, [5.0, 0.0, 0.0]),
            make_cell(1, 0, 0, [9.9, 0.0, 0.0])]
    assert should_flag_overflow(grid) is False

def test_overflow_respects_custom_threshold():
    grid = [make_cell(0, 0, 0, [8.0, 0.0, 0.0])]
    assert should_flag_overflow(grid, threshold=7.5) is True
    assert should_flag_overflow(grid, threshold=8.5) is False

def test_overflow_ignores_invalid_cells():
    grid = [
        make_cell(0, 0, 0, [1.0, 0.0, 0.0]),
        make_cell(1, 0, 0, None, fluid_mask=False),
        make_cell(2, 0, 0, "bad", fluid_mask=True)
    ]
    assert should_flag_overflow(grid, threshold=10.0) is False

# --------------------------
# adjust_time_step() Tests
# --------------------------

def base_config(time_step=0.1, min_x=0.0, max_x=1.0, nx=10):
    return {
        "simulation_parameters": {
            "time_step": time_step
        },
        "domain_definition": {
            "min_x": min_x,
            "max_x": max_x,
            "nx": nx
        }
    }

def test_adjust_step_returns_original_if_cfl_safe():
    grid = [make_cell(0, 0, 0, [0.1, 0.0, 0.0])]
    config = base_config(time_step=0.1)
    dt = adjust_time_step(grid, config, cfl_limit=1.0)
    assert isinstance(dt, float)
    assert abs(dt - 0.1) < 1e-6

def test_adjust_step_reduces_dt_if_cfl_exceeds():
    grid = [make_cell(0, 0, 0, [10.0, 0.0, 0.0])]  # high velocity
    config = base_config(time_step=0.1)
    new_dt = adjust_time_step(grid, config, cfl_limit=1.0)
    assert new_dt < 0.1

def test_adjust_step_returns_safe_default_for_missing_config():
    dt = adjust_time_step([], {}, cfl_limit=1.0)
    assert isinstance(dt, float)
    assert dt == 0.1

def test_adjust_step_handles_zero_resolution():
    grid = [make_cell(0, 0, 0, [1.0, 0.0, 0.0])]
    config = base_config(time_step=0.1, nx=0)
    dt = adjust_time_step(grid, config)
    assert isinstance(dt, float)
    assert dt == 0.1

def test_adjust_step_ignores_malformed_velocity():
    grid = [make_cell(0, 0, 0, "bad")]
    config = base_config(time_step=0.2)
    dt = adjust_time_step(grid, config)
    assert dt == 0.2



