# tests/test_reflex_logic.py
# 🧪 Validates reflex logic functions: damping triggers, overflow detection, and CFL-based time-step adjustment

from src.reflex.reflex_logic import should_dampen, should_flag_overflow, adjust_time_step
from src.grid_modules.cell import Cell
import math

def make_cell(vx, vy, vz, fluid=True):
    return Cell(x=0.0, y=0.0, z=0.0, velocity=[vx, vy, vz], pressure=1.0, fluid_mask=fluid)

def test_should_dampen_triggers_on_spike():
    grid = [
        make_cell(1.0, 0.0, 0.0),
        make_cell(1.0, 0.0, 0.0),
        make_cell(5.0, 0.0, 0.0)
    ]
    assert should_dampen(grid, volatility_threshold=0.5) is True

def test_should_dampen_false_on_uniform_velocity():
    grid = [make_cell(2.0, 2.0, 2.0)] * 5
    assert should_dampen(grid) is False

def test_should_dampen_skips_nonfluid_and_malformed():
    bad1 = Cell(x=0.0, y=0.0, z=0.0, velocity="bad", pressure=0.0, fluid_mask=True)
    bad2 = Cell(x=0.0, y=0.0, z=0.0, velocity=None, pressure=0.0, fluid_mask=True)
    solid = make_cell(1.0, 1.0, 1.0, fluid=False)
    grid = [bad1, bad2, solid]
    assert should_dampen(grid) is False

def test_should_dampen_empty_returns_false():
    assert should_dampen([]) is False

def test_should_flag_overflow_detects_exceeding():
    cell = make_cell(15.0, 0.0, 0.0)
    assert should_flag_overflow([cell], threshold=10.0) is True

def test_should_flag_overflow_skips_nonfluid():
    cell = make_cell(15.0, 0.0, 0.0, fluid=False)
    assert should_flag_overflow([cell]) is False

def test_should_flag_overflow_skips_malformed():
    bad = Cell(x=0.0, y=0.0, z=0.0, velocity="bad", pressure=0.0, fluid_mask=True)
    assert should_flag_overflow([bad]) is False

def test_should_flag_overflow_returns_false_when_safe():
    cell = make_cell(2.0, 2.0, 2.0)  # magnitude ≈ 3.46
    assert should_flag_overflow([cell], threshold=10.0) is False

def test_adjust_time_step_reduces_on_excessive_cfl():
    config = {
        "simulation_parameters": {"time_step": 0.1},
        "domain_definition": {"min_x": 0.0, "max_x": 2.0, "nx": 2}
    }
    cell = make_cell(15.0, 0.0, 0.0)
    result = adjust_time_step([cell], config, cfl_limit=1.0)
    assert result < 0.1

def test_adjust_time_step_returns_original_when_stable():
    config = {
        "simulation_parameters": {"time_step": 0.1},
        "domain_definition": {"min_x": 0.0, "max_x": 2.0, "nx": 2}
    }
    cell = make_cell(1.0, 1.0, 1.0)  # magnitude ≈ 1.732
    result = adjust_time_step([cell], config)
    assert result == 0.1

def test_adjust_time_step_handles_zero_dx_gracefully():
    config = {
        "simulation_parameters": {"time_step": 0.1},
        "domain_definition": {"min_x": 0.0, "max_x": 0.0, "nx": 0}
    }
    cell = make_cell(1.0, 0.0, 0.0)
    result = adjust_time_step([cell], config)
    assert result == 0.1  # fallback dx = 1.0

def test_adjust_time_step_empty_grid_returns_original():
    config = {
        "simulation_parameters": {"time_step": 0.2},
        "domain_definition": {"min_x": 0.0, "max_x": 2.0, "nx": 2}
    }
    assert adjust_time_step([], config) == 0.2