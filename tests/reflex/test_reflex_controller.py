# tests/reflex/test_reflex_controller.py
# 🧪 Unit tests for reflex_controller.py — verifies reflex logic, metric generation, and snapshot metadata structure

import pytest
from src.grid_modules.cell import Cell
from src.reflex.reflex_controller import apply_reflex

def mock_config(time_step=0.1, viscosity=0.01):
    return {
        "simulation_parameters": {
            "time_step": time_step
        },
        "fluid_properties": {
            "viscosity": viscosity
        },
        "domain_definition": {
            "min_x": 0.0,
            "max_x": 1.0,
            "nx": 10
        }
    }

def test_reflex_output_keys_complete():
    grid = [
        Cell(x=0, y=0, z=0, velocity=[0.01, 0.0, 0.0], pressure=100.0, fluid_mask=True),
        Cell(x=1, y=0, z=0, velocity=None, pressure=None, fluid_mask=False)
    ]
    flags = apply_reflex(grid, mock_config(), step=0)

    expected_keys = {
        "max_velocity",
        "max_divergence",
        "global_cfl",
        "overflow_detected",
        "damping_enabled",
        "adjusted_time_step",
        "projection_passes"
    }

    assert isinstance(flags, dict)
    assert expected_keys.issubset(flags.keys())

def test_reflex_boolean_flags_are_valid():
    grid = [
        Cell(x=0, y=0, z=0, velocity=[0.0, 0.0, 0.0], pressure=100.0, fluid_mask=True)
    ]
    result = apply_reflex(grid, mock_config(), step=1)

    assert isinstance(result["damping_enabled"], bool)
    assert isinstance(result["overflow_detected"], bool)

def test_adjusted_time_step_matches_config_when_stable():
    grid = [
        Cell(x=0.0, y=0.0, z=0.0, velocity=[0.01, 0.02, 0.03], pressure=99.0, fluid_mask=True)
    ]
    flags = apply_reflex(grid, mock_config(time_step=0.05), step=2)

    assert isinstance(flags["adjusted_time_step"], float)
    assert abs(flags["adjusted_time_step"] - 0.05) < 1e-6

def test_max_velocity_computation_is_correct():
    grid = [
        Cell(x=0.0, y=0.0, z=0.0, velocity=[0.1, 0.2, 0.2], pressure=101.0, fluid_mask=True),
        Cell(x=1.0, y=0.0, z=0.0, velocity=[0.0, 0.0, 0.0], pressure=102.0, fluid_mask=True)
    ]
    result = apply_reflex(grid, mock_config(), step=3)
    expected = (0.1**2 + 0.2**2 + 0.2**2)**0.5

    assert isinstance(result["max_velocity"], float)
    assert abs(result["max_velocity"] - expected) < 1e-6

def test_global_cfl_computation_is_stable():
    grid = [
        Cell(x=0.0, y=0.0, z=0.0, velocity=[0.01, 0.0, 0.0], pressure=101.0, fluid_mask=True)
    ]
    result = apply_reflex(grid, mock_config(time_step=0.1), step=4)

    assert isinstance(result["global_cfl"], float)
    assert result["global_cfl"] <= 1.0

def test_max_divergence_is_float():
    grid = [
        Cell(x=0.0, y=0.0, z=0.0, velocity=[0.02, -0.01, 0.03], pressure=100.0, fluid_mask=True)
    ]
    result = apply_reflex(grid, mock_config(), step=5)
    assert isinstance(result["max_divergence"], float)

def test_projection_pass_count_is_int():
    grid = [
        Cell(x=0.0, y=0.0, z=0.0, velocity=[0.03, 0.01, 0.02], pressure=100.0, fluid_mask=True)
    ]
    result = apply_reflex(grid, mock_config(), step=6)
    assert isinstance(result["projection_passes"], int)
    assert result["projection_passes"] >= 0

def test_empty_grid_returns_safe_values():
    result = apply_reflex([], mock_config(), step=7)

    assert isinstance(result["max_velocity"], float)
    assert isinstance(result["global_cfl"], float)
    assert isinstance(result["max_divergence"], float)
    assert isinstance(result["projection_passes"], int)
    assert isinstance(result["damping_enabled"], bool)
    assert isinstance(result["overflow_detected"], bool)
    assert isinstance(result["adjusted_time_step"], float)



