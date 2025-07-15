# tests/test_reflex_controller.py
# 🧪 Validates reflex metadata generation, pressure flags, overflow/damping logic, and verbosity diagnostics

import pytest
from src.grid_modules.cell import Cell
from src.reflex.reflex_controller import apply_reflex

def make_cell(x=0.0, y=0.0, z=0.0, velocity=None, pressure=None, fluid=True, influenced=False):
    cell = Cell(x=x, y=y, z=z, velocity=velocity, pressure=pressure, fluid_mask=fluid)
    if influenced:
        setattr(cell, "influenced_by_ghost", True)
    return cell

@pytest.fixture
def config_dict():
    return {
        "reflex_verbosity": "high",
        "include_divergence_delta": True,
        "include_pressure_mutation_map": True,
        "log_projection_trace": True
    }

@pytest.fixture
def input_data():
    return {
        "domain_definition": {
            "min_x": 0.0, "max_x": 2.0, "nx": 2,
            "min_y": 0.0, "max_y": 2.0, "ny": 2,
            "min_z": 0.0, "max_z": 2.0, "nz": 2
        },
        "simulation_parameters": {
            "time_step": 0.1
        }
    }

def test_reflex_output_contains_expected_keys(config_dict, input_data):
    cell = make_cell(velocity=[1.0, 0.0, 0.0], pressure=10.0)
    result = apply_reflex([cell], input_data, step=1, config=config_dict)
    expected_keys = [
        "max_velocity", "max_divergence", "global_cfl", "overflow_detected",
        "damping_enabled", "adjusted_time_step", "projection_passes",
        "divergence_zero", "projection_skipped", "ghost_influence_count",
        "fluid_cells_modified_by_ghost", "triggered_by",
        "pressure_solver_invoked", "pressure_mutated",
        "post_projection_divergence"
    ]
    for key in expected_keys:
        assert key in result

def test_reflex_mutation_flags(config_dict, input_data):
    cell = make_cell(velocity=[0.1, 0.2, 0.3], pressure=1.0)
    result = apply_reflex([cell], input_data, step=5,
                          ghost_influence_count=2,
                          pressure_solver_invoked=True,
                          pressure_mutated=True,
                          post_projection_divergence=1e-9,
                          config=config_dict)
    assert result["pressure_solver_invoked"] is True
    assert result["pressure_mutated"] is True
    assert result["divergence_zero"] is True
    assert "ghost_influence" in result["triggered_by"]

def test_reflex_zero_divergence_threshold(config_dict, input_data):
    cell = make_cell(velocity=[0.0, 0.0, 0.0], pressure=0.0)
    result = apply_reflex([cell], input_data, step=6,
                          post_projection_divergence=0.0,
                          config=config_dict)
    assert result["divergence_zero"] is True

def test_reflex_skips_projection_if_zero_passes(config_dict, input_data):
    cell = make_cell(velocity=[1.0, 0.0, 0.0], pressure=10.0)
    # Override evaluator with zero output
    original = __import__("src.metrics.projection_evaluator", fromlist=["calculate_projection_passes"])
    setattr(original, "calculate_projection_passes", lambda grid: 0)
    result = apply_reflex([cell], input_data, step=8, config=config_dict)
    assert result["projection_skipped"] is True

def test_reflex_detects_overflow_and_damping(input_data):
    cell = make_cell(velocity=[15.0, 0.0, 0.0], pressure=0.0)
    result = apply_reflex([cell], input_data, step=3)
    assert result["overflow_detected"] is True or result["damping_enabled"] is True
    assert "overflow_detected" in result["triggered_by"] or "damping_enabled" in result["triggered_by"]

def test_fluid_influence_tagged_cells_counted(config_dict, input_data):
    c1 = make_cell(velocity=[1, 0, 0], pressure=10.0, influenced=True)
    c2 = make_cell(velocity=[0, 0, 1], pressure=5.0, influenced=False)
    c3 = make_cell(velocity=[0, 0, 0], pressure=3.0, fluid=False)
    grid = [c1, c2, c3]
    result = apply_reflex(grid, input_data, step=4, config=config_dict)
    assert result["fluid_cells_modified_by_ghost"] == 1

def test_reflex_handles_missing_fields_gracefully(input_data):
    cell = make_cell(velocity=[1.0, 0.0, 0.0], pressure=10.0)
    result = apply_reflex([cell], input_data, step=9)
    assert isinstance(result["max_velocity"], float)
    assert result["pressure_solver_invoked"] is False
    assert result["pressure_mutated"] is False

def test_verbose_output_triggers(capsys, config_dict, input_data):
    cell = make_cell(velocity=[2.0, 0.0, 0.0], pressure=5.0)
    apply_reflex([cell], input_data, step=10, config=config_dict)
    out = capsys.readouterr().out
    assert "[reflex]" in out
    assert "Max velocity" in out
    assert "Projection passes" in out