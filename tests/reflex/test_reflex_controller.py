# tests/reflex/test_reflex_controller.py
# 🧪 Unit tests for src/reflex/reflex_controller.py

from src.grid_modules.cell import Cell
from src.reflex.reflex_controller import apply_reflex

def make_fluid_cell(x, y, z, velocity=None, pressure=None, influenced=False):
    cell = Cell(x=x, y=y, z=z, velocity=velocity, pressure=pressure, fluid_mask=True)
    if influenced:
        setattr(cell, "influenced_by_ghost", True)
    return cell

def test_reflex_output_contains_expected_keys():
    cell = make_fluid_cell(0.0, 0.0, 0.0, velocity=[1.0, 0.0, 0.0], pressure=1.0)
    input_data = {
        "domain_definition": {
            "nx": 1, "ny": 1, "nz": 1,
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0
        },
        "simulation_parameters": {
            "time_step": 0.1
        }
    }
    result = apply_reflex([cell], input_data, step=1)
    required_keys = [
        "max_velocity", "max_divergence", "global_cfl",
        "overflow_detected", "damping_enabled",
        "adjusted_time_step", "projection_passes",
        "divergence_zero", "projection_skipped",
        "ghost_influence_count", "fluid_cells_modified_by_ghost",
        "triggered_by", "pressure_solver_invoked",
        "pressure_mutated", "post_projection_divergence", "reflex_score"
    ]
    for key in required_keys:
        assert key in result

def test_reflex_flags_trigger_on_influence_and_mutation():
    cell = make_fluid_cell(0.0, 0.0, 0.0, velocity=[0.0, 0.0, 0.0], pressure=0.0, influenced=True)
    input_data = {
        "domain_definition": {
            "nx": 1, "ny": 1, "nz": 1,
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0
        },
        "simulation_parameters": {
            "time_step": 0.05
        }
    }
    config = {
        "reflex_verbosity": "low"
    }
    result = apply_reflex(
        [cell], input_data, step=3,
        ghost_influence_count=1,
        config=config,
        pressure_solver_invoked=True,
        pressure_mutated=True,
        post_projection_divergence=1e-9
    )
    assert "ghost_influence" in result["triggered_by"]
    assert result["divergence_zero"] is True
    assert result["pressure_mutated"] is True
    assert result["fluid_cells_modified_by_ghost"] == 1

def test_reflex_skips_projection_if_passes_zero():
    cell = make_fluid_cell(1.0, 1.0, 1.0, velocity=[0.0, 0.0, 0.0], pressure=2.0)
    input_data = {
        "domain_definition": {
            "nx": 1, "ny": 1, "nz": 1,
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0
        },
        "simulation_parameters": {
            "time_step": 0.02
        }
    }
    result = apply_reflex([cell], input_data, step=5, pressure_solver_invoked=True)
    assert isinstance(result["projection_passes"], int)
    assert result["pressure_solver_invoked"] is True
    assert isinstance(result["reflex_score"], float)

def test_reflex_suppression_check_triggers_warning():
    cell = make_fluid_cell(0.0, 0.0, 0.0, velocity=[0.0, 0.0, 0.0], pressure=1.0)
    input_data = {
        "domain_definition": {
            "nx": 1, "ny": 1, "nz": 1,
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0
        },
        "simulation_parameters": {
            "time_step": 0.01
        }
    }
    result = apply_reflex(
        [cell], input_data, step=6,
        ghost_influence_count=1,
        pressure_mutated=True
    )
    assert result["fluid_cells_modified_by_ghost"] == 0



