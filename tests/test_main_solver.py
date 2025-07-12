# tests/test_main_solver.py
# 🧪 Comprehensive unit tests for main_solver.py — includes reflex logic, masking, and snapshot integrity

import pytest
from dataclasses import asdict
from src.main_solver import generate_snapshots
from src.grid_generator import generate_grid, generate_grid_with_mask
from src.metrics.velocity_metrics import compute_max_velocity

MASKED_INPUT = {
    "domain_definition": {
        "min_x": 0.0, "max_x": 3.0,
        "min_y": 0.0, "max_y": 2.0,
        "min_z": 0.0, "max_z": 1.0,
        "nx": 3, "ny": 2, "nz": 1
    },
    "fluid_properties": {
        "density": 1.0, "viscosity": 0.01
    },
    "initial_conditions": {
        "initial_velocity": [0.1, 0.0, 0.0],
        "initial_pressure": 100.0
    },
    "simulation_parameters": {
        "time_step": 0.1,
        "total_time": 1.0,
        "output_interval": 2
    },
    "boundary_conditions": [],
    "geometry_definition": {
        "geometry_mask_flat": [
            1, 1, 0,
            0, 1, 1
        ],
        "geometry_mask_shape": [3, 2, 1],
        "mask_encoding": {
            "fluid": 1,
            "solid": 0
        },
        "flattening_order": "x-major"
    }
}

@pytest.fixture
def input_with_mask():
    return MASKED_INPUT.copy()

def test_snapshot_count_matches_interval(input_with_mask):
    snaps = generate_snapshots(input_with_mask, "mask_test")
    assert len(snaps) == 6  # steps 0,2,4,6,8,10

def test_snapshot_contains_expected_keys(input_with_mask):
    snaps = generate_snapshots(input_with_mask, "mask_test")
    for _, snap in snaps:
        assert "step_index" in snap
        assert "grid" in snap
        assert "reflex_flags" in snap

def test_reflex_flags_structure(input_with_mask):
    snaps = generate_snapshots(input_with_mask, "reflex_test")
    for _, snap in snaps:
        reflex = snap["reflex_flags"]
        expected_keys = {
            "damping_enabled",
            "overflow_detected",
            "adjusted_time_step",
            "max_velocity",
            "global_cfl"
        }
        assert isinstance(reflex, dict)
        assert expected_keys.issubset(reflex.keys())
        assert isinstance(reflex["damping_enabled"], bool)
        assert isinstance(reflex["overflow_detected"], bool)
        assert isinstance(reflex["adjusted_time_step"], float)
        assert isinstance(reflex["max_velocity"], float)
        assert isinstance(reflex["global_cfl"], float)

def test_grid_has_correct_field_types(input_with_mask):
    snaps = generate_snapshots(input_with_mask, "field_check")
    for _, snap in snaps:
        for cell in snap["grid"]:
            assert isinstance(cell["x"], (int, float))
            assert isinstance(cell["y"], (int, float))
            assert isinstance(cell["z"], (int, float))
            assert isinstance(cell["fluid_mask"], bool)
            if cell["fluid_mask"]:
                assert isinstance(cell["velocity"], list)
                assert isinstance(cell["pressure"], (int, float))
                assert len(cell["velocity"]) == 3
            else:
                assert cell["velocity"] is None
                assert cell["pressure"] is None

def test_masked_grid_initialization_matches_expected(input_with_mask):
    grid = generate_grid_with_mask(
        input_with_mask["domain_definition"],
        input_with_mask["initial_conditions"],
        input_with_mask["geometry_definition"]
    )
    fluid_count = sum(1 for cell in grid if cell.fluid_mask)
    solid_count = sum(1 for cell in grid if not cell.fluid_mask)
    assert fluid_count == 4
    assert solid_count == 2

def test_velocity_magnitude_consistency(input_with_mask):
    grid = generate_grid_with_mask(
        input_with_mask["domain_definition"],
        input_with_mask["initial_conditions"],
        input_with_mask["geometry_definition"]
    )
    expected_velocity = input_with_mask["initial_conditions"]["initial_velocity"]
    expected_mag = sum(v**2 for v in expected_velocity)**0.5
    actual_mag = compute_max_velocity(grid)
    assert round(actual_mag, 5) == round(expected_mag, 5)

def test_zero_output_interval_is_handled_gracefully(input_with_mask):
    input_with_mask["simulation_parameters"]["output_interval"] = 0
    snaps = generate_snapshots(input_with_mask, "zero_interval")
    assert len(snaps) > 0  # Should not crash or hang

def test_snapshot_step_index_formatting():
    steps = [f"{i:04d}" for i in [0, 1, 12, 123, 1234]]
    assert steps == ["0000", "0001", "0012", "0123", "1234"]



