# tests/test_main_solver.py
# 🧪 Comprehensive unit tests for main_solver.py — includes reflex metrics, masking, and snapshot integrity

import pytest
from src.main_solver import generate_snapshots
from src.grid_generator import generate_grid_with_mask
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
    "boundary_conditions": {
        "x_min": "inlet",
        "x_max": "outlet",
        "y_min": "wall",
        "y_max": "wall",
        "z_min": "symmetry",
        "z_max": "symmetry",
        "faces": [1, 2, 3, 4, 5, 6],
        "type": "dirichlet",
        "apply_to": ["pressure", "velocity"],
        "pressure": 100.0,
        "velocity": [0.0, 0.0, 0.0],
        "no_slip": True
    },
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

def test_snapshot_contains_flat_metrics(input_with_mask):
    snaps = generate_snapshots(input_with_mask, "flat_metrics_check")
    required_keys = {
        "step_index", "grid",
        "max_velocity", "max_divergence", "global_cfl",
        "overflow_detected", "damping_enabled",
        "adjusted_time_step", "projection_passes"
    }
    for _, snap in snaps:
        assert required_keys.issubset(snap.keys())

def test_metrics_have_correct_types(input_with_mask):
    snaps = generate_snapshots(input_with_mask, "type_check")
    for _, snap in snaps:
        assert isinstance(snap["max_velocity"], float)
        assert isinstance(snap["max_divergence"], float)
        assert isinstance(snap["global_cfl"], float)
        assert isinstance(snap["overflow_detected"], bool)
        assert isinstance(snap["damping_enabled"], bool)
        assert isinstance(snap["adjusted_time_step"], float)
        assert isinstance(snap["projection_passes"], int)

def test_grid_has_correct_field_types_and_nulling(input_with_mask):
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
    assert len(snaps) > 0  # Should fallback to interval=1

def test_snapshot_step_index_formatting():
    steps = [f"{i:04d}" for i in [0, 1, 12, 123, 1234]]
    assert steps == ["0000", "0001", "0012", "0123", "1234"]



