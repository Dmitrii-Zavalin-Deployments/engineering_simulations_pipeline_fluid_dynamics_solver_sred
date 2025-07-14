# tests/test_main_solver.py
# 🧪 Comprehensive unit tests for main_solver.py — includes reflex metrics, masking, influence flag export, and snapshot integrity

import os
import json
import pytest
from src.main_solver import generate_snapshots, load_reflex_config
from src.grid_generator import generate_grid_with_mask
from src.metrics.velocity_metrics import compute_max_velocity

MASKED_INPUT = {
    "domain_definition": {
        "min_x": 0.0, "max_x": 3.0,
        "min_y": 0.0, "max_y": 2.0,
        "min_z": 0.0, "max_z": 1.0,
        "nx": 3, "ny": 2, "nz": 1
    },
    "fluid_properties": { "density": 1.0, "viscosity": 0.01 },
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
        "x_min": "inlet", "x_max": "outlet", "y_min": "wall", "y_max": "wall",
        "z_min": "symmetry", "z_max": "symmetry", "faces": [1,2,3,4,5,6],
        "type": "dirichlet", "apply_to": ["pressure", "velocity"],
        "pressure": 100.0, "velocity": [0.0, 0.0, 0.0], "no_slip": True
    },
    "geometry_definition": {
        "geometry_mask_flat": [1, 1, 0, 0, 1, 1],
        "geometry_mask_shape": [3, 2, 1],
        "mask_encoding": { "fluid": 1, "solid": 0 },
        "flattening_order": "x-major"
    }
}

@pytest.fixture
def input_with_mask():
    return MASKED_INPUT.copy()

@pytest.fixture
def reflex_config():
    return load_reflex_config("config/reflex_debug_config.yaml")

def test_snapshot_count_matches_interval(input_with_mask, reflex_config):
    snaps = generate_snapshots(input_with_mask, "mask_test", config=reflex_config)
    assert len(snaps) == 6

def test_snapshot_includes_all_required_keys(input_with_mask, reflex_config):
    snaps = generate_snapshots(input_with_mask, "key_check", config=reflex_config)
    required_keys = {
        "step_index", "grid", "max_velocity", "max_divergence", "global_cfl",
        "overflow_detected", "damping_enabled", "adjusted_time_step", "projection_passes",
        "ghost_influence_count", "fluid_cells_modified_by_ghost", "ghost_registry",
        "pressure_mutated", "velocity_projected", "projection_skipped", "divergence_zero",
        "pressure_solver_invoked", "post_projection_divergence"
    }
    for _, snap in snaps:
        assert required_keys.issubset(snap.keys())

def test_metrics_have_correct_types(input_with_mask, reflex_config):
    snaps = generate_snapshots(input_with_mask, "type_check", config=reflex_config)
    for _, snap in snaps:
        assert isinstance(snap["max_velocity"], float)
        assert isinstance(snap["max_divergence"], float)
        assert isinstance(snap["post_projection_divergence"], float)
        assert isinstance(snap["global_cfl"], float)
        assert isinstance(snap["overflow_detected"], bool)
        assert isinstance(snap["damping_enabled"], bool)
        assert isinstance(snap["adjusted_time_step"], float)
        assert isinstance(snap["projection_passes"], int)
        assert isinstance(snap["ghost_influence_count"], int)
        assert isinstance(snap["fluid_cells_modified_by_ghost"], int)
        assert isinstance(snap["ghost_registry"], dict)
        assert isinstance(snap["pressure_solver_invoked"], bool)
        assert isinstance(snap["pressure_mutated"], bool)

def test_projection_flag_consistency(input_with_mask, reflex_config):
    snaps = generate_snapshots(input_with_mask, "projection_consistency", config=reflex_config)
    for _, snap in snaps:
        if snap["divergence_zero"] or snap["projection_passes"] == 0:
            assert snap["projection_skipped"]
        else:
            assert not snap["projection_skipped"]
        if not snap["projection_skipped"]:
            assert snap["pressure_solver_invoked"]

def test_pressure_mutation_causality(input_with_mask, reflex_config):
    snaps = generate_snapshots(input_with_mask, "pressure_trace", config=reflex_config)
    for _, snap in snaps:
        if snap["pressure_mutated"]:
            assert snap["pressure_solver_invoked"]

def test_snapshot_grid_size_matches_geometry(input_with_mask, reflex_config):
    nx, ny, nz = (input_with_mask["domain_definition"][key] for key in ("nx", "ny", "nz"))
    expected_size = nx * ny * nz
    snaps = generate_snapshots(input_with_mask, "grid_size_check", config=reflex_config)
    for _, snap in snaps:
        assert len(snap["grid"]) == expected_size

def test_snapshot_nulling_on_nonfluid_cells(input_with_mask, reflex_config):
    snaps = generate_snapshots(input_with_mask, "nulling_check", config=reflex_config)
    for _, snap in snaps:
        for cell in snap["grid"]:
            if not cell["fluid_mask"]:
                assert cell["velocity"] is None
                assert cell["pressure"] is None

def test_step_index_formatting_logic():
    steps = [f"{i:04d}" for i in [0, 1, 12, 123, 1234]]
    assert steps == ["0000", "0001", "0012", "0123", "1234"]

def test_zero_output_interval_fallback(input_with_mask, reflex_config):
    input_with_mask["simulation_parameters"]["output_interval"] = 0
    snaps = generate_snapshots(input_with_mask, "interval_fallback", config=reflex_config)
    assert len(snaps) > 0

def test_summary_file_written_to_disk(input_with_mask, reflex_config):
    path = os.path.join("data/testing-input-output/navier_stokes_output/step_summary.txt")
    if os.path.exists(path): os.remove(path)
    generate_snapshots(input_with_mask, "summary_test", config=reflex_config)
    assert os.path.exists(path)
    with open(path) as f:
        lines = f.readlines()
    assert any("Step" in line and "Summary" in line for line in lines)

def test_masked_grid_initialization_matches_expected(input_with_mask):
    grid = generate_grid_with_mask(
        input_with_mask["domain_definition"],
        input_with_mask["initial_conditions"],
        input_with_mask["geometry_definition"]
    )
    fluid = sum(1 for cell in grid if cell.fluid_mask)
    solid = sum(1 for cell in grid if not cell.fluid_mask)
    assert fluid == 4
    assert solid == 2

def test_velocity_magnitude_consistency(input_with_mask):
    grid = generate_grid_with_mask(
        input_with_mask["domain_definition"],
        input_with_mask["initial_conditions"],
        input_with_mask["geometry_definition"]
    )
    expected = input_with_mask["initial_conditions"]["initial_velocity"]
    expected_mag = sum(v**2 for v in expected)**0.5
    actual_mag = compute_max_velocity(grid)
    assert round(actual_mag, 5) == round(expected_mag, 5)

def test_influence_flags_log_exported_and_matches_snapshot(input_with_mask, reflex_config):
    path = os.path.join("data/testing-input-output/navier_stokes_output/influence_flags_log.json")
    if os.path.exists(path): os.remove(path)
    snaps = generate_snapshots(input_with_mask, "influence_check", config=reflex_config)
    assert os.path.exists(path)
    with open(path) as f:
        log = json.load(f)
    for entry, (_, snap) in zip(log, snaps):
        assert entry["step_index"] == snap["step_index"]
        assert entry["influenced_cell_count"] == snap["fluid_cells_modified_by_ghost"]

def test_mutation_pathways_log_exported(input_with_mask, reflex_config):
    path = os.path.join("data/testing-input-output/navier_stokes_output/mutation_pathways_log.json")
    if os.path.exists(path): os.remove(path)
    generate_snapshots(input_with_mask, "mutation_export_check", config=reflex_config)
    assert os.path.exists(path)
    with open(path) as f:
        log = json.load(f)
    for entry in log:
        assert "step_index" in entry
        assert isinstance(entry.get("pressure_mutated"), bool)
        assert isinstance(entry.get("triggered_by"), list)



