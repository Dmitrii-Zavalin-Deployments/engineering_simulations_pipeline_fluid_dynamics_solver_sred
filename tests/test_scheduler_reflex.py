# tests/test_scheduler_reflex.py

import os
import json
import pytest

SNAPSHOT_ROOT = "data/testing-input-output/navier_stokes_output"

SCENARIOS = {
    "fluid_simulation_input": {
        "expect_damping_enabled": False,
        "expect_overflow_detected": False,
        "expect_pressure_mutated": True,
        "min_projection_passes": 1,
        "max_velocity_limit": 1.5,
        "max_divergence_limit": 0.2,
        "global_cfl_limit": 1.0
    }
}

def discover_snapshot_files():
    if not os.path.isdir(SNAPSHOT_ROOT):
        return []
    return [
        filename for filename in os.listdir(SNAPSHOT_ROOT)
        if filename.endswith("_step_0000.json")
    ]

SNAPSHOT_FILES = discover_snapshot_files()
skip_reason = "❌ Snapshot files not found — run simulation before testing."

@pytest.mark.parametrize("snapshot_file", SNAPSHOT_FILES or ["_placeholder"])
def test_reflex_response_and_flags(snapshot_file):
    if snapshot_file == "_placeholder":
        pytest.skip(skip_reason)

    scenario_prefix = snapshot_file.replace("_step_0000.json", "")
    if scenario_prefix not in SCENARIOS:
        pytest.skip(f"⚠️ No reflex expectations configured for {snapshot_file}")

    path = os.path.join(SNAPSHOT_ROOT, snapshot_file)
    assert os.path.isfile(path), f"❌ Missing snapshot file: {path}"

    with open(path) as f:
        snap = json.load(f)

    expected = SCENARIOS[scenario_prefix]

    required_keys = [
        "damping_enabled", "overflow_detected", "projection_passes",
        "pressure_mutated", "pressure_solver_invoked",
        "post_projection_divergence", "mutated_cells"
    ]
    for key in required_keys:
        assert key in snap, f"❌ Missing key '{key}' in {snapshot_file}"

    assert isinstance(snap["damping_enabled"], bool)
    assert isinstance(snap["overflow_detected"], bool)
    assert isinstance(snap["pressure_mutated"], bool)
    assert isinstance(snap["pressure_solver_invoked"], bool)
    assert isinstance(snap["projection_passes"], int)
    assert isinstance(snap["post_projection_divergence"], (int, float))
    assert isinstance(snap["mutated_cells"], list)

    assert snap["damping_enabled"] == expected["expect_damping_enabled"], (
        f"⚠️ Damping mismatch in {snapshot_file}"
    )
    assert snap["overflow_detected"] == expected["expect_overflow_detected"], (
        f"⚠️ Overflow mismatch in {snapshot_file}"
    )
    assert snap["pressure_mutated"] == expected["expect_pressure_mutated"], (
        f"⚠️ Pressure mutation mismatch in {snapshot_file}"
    )
    assert snap["projection_passes"] >= expected["min_projection_passes"], (
        f"⚠️ Projection passes too low in {snapshot_file}"
    )

    if "max_velocity_limit" in expected:
        assert snap["max_velocity"] <= expected["max_velocity_limit"], (
            f"⚠️ Velocity exceeds limit in {snapshot_file}"
        )

    if "max_divergence_limit" in expected:
        assert snap["max_divergence"] <= expected["max_divergence_limit"], (
            f"⚠️ Divergence exceeds limit in {snapshot_file}"
        )

    if "global_cfl_limit" in expected:
        assert snap["global_cfl"] <= expected["global_cfl_limit"], (
            f"⚠️ CFL exceeds limit in {snapshot_file}"
        )



