# tests/test_scheduler_reflex.py

import os
import json
import pytest

SNAPSHOT_ROOT = "data/testing-input-output/navier_stokes_output"

# Reflex expectations keyed by snapshot filename prefix
SCENARIOS = {
    "cfl_spike": {
        "expect_damping_enabled": True,
        "expect_overflow_detected": False,
        "min_projection_passes": 1
    },
    "projection_overload": {
        "expect_damping_enabled": False,
        "expect_overflow_detected": False,
        "min_projection_passes": 2
    },
    "velocity_burst": {
        "expect_damping_enabled": True,
        "expect_overflow_detected": True,
        "min_projection_passes": 2
    },
    "stable_flow": {
        "expect_damping_enabled": False,
        "expect_overflow_detected": False,
        "min_projection_passes": 1
    }
}

def discover_snapshot_files():
    """Find all step_0000 snapshots in the flat output directory"""
    return [
        filename for filename in os.listdir(SNAPSHOT_ROOT)
        if filename.endswith("_step_0000.json")
    ]

@pytest.mark.parametrize("snapshot_file", discover_snapshot_files())
def test_reflex_response_and_flags(snapshot_file):
    scenario_prefix = snapshot_file.replace("_step_0000.json", "")
    if scenario_prefix not in SCENARIOS:
        pytest.skip(f"⚠️ No reflex expectations configured for {snapshot_file}")

    path = os.path.join(SNAPSHOT_ROOT, snapshot_file)
    with open(path) as f:
        snap = json.load(f)

    expected = SCENARIOS[scenario_prefix]

    # Required reflex fields
    for key in ["damping_enabled", "overflow_detected", "projection_passes"]:
        assert key in snap, f"❌ Missing key '{key}' in {snapshot_file}"

    # Damping check
    assert snap["damping_enabled"] == expected["expect_damping_enabled"], (
        f"⚠️ Damping mismatch in {snapshot_file}: expected {expected['expect_damping_enabled']}, got {snap['damping_enabled']}"
    )

    # Overflow check
    assert snap["overflow_detected"] == expected["expect_overflow_detected"], (
        f"⚠️ Overflow mismatch in {snapshot_file}: expected {expected['expect_overflow_detected']}, got {snap['overflow_detected']}"
    )

    # Projection depth check
    actual_passes = snap["projection_passes"]
    assert actual_passes >= expected["min_projection_passes"], (
        f"⚠️ Projection pass count too low in {snapshot_file}: expected ≥ {expected['min_projection_passes']}, got {actual_passes}"
    )



