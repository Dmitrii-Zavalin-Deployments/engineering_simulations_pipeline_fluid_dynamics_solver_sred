import os
import json
import pytest
import subprocess

SCENARIOS = [
    ("stable_flow.json", {"expect_reflex": False}),
    ("cfl_spike.json", {"expect_damping": True}),
    ("projection_overload.json", {"projection_passes": 5}),
    ("velocity_burst.json", {"expect_overflow": True}),
    ("damped_cavity.json", {"max_velocity_expected": 120.0})
]

@pytest.mark.parametrize("scenario_file, expectations", SCENARIOS)
def test_solver_behavior_snapshot(tmp_path, scenario_file, expectations):
    input_path = f"tests/inputs/{scenario_file}"
    output_dir = tmp_path / "solver_output"
    output_dir.mkdir(exist_ok=True)

    # Run main_solver.py with scenario input
    subprocess.run([
        "python", "src/main_solver.py",
        str(input_path),
        str(output_dir)
    ], check=True)

    # Load snapshot
    snapshot_path = output_dir / "navier_stokes_output" / "divergence_snapshot.json"
    assert snapshot_path.is_file(), f"Missing snapshot: {snapshot_path}"
    with open(snapshot_path) as f:
        snap = json.load(f)

    # Validate expectations
    if "expect_reflex" in expectations:
        assert not snap.get("overflow_detected", False)
        assert snap["global_cfl"] < 1.0

    if expectations.get("expect_damping"):
        assert snap["global_cfl"] > 1.0
        assert snap.get("damping_enabled") is True

    if "projection_passes" in expectations:
        assert snap["projection_passes"] >= expectations["projection_passes"]

    if expectations.get("expect_overflow"):
        assert snap.get("overflow_detected") is True

    if "max_velocity_expected" in expectations:
        assert snap["max_velocity"] <= expectations["max_velocity_expected"]



