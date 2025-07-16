# tests/test_stability_pipeline.py

import os
import json
import pytest
import re
from collections import defaultdict

SNAPSHOT_ROOT = "data/testing-input-output/navier_stokes_output"
skip_reason = "❌ Snapshot files not found — run simulation before testing."

def discover_snapshot_series():
    """
    Groups snapshot files by scenario prefix and sorts them by step index.
    Example result: {"fluid_simulation_input": [step_0000, step_0001, ...], ...}
    """
    if not os.path.isdir(SNAPSHOT_ROOT):
        return {}

    snapshots = defaultdict(list)
    pattern = re.compile(r"^(.*?)_step_(\d{4})\.json$")
    for filename in os.listdir(SNAPSHOT_ROOT):
        match = pattern.match(filename)
        if match:
            prefix, step = match.groups()
            snapshots[prefix].append((int(step), filename))

    sorted_series = {
        prefix: [fname for _, fname in sorted(steps)]
        for prefix, steps in snapshots.items()
    }
    return sorted_series

SCENARIO_SERIES = discover_snapshot_series()

@pytest.mark.parametrize("scenario_prefix,snapshot_files", SCENARIO_SERIES.items() or [("_placeholder", [])])
def test_stability_over_time(scenario_prefix, snapshot_files):
    if scenario_prefix == "_placeholder":
        pytest.skip(skip_reason)

    divergence_trend = []
    velocity_trend = []
    projection_passes = []
    damping_flags = []
    overflow_flags = []

    for filename in snapshot_files:
        path = os.path.join(SNAPSHOT_ROOT, filename)
        if not os.path.isfile(path):
            pytest.skip(f"❌ Missing snapshot file: {filename}")
        with open(path) as f:
            snap = json.load(f)

        for key in ["max_divergence", "max_velocity", "projection_passes", "damping_enabled", "overflow_detected"]:
            assert key in snap, f"❌ Missing '{key}' in {filename}"

        divergence_trend.append(snap["max_divergence"])
        velocity_trend.append(snap["max_velocity"])
        projection_passes.append(snap["projection_passes"])
        damping_flags.append(snap["damping_enabled"])
        overflow_flags.append(snap["overflow_detected"])

    assert all(d >= 0 for d in divergence_trend), f"❌ Negative divergence in {scenario_prefix}"
    assert all(v >= 0 for v in velocity_trend), f"❌ Negative velocity in {scenario_prefix}"
    assert any(projection_passes), f"⚠️ No projection passes logged in {scenario_prefix}"
    assert all(isinstance(flag, bool) for flag in damping_flags), f"⚠️ Damping flags must be boolean in {scenario_prefix}"
    assert all(isinstance(flag, bool) for flag in overflow_flags), f"⚠️ Overflow flags must be boolean in {scenario_prefix}"

    # Example behavioral expectations (optional)
    if "cfl_spike" in scenario_prefix:
        assert damping_flags.count(True) >= 2, f"⚠️ Damping should trigger ≥ 2 times under CFL spike in {scenario_prefix}"

    if "projection_overload" in scenario_prefix:
        assert max(projection_passes) >= 2, f"⚠️ Expected projection escalation in {scenario_prefix}"

    if "velocity_burst" in scenario_prefix:
        assert any(overflow_flags), f"⚠️ Expected overflow in {scenario_prefix}"
        assert any(v > 2.0 for v in velocity_trend), f"⚠️ Velocity burst threshold not reached in {scenario_prefix}"



