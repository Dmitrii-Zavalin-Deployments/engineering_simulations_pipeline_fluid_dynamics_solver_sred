# tests/test_stability_utils.py

import os
import json
import re
import pytest
from collections import defaultdict

SNAPSHOT_ROOT = "data/testing-input-output/navier_stokes_output"

def discover_snapshot_series():
    """
    Groups snapshot files by scenario prefix and sorts them by step index.
    Returns: dict { "fluid_simulation_input": [filename, filename, ...], ... }
    """
    snapshots = defaultdict(list)
    pattern = re.compile(r"^(.*?)_step_(\d{4})\.json$")
    for fname in os.listdir(SNAPSHOT_ROOT):
        match = pattern.match(fname)
        if match:
            prefix, step = match.groups()
            snapshots[prefix].append((int(step), fname))
    return {
        prefix: [fname for _, fname in sorted(files)]
        for prefix, files in snapshots.items()
    }

@pytest.mark.parametrize("scenario_prefix,snapshot_files", discover_snapshot_series().items())
def test_volatility_metrics_across_timesteps(scenario_prefix, snapshot_files):
    divergence_slope = []
    volatility_delta = []
    overflow_flags = []

    for filename in snapshot_files:
        path = os.path.join(SNAPSHOT_ROOT, filename)
        with open(path) as f:
            snap = json.load(f)

        for key in ["max_divergence", "volatility_slope", "volatility_delta", "overflow_detected"]:
            assert key in snap, f"❌ Missing '{key}' in {filename}"

        assert snap["max_divergence"] >= 0, f"❌ Invalid divergence in {filename}"
        assert isinstance(snap["volatility_slope"], str), f"❌ volatility_slope must be string in {filename}"
        assert isinstance(snap["volatility_delta"], (int, float)), f"❌ volatility_delta must be numeric in {filename}"
        assert isinstance(snap["overflow_detected"], bool), f"❌ overflow_detected must be boolean in {filename}"

        divergence_slope.append(snap["volatility_slope"])
        volatility_delta.append(snap["volatility_delta"])
        overflow_flags.append(snap["overflow_detected"])

    # Behavioral assertions (example logic)
    if "velocity_burst" in scenario_prefix:
        assert any(overflow_flags), f"⚠️ Overflow expected in {scenario_prefix}"
        assert any(s == "increasing" for s in divergence_slope), f"⚠️ Divergence slope should increase in {scenario_prefix}"

    if "cfl_spike" in scenario_prefix:
        assert volatility_delta[-1] > volatility_delta[0], f"⚠️ Volatility should grow in {scenario_prefix}"

    if "stable_flow" in scenario_prefix:
        assert not any(overflow_flags), f"⚠️ No overflow expected in {scenario_prefix}"
        assert all(s == "flat" for s in divergence_slope), f"⚠️ Divergence slope should remain flat in {scenario_prefix}"



