# tests/test_stability_utils.py

import os
import json
import re
import pytest
from collections import defaultdict

SNAPSHOT_ROOT = "data/testing-input-output/navier_stokes_output"
skip_reason = "❌ Snapshot files not found — run simulation before testing."

def discover_snapshot_series():
    """Groups snapshot files by scenario prefix and sorts them by step index."""
    if not os.path.isdir(SNAPSHOT_ROOT):
        return {}

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

SCENARIO_SERIES = discover_snapshot_series()

@pytest.mark.parametrize("scenario_prefix,snapshot_files", SCENARIO_SERIES.items() or [("_placeholder", [])])
def test_volatility_metrics_across_timesteps(scenario_prefix, snapshot_files):
    if scenario_prefix == "_placeholder":
        pytest.skip(skip_reason)

    divergence_values = []
    overflow_flags = []

    for filename in snapshot_files:
        path = os.path.join(SNAPSHOT_ROOT, filename)
        if not os.path.isfile(path):
            pytest.skip(f"❌ Missing snapshot file: {filename}")
        with open(path) as f:
            snap = json.load(f)

        for key in ["max_divergence", "overflow_detected"]:
            assert key in snap, f"❌ Missing '{key}' in {filename}"

        assert isinstance(snap["max_divergence"], (int, float)) and snap["max_divergence"] >= 0, f"❌ Invalid divergence in {filename}"
        assert isinstance(snap["overflow_detected"], bool), f"❌ overflow_detected must be boolean in {filename}"

        divergence_values.append(snap["max_divergence"])
        overflow_flags.append(snap["overflow_detected"])

    # Behavioral assertions
    if "velocity_burst" in scenario_prefix:
        assert any(overflow_flags), f"⚠️ Overflow expected in {scenario_prefix}"

    if "stable_flow" in scenario_prefix:
        assert not any(overflow_flags), f"⚠️ No overflow expected in {scenario_prefix}"



