# tests/validator/test_trace_resilience.py
# ✅ Reflex Snapshot Integrity Test

import pytest

def load_snapshot(path: str) -> dict:
    import json
    with open(path, 'r') as f:
        return json.load(f)

def validate_trace(snapshot: dict):
    """
    Asserts core simulation integrity metrics from snapshot.
    Raises AssertionError if any condition fails.
    """
    step = snapshot.get("step_index", "unknown")

    pre_div = snapshot.get("pre_divergence", None)
    post_div = snapshot.get("post_divergence", None)
    mutation_count = snapshot.get("pressure_mutation_count", None)
    ghost_adjacent_count = snapshot.get("ghost_adjacent_count", None)

    assert isinstance(pre_div, float), f"[step {step}] Missing pre_divergence"
    assert isinstance(post_div, float), f"[step {step}] Missing post_divergence"
    assert post_div < pre_div + 1e-6, f"[step {step}] Divergence did not collapse: pre={pre_div}, post={post_div}"
    assert mutation_count and mutation_count > 0, f"[step {step}] No pressure mutations recorded"
    assert ghost_adjacent_count and ghost_adjacent_count > 0, f"[step {step}] No ghost–fluid adjacency detected"

@pytest.mark.parametrize("snapshot_path", [
    "data/testing-input-output/navier_stokes_output/step_0008_snapshot.json",
    "data/testing-input-output/navier_stokes_output/step_0010_snapshot.json"
])
def test_snapshot_resilience(snapshot_path):
    snapshot = load_snapshot(snapshot_path)
    validate_trace(snapshot)



