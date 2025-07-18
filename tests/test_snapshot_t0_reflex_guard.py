# tests/test_snapshot_t0_reflex_guard.py
# 🧪 Reflex Mutation Trace and Influence Guard — Snapshot t=0

from tests.snapshot_t0_shared import snapshot

def test_mutation_trace_presence(snapshot):
    mutated_cells = snapshot.get("mutated_cells", [])
    assert isinstance(mutated_cells, list), "❌ mutated_cells must be a list"
    for entry in mutated_cells:
        assert isinstance(entry, list) and len(entry) == 3, f"❌ Invalid mutated coordinate format: {entry}"
        assert all(isinstance(v, (int, float)) for v in entry), f"❌ Non-numeric mutation coordinates: {entry}"

def test_triggered_by_flags(snapshot):
    triggers = snapshot.get("triggered_by", [])
    assert isinstance(triggers, list), "❌ triggered_by must be a list"
    for reason in triggers:
        assert isinstance(reason, str), f"❌ trigger reason must be string: {reason}"

def test_fluid_cells_modified_by_ghost(snapshot):
    modified_count = snapshot.get("fluid_cells_modified_by_ghost", None)
    assert isinstance(modified_count, int), "❌ fluid_cells_modified_by_ghost must be an integer"
    assert modified_count >= 0, "❌ fluid_cells_modified_by_ghost must not be negative"

def test_post_projection_divergence(snapshot):
    divergence = snapshot.get("post_projection_divergence", None)
    assert isinstance(divergence, (int, float)), "❌ post_projection_divergence must be numeric"
    assert divergence >= 0.0, "❌ post_projection_divergence must not be negative"



