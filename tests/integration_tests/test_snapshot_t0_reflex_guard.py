# tests/test_snapshot_t0_reflex_guard.py
# 🧪 Reflex Metadata Validation for t=0 Snapshot

from tests.snapshot_t0_shared import snapshot

def test_mutation_trace_presence(snapshot):
    mutated = snapshot.get("mutated_cells", [])
    assert isinstance(mutated, list), "❌ mutated_cells must be a list"
    for entry in mutated:
        assert isinstance(entry, list) and len(entry) == 3, f"❌ Invalid mutation coordinate: {entry}"
        assert all(isinstance(v, (int, float)) for v in entry), f"❌ Mutation components must be numeric: {entry}"

def test_triggered_by_reasons(snapshot):
    triggers = snapshot.get("triggered_by", [])
    assert isinstance(triggers, list), "❌ triggered_by must be a list"
    for reason in triggers:
        assert isinstance(reason, str), f"❌ trigger value must be string: {reason}"

def test_ghost_cell_modification(snapshot):
    ghost_mods = snapshot.get("fluid_cells_modified_by_ghost", None)
    assert isinstance(ghost_mods, int), "❌ fluid_cells_modified_by_ghost must be integer"
    assert ghost_mods >= 0, "❌ ghost modification count cannot be negative"

def test_post_projection_divergence(snapshot):
    divergence = snapshot.get("post_projection_divergence", None)
    assert isinstance(divergence, (float, int)), "❌ post_projection_divergence must be numeric"
    assert divergence >= 0.0, "❌ post_projection_divergence must be non-negative"



