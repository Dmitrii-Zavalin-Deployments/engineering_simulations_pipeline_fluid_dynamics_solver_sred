# tests/test_snapshot_t0_diagnostics.py
# 🧪 Reflex and Ghost Diagnostic Integrity — Snapshot t=0

from tests.snapshot_t0_shared import snapshot

def test_basic_reflex_flags(snapshot):
    assert isinstance(snapshot.get("max_divergence"), (int, float)), "❌ max_divergence must be numeric"
    assert isinstance(snapshot.get("projection_passes"), int), "❌ projection_passes must be an integer"
    assert "divergence_zero" in snapshot, "❌ Missing divergence_zero flag"
    assert "projection_skipped" in snapshot, "❌ Missing projection_skipped flag"

def test_ghost_diagnostic_fields_present(snapshot):
    ghost_diag = snapshot.get("ghost_diagnostics", {})
    assert isinstance(ghost_diag, dict), "❌ ghost_diagnostics must be a dictionary"
    for key in ["total", "per_face", "pressure_overrides", "no_slip_enforced"]:
        assert key in ghost_diag, f"❌ Missing ghost diagnostic field: {key}"



