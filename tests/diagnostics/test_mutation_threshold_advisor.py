# tests/diagnostics/test_mutation_threshold_advisor.py
# ✅ Validation suite for src/diagnostics/mutation_threshold_advisor.py

import pytest
from src.diagnostics.mutation_threshold_advisor import get_delta_threshold

def test_default_context_behavior(capsys):
    threshold = get_delta_threshold(cell={}, context={})
    assert threshold == pytest.approx(1e-8)
    assert "[THRESHOLD] Computed delta threshold" in capsys.readouterr().out

@pytest.mark.parametrize("resolution,expected", [
    ("high", 1e-9),
    ("low", 5e-8),
    ("normal", 1e-8),
    ("unknown", 1e-8),
])
def test_resolution_scaling(resolution, expected):
    threshold = get_delta_threshold(cell={}, context={"resolution": resolution})
    assert threshold == pytest.approx(expected)

@pytest.mark.parametrize("divergence,expected_factor", [
    (0.0, 2.0),
    (0.005, 2.0),
    (0.02, 1.0),
    (0.2, 0.5),
])
def test_divergence_scaling(divergence, expected_factor):
    base = 1e-8
    context = {"divergence": divergence}
    threshold = get_delta_threshold(cell={}, context=context)
    assert threshold == pytest.approx(base * expected_factor, rel=1e-6)

@pytest.mark.parametrize("time_step,expected_factor", [
    (0.005, 2.0),
    (0.05, 1.0),
    (0.25, 0.5),
])
def test_time_step_scaling(time_step, expected_factor):
    base = 1e-8
    context = {"time_step": time_step}
    threshold = get_delta_threshold(cell={}, context=context)
    assert threshold == pytest.approx(base * expected_factor, rel=1e-6)

@pytest.mark.parametrize("reflex_score,expected_factor", [
    (0.1, 0.5),
    (0.5, 1.0),
    (0.9, 1.5),
])
def test_reflex_score_scaling(reflex_score, expected_factor):
    base = 1e-8
    context = {"reflex_score": reflex_score}
    threshold = get_delta_threshold(cell={}, context=context)
    assert threshold == pytest.approx(base * expected_factor, rel=1e-6)

@pytest.mark.parametrize("mutation_density,expected_factor", [
    (0.01, 1.25),
    (0.1, 1.0),
    (0.4, 0.75),
])
def test_mutation_density_scaling(mutation_density, expected_factor):
    base = 1e-8
    context = {"mutation_density": mutation_density}
    threshold = get_delta_threshold(cell={}, context=context)
    assert threshold == pytest.approx(base * expected_factor, rel=1e-6)

def test_combined_context_scaling():
    context = {
        "resolution": "high",           # ×0.1 → 1e-9
        "divergence": 0.0,              # ×2   → 2e-9
        "time_step": 0.25,              # ×0.5 → 1e-9
        "reflex_score": 0.9,            # ×1.5 → 1.5e-9
        "mutation_density": 0.01        # ×1.25 → 1.875e-9
    }
    threshold = get_delta_threshold(cell={}, context=context)
    assert threshold == pytest.approx(1.875e-9, rel=1e-6)

def test_threshold_never_below_minimum():
    context = {
        "resolution": "high",
        "divergence": 0.2,
        "time_step": 0.25,
        "reflex_score": 0.0,
        "mutation_density": 0.9
    }
    threshold = get_delta_threshold(cell={}, context=context)
    assert threshold >= 1e-15



