# tests/diagnostics/test_mutation_threshold_advisor.py

import pytest
from src.diagnostics.mutation_threshold_advisor import get_delta_threshold

class DummyCell:
    def __init__(self, x=0.0, y=0.0, z=0.0, velocity=None):
        self.x = x
        self.y = y
        self.z = z
        self.velocity = velocity or [0.0, 0.0, 0.0]

def test_default_threshold_is_base():
    cell = DummyCell()
    context = {}
    threshold = get_delta_threshold(cell, context)
    assert threshold == pytest.approx(1e-8)

def test_resolution_scaling():
    cell = DummyCell()
    high = get_delta_threshold(cell, {"resolution": "high"})
    low = get_delta_threshold(cell, {"resolution": "low"})
    normal = get_delta_threshold(cell, {"resolution": "normal"})
    assert high == pytest.approx(1e-9)
    assert low == pytest.approx(5e-8)
    assert normal == pytest.approx(1e-8)

def test_divergence_scaling():
    cell = DummyCell()
    low_div = get_delta_threshold(cell, {"divergence": 0.001})
    high_div = get_delta_threshold(cell, {"divergence": 0.2})
    assert low_div > 1e-8
    assert high_div < 1e-8

def test_time_step_scaling():
    cell = DummyCell()
    small_dt = get_delta_threshold(cell, {"time_step": 0.005})
    large_dt = get_delta_threshold(cell, {"time_step": 0.25})
    assert small_dt > 1e-8
    assert large_dt < 1e-8

def test_combined_context_adjustment():
    cell = DummyCell()
    context = {
        "resolution": "high",
        "divergence": 0.001,
        "time_step": 0.005
    }
    threshold = get_delta_threshold(cell, context)
    # Expected: 1e-8 * 0.1 (high) * 2 (low div) * 2 (small dt) = 4e-8
    assert threshold == pytest.approx(4e-8)

def test_clamping_to_machine_precision():
    cell = DummyCell()
    context = {
        "resolution": "high",
        "divergence": 0.0,
        "time_step": 0.0
    }
    threshold = get_delta_threshold(cell, context)
    assert threshold >= 1e-15



