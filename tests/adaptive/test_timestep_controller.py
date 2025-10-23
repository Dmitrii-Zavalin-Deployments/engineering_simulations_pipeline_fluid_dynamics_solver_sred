# tests/adaptive/test_timestep_controller.py
# ✅ Validation suite for src/adaptive/timestep_controller.py

import json
import tempfile
import pytest

from src.adaptive.timestep_controller import (
    load_pressure_delta,
    load_mutation_trace,
    compute_mutation_density,
    compute_mutation_frequency,
    suggest_timestep
)

@pytest.fixture
def valid_pressure_delta(tmp_path):
    path = tmp_path / "delta_map.json"
    data = {
        "(0.0, 0.0, 0.0)": {"delta": 0.1},
        "(1.0, 0.0, 0.0)": {"delta": 0.0},
        "(2.0, 0.0, 0.0)": {"delta": 0.2},
        "(3.0, 0.0, 0.0)": {"delta": 0.0},
        "(4.0, 0.0, 0.0)": {"delta": 0.3}
    }
    with open(path, "w") as f:
        json.dump(data, f)
    return str(path)

@pytest.fixture
def high_mutation_trace(tmp_path):
    path = tmp_path / "trace.json"
    trace = [{"pressure_mutated": True} for _ in range(5)]
    with open(path, "w") as f:
        json.dump(trace, f)
    return str(path)

@pytest.fixture
def low_mutation_trace(tmp_path):
    path = tmp_path / "trace.json"
    trace = [{"pressure_mutated": False} for _ in range(5)]
    with open(path, "w") as f:
        json.dump(trace, f)
    return str(path)

@pytest.fixture
def mixed_mutation_trace(tmp_path):
    path = tmp_path / "trace.json"
    trace = [
        {"pressure_mutated": True},
        {"pressure_mutated": False},
        {"pressure_mutated": True},
        {"pressure_mutated": False},
        {"pressure_mutated": False}
    ]
    with open(path, "w") as f:
        json.dump(trace, f)
    return str(path)

def test_load_pressure_delta_valid(valid_pressure_delta):
    result = load_pressure_delta(valid_pressure_delta)
    assert isinstance(result, dict)
    assert len(result) == 5

def test_load_pressure_delta_invalid(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("not valid json")
    result = load_pressure_delta(str(broken))
    assert result == {}

def test_load_mutation_trace_valid(high_mutation_trace):
    result = load_mutation_trace(high_mutation_trace)
    assert isinstance(result, list)
    assert len(result) == 5
    assert all(step["pressure_mutated"] for step in result)

def test_load_mutation_trace_invalid(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("not valid json")
    result = load_mutation_trace(str(broken))
    assert result == []

def test_compute_mutation_density_valid():
    delta_map = {
        "a": {"delta": 0.1},
        "b": {"delta": 0.0},
        "c": {"delta": 0.2}
    }
    density = compute_mutation_density(delta_map)
    assert density == pytest.approx(2 / 3)

def test_compute_mutation_density_empty():
    assert compute_mutation_density({}) == 0.0

def test_compute_mutation_frequency_high():
    trace = [{"pressure_mutated": True} for _ in range(5)]
    freq = compute_mutation_frequency(trace)
    assert freq == 1.0

def test_compute_mutation_frequency_low():
    trace = [{"pressure_mutated": False} for _ in range(5)]
    freq = compute_mutation_frequency(trace)
    assert freq == 0.0

def test_compute_mutation_frequency_mixed():
    trace = [{"pressure_mutated": True}, {"pressure_mutated": False}]
    freq = compute_mutation_frequency(trace, recent_steps=2)
    assert freq == 0.5

def test_suggest_timestep_reflex_score_block(valid_pressure_delta, high_mutation_trace):
    result = suggest_timestep(
        pressure_delta_path=valid_pressure_delta,
        mutation_trace_path=high_mutation_trace,
        base_dt=0.01,
        reflex_score=2,
        min_score=4
    )
    assert result == 0.01  # score too low → no change

def test_suggest_timestep_high_mutation(valid_pressure_delta, high_mutation_trace):
    result = suggest_timestep(
        pressure_delta_path=valid_pressure_delta,
        mutation_trace_path=high_mutation_trace,
        base_dt=0.01,
        reflex_score=5
    )
    assert result == pytest.approx(0.005)

def test_suggest_timestep_low_mutation(valid_pressure_delta, low_mutation_trace):
    result = suggest_timestep(
        pressure_delta_path=valid_pressure_delta,
        mutation_trace_path=low_mutation_trace,
        base_dt=0.01,
        reflex_score=5
    )
    assert result == pytest.approx(0.015)

def test_suggest_timestep_mixed_mutation(valid_pressure_delta, mixed_mutation_trace):
    result = suggest_timestep(
        pressure_delta_path=valid_pressure_delta,
        mutation_trace_path=mixed_mutation_trace,
        base_dt=0.01,
        reflex_score=5
    )
    assert result == 0.01  # no change



