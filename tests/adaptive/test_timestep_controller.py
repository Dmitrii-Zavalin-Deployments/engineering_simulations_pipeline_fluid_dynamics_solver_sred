# tests/adaptive/test_timestep_controller.py
# 🧪 Unit tests for src/adaptive/timestep_controller.py

import os
import json
import tempfile
import pytest
from src.adaptive import timestep_controller

def make_pressure_delta_file(data: dict) -> str:
    with tempfile.NamedTemporaryFile("w+", delete=False) as f:
        json.dump(data, f)
        return f.name

def make_mutation_trace_file(trace: list) -> str:
    with tempfile.NamedTemporaryFile("w+", delete=False) as f:
        json.dump(trace, f)
        return f.name

def test_high_mutation_density_and_frequency_reduces_dt():
    delta_data = {
        "cell1": {"delta": 1.0},
        "cell2": {"delta": 0.5},
        "cell3": {"delta": 0.4},
        "cell4": {"delta": 0.3},
        "cell5": {"delta": 0.1},
    }
    trace_data = [{"pressure_mutated": True} for _ in range(5)]

    delta_path = make_pressure_delta_file(delta_data)
    trace_path = make_mutation_trace_file(trace_data)

    dt = timestep_controller.suggest_timestep(delta_path, trace_path, base_dt=0.02, reflex_score=5)
    os.unlink(delta_path)
    os.unlink(trace_path)

    assert dt == 0.01  # reduced to 50%

def test_low_mutation_density_and_frequency_increases_dt():
    delta_data = {
        "cell1": {"delta": 0.0},
        "cell2": {"delta": 0.0},
        "cell3": {"delta": 0.0},
        "cell4": {"delta": 0.0},
        "cell5": {"delta": 0.0},  # ✅ Ensures mutation_density == 0.0
    }
    trace_data = [{"pressure_mutated": False} for _ in range(5)]

    delta_path = make_pressure_delta_file(delta_data)
    trace_path = make_mutation_trace_file(trace_data)

    dt = timestep_controller.suggest_timestep(delta_path, trace_path, base_dt=0.02, reflex_score=5)
    os.unlink(delta_path)
    os.unlink(trace_path)

    assert dt == pytest.approx(0.03)  # ✅ 0.02 * 1.5

def test_medium_mutation_conditions_preserve_dt():
    delta_data = {
        "cell1": {"delta": 0.1},
        "cell2": {"delta": 0.0},
        "cell3": {"delta": 0.0},
        "cell4": {"delta": 0.0},
        "cell5": {"delta": 0.0},
    }
    trace_data = [{"pressure_mutated": True}, {"pressure_mutated": False}] * 3

    delta_path = make_pressure_delta_file(delta_data)
    trace_path = make_mutation_trace_file(trace_data)

    dt = timestep_controller.suggest_timestep(delta_path, trace_path, base_dt=0.02, reflex_score=5)
    os.unlink(delta_path)
    os.unlink(trace_path)

    assert dt == pytest.approx(0.02)

def test_reflex_score_below_threshold_preserves_dt():
    delta_data = {
        "cell1": {"delta": 1.0},
        "cell2": {"delta": 1.0},
        "cell3": {"delta": 1.0},
        "cell4": {"delta": 1.0},
    }
    trace_data = [{"pressure_mutated": True} for _ in range(5)]

    delta_path = make_pressure_delta_file(delta_data)
    trace_path = make_mutation_trace_file(trace_data)

    dt = timestep_controller.suggest_timestep(delta_path, trace_path, base_dt=0.02, reflex_score=2, min_score=4)
    os.unlink(delta_path)
    os.unlink(trace_path)

    assert dt == 0.02  # base_dt preserved

def test_empty_inputs_return_base_dt():
    delta_path = make_pressure_delta_file({})
    trace_path = make_mutation_trace_file([])

    dt = timestep_controller.suggest_timestep(delta_path, trace_path, base_dt=0.015, reflex_score=5)
    os.unlink(delta_path)
    os.unlink(trace_path)

    assert dt == pytest.approx(0.0225)  # ✅ 0.015 * 1.5



