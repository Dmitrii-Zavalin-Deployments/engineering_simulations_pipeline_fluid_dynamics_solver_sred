# tests/adaptive/test_timestep_controller.py
# ✅ Unit tests for timestep_controller.py

import os
import json
import pytest
from src.adaptive.timestep_controller import (
    compute_mutation_density,
    compute_mutation_frequency,
    suggest_timestep
)

@pytest.fixture
def delta_map_high_mutation(tmp_path):
    path = tmp_path / "pressure_delta.json"
    data = {
        "(0.0, 0.0, 0.0)": {"delta": 2.0},
        "(1.0, 0.0, 0.0)": {"delta": 0.0},
        "(2.0, 0.0, 0.0)": {"delta": 1.0},
        "(3.0, 0.0, 0.0)": {"delta": 0.0},
        "(4.0, 0.0, 0.0)": {"delta": 2.5}
    }
    with open(path, "w") as f:
        json.dump(data, f)
    return path

@pytest.fixture
def delta_map_low_mutation(tmp_path):
    path = tmp_path / "pressure_delta_low.json"
    data = {
        "(0.0, 0.0, 0.0)": {"delta": 0.0},
        "(1.0, 0.0, 0.0)": {"delta": 0.0},
        "(2.0, 0.0, 0.0)": {"delta": 0.0}
    }
    with open(path, "w") as f:
        json.dump(data, f)
    return path

@pytest.fixture
def mutation_trace_high(tmp_path):
    path = tmp_path / "mutation_trace_high.json"
    data = [
        {"pressure_mutated": True},
        {"pressure_mutated": True},
        {"pressure_mutated": True},
        {"pressure_mutated": False},
        {"pressure_mutated": True}
    ]
    with open(path, "w") as f:
        json.dump(data, f)
    return path

@pytest.fixture
def mutation_trace_low(tmp_path):
    path = tmp_path / "mutation_trace_low.json"
    data = [
        {"pressure_mutated": False},
        {"pressure_mutated": False},
        {"pressure_mutated": False}
    ]
    with open(path, "w") as f:
        json.dump(data, f)
    return path

def test_compute_mutation_density_high(delta_map_high_mutation):
    with open(delta_map_high_mutation, "r") as f:
        data = json.load(f)
    density = compute_mutation_density(data)
    assert 0.3 <= density <= 0.8  # 3 of 5 cells mutated

def test_compute_mutation_density_low(delta_map_low_mutation):
    with open(delta_map_low_mutation, "r") as f:
        data = json.load(f)
    density = compute_mutation_density(data)
    assert density == 0.0

def test_compute_mutation_frequency_high(mutation_trace_high):
    with open(mutation_trace_high, "r") as f:
        trace = json.load(f)
    freq = compute_mutation_frequency(trace)
    assert freq == 0.8

def test_compute_mutation_frequency_low(mutation_trace_low):
    with open(mutation_trace_low, "r") as f:
        trace = json.load(f)
    freq = compute_mutation_frequency(trace)
    assert freq == 0.0

def test_suggest_timestep_reduce(delta_map_high_mutation, mutation_trace_high):
    dt = suggest_timestep(str(delta_map_high_mutation), str(mutation_trace_high), base_dt=0.02)
    assert dt < 0.02

def test_suggest_timestep_increase(delta_map_low_mutation, mutation_trace_low):
    dt = suggest_timestep(str(delta_map_low_mutation), str(mutation_trace_low), base_dt=0.02)
    assert dt > 0.02

def test_suggest_timestep_stable(delta_map_high_mutation, mutation_trace_low):
    dt = suggest_timestep(str(delta_map_high_mutation), str(mutation_trace_low), base_dt=0.01)
    assert dt == 0.01

def test_suggest_timestep_reflex_score_block(delta_map_high_mutation, mutation_trace_high):
    dt = suggest_timestep(str(delta_map_high_mutation), str(mutation_trace_high), base_dt=0.01, reflex_score=3, min_score=4)
    assert dt == 0.01

def test_suggest_timestep_no_data(tmp_path):
    empty_delta = tmp_path / "empty_delta.json"
    empty_trace = tmp_path / "empty_trace.json"
    empty_delta.write_text("{}")
    empty_trace.write_text("[]")
    dt = suggest_timestep(str(empty_delta), str(empty_trace), base_dt=0.02)
    assert dt > 0.02  # triggers default increase on low activity



