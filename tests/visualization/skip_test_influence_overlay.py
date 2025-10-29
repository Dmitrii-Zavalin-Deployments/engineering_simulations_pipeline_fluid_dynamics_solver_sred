import os
import pytest
import matplotlib
matplotlib.use("Agg")  # Disable GUI backend for headless testing

from src.visualization.influence_overlay import render_influence_overlay

@pytest.fixture
def influence_log_full():
    return {
        "step_score": 0.9,
        "adjacency_zones": [
            {"x": 1.0, "y": 2.0, "radius": 0.5},
            {"x": 3.0, "y": 4.0, "radius": 0.3}
        ],
        "suppression_zones": [
            {"x": 5.0, "y": 6.0, "radius": 0.4}
        ],
        "boundary_cells": [
            {"x": 0.0, "y": 0.0},
            {"x": 2.0, "y": 2.0}
        ]
    }

@pytest.fixture
def influence_log_minimal():
    return {
        "step_score": 0.9,
        "adjacency_zones": [],
        "suppression_zones": []
    }

@pytest.fixture
def influence_log_low_score():
    return {
        "step_score": 0.5,
        "adjacency_zones": [{"x": 1.0, "y": 2.0, "radius": 0.5}],
        "suppression_zones": [{"x": 3.0, "y": 4.0, "radius": 0.3}]
    }

def test_overlay_renders_full(tmp_path, influence_log_full):
    output_path = tmp_path / "overlay_full.png"
    render_influence_overlay(influence_log_full, str(output_path))
    assert output_path.exists()

def test_overlay_renders_minimal(tmp_path, influence_log_minimal):
    output_path = tmp_path / "overlay_minimal.png"
    render_influence_overlay(influence_log_minimal, str(output_path))
    assert output_path.exists()

def test_overlay_skips_if_score_too_low(tmp_path, influence_log_low_score, capsys):
    output_path = tmp_path / "overlay_skipped.png"
    render_influence_overlay(influence_log_low_score, str(output_path), score_threshold=0.85)
    assert not output_path.exists()
    output = capsys.readouterr().out
    assert "Skipping overlay" in output

def test_overlay_creates_directory(tmp_path, influence_log_full):
    nested_path = tmp_path / "nested" / "overlay.png"
    render_influence_overlay(influence_log_full, str(nested_path))
    assert nested_path.exists()

def test_overlay_handles_missing_score_field(tmp_path):
    influence_log = {
        "adjacency_zones": [{"x": 1.0, "y": 2.0, "radius": 0.5}],
        "suppression_zones": [{"x": 3.0, "y": 4.0, "radius": 0.3}]
    }
    output_path = tmp_path / "overlay_missing_score.png"
    render_influence_overlay(influence_log, str(output_path))
    assert not output_path.exists()  # score defaults to 0.0, below threshold



