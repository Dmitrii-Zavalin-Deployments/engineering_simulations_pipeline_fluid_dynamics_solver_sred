# tests/visualization/test_influence_overlay.py

import os
import shutil
import pytest
from src.visualization.influence_overlay import render_influence_overlay

TEST_OUTPUT_DIR = "test_output/overlay_tests"

@pytest.fixture(autouse=True)
def setup_and_cleanup():
    # Ensure clean test output directory
    shutil.rmtree(TEST_OUTPUT_DIR, ignore_errors=True)
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    yield
    shutil.rmtree(TEST_OUTPUT_DIR, ignore_errors=True)

def test_overlay_skipped_when_score_below_threshold(capfd):
    influence_log = {
        "step_score": 0.5,  # below default threshold
        "adjacency_zones": [{"x": 0, "y": 0, "radius": 1}],
        "suppression_zones": [{"x": 1, "y": 1, "radius": 0.5}]
    }
    output_file = os.path.join(TEST_OUTPUT_DIR, "low_score.png")
    render_influence_overlay(influence_log, output_file)
    
    assert not os.path.exists(output_file)
    captured = capfd.readouterr()
    assert "Skipping overlay" in captured.out

def test_overlay_rendered_when_score_meets_threshold():
    influence_log = {
        "step_score": 0.95,  # above threshold
        "adjacency_zones": [{"x": 2, "y": 2, "radius": 1}],
        "suppression_zones": [{"x": 3, "y": 3, "radius": 0.5}]
    }
    output_file = os.path.join(TEST_OUTPUT_DIR, "high_score.png")
    render_influence_overlay(influence_log, output_file)
    
    assert os.path.exists(output_file)

def test_overlay_handles_missing_zone_data_gracefully():
    influence_log = {
        "step_score": 1.0  # trigger rendering
        # no adjacency_zones or suppression_zones provided
    }
    output_file = os.path.join(TEST_OUTPUT_DIR, "empty_zones.png")
    render_influence_overlay(influence_log, output_file)
    
    assert os.path.exists(output_file)

def test_overlay_threshold_override():
    influence_log = {
        "step_score": 0.7,
        "adjacency_zones": [],
        "suppression_zones": []
    }
    output_file = os.path.join(TEST_OUTPUT_DIR, "custom_threshold.png")
    render_influence_overlay(influence_log, output_file, score_threshold=0.65)
    
    assert os.path.exists(output_file)



