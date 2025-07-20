# ✅ Unit Test Suite — Influence Overlay Renderer
# 📄 Full Path: tests/visualization/test_influence_overlay.py

import pytest
import os
import matplotlib
matplotlib.use("Agg")  # Disable GUI backend for test environments
from src.visualization.influence_overlay import render_influence_overlay
from tempfile import TemporaryDirectory

def test_overlay_skipped_when_score_too_low(capsys):
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "step_overlay.png")
        log = {
            "step_score": 0.2,
            "adjacency_zones": [{"x": 0, "y": 0, "radius": 1}],
            "suppression_zones": [{"x": 1, "y": 1, "radius": 1}]
        }
        render_influence_overlay(log, path, score_threshold=0.5)
        captured = capsys.readouterr().out
        assert "Skipping overlay" in captured
        assert not os.path.exists(path)

def test_overlay_created_for_valid_score():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "overlay.png")
        log = {
            "step_score": 0.9,
            "adjacency_zones": [{"x": 2, "y": 2, "radius": 0.5}],
            "suppression_zones": [{"x": 3, "y": 3, "radius": 0.5}]
        }
        render_influence_overlay(log, path)
        assert os.path.exists(path)

def test_overlay_handles_missing_score_key():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "no_score_overlay.png")
        log = {
            "adjacency_zones": [{"x": 0, "y": 0, "radius": 0.3}],
            "suppression_zones": [{"x": 1, "y": 1, "radius": 0.3}]
        }
        render_influence_overlay(log, path)
        # Expected to skip due to fallback score of 0.0
        assert not os.path.exists(path)

def test_overlay_handles_non_numeric_score(capsys):
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bad_score_overlay.png")
        log = {
            "step_score": "not_a_number",
            "adjacency_zones": [{"x": 0, "y": 0, "radius": 0.3}]
        }
        render_influence_overlay(log, path)
        captured = capsys.readouterr().out
        assert "Skipping overlay" in captured
        assert not os.path.exists(path)

def test_overlay_writes_to_nested_folder():
    with TemporaryDirectory() as tmp:
        subdir = os.path.join(tmp, "nested", "path")
        path = os.path.join(subdir, "overlay.png")
        log = {
            "step_score": 1.0,
            "adjacency_zones": [],
            "suppression_zones": []
        }
        render_influence_overlay(log, path)
        assert os.path.exists(path)



