# tests/adaptive/test_grid_refiner.py
# ✅ Validation suite for src/adaptive/grid_refiner.py

import os
import json
import tempfile
import pytest

from src.adaptive.grid_refiner import (
    load_delta_map,
    detect_mutation_clusters,
    propose_refinement_zones
)

@pytest.fixture
def delta_map_with_mutations(tmp_path):
    path = tmp_path / "delta_map.json"
    data = {
        "(0.0, 0.0, 0.0)": {"delta": 0.1},
        "(0.1, 0.0, 0.0)": {"delta": 0.2},
        "(0.2, 0.0, 0.0)": {"delta": 0.3},
        "(0.3, 0.0, 0.0)": {"delta": 0.4},
        "(0.4, 0.0, 0.0)": {"delta": 0.5},
        "(1.0, 1.0, 1.0)": {"delta": 0.0},  # Should be excluded
    }
    with open(path, "w") as f:
        json.dump(data, f)
    return str(path)

@pytest.fixture
def empty_delta_map(tmp_path):
    path = tmp_path / "empty_map.json"
    with open(path, "w") as f:
        json.dump({}, f)
    return str(path)

def test_load_delta_map_filters_zero_deltas(delta_map_with_mutations):
    coords = load_delta_map(delta_map_with_mutations)
    assert len(coords) == 5
    assert (1.0, 1.0, 1.0) not in coords

def test_detect_mutation_clusters_threshold_met():
    coords = [
        (0.0, 0.0, 0.0),
        (0.05, 0.0, 0.0),
        (0.10, 0.0, 0.0),
        (0.15, 0.0, 0.0),
        (0.20, 0.0, 0.0),
        (0.25, 0.0, 0.0),
    ]
    spacing = (0.1, 0.1, 0.1)
    clusters = detect_mutation_clusters(coords, spacing, radius=1, threshold=3)
    assert len(clusters) > 0
    assert any(c == (0.10, 0.0, 0.0) for c in clusters)

def test_detect_mutation_clusters_threshold_not_met():
    coords = [
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (2.0, 2.0, 2.0),
    ]
    spacing = (1.0, 1.0, 1.0)
    clusters = detect_mutation_clusters(coords, spacing, radius=1, threshold=3)
    assert clusters == []

def test_detect_mutation_clusters_empty_input():
    clusters = detect_mutation_clusters([], spacing=(1.0, 1.0, 1.0))
    assert clusters == []

def test_propose_refinement_zones_creates_output(delta_map_with_mutations):
    with tempfile.TemporaryDirectory() as tmpdir:
        result = propose_refinement_zones(
            delta_map_path=delta_map_with_mutations,
            spacing=(0.1, 0.1, 0.1),
            step_index=7,
            output_folder=tmpdir,
            threshold=2
        )
        assert isinstance(result, list)
        assert len(result) > 0
        output_path = os.path.join(tmpdir, "refinement_step_0007.json")
        assert os.path.exists(output_path)
        with open(output_path) as f:
            data = json.load(f)
        assert "refinement_zones" in data
        assert isinstance(data["refinement_zones"], list)

def test_propose_refinement_zones_skips_empty(empty_delta_map):
    with tempfile.TemporaryDirectory() as tmpdir:
        result = propose_refinement_zones(
            delta_map_path=empty_delta_map,
            spacing=(1.0, 1.0, 1.0),
            step_index=99,
            output_folder=tmpdir,
            threshold=1
        )
        assert result == []
        output_path = os.path.join(tmpdir, "refinement_step_0099.json")
        assert not os.path.exists(output_path)



