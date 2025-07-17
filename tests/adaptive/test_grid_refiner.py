# tests/adaptive/test_grid_refiner.py
# ✅ Unit tests for src/adaptive/grid_refiner.py

import os
import json
import pytest
from src.adaptive.grid_refiner import (
    load_delta_map,
    detect_mutation_clusters,
    propose_refinement_zones
)

@pytest.fixture
def delta_map_dense(tmp_path):
    path = tmp_path / "delta_dense.json"
    coords = {
        f"({x}, {y}, 0.0)": {"delta": 0.01}
        for x in range(5) for y in range(5)  # 25 nearby points
    }
    with open(path, "w") as f:
        json.dump(coords, f)
    return path

@pytest.fixture
def delta_map_sparse(tmp_path):
    path = tmp_path / "delta_sparse.json"
    coords = {
        "(0.0, 0.0, 0.0)": {"delta": 0.02},
        "(10.0, 0.0, 0.0)": {"delta": 0.02},
        "(20.0, 0.0, 0.0)": {"delta": 0.02}
    }
    with open(path, "w") as f:
        json.dump(coords, f)
    return path

@pytest.fixture
def delta_map_empty(tmp_path):
    path = tmp_path / "delta_empty.json"
    with open(path, "w") as f:
        json.dump({}, f)
    return path

def test_load_delta_map_parses_valid_keys(delta_map_dense):
    coords = load_delta_map(str(delta_map_dense))
    assert len(coords) == 25
    assert isinstance(coords[0], tuple)
    assert all(len(c) == 3 for c in coords)

def test_detect_clusters_identifies_dense_patch():
    coords = [(x, y, 0.0) for x in range(5) for y in range(5)]
    spacing = (1.0, 1.0, 1.0)
    clusters = detect_mutation_clusters(coords, spacing, radius=1)
    assert len(clusters) > 0
    assert all(isinstance(c, tuple) for c in clusters)

def test_detect_clusters_ignores_sparse_points():
    coords = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (20.0, 0.0, 0.0)]
    spacing = (1.0, 1.0, 1.0)
    clusters = detect_mutation_clusters(coords, spacing, radius=1)
    assert clusters == []

def test_propose_refinement_zones_creates_file(delta_map_dense, tmp_path):
    spacing = (1.0, 1.0, 1.0)
    output_dir = tmp_path / "refine"
    zones = propose_refinement_zones(str(delta_map_dense), spacing, step_index=9, output_folder=str(output_dir))
    assert len(zones) > 0
    expected_file = output_dir / "refinement_step_0009.json"
    assert expected_file.exists()

def test_propose_refinement_zones_handles_sparse(delta_map_sparse, tmp_path):
    spacing = (1.0, 1.0, 1.0)
    zones = propose_refinement_zones(str(delta_map_sparse), spacing, step_index=3, output_folder=str(tmp_path))
    assert zones == []

def test_propose_refinement_zones_handles_empty(delta_map_empty, tmp_path):
    spacing = (1.0, 1.0, 1.0)
    zones = propose_refinement_zones(str(delta_map_empty), spacing, step_index=1, output_folder=str(tmp_path))
    assert zones == []

def test_load_delta_map_gracefully_handles_bad_file(tmp_path):
    bad_file = tmp_path / "corrupt.json"
    bad_file.write_text("not valid json")
    coords = load_delta_map(str(bad_file))
    assert coords == []



