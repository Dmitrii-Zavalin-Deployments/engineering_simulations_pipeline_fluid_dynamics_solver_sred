# tests/adaptive/test_grid_refiner.py
# 🧪 Unit tests for src/adaptive/grid_refiner.py

import os
import json
import tempfile
import shutil
import pytest
from src.adaptive import grid_refiner

def test_load_delta_map_valid():
    with tempfile.NamedTemporaryFile("w+", delete=False) as f:
        json.dump({
            "(1.0,2.0,3.0)": {"delta": 0.5},
            "(4.0,5.0,6.0)": {"delta": 0.0},
            "(7.0,8.0,9.0)": {"delta": -1.2}
        }, f)
        f.close()
        result = grid_refiner.load_delta_map(f.name)
        os.unlink(f.name)

    assert (1.0,2.0,3.0) in result
    assert (7.0,8.0,9.0) in result
    assert (4.0,5.0,6.0) not in result
    assert len(result) == 2

def test_load_delta_map_invalid_json():
    with tempfile.NamedTemporaryFile("w+", delete=False) as f:
        f.write("not a json string")
        f.close()
        result = grid_refiner.load_delta_map(f.name)
        os.unlink(f.name)

    assert result == []

def test_detect_mutation_clusters_none():
    coords = [
        (0.0, 0.0, 0.0),
        (10.0, 10.0, 10.0),
        (20.0, 20.0, 20.0)
    ]
    spacing = (1.0, 1.0, 1.0)
    result = grid_refiner.detect_mutation_clusters(coords, spacing)
    assert result == []

def test_detect_mutation_clusters_dense():
    coords = [(i*1.0, 0.0, 0.0) for i in range(6)]
    spacing = (1.0, 1.0, 1.0)
    result = grid_refiner.detect_mutation_clusters(coords, spacing, threshold=2)
    assert len(result) >= 1
    for coord in result:
        assert isinstance(coord, tuple) and len(coord) == 3

def test_propose_refinement_zones_empty_map():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "delta.json")
        with open(path, "w") as f:
            json.dump({}, f)

        result = grid_refiner.propose_refinement_zones(path, (1.0, 1.0, 1.0), 5, tmpdir)
        assert result == []

def test_propose_refinement_zones_cluster_detected():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "delta.json")
        data = {
            f"({i}.0,0.0,0.0)": {"delta": 0.9} for i in range(6)
        }
        with open(path, "w") as f:
            json.dump(data, f)

        result = grid_refiner.propose_refinement_zones(path, (1.0, 1.0, 1.0), 2, tmpdir, threshold=2)
        assert len(result) >= 1

        output_file = os.path.join(tmpdir, "refinement_step_0002.json")
        assert os.path.isfile(output_file)

        with open(output_file) as f:
            saved = json.load(f)
            assert "refinement_zones" in saved
            assert isinstance(saved["refinement_zones"], list)
            assert all(len(c) == 3 for c in saved["refinement_zones"])



