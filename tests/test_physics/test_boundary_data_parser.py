import numpy as np
import pytest
from src.physics.boundary_data_parser import identify_boundary_nodes


def minimal_mesh_info():
    return {
        "min_x": 0.0, "max_x": 1.0,
        "min_y": 0.0, "max_y": 1.0,
        "min_z": 0.0, "max_z": 1.0,
        "dx": 0.5, "dy": 0.5, "dz": 0.5,
        "grid_shape": (2, 2, 2)
    }


def test_skips_bc_with_no_faces(capfd):
    bc_def = {"wall": {"type": "dirichlet"}}
    bcs = identify_boundary_nodes(bc_def, [], minimal_mesh_info())
    out = capfd.readouterr().err
    assert "has no 'faces' specified" in out
    assert bcs == {}


def test_skips_face_id_not_in_lookup(capfd):
    face = {"face_id": "good_face", "nodes": {"n0": [0, 0, 0], "n1": [0.5, 0, 0], "n2": [0.5, 0.5, 0], "n3": [0, 0.5, 0]}}
    bc_def = {"bc1": {"faces": ["missing_id"]}}
    bcs = identify_boundary_nodes(bc_def, [face], minimal_mesh_info())
    err = capfd.readouterr().err
    assert "not found for boundary" in err
    assert bcs == {}


def test_warning_on_non_axis_aligned_boundary(capfd):
    tilted_face = {
        "face_id": "diagonal",
        "nodes": {
            "n0": [0.0, 0.0, 0.0],
            "n1": [0.1, 0.1, 0.0],
            "n2": [0.1, 0.1, 0.1],
            "n3": [0.0, 0.0, 0.1]
        }
    }
    bc_def = {"tilted": {"faces": ["diagonal"]}}
    bcs = identify_boundary_nodes(bc_def, [tilted_face], minimal_mesh_info())
    assert "tilted" in bcs
    assert bcs["tilted"]["boundary_dim"] is None
    err = capfd.readouterr().err
    assert "does not perfectly align" in err


def test_wall_with_no_slip_suppresses_pressure():
    face = {
        "face_id": "wallface",
        "nodes": {
            "n0": [0.0, 0.0, 0.0],
            "n1": [0.5, 0.0, 0.0],
            "n2": [0.5, 0.5, 0.0],
            "n3": [0.0, 0.5, 0.0],
        }
    }
    bc_def = {"wall": {"faces": ["wallface"], "no_slip": True, "apply_to": ["velocity", "pressure"]}}
    bcs = identify_boundary_nodes(bc_def, [face], minimal_mesh_info())
    assert bcs["wall"]["apply_to"] == ["velocity"]


def test_defaults_for_missing_fields():
    face = {
        "face_id": "basic",
        "nodes": {
            "n0": [0.0, 0.0, 0.0],
            "n1": [0.5, 0.0, 0.0],
            "n2": [0.5, 0.5, 0.0],
            "n3": [0.0, 0.5, 0.0],
        }
    }
    bc_def = {"inlet": {"faces": ["basic"]}}  # No velocity, pressure, or apply_to
    bcs = identify_boundary_nodes(bc_def, [face], minimal_mesh_info())
    bc = bcs["inlet"]
    assert np.allclose(bc["velocity"], [0.0, 0.0, 0.0])
    assert bc["pressure"] == 0.0
    assert "velocity" in bc["apply_to"] and "pressure" in bc["apply_to"]



