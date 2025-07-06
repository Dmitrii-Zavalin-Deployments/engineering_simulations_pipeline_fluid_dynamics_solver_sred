import numpy as np
import json
import pytest
from pathlib import Path
from src.solver.results_handler import save_field_snapshot

def test_save_field_snapshot_creates_json(tmp_path):
    step = 3
    fields_dir = tmp_path / "output"
    velocity = np.full((2, 2, 2, 3), 1.23)  # Non-zero velocity field
    pressure = np.full((2, 2, 2), 101325.0)  # Non-zero pressure

    save_field_snapshot(step, velocity, pressure, str(fields_dir))

    expected_file = fields_dir / "step_0003.json"
    assert expected_file.exists(), "Snapshot file was not created"

    with open(expected_file, "r") as f:
        data = json.load(f)

    assert data["step"] == step
    assert isinstance(data["velocity"], list)
    assert isinstance(data["pressure"], list)

    vel_array = np.array(data["velocity"])
    pres_array = np.array(data["pressure"])

    assert vel_array.shape == (2, 2, 2, 3)
    assert pres_array.shape == (2, 2, 2)
    assert np.allclose(vel_array, 1.23)
    assert np.allclose(pres_array, 101325.0)



