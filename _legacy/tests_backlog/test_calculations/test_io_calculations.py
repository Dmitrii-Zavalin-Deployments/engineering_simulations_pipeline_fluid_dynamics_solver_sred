import numpy as np
import json
import gzip
import os
import pytest
from pathlib import Path
from src.utils import io as io_utils

@pytest.fixture
def dummy_fields():
    velocity = np.ones((2, 2, 2, 3), dtype=np.float64) * 2.5
    pressure = np.full((2, 2, 2), 101325.0, dtype=np.float64)
    return velocity, pressure

def test_save_and_load_json_plain(tmp_path):
    data = {"a": 1, "b": [1.0, 2.0, 3.0]}
    filepath = tmp_path / "plain.json"

    io_utils.save_json(data, str(filepath))
    loaded = io_utils.load_json(str(filepath))

    assert loaded == data

def test_save_and_load_json_gzip(tmp_path):
    data = {"grid": [[1, 2], [3, 4]], "note": "compressed"}
    filepath = tmp_path / "compressed.json.gz"

    io_utils.save_json(data, str(filepath), use_gzip=True)
    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded == data

def test_save_checkpoint_creates_npz(tmp_path, dummy_fields):
    velocity, pressure = dummy_fields
    time = 0.75
    path = tmp_path / "state_checkpoint.npz"

    io_utils.save_checkpoint(str(path), velocity, pressure, time)
    assert path.exists()

    loaded = np.load(path)
    assert np.allclose(loaded["velocity"], velocity)
    assert np.allclose(loaded["pressure"], pressure)
    assert loaded["current_time"] == pytest.approx(time)

def test_write_output_to_vtk_creates_file(tmp_path, dummy_fields):
    velocity, pressure = dummy_fields
    x = np.linspace(0, 2, 3)
    y = np.linspace(0, 2, 3)
    z = np.linspace(0, 2, 3)
    vtk_path = tmp_path / "test_output.vti"

    io_utils.write_output_to_vtk(
        velocity_field=velocity,
        pressure_field=pressure,
        x_coords_grid_lines=x,
        y_coords_grid_lines=y,
        z_coords_grid_lines=z,
        output_filepath=str(vtk_path)
    )

    assert vtk_path.exists()
    content = vtk_path.read_text()

    # Sanity checks on the file content
    assert "<VTKFile" in content
    assert "<ImageData" in content
    assert "Velocity" in content
    assert "Pressure" in content
    assert "_".join(["0", "1", "2"]) in content or "<AppendedData" in content



