# tests/exporters/test_velocity_field_writer.py
# ✅ Validation suite for src/exporters/velocity_field_writer.py

import os
import json
import tempfile
from src.exporters.velocity_field_writer import write_velocity_field

class MockCell:
    def __init__(self, x, y, z, velocity, fluid_mask=True):
        self.x = x
        self.y = y
        self.z = z
        self.velocity = velocity
        self.fluid_mask = fluid_mask

def test_write_velocity_field_creates_file_and_directory():
    grid = [
        MockCell(0, 0, 0, [1.0, 0.0, 0.0]),
        MockCell(1, 2, 3, [0.5, -0.5, 0.25])
    ]

    with tempfile.TemporaryDirectory() as tmp:
        write_velocity_field(grid, step=42, output_dir=tmp)
        expected_file = os.path.join(tmp, "velocity_field_step_0042.json")
        assert os.path.exists(expected_file)

        with open(expected_file) as f:
            data = json.load(f)

        assert "(0.00, 0.00, 0.00)" in data
        assert "(1.00, 2.00, 3.00)" in data
        assert data["(0.00, 0.00, 0.00)"]["vx"] == 1.0
        assert data["(1.00, 2.00, 3.00)"]["vy"] == -0.5

def test_write_velocity_field_excludes_non_fluid_cells():
    grid = [
        MockCell(0, 0, 0, [1.0, 0.0, 0.0], fluid_mask=False),
        MockCell(1, 1, 1, [0.0, 1.0, 0.0], fluid_mask=True)
    ]

    with tempfile.TemporaryDirectory() as tmp:
        write_velocity_field(grid, step=7, output_dir=tmp)
        expected_file = os.path.join(tmp, "velocity_field_step_0007.json")
        with open(expected_file) as f:
            data = json.load(f)

        assert "(1.00, 1.00, 1.00)" in data
        assert "(0.00, 0.00, 0.00)" not in data

def test_write_velocity_field_handles_missing_velocity_attribute():
    class IncompleteCell:
        def __init__(self, x, y, z, fluid_mask=True):
            self.x = x
            self.y = y
            self.z = z
            self.fluid_mask = fluid_mask

    grid = [
        IncompleteCell(0, 0, 0),
        MockCell(1, 1, 1, [0.0, 0.0, 0.0])
    ]

    with tempfile.TemporaryDirectory() as tmp:
        write_velocity_field(grid, step=8, output_dir=tmp)
        expected_file = os.path.join(tmp, "velocity_field_step_0008.json")
        with open(expected_file) as f:
            data = json.load(f)

        assert "(1.00, 1.00, 1.00)" in data
        assert "(0.00, 0.00, 0.00)" not in data

def test_write_velocity_field_empty_grid_creates_valid_file():
    grid = []

    with tempfile.TemporaryDirectory() as tmp:
        write_velocity_field(grid, step=0, output_dir=tmp)
        expected_file = os.path.join(tmp, "velocity_field_step_0000.json")
        assert os.path.exists(expected_file)

        with open(expected_file) as f:
            data = json.load(f)
        assert data == {}

def test_write_velocity_field_default_output_dir(tmp_path):
    grid = [MockCell(0, 0, 0, [0.0, 0.0, 0.0])]

    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)  # simulate working directory
        write_velocity_field(grid, step=1)
        expected_file = os.path.join("data", "snapshots", "velocity_field_step_0001.json")
        assert os.path.exists(expected_file)



