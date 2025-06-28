# tests/conftest.py

import pytest
import tempfile
import shutil
import json
import numpy as np
from pathlib import Path

# -------------------------- Project Fixture Paths --------------------------

@pytest.fixture(scope="session")
def project_root():
    """Returns the root directory of the project."""
    return Path(__file__).resolve().parents[1]

@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """Path to shared test data directory (e.g. mock configs, meshes)."""
    return project_root / "tests" / "utilities" / "mock_data"

@pytest.fixture(scope="function")
def temp_output_dir():
    """Creates a temporary output directory for tests and cleans it up after."""
    temp_dir = tempfile.mkdtemp(prefix="fluid_output_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="function")
def basic_solver_config(test_data_dir):
    """Loads a minimal valid solver config for simulation or integration tests."""
    config_path = test_data_dir / "basic_config.json"
    with open(config_path, "r") as f:
        return json.load(f)

# -------------------------- Test Utilities --------------------------

def generate_velocity_with_divergence(grid_shape, pattern="x-ramp"):
    """
    Generates a 3D velocity field with built-in divergence.

    Args:
        grid_shape (tuple): Shape of the grid (with ghost cells).
        pattern (str): One of ['x-ramp', 'random', 'radial'].

    Returns:
        np.ndarray: Velocity field with shape (..., 3)
    """
    u = np.zeros(grid_shape + (3,), dtype=np.float64)

    if pattern == "x-ramp":
        for i in range(1, grid_shape[0] - 1):
            u[i, :, :, 0] = i * 0.1
    elif pattern == "random":
        u = np.random.randn(*grid_shape, 3) * 0.2
    elif pattern == "radial":
        x = np.linspace(-1, 1, grid_shape[0])
        y = np.linspace(-1, 1, grid_shape[1])
        z = np.linspace(-1, 1, grid_shape[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        r = np.sqrt(X**2 + Y**2 + Z**2) + 1e-6
        u[..., 0] = X / r
        u[..., 1] = Y / r
        u[..., 2] = Z / r
    else:
        raise ValueError(f"Unknown pattern '{pattern}'")

    return u



