# conftest.py

import pytest
import tempfile
import shutil
import json
from pathlib import Path

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



