import numpy as np
import json
import pytest
from pathlib import Path
from src.utils import io_utils

def test_convert_numpy_to_list_nested():
    data = {
        "array": np.array([1, 2, 3]),
        "nested": {
            "matrix": np.array([[1.1, 2.2], [3.3, 4.4]]),
            "list_of_arrays": [np.array([5, 6]), np.array([7])]
        }
    }
    result = io_utils.convert_numpy_to_list(data)

    assert isinstance(result["array"], list)
    assert result["array"] == [1, 2, 3]
    assert isinstance(result["nested"]["matrix"], list)
    assert isinstance(result["nested"]["list_of_arrays"][0], list)
    assert result["nested"]["matrix"][0][1] == pytest.approx(2.2)

def test_save_json_handles_numpy_and_writes(tmp_path):
    sample = {
        "velocity": np.array([[1.0, 2.0, 3.0]]),
        "pressure": np.array([101325.0])
    }
    filepath = tmp_path / "test_sim.json"
    serializable = io_utils.convert_numpy_to_list(sample)

    io_utils.save_json(serializable, filepath)
    assert filepath.exists()

    with open(filepath) as f:
        loaded = json.load(f)
        assert loaded["pressure"] == [101325.0]
        assert loaded["velocity"][0][1] == pytest.approx(2.0)

def test_apply_config_defaults_adds_missing_keys():
    minimal = {
        "fluid": {"density": 1.23}
    }
    cfg = io_utils.apply_config_defaults(minimal)

    assert cfg["grid"]["dx"] == 1.0
    assert cfg["time"]["time_step"] == 0.01
    assert cfg["solver"]["method"] == "explicit"
    assert cfg["initial_conditions"]["velocity_magnitude"] == 1.0
    assert cfg["fluid"]["density"] == 1.23  # Must still exist and not be overwritten

def test_apply_config_defaults_raises_for_missing_density():
    with pytest.raises(KeyError, match="fluid.density"):
        io_utils.apply_config_defaults({})



