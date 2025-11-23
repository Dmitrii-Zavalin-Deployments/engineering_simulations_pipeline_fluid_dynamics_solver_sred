# tests/test_force_utils.py
# ✅ Validation tests for src/step_2_time_stepping_loop/force_utils.py

import pytest
from src.step_2_time_stepping_loop.force_utils import load_external_forces


# ---------------- Happy Path ----------------

def test_load_external_forces_valid_config():
    """Ensure valid external_forces block returns correct Fx, Fy, Fz dict."""
    config = {
        "external_forces": {
            "force_vector": [1.0, -2.0, 3.5],
            "force_units": "N/m^3",
            "force_comment": "Test forces"
        }
    }
    result = load_external_forces(config)
    assert isinstance(result, dict)
    assert result == {"Fx": 1.0, "Fy": -2.0, "Fz": 3.5}


# ---------------- Error Cases ----------------

def test_missing_external_forces_block_raises_keyerror():
    """Missing external_forces block should raise KeyError."""
    config = {}
    with pytest.raises(KeyError) as excinfo:
        load_external_forces(config)
    assert "Missing 'external_forces'" in str(excinfo.value)


def test_missing_force_vector_raises_valueerror():
    """Missing force_vector inside external_forces should raise ValueError."""
    config = {
        "external_forces": {
            "force_units": "N/m^3",
            "force_comment": "No vector provided"
        }
    }
    with pytest.raises(ValueError) as excinfo:
        load_external_forces(config)
    assert "Invalid or missing 'force_vector'" in str(excinfo.value)


def test_force_vector_wrong_length_raises_valueerror():
    """Force vector with wrong length should raise ValueError."""
    config = {
        "external_forces": {
            "force_vector": [1.0, 2.0],  # only 2 values
            "force_units": "N/m^3",
            "force_comment": "Invalid length"
        }
    }
    with pytest.raises(ValueError) as excinfo:
        load_external_forces(config)
    assert "Expected [Fx, Fy, Fz]" in str(excinfo.value)


def test_force_vector_none_raises_valueerror():
    """Force vector explicitly set to None should raise ValueError."""
    config = {
        "external_forces": {
            "force_vector": None,
            "force_units": "N/m^3",
            "force_comment": "Invalid None"
        }
    }
    with pytest.raises(ValueError):
        load_external_forces(config)


# ---------------- Edge Cases ----------------

def test_force_vector_with_zero_values():
    """Force vector with all zeros should return dict of zeros."""
    config = {
        "external_forces": {
            "force_vector": [0.0, 0.0, 0.0],
            "force_units": "N/m^3",
            "force_comment": "Zero forces"
        }
    }
    result = load_external_forces(config)
    assert result == {"Fx": 0.0, "Fy": 0.0, "Fz": 0.0}


def test_force_vector_with_large_values():
    """Force vector with large magnitudes should be handled correctly."""
    config = {
        "external_forces": {
            "force_vector": [1e6, -1e6, 3.14159],
            "force_units": "N/m^3",
            "force_comment": "Stress test with large values"
        }
    }
    result = load_external_forces(config)
    assert result["Fx"] == pytest.approx(1e6)
    assert result["Fy"] == pytest.approx(-1e6)
    assert result["Fz"] == pytest.approx(3.14159)



