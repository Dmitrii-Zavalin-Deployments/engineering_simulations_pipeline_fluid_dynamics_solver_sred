# tests/initialization/test_domain_bounds_validation.py
# ✅ Unit tests for extract_domain_bounds() — strict schema enforcement

import pytest
from src.initialization.fluid_mask_initializer import extract_domain_bounds

def test_extract_domain_bounds_raises_on_missing_keys():
    incomplete_domain = {
        "min_x": 0.0, "max_x": 1.0,
        "min_y": 0.0, "max_y": 1.0,
        "min_z": 0.0, "max_z": 1.0,
        # Missing nx, ny, nz
    }

    with pytest.raises(ValueError) as excinfo:
        extract_domain_bounds(incomplete_domain)

    assert "Missing required domain keys" in str(excinfo.value)
    assert "nx" in str(excinfo.value)
    assert "ny" in str(excinfo.value)
    assert "nz" in str(excinfo.value)

def test_extract_domain_bounds_passes_with_all_keys():
    complete_domain = {
        "min_x": 0.0, "max_x": 1.0,
        "min_y": 0.0, "max_y": 1.0,
        "min_z": 0.0, "max_z": 1.0,
        "nx": 10, "ny": 10, "nz": 10
    }

    result = extract_domain_bounds(complete_domain)
    assert result == (
        10, 10, 10,
        0.0, 1.0,
        0.0, 1.0,
        0.0, 1.0
    )
