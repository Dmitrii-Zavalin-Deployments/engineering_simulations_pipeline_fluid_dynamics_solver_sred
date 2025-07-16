# tests/utils/test_domain_normalizer.py
# 🧪 Validates domain normalization logic and default injection behavior

import pytest
import logging
from src.utils.domain_normalizer import normalize_domain, REQUIRED_KEYS

@pytest.fixture(autouse=True)
def capture_logs(caplog):
    caplog.set_level(logging.WARNING)
    yield

def test_normalize_domain_preserves_complete_input(caplog):
    domain = REQUIRED_KEYS.copy()
    result = normalize_domain(domain)
    assert result == REQUIRED_KEYS
    assert "missing keys" not in caplog.text

def test_normalize_domain_fills_partial_missing_keys(caplog):
    partial = {
        "min_x": 0.0, "max_x": 1.0,  # nx missing
        "min_y": 0.0, "ny": 2,       # max_y missing
        "max_z": 5.0                 # missing min_z and nz
    }
    result = normalize_domain(partial)
    for key in REQUIRED_KEYS:
        assert key in result
    assert result["nx"] == 1
    assert result["max_y"] == 1.0
    assert result["min_z"] == 0.0
    assert result["nz"] == 1
    assert "missing keys" in caplog.text
    assert "Domain normalization applied" in caplog.text

def test_normalize_domain_empty_input(caplog):
    result = normalize_domain({})
    for key in REQUIRED_KEYS:
        assert key in result
        assert result[key] == REQUIRED_KEYS[key]
    assert "missing keys" in caplog.text

def test_normalize_domain_overrides_none_values(caplog):
    incomplete = {
        "min_x": None,
        "max_x": None
    }
    result = normalize_domain(incomplete)
    assert result["min_x"] == 0.0
    assert result["max_x"] == 1.0
    assert "missing keys" in caplog.text
    assert "Domain normalization applied" in caplog.text

def test_normalize_domain_ignores_extra_keys():
    extra = {
        "min_x": 0.0, "max_x": 1.0, "nx": 4,
        "min_y": 0.0, "max_y": 2.0, "ny": 2,
        "min_z": 0.0, "max_z": 3.0, "nz": 3,
        "custom_key": "value"
    }
    result = normalize_domain(extra)
    assert result["custom_key"] == "value"
    assert result["nx"] == 4
    assert result["max_z"] == 3.0

def test_normalize_domain_nested_input_safe(caplog):
    nested = {
        "min_x": 0.0,
        "metadata": {"description": "test domain"}
    }
    result = normalize_domain(nested)
    for key in REQUIRED_KEYS:
        assert key in result
    assert result["metadata"] == {"description": "test domain"}
    assert "missing keys" in caplog.text

def test_normalize_domain_warns_once_per_call(caplog):
    input1 = {"min_x": 0.0}
    input2 = {"max_x": None}
    normalize_domain(input1)
    normalize_domain(input2)
    warnings = [record.message for record in caplog.records if "Domain normalization applied" in record.message]
    assert len(warnings) == 2  # one warning per call