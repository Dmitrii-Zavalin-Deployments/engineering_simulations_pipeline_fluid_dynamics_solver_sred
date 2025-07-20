# ✅ Unit Test Suite — Domain Normalizer
# 📄 Full Path: tests/utils/test_domain_normalizer.py

import pytest
import logging
from src.utils import domain_normalizer as dn

def test_all_required_keys_present_returns_unchanged():
    domain = {
        "min_x": 0.0, "max_x": 2.0, "nx": 10,
        "min_y": 0.0, "max_y": 2.0, "ny": 10,
        "min_z": 0.0, "max_z": 2.0, "nz": 10
    }
    result = dn.normalize_domain(domain)
    assert result == domain  # No change
    assert all(k in result for k in dn.REQUIRED_KEYS)

def test_some_keys_missing_gets_defaulted(caplog):
    partial = {
        "min_x": -1.0, "max_x": 2.0, "nx": 4,
        "min_z": 0.0
    }
    caplog.set_level(logging.WARNING)
    result = dn.normalize_domain(partial)

    for k in dn.REQUIRED_KEYS:
        assert k in result
        assert isinstance(result[k], float) or isinstance(result[k], int)

    assert result["min_x"] == -1.0  # preserved
    assert result["min_y"] == 0.0  # defaulted
    assert "Domain normalization applied" in caplog.text
    assert "missing keys substituted" in caplog.text

def test_empty_dict_yields_all_defaults(caplog):
    caplog.set_level(logging.WARNING)
    result = dn.normalize_domain({})
    assert len(result) == len(dn.REQUIRED_KEYS)
    for k, v in dn.REQUIRED_KEYS.items():
        assert result[k] == v
    assert "missing keys substituted" in caplog.text

def test_original_dict_not_mutated():
    original = {"min_x": 5.0}
    result = dn.normalize_domain(original)
    assert original == {"min_x": 5.0}
    assert result["min_x"] == 5.0
    assert all(k in result for k in dn.REQUIRED_KEYS)



