# tests/test_grid_spacing.py

import pytest
from src.step_2_time_stepping_loop.grid_spacing import compute_grid_spacings


# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture
def valid_config():
    return {
        "domain_definition": {
            "x_min": 0.0,
            "x_max": 4.0,
            "y_min": -2.0,
            "y_max": 2.0,
            "z_min": -1.0,
            "z_max": 3.0,
            "nx": 4,
            "ny": 4,
            "nz": 4,
        }
    }


# -----------------------------
# Positive cases
# -----------------------------

def test_compute_grid_spacings_default_mode(valid_config):
    dx, dy, dz = compute_grid_spacings(valid_config, mode="nx")
    assert dx == pytest.approx(1.0)
    assert dy == pytest.approx(1.0)
    assert dz == pytest.approx(1.0)


def test_compute_grid_spacings_nx_minus_one(valid_config):
    dx, dy, dz = compute_grid_spacings(valid_config, mode="nx_minus_one")
    # (x_max - x_min) / (nx - 1) = 4 / 3
    assert dx == pytest.approx(4.0 / 3.0)
    assert dy == pytest.approx(4.0 / 3.0)
    assert dz == pytest.approx(4.0 / 3.0)


def test_compute_grid_spacings_non_uniform(valid_config):
    # Adjust domain to produce different spacings
    valid_config["domain_definition"]["x_max"] = 8.0
    valid_config["domain_definition"]["y_max"] = 6.0
    valid_config["domain_definition"]["z_max"] = 2.0
    dx, dy, dz = compute_grid_spacings(valid_config, mode="nx")
    assert dx == pytest.approx(2.0)
    assert dy == pytest.approx(2.0)
    assert dz == pytest.approx(0.75)


# -----------------------------
# Error cases: missing keys
# -----------------------------

def test_missing_domain_definition_key():
    config = {}
    with pytest.raises(ValueError) as e:
        compute_grid_spacings(config)
    assert "domain_definition" in str(e.value)


def test_missing_required_fields(valid_config):
    del valid_config["domain_definition"]["nx"]
    with pytest.raises(ValueError) as e:
        compute_grid_spacings(valid_config)
    assert "missing keys" in str(e.value)


# -----------------------------
# Error cases: invalid values
# -----------------------------

def test_invalid_nx_type(valid_config):
    valid_config["domain_definition"]["nx"] = "four"
    with pytest.raises(ValueError) as e:
        compute_grid_spacings(valid_config)
    assert "must be a positive integer" in str(e.value)


def test_invalid_negative_ny(valid_config):
    valid_config["domain_definition"]["ny"] = -5
    with pytest.raises(ValueError) as e:
        compute_grid_spacings(valid_config)
    assert "must be a positive integer" in str(e.value)


def test_invalid_non_numeric_bounds(valid_config):
    valid_config["domain_definition"]["x_min"] = "zero"
    with pytest.raises(ValueError) as e:
        compute_grid_spacings(valid_config)
    assert "must be numeric" in str(e.value)


def test_invalid_bounds_order(valid_config):
    valid_config["domain_definition"]["x_max"] = -1.0
    with pytest.raises(ValueError) as e:
        compute_grid_spacings(valid_config)
    assert "must be greater than" in str(e.value)


# -----------------------------
# Error cases: nx_minus_one mode
# -----------------------------

def test_nx_minus_one_requires_minimum_points(valid_config):
    valid_config["domain_definition"]["nx"] = 1
    with pytest.raises(ValueError) as e:
        compute_grid_spacings(valid_config, mode="nx_minus_one")
    assert "requires nx, ny, nz >= 2" in str(e.value)


# -----------------------------
# Error cases: final sanity checks
# -----------------------------

def test_zero_spacing(valid_config):
    valid_config["domain_definition"]["x_max"] = valid_config["domain_definition"]["x_min"]
    with pytest.raises(ValueError) as e:
        compute_grid_spacings(valid_config)
    assert "must be greater than" in str(e.value)


def test_infinite_spacing(valid_config):
    valid_config["domain_definition"]["x_max"] = float("inf")
    with pytest.raises(ValueError) as e:
        compute_grid_spacings(valid_config)
    assert "must be > 0 and finite" in str(e.value)



