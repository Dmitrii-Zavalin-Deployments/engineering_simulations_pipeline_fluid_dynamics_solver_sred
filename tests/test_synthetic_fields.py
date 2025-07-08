# tests/test_synthetic_fields.py

import pytest
import numpy as np
from src.stability_utils import check_field_validity, test_velocity_bounds  # ✅ Confirmed accurate import

def generate_nan_field(shape):
    field = np.zeros(shape)
    field[0, 0] = np.nan
    return field

def generate_inf_field(shape):
    field = np.ones(shape)
    field[1, 1] = np.inf
    return field

def generate_spike_velocity_field(shape, spike_value=1e6):
    field = np.ones((*shape, 3))
    field[2, 2, 2] = spike_value
    return field

def generate_malformed_velocity_field():
    velocity_field = np.ones((5, 5, 5, 3))
    velocity_field[1, 1, 1, :] = np.nan
    return velocity_field

@pytest.mark.parametrize("field_generator,label", [
    (generate_nan_field, "NaNField"),
    (generate_inf_field, "InfField")
])
def test_synthetic_corrupted_fields_fail_validation(field_generator, label):
    field = field_generator((4, 4))
    assert check_field_validity(field, label=label) is False

def test_spike_velocity_exceeds_bounds():
    velocity_field = generate_spike_velocity_field((5, 5, 5), spike_value=99999.0)
    assert test_velocity_bounds(velocity_field, velocity_limit=100.0) is False

def test_clean_velocity_passes_bounds_check():
    velocity_field = np.ones((4, 4, 4, 3)) * 9.0
    assert test_velocity_bounds(velocity_field, velocity_limit=100.0) is True

def test_malformed_velocity_field_detects_nan():
    velocity_field = generate_malformed_velocity_field()
    assert check_field_validity(velocity_field, label="MalformedVelocity") is False



