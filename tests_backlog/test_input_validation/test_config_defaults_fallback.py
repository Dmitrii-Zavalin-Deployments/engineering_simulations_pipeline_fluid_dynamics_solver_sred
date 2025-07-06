# tests/test_input_validation/test_config_defaults_fallback.py

import pytest
from copy import deepcopy

# This is your real implementation
from src.utils.io_utils import apply_config_defaults

def test_defaults_are_applied_when_fields_are_missing(basic_solver_config):
    """
    Validates that apply_config_defaults fills in reasonable defaults when optional fields are omitted.
    """

    config = deepcopy(basic_solver_config)

    # Remove optional entries to simulate incomplete input
    config["grid"].pop("dx", None)
    config["time"].pop("time_step", None)
    config["solver"]["method"] = ""  # Clear value

    updated = apply_config_defaults(config)

    assert "dx" in updated["grid"] and updated["grid"]["dx"] > 0, "Default dx not set."
    assert "time_step" in updated["time"] and updated["time"]["time_step"] > 0, "Default time_step not set."
    assert updated["solver"]["method"], "Default solver method not applied."

def test_missing_required_field_raises_exception(basic_solver_config):
    """
    Validates that apply_config_defaults raises an error when a required field is missing.
    """
    config = deepcopy(basic_solver_config)
    config["fluid"].pop("density", None)

    with pytest.raises(KeyError, match="fluid.density"):
        apply_config_defaults(config)



