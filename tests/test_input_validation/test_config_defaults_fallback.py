# tests/test_input_validation/test_config_defaults_fallback.py

import pytest

def test_defaults_applied_when_optional_fields_missing(monkeypatch, basic_solver_config):
    """
    Simulates missing optional keys in the config and checks fallback behavior.
    Assumes your simulation code provides default values for these keys internally.
    """

    # Remove some optional fields
    basic_solver_config["grid"].pop("dx", None)
    basic_solver_config["time"].pop("time_step", None)
    basic_solver_config["solver"]["method"] = ""  # Empty method string

    # Assume there's a function in your pre-processing pipeline that applies defaults
    # You may need to replace this with the actual import path
    from src.utils.io_utils import apply_config_defaults  

    try:
        filled = apply_config_defaults(basic_solver_config)
    except Exception as e:
        pytest.fail(f"Default application raised an unexpected error: {e}")

    # Now validate that defaults have been injected
    assert "dx" in filled["grid"], "Default dx not applied to grid."
    assert filled["grid"]["dx"] > 0, "Default dx must be positive."

    assert "time_step" in filled["time"], "Default time_step not applied."
    assert filled["time"]["time_step"] > 0, "Default time_step must be positive."

    assert filled["solver"]["method"], "Default solver method should not be empty."

def test_missing_required_fields_should_raise(basic_solver_config):
    """
    Test that truly required fields still cause a failure if removed.
    """
    basic_solver_config["fluid"].pop("density", None)

    from src.utils.io_utils import apply_config_defaults

    with pytest.raises(KeyError):
        apply_config_defaults(basic_solver_config)



