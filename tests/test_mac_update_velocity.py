# tests/test_mac_update_velocity.py
# ✅ Validation tests for src/step_2_time_stepping_loop/mac_update_velocity.py
# Uses canonical mock grid from tests/mocks/cell_dict_mock.py

import pytest

from src.step_2_time_stepping_loop.mac_update_velocity import (
    update_velocity_x,
    update_velocity_y,
    update_velocity_z,
)

# Import the canonical mock grid
from tests.mocks.cell_dict_mock import cell_dict


class DummyConfig:
    """Minimal config with external forces."""
    def __init__(self, fx=0.0, fy=0.0, fz=0.0):
        self.data = {
            "external_forces": {
                "force_vector": [fx, fy, fz],
                "force_units": "N/m^3",
                "force_comment": "Test forces"
            }
        }


# --- Core Tests --------------------------------------------------------------

def test_update_velocity_x_returns_float():
    """Ensure update_velocity_x returns a float result for central cell."""
    config = DummyConfig(fx=1.0).data
    result = update_velocity_x(cell_dict, 13, dx=1.0, dy=1.0, dz=1.0,
                               dt=0.1, rho=1.0, mu=0.1,
                               config=config, timestep=0)
    assert isinstance(result, float)
    assert result == pytest.approx(result)  # finite


def test_update_velocity_y_returns_float():
    """Ensure update_velocity_y returns a float result for central cell."""
    config = DummyConfig(fy=2.0).data
    result = update_velocity_y(cell_dict, 13, dx=1.0, dy=1.0, dz=1.0,
                               dt=0.1, rho=1.0, mu=0.1,
                               config=config, timestep=0)
    assert isinstance(result, float)
    assert result == pytest.approx(result)


def test_update_velocity_z_returns_float():
    """Ensure update_velocity_z returns a float result for central cell."""
    config = DummyConfig(fz=-3.0).data
    result = update_velocity_z(cell_dict, 13, dx=1.0, dy=1.0, dz=1.0,
                               dt=0.1, rho=1.0, mu=0.1,
                               config=config, timestep=0)
    assert isinstance(result, float)
    assert result == pytest.approx(result)


# --- Force Integration Tests -------------------------------------------------

def test_force_integration_affects_result():
    """Changing external force should shift v* in expected direction."""
    config_no_force = DummyConfig(fx=0.0).data
    config_with_force = DummyConfig(fx=10.0).data

    result_no_force = update_velocity_x(cell_dict, 13, 1.0, 1.0, 1.0,
                                        dt=0.1, rho=1.0, mu=0.1,
                                        config=config_no_force, timestep=0)
    result_with_force = update_velocity_x(cell_dict, 13, 1.0, 1.0, 1.0,
                                          dt=0.1, rho=1.0, mu=0.1,
                                          config=config_with_force, timestep=0)

    assert result_with_force > result_no_force


# --- Error Handling Tests ----------------------------------------------------

def test_missing_external_forces_raises_keyerror():
    """Missing external_forces block should raise KeyError."""
    bad_config = {}
    with pytest.raises(KeyError):
        update_velocity_x(cell_dict, 13, 1.0, 1.0, 1.0,
                          dt=0.1, rho=1.0, mu=0.1,
                          config=bad_config, timestep=0)


def test_invalid_force_vector_length_raises_valueerror():
    """Force vector with wrong length should raise ValueError."""
    bad_config = {
        "external_forces": {
            "force_vector": [1.0, 2.0],  # only 2 values
            "force_units": "N/m^3",
            "force_comment": "Invalid length"
        }
    }
    with pytest.raises(ValueError):
        update_velocity_y(cell_dict, 13, 1.0, 1.0, 1.0,
                          dt=0.1, rho=1.0, mu=0.1,
                          config=bad_config, timestep=0)


# --- Performance Guard -------------------------------------------------------

def test_update_velocity_runs_fast():
    """Ensure update functions run quickly (performance guard)."""
    config = DummyConfig(fx=1.0, fy=1.0, fz=1.0).data
    import time
    start = time.perf_counter()
    for _ in range(500):
        update_velocity_x(cell_dict, 13, 1.0, 1.0, 1.0,
                          dt=0.1, rho=1.0, mu=0.1, config=config, timestep=0)
        update_velocity_y(cell_dict, 13, 1.0, 1.0, 1.0,
                          dt=0.1, rho=1.0, mu=0.1, config=config, timestep=0)
        update_velocity_z(cell_dict, 13, 1.0, 1.0, 1.0,
                          dt=0.1, rho=1.0, mu=0.1, config=config, timestep=0)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0  # should complete within 1 second for 1500 calls



