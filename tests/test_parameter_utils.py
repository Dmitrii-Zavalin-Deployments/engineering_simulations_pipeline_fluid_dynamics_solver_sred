# tests/test_parameter_utils.py
# ✅ Validation tests for src/step_2_time_stepping_loop/parameter_utils.py
#
# Covers:
#   - Happy path (valid config)
#   - Missing required blocks
#   - Invalid scalar values (dt, rho, mu)
#   - Invalid domain definitions (nx, ny, nz, spacings)
#   - Invalid external forces
#   - Performance guard (function runs quickly)

import pytest
from src.step_2_time_stepping_loop.parameter_utils import load_solver_parameters


# --- Fixtures ---------------------------------------------------------------

@pytest.fixture
def valid_config():
    return {
        "simulation_parameters": {"time_step": 0.1},
        "fluid_properties": {"density": 1.137, "viscosity": 0.09},
        "domain_definition": {
            "x_min": 0.0, "x_max": 3.0,
            "y_min": -1.5, "y_max": 1.5,
            "z_min": -1.5, "z_max": 1.5,
            "nx": 3, "ny": 3, "nz": 3,
        },
        "external_forces": {"force_vector": [0.0, 0.0, 0.0]},
    }


# --- Happy Path -------------------------------------------------------------

def test_valid_config_returns_expected_dict(valid_config):
    params = load_solver_parameters(valid_config)
    assert isinstance(params, dict)
    assert set(params.keys()) == {"dt", "rho", "mu", "dx", "dy", "dz", "Fx", "Fy", "Fz"}
    assert params["dt"] == pytest.approx(0.1)
    assert params["rho"] == pytest.approx(1.137)
    assert params["mu"] == pytest.approx(0.09)
    assert params["dx"] == pytest.approx(1.0)   # (3.0 - 0.0)/3
    assert params["dy"] == pytest.approx(1.0)   # (1.5 - (-1.5))/3
    assert params["dz"] == pytest.approx(1.0)   # (1.5 - (-1.5))/3
    assert params["Fx"] == 0.0
    assert params["Fy"] == 0.0
    assert params["Fz"] == 0.0


# --- Missing Blocks ---------------------------------------------------------

@pytest.mark.parametrize("missing_block", [
    "simulation_parameters", "fluid_properties", "domain_definition", "external_forces"
])
def test_missing_required_blocks_raise_keyerror(valid_config, missing_block):
    bad_config = dict(valid_config)
    bad_config.pop(missing_block)
    with pytest.raises(KeyError):
        load_solver_parameters(bad_config)


# --- Invalid Scalars --------------------------------------------------------

def test_invalid_dt_raises_valueerror(valid_config):
    valid_config["simulation_parameters"]["time_step"] = -0.1
    with pytest.raises(ValueError):
        load_solver_parameters(valid_config)


def test_invalid_rho_raises_valueerror(valid_config):
    valid_config["fluid_properties"]["density"] = 0.0
    with pytest.raises(ValueError):
        load_solver_parameters(valid_config)


def test_invalid_mu_raises_valueerror(valid_config):
    valid_config["fluid_properties"]["viscosity"] = -1.0
    with pytest.raises(ValueError):
        load_solver_parameters(valid_config)


# --- Invalid Domain ---------------------------------------------------------

def test_missing_domain_field_raises_keyerror(valid_config):
    valid_config["domain_definition"].pop("nx")
    with pytest.raises(KeyError):
        load_solver_parameters(valid_config)


@pytest.mark.parametrize("nx,ny,nz", [
    (0, 3, 3), (3, 0, 3), (3, 3, 0)
])
def test_invalid_grid_resolution_raises_valueerror(valid_config, nx, ny, nz):
    valid_config["domain_definition"]["nx"] = nx
    valid_config["domain_definition"]["ny"] = ny
    valid_config["domain_definition"]["nz"] = nz
    with pytest.raises(ValueError):
        load_solver_parameters(valid_config)


def test_invalid_spacings_raise_valueerror(valid_config):
    valid_config["domain_definition"]["x_min"] = 3.0
    valid_config["domain_definition"]["x_max"] = 3.0
    with pytest.raises(ValueError):
        load_solver_parameters(valid_config)


# --- Invalid Forces ---------------------------------------------------------

def test_missing_force_vector_raises_valueerror(valid_config):
    valid_config["external_forces"].pop("force_vector")
    with pytest.raises(ValueError):
        load_solver_parameters(valid_config)


def test_force_vector_wrong_length_raises_valueerror(valid_config):
    valid_config["external_forces"]["force_vector"] = [1.0, 2.0]
    with pytest.raises(ValueError):
        load_solver_parameters(valid_config)


# --- Performance Guard ------------------------------------------------------

def test_load_solver_parameters_runs_fast(valid_config):
    import time
    start = time.perf_counter()
    for _ in range(1000):
        load_solver_parameters(valid_config)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0  # should complete within 1 second for 1000 calls



