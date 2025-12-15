# tests/test_mac_interpolation_vx.py
# Comprehensive unit tests for src/step_2_time_stepping_loop/mac_interpolation/vx.py

import pytest
from src.step_2_time_stepping_loop.mac_interpolation import vx
from tests.mocks.cell_dict_mock import cell_dict


# --- Basic interpolation correctness -----------------------------------------

def test_vx_i_plus_half_t0():
    result = vx.vx_i_plus_half(cell_dict, 13, timestep=0)
    central = cell_dict["13"]["time_history"]["0"]["velocity"]["vx"]
    neighbor_index = cell_dict["13"]["flat_index_i_plus_1"]
    neighbor = cell_dict[str(neighbor_index)]["time_history"]["0"]["velocity"]["vx"]
    expected = 0.5 * (central + neighbor)
    assert result == pytest.approx(expected)


def test_vx_i_plus_half_latest_equals_t1():
    result_latest = vx.vx_i_plus_half(cell_dict, 13)
    result_t1 = vx.vx_i_plus_half(cell_dict, 13, timestep=1)
    assert result_latest == pytest.approx(result_t1)


def test_vx_i_minus_half_t0():
    result = vx.vx_i_minus_half(cell_dict, 13, timestep=0)
    central = cell_dict["13"]["time_history"]["0"]["velocity"]["vx"]
    neighbor_index = cell_dict["13"]["flat_index_i_minus_1"]
    neighbor = cell_dict[str(neighbor_index)]["time_history"]["0"]["velocity"]["vx"]
    expected = 0.5 * (central + neighbor)
    assert result == pytest.approx(expected)


def test_vx_i_plus_three_half_t0():
    result = vx.vx_i_plus_three_half(cell_dict, 13, timestep=0)
    ip1 = cell_dict["13"]["flat_index_i_plus_1"]
    ip2 = cell_dict[str(ip1)]["flat_index_i_plus_1"]
    v_ip1 = cell_dict[str(ip1)]["time_history"]["0"]["velocity"]["vx"]
    v_ip2 = cell_dict[str(ip2)]["time_history"]["0"]["velocity"]["vx"]
    expected = 0.5 * (v_ip1 + v_ip2)
    assert result == pytest.approx(expected)


def test_vx_i_minus_three_half_t0(monkeypatch):
    # Force missing left neighbor to simulate boundary
    monkeypatch.setitem(cell_dict["12"], "flat_index_i_minus_1", None)
    result = vx.vx_i_minus_three_half(cell_dict, 13, timestep=0)
    # Should fallback to velocity at i-1
    expected = cell_dict["12"]["time_history"]["0"]["velocity"]["vx"]
    assert result == pytest.approx(expected)


def test_vx_j_plus_one_t0():
    result = vx.vx_j_plus_one(cell_dict, 13, timestep=0)
    central = cell_dict["13"]["time_history"]["0"]["velocity"]["vx"]
    neighbor_index = cell_dict["13"]["flat_index_j_plus_1"]
    neighbor = cell_dict[str(neighbor_index)]["time_history"]["0"]["velocity"]["vx"]
    expected = 0.5 * (central + neighbor)
    assert result == pytest.approx(expected)


def test_vx_k_minus_one_t1():
    result = vx.vx_k_minus_one(cell_dict, 13, timestep=1)
    central = cell_dict["13"]["time_history"]["1"]["velocity"]["vx"]
    neighbor_index = cell_dict["13"]["flat_index_k_minus_1"]
    neighbor = cell_dict[str(neighbor_index)]["time_history"]["1"]["velocity"]["vx"]
    expected = 0.5 * (central + neighbor)
    assert result == pytest.approx(expected)


# --- Boundary fallback (Neumann condition) -----------------------------------

def test_vx_i_plus_half_fallback(monkeypatch):
    monkeypatch.setitem(cell_dict["13"], "flat_index_i_plus_1", None)
    result = vx.vx_i_plus_half(cell_dict, 13, timestep=0)
    central = cell_dict["13"]["time_history"]["0"]["velocity"]["vx"]
    assert result == pytest.approx(central)


def test_vx_i_plus_three_half_missing_second(monkeypatch):
    ip1 = cell_dict["13"]["flat_index_i_plus_1"]
    monkeypatch.setitem(cell_dict[str(ip1)], "flat_index_i_plus_1", None)
    result = vx.vx_i_plus_three_half(cell_dict, 13, timestep=0)
    expected = cell_dict[str(ip1)]["time_history"]["0"]["velocity"]["vx"]
    assert result == pytest.approx(expected)


def test_vx_j_minus_one_fallback(monkeypatch):
    monkeypatch.setitem(cell_dict["13"], "flat_index_j_minus_1", None)
    result = vx.vx_j_minus_one(cell_dict, 13, timestep=0)
    central = cell_dict["13"]["time_history"]["0"]["velocity"]["vx"]
    assert result == pytest.approx(central)


def test_vx_k_plus_one_fallback(monkeypatch):
    monkeypatch.setitem(cell_dict["13"], "flat_index_k_plus_1", None)
    result = vx.vx_k_plus_one(cell_dict, 13, timestep=0)
    central = cell_dict["13"]["time_history"]["0"]["velocity"]["vx"]
    assert result == pytest.approx(central)


# --- Debug logging -----------------------------------------------------------

def test_vx_debug_logging(capsys):
    vx.debug = True
    result = vx.vx_j_plus_one(cell_dict, 13, timestep=0)
    captured = capsys.readouterr()
    assert "vx_j+1" in captured.out
    vx.debug = False


# --- Robustness against malformed input --------------------------------------

def test_vx_invalid_timestep_raises_valueerror():
    with pytest.raises(ValueError):
        vx.vx_i_plus_half(cell_dict, 13, timestep=99)


def test_vx_missing_velocity_key(monkeypatch):
    # Temporarily remove velocity dict
    monkeypatch.setitem(cell_dict["13"]["time_history"]["0"], "velocity", {})
    with pytest.raises(KeyError):
        vx.vx_i_plus_half(cell_dict, 13, timestep=0)


# --- Symmetry sanity checks --------------------------------------------------

def test_vx_symmetry_equal_neighbors(monkeypatch):
    # Force central and neighbor vx equal
    monkeypatch.setitem(cell_dict["13"]["time_history"]["0"]["velocity"], "vx", 1.0)
    neighbor_index = cell_dict["13"]["flat_index_i_plus_1"]
    monkeypatch.setitem(cell_dict[str(neighbor_index)]["time_history"]["0"]["velocity"], "vx", 1.0)
    result = vx.vx_i_plus_half(cell_dict, 13, timestep=0)
    assert result == pytest.approx(1.0)



