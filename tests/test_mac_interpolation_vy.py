# tests/test_mac_interpolation_vy.py
# Comprehensive unit tests for src/step_2_time_stepping_loop/mac_interpolation/vy.py

import pytest
from src.step_2_time_stepping_loop.mac_interpolation import vy
from tests.mocks.cell_dict_mock import cell_dict


# --- Basic interpolation correctness -----------------------------------------

def test_vy_j_plus_half_t0():
    result = vy.vy_j_plus_half(cell_dict, 13, timestep=0)
    central = cell_dict["13"]["time_history"]["0"]["velocity"]["vy"]
    neighbor_index = cell_dict["13"]["flat_index_j_plus_1"]
    neighbor = cell_dict[str(neighbor_index)]["time_history"]["0"]["velocity"]["vy"]
    expected = 0.5 * (central + neighbor)
    assert result == pytest.approx(expected)


def test_vy_j_plus_half_latest_equals_t1():
    result_latest = vy.vy_j_plus_half(cell_dict, 13)
    result_t1 = vy.vy_j_plus_half(cell_dict, 13, timestep=1)
    assert result_latest == pytest.approx(result_t1)


def test_vy_j_minus_half_t0():
    result = vy.vy_j_minus_half(cell_dict, 13, timestep=0)
    central = cell_dict["13"]["time_history"]["0"]["velocity"]["vy"]
    neighbor_index = cell_dict["13"]["flat_index_j_minus_1"]
    neighbor = cell_dict[str(neighbor_index)]["time_history"]["0"]["velocity"]["vy"]
    expected = 0.5 * (central + neighbor)
    assert result == pytest.approx(expected)


def test_vy_j_plus_three_half_t0():
    result = vy.vy_j_plus_three_half(cell_dict, 10, timestep=0)
    jp1 = cell_dict["10"]["flat_index_j_plus_1"]
    jp2 = cell_dict[str(jp1)].get("flat_index_j_plus_1")
    if jp2 is None:
        # Fallback: only j+1 available
        expected = cell_dict[str(jp1)]["time_history"]["0"]["velocity"]["vy"]
    else:
        v_jp1 = cell_dict[str(jp1)]["time_history"]["0"]["velocity"]["vy"]
        v_jp2 = cell_dict[str(jp2)]["time_history"]["0"]["velocity"]["vy"]
        expected = 0.5 * (v_jp1 + v_jp2)
    assert result == pytest.approx(expected)


def test_vy_j_minus_three_half_t0(monkeypatch):
    # Force missing deeper neighbor to simulate boundary
    monkeypatch.setitem(cell_dict["10"], "flat_index_j_minus_1", None)
    result = vy.vy_j_minus_three_half(cell_dict, 16, timestep=0)
    # Should fallback to velocity at j-1 (central cell)
    expected = cell_dict["13"]["time_history"]["0"]["velocity"]["vy"]
    assert result == pytest.approx(expected)


def test_vy_i_plus_one_t0():
    result = vy.vy_i_plus_one(cell_dict, 13, timestep=0)
    central = cell_dict["13"]["time_history"]["0"]["velocity"]["vy"]
    neighbor_index = cell_dict["13"]["flat_index_i_plus_1"]
    neighbor = cell_dict[str(neighbor_index)]["time_history"]["0"]["velocity"]["vy"]
    expected = 0.5 * (central + neighbor)
    assert result == pytest.approx(expected)


def test_vy_k_minus_one_t1():
    result = vy.vy_k_minus_one(cell_dict, 13, timestep=1)
    central = cell_dict["13"]["time_history"]["1"]["velocity"]["vy"]
    neighbor_index = cell_dict["13"]["flat_index_k_minus_1"]
    neighbor = cell_dict[str(neighbor_index)]["time_history"]["1"]["velocity"]["vy"]
    expected = 0.5 * (central + neighbor)
    assert result == pytest.approx(expected)


# --- Boundary fallback (Neumann condition) -----------------------------------

def test_vy_j_plus_half_fallback(monkeypatch):
    monkeypatch.setitem(cell_dict["13"], "flat_index_j_plus_1", None)
    result = vy.vy_j_plus_half(cell_dict, 13, timestep=0)
    central = cell_dict["13"]["time_history"]["0"]["velocity"]["vy"]
    assert result == pytest.approx(central)


def test_vy_j_plus_three_half_missing_second(monkeypatch):
    jp1 = cell_dict["13"]["flat_index_j_plus_1"]
    monkeypatch.setitem(cell_dict[str(jp1)], "flat_index_j_plus_1", None)
    result = vy.vy_j_plus_three_half(cell_dict, 10, timestep=0)
    # Fallback: only j+1 available
    expected = cell_dict[str(jp1)]["time_history"]["0"]["velocity"]["vy"]
    assert result == pytest.approx(expected)


def test_vy_i_minus_one_fallback(monkeypatch):
    monkeypatch.setitem(cell_dict["13"], "flat_index_i_minus_1", None)
    result = vy.vy_i_minus_one(cell_dict, 13, timestep=0)
    central = cell_dict["13"]["time_history"]["0"]["velocity"]["vy"]
    assert result == pytest.approx(central)


def test_vy_k_plus_one_fallback(monkeypatch):
    monkeypatch.setitem(cell_dict["13"], "flat_index_k_plus_1", None)
    result = vy.vy_k_plus_one(cell_dict, 13, timestep=0)
    central = cell_dict["13"]["time_history"]["0"]["velocity"]["vy"]
    assert result == pytest.approx(central)


# --- Debug logging -----------------------------------------------------------

def test_vy_debug_logging(capsys):
    vy.debug = True
    result = vy.vy_i_plus_one(cell_dict, 13, timestep=0)
    captured = capsys.readouterr()
    assert "vy_i+1" in captured.out
    vy.debug = False


# --- Robustness against malformed input --------------------------------------

def test_vy_invalid_timestep_raises_valueerror():
    with pytest.raises(ValueError):
        vy.vy_j_plus_half(cell_dict, 13, timestep=99)


def test_vy_missing_velocity_key(monkeypatch):
    # Temporarily remove velocity dict
    monkeypatch.setitem(cell_dict["13"]["time_history"]["0"], "velocity", {})
    with pytest.raises(KeyError):
        vy.vy_j_plus_half(cell_dict, 13, timestep=0)


# --- Symmetry sanity checks --------------------------------------------------

def test_vy_symmetry_equal_neighbors(monkeypatch):
    # Force central and neighbor vy equal
    monkeypatch.setitem(cell_dict["13"]["time_history"]["0"]["velocity"], "vy", 2.0)
    neighbor_index = cell_dict["13"]["flat_index_j_plus_1"]
    monkeypatch.setitem(cell_dict[str(neighbor_index)]["time_history"]["0"]["velocity"], "vy", 2.0)
    result = vy.vy_j_plus_half(cell_dict, 13, timestep=0)
    assert result == pytest.approx(2.0)



