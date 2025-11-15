# tests/test_cell_builder.py
# ✅ Unit tests for step_1_solver_initialization/cell_builder.py

import pytest
from src.step_1_solver_initialization.cell_builder import build_cell_dict, debug

# --- Helper config generator ---
def make_config(nx, ny, nz, mask_flat, mask_encoding, init_pressure=100.0, init_velocity=(1.0, 2.0, 3.0), boundary_conditions=None):
    return {
        "domain_definition": {"nx": nx, "ny": ny, "nz": nz},
        "geometry_definition": {
            "geometry_mask_flat": mask_flat,
            "mask_encoding": mask_encoding,
        },
        "initial_conditions": {
            "initial_pressure": init_pressure,
            "initial_velocity": init_velocity,
        },
        "boundary_conditions": boundary_conditions or [],
    }

# --- Fluid cell classification ---
def test_fluid_cell_classification():
    mask_encoding = {"fluid": 0, "solid": 1, "boundary": 2}
    config = make_config(1, 1, 1, [0], mask_encoding)
    cell_dict = build_cell_dict(config)
    cell = cell_dict[0]
    assert cell["cell_type"] == "fluid"
    assert cell["boundary_role"] is None

# --- Solid cell classification ---
def test_solid_cell_classification():
    mask_encoding = {"fluid": 0, "solid": 1, "boundary": 2}
    config = make_config(1, 1, 1, [1], mask_encoding)
    cell_dict = build_cell_dict(config)
    cell = cell_dict[0]
    assert cell["cell_type"] == "solid"
    assert cell["boundary_role"] is None

# --- Boundary cell classification with roles ---
@pytest.mark.parametrize("apply_faces, expected_role", [
    (["x_min"], "inlet"),
    (["x_max"], "outlet"),
    (["y_min"], "wall"),
    (["y_max"], "wall"),
    (["z_min"], "inlet"),
    (["z_max"], "outlet"),
    (["wall"], "wall"),
])
def test_boundary_roles(apply_faces, expected_role):
    mask_encoding = {"fluid": 0, "solid": 1, "boundary": 2}
    bc = [{"apply_faces": apply_faces, "role": expected_role}]
    config = make_config(2, 2, 2, [2]*8, mask_encoding, boundary_conditions=bc)
    cell_dict = build_cell_dict(config)
    # Check at least one boundary cell has the expected role
    roles = [cell["boundary_role"] for cell in cell_dict.values()]
    assert expected_role in roles

# --- Fallback classification ---
def test_fallback_classification():
    mask_encoding = {"fluid": 0, "solid": 1, "boundary": 2}
    config = make_config(1, 1, 1, [99], mask_encoding)
    cell_dict = build_cell_dict(config)
    cell = cell_dict[0]
    assert cell["cell_type"] == "fluid"  # fallback

# --- Time history initialization ---
def test_time_history_initialization():
    mask_encoding = {"fluid": 0, "solid": 1, "boundary": 2}
    config = make_config(1, 1, 1, [0], mask_encoding, init_pressure=200.0, init_velocity=(5.0, 6.0, 7.0))
    cell_dict = build_cell_dict(config)
    history = cell_dict[0]["time_history"][0]
    assert history["pressure"] == 200.0
    assert history["velocity"] == {"vx": 5.0, "vy": 6.0, "vz": 7.0}

# --- Neighbor mapping integration ---
def test_neighbor_keys_exist():
    mask_encoding = {"fluid": 0, "solid": 1, "boundary": 2}
    config = make_config(2, 2, 2, [0]*8, mask_encoding)
    cell_dict = build_cell_dict(config)
    cell = cell_dict[0]
    for key in ["flat_index_i_plus_1","flat_index_i_minus_1","flat_index_j_plus_1","flat_index_j_minus_1","flat_index_k_plus_1","flat_index_k_minus_1"]:
        assert key in cell

# --- Domain shape variations ---
def test_single_cell_domain():
    mask_encoding = {"fluid": 0, "solid": 1, "boundary": 2}
    config = make_config(1, 1, 1, [0], mask_encoding)
    cell_dict = build_cell_dict(config)
    assert len(cell_dict) == 1

def test_multi_cell_domain():
    mask_encoding = {"fluid": 0, "solid": 1, "boundary": 2}
    config = make_config(2, 2, 2, [0]*8, mask_encoding)
    cell_dict = build_cell_dict(config)
    assert len(cell_dict) == 8

# --- Boundary conditions absent ---
def test_boundary_conditions_absent():
    mask_encoding = {"fluid": 0, "solid": 1, "boundary": 2}
    config = make_config(1, 1, 1, [2], mask_encoding, boundary_conditions=None)
    cell_dict = build_cell_dict(config)
    assert cell_dict[0]["boundary_role"] is None

# --- Debug flag coverage ---
def test_debug_flag_output(capsys):
    from src.step_1_solver_initialization import cell_builder
    cell_builder.debug = True
    mask_encoding = {"fluid": 0, "solid": 1, "boundary": 2}
    config = make_config(1, 1, 1, [0], mask_encoding)
    _ = cell_builder.build_cell_dict(config)
    captured = capsys.readouterr()
    assert "🧱 Built cell" in captured.out
    cell_builder.debug = False



