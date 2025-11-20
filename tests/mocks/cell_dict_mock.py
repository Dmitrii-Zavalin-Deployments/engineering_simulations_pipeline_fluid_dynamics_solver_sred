# tests/mocks/cell_dict_mock.py
# Central + axis neighbors only, 3×3×3 grid, x-major flattening
# Flat index formula: flat_index = x + nx * (y + ny * z), with nx=ny=nz=3

cell_dict = {
    # Central cell (1,1,1) -> flat_index = 13
    "13": {
        "flat_index": 13,
        "grid_index": [1, 1, 1],
        "cell_type": "fluid",
        "boundary_role": None,
        "flat_index_i_minus_1": 12,  # (0,1,1)
        "flat_index_i_plus_1": 14,   # (2,1,1)
        "flat_index_j_minus_1": 10,  # (1,0,1)
        "flat_index_j_plus_1": 16,   # (1,2,1)
        "flat_index_k_minus_1": 4,   # (1,1,0)
        "flat_index_k_plus_1": 22,   # (1,1,2)
        "time_history": {
            "0": {"pressure": 100.0, "velocity": {"vx": 1.0, "vy": 1.0, "vz": 1.0}},
            "1": {"pressure": 101.0, "velocity": {"vx": 1.1, "vy": 1.1, "vz": 1.1}},
        },
    },

    # X- neighbor (0,1,1) -> flat_index = 12
    "12": {
        "flat_index": 12,
        "grid_index": [0, 1, 1],
        "cell_type": "fluid",
        "boundary_role": None,
        "flat_index_i_plus_1": 13,
        "time_history": {
            "0": {"pressure": 99.0, "velocity": {"vx": 0.5, "vy": 1.0, "vz": 1.0}},
            "1": {"pressure": 99.5, "velocity": {"vx": 0.6, "vy": 1.1, "vz": 1.1}},
        },
    },

    # X+ neighbor (2,1,1) -> flat_index = 14
    "14": {
        "flat_index": 14,
        "grid_index": [2, 1, 1],
        "cell_type": "fluid",
        "boundary_role": None,
        "flat_index_i_minus_1": 13,
        "time_history": {
            "0": {"pressure": 101.0, "velocity": {"vx": 1.5, "vy": 1.0, "vz": 1.0}},
            "1": {"pressure": 101.5, "velocity": {"vx": 1.6, "vy": 1.1, "vz": 1.1}},
        },
    },

    # Y- neighbor (1,0,1) -> flat_index = 10
    "10": {
        "flat_index": 10,
        "grid_index": [1, 0, 1],
        "cell_type": "fluid",
        "boundary_role": None,
        "flat_index_j_plus_1": 13,
        "time_history": {
            "0": {"pressure": 98.0, "velocity": {"vx": 1.0, "vy": 0.5, "vz": 1.0}},
            "1": {"pressure": 98.5, "velocity": {"vx": 1.1, "vy": 0.6, "vz": 1.1}},
        },
    },

    # Y+ neighbor (1,2,1) -> flat_index = 16
    "16": {
        "flat_index": 16,
        "grid_index": [1, 2, 1],
        "cell_type": "fluid",
        "boundary_role": None,
        "flat_index_j_minus_1": 13,
        "time_history": {
            "0": {"pressure": 102.0, "velocity": {"vx": 1.0, "vy": 1.5, "vz": 1.0}},
            "1": {"pressure": 102.5, "velocity": {"vx": 1.1, "vy": 1.6, "vz": 1.1}},
        },
    },

    # Z- neighbor (1,1,0) -> flat_index = 4
    "4": {
        "flat_index": 4,
        "grid_index": [1, 1, 0],
        "cell_type": "fluid",
        "boundary_role": None,
        "flat_index_k_plus_1": 13,
        "time_history": {
            "0": {"pressure": 97.0, "velocity": {"vx": 1.0, "vy": 1.0, "vz": 0.5}},
            "1": {"pressure": 97.5, "velocity": {"vx": 1.1, "vy": 1.1, "vz": 0.6}},
        },
    },

    # Z+ neighbor (1,1,2) -> flat_index = 22
    "22": {
        "flat_index": 22,
        "grid_index": [1, 1, 2],
        "cell_type": "fluid",
        "boundary_role": None,
        "flat_index_k_minus_1": 13,
        "time_history": {
            "0": {"pressure": 103.0, "velocity": {"vx": 1.0, "vy": 1.0, "vz": 1.5}},
            "1": {"pressure": 103.5, "velocity": {"vx": 1.1, "vy": 1.1, "vz": 1.6}},
        },
    },
}



