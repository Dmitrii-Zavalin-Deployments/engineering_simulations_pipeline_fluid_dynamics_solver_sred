# src/step_1_solver_initialization/neighbor_mapper.py
# 🧭 Maps stencil-safe neighbors for each flat_index in a 3D grid

from src.step_1_solver_initialization.indexing_utils import (
    grid_to_flat,
    flat_to_grid,
    is_valid_grid_index
)

# ✅ Centralized debug flag
debug = true

def get_stencil_neighbors(flat_index: int, shape: tuple[int, int, int]) -> dict:
    """
    Given a flat_index and grid shape, return stencil-safe neighbor flat indices.
    If a neighbor is out of bounds, its value will be None.
    """
    x, y, z = flat_to_grid(flat_index, shape)
    if debug:
        print(f"🧭 get_stencil_neighbors → flat_index={flat_index}, shape={shape}")
        print(f"   Coordinates: (x={x}, y={y}, z={z})")

    neighbors = {}

    # i-direction neighbors
    for dx, label in [(-1, "flat_index_i_minus_1"), (1, "flat_index_i_plus_1")]:
        nx = x + dx
        if is_valid_grid_index(nx, y, z, shape):
            neighbors[label] = grid_to_flat(nx, y, z, shape)
            if debug:
                print(f"   {label}: valid → {neighbors[label]} (from (x={nx}, y={y}, z={z}))")
        else:
            neighbors[label] = None
            if debug:
                print(f"   {label}: out of bounds → None")

    # j-direction neighbors
    for dy, label in [(-1, "flat_index_j_minus_1"), (1, "flat_index_j_plus_1")]:
        ny = y + dy
        if is_valid_grid_index(x, ny, z, shape):
            neighbors[label] = grid_to_flat(x, ny, z, shape)
            if debug:
                print(f"   {label}: valid → {neighbors[label]} (from (x={x}, y={ny}, z={z}))")
        else:
            neighbors[label] = None
            if debug:
                print(f"   {label}: out of bounds → None")

    # k-direction neighbors
    for dz, label in [(-1, "flat_index_k_minus_1"), (1, "flat_index_k_plus_1")]:
        nz = z + dz
        if is_valid_grid_index(x, y, nz, shape):
            neighbors[label] = grid_to_flat(x, y, nz, shape)
            if debug:
                print(f"   {label}: valid → {neighbors[label]} (from (x={x}, y={y}, z={nz}))")
        else:
            neighbors[label] = None
            if debug:
                print(f"   {label}: out of bounds → None")

    if debug:
        print(f"🧭 Neighbor mapping complete for flat_index={flat_index}\n")

    return neighbors



