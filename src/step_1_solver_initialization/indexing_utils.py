# step_1_solver_initialization/indexing_utils.py
# 🔁 Converts between flat_index and grid_index [x, y, z] using x-major (row-major) flattening logic

def grid_to_flat(x: int, y: int, z: int, shape: tuple[int, int, int]) -> int:
    """
    Convert 3D grid index (x, y, z) to flat_index using x-major order.
    shape = (nx, ny, nz)
    """
    nx, ny, nz = shape
    return x + nx * (y + ny * z)


def flat_to_grid(flat_index: int, shape: tuple[int, int, int]) -> list[int]:
    """
    Convert flat_index to 3D grid index [x, y, z] using x-major order.
    shape = (nx, ny, nz)
    """
    nx, ny, nz = shape
    z = flat_index // (nx * ny)
    y = (flat_index % (nx * ny)) // nx
    x = flat_index % nx
    return [x, y, z]


def is_valid_grid_index(x: int, y: int, z: int, shape: tuple[int, int, int]) -> bool:
    """
    Check if the grid index [x, y, z] is within bounds of the domain shape.
    """
    nx, ny, nz = shape
    return 0 <= x < nx and 0 <= y < ny and 0 <= z < nz


def is_valid_flat_index(flat_index: int, shape: tuple[int, int, int]) -> bool:
    """
    Check if the flat index is within bounds of the flattened domain.
    """
    nx, ny, nz = shape
    return 0 <= flat_index < nx * ny * nz



