# src/step_1_solver_initialization/indexing_utils.py
# 🔁 Converts between flat_index and grid_index [x, y, z] using x-major (row-major) flattening logic

# ✅ Centralized debug flag
debug = False

def grid_to_flat(x: int, y: int, z: int, shape: tuple[int, int, int]) -> int:
    """
    Convert 3D grid index (x, y, z) to flat_index using x-major order.
    Formula: flat_index = x + nx * (y + ny * z)
    """
    nx, ny, nz = shape
    flat_index = x + nx * (y + ny * z)
    if debug:
        print(f"📐 grid_to_flat → (x={x}, y={y}, z={z}, shape={shape})")
        print(f"   Formula: flat_index = x + nx*(y + ny*z)")
        print(f"   Result: flat_index = {flat_index}")
    return flat_index

def flat_to_grid(flat_index: int, shape: tuple[int, int, int]) -> list[int]:
    """
    Convert flat_index to 3D grid index [x, y, z] using x-major order.
    Reverse mapping:
      z = flat_index // (nx*ny)
      y = (flat_index % (nx*ny)) // nx
      x = flat_index % nx
    """
    nx, ny, nz = shape
    z = flat_index // (nx * ny)
    y = (flat_index % (nx * ny)) // nx
    x = flat_index % nx
    if debug:
        print(f"📐 flat_to_grid → flat_index={flat_index}, shape={shape}")
        print(f"   Reverse mapping formulas:")
        print(f"   z = flat_index // (nx*ny) → {z}")
        print(f"   y = (flat_index % (nx*ny)) // nx → {y}")
        print(f"   x = flat_index % nx → {x}")
        print(f"   Result: (x={x}, y={y}, z={z})")
    return [x, y, z]

def is_valid_grid_index(x: int, y: int, z: int, shape: tuple[int, int, int]) -> bool:
    nx, ny, nz = shape
    valid = 0 <= x < nx and 0 <= y < ny and 0 <= z < nz
    if debug:
        print(f"🔍 is_valid_grid_index → (x={x}, y={y}, z={z}, shape={shape}) → {valid}")
    return valid

def is_valid_flat_index(flat_index: int, shape: tuple[int, int, int]) -> bool:
    nx, ny, nz = shape
    valid = 0 <= flat_index < nx * ny * nz
    if debug:
        print(f"🔍 is_valid_flat_index → flat_index={flat_index}, shape={shape} → {valid}")
    return valid



