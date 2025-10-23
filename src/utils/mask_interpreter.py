# src/utils/mask_interpreter.py
# 🧩 Shared utility for interpreting geometry mask flattening
# 📌 This module decodes flattened geometry masks into fluid/solid booleans for grid initialization.
# It includes only cells explicitly marked as fluid via encoding["fluid"].
# It does NOT skip based on adjacency or ghost proximity — all logic is encoding-driven.

from typing import List

# ✅ Centralized debug flag for GitHub Actions logging
debug = True

def decode_geometry_mask_flat(
    flat_mask: List[int],
    shape: List[int],
    encoding: dict = {"fluid": 1, "solid": 0},
    order: str = "x-major"
) -> List[bool]:
    """
    Decodes a flattened geometry mask into a bool list matching grid indexing.

    Args:
        flat_mask (List[int]): Flattened mask (e.g. [1,0,1,...])
        shape (List[int]): [nx, ny, nz] shape used during flattening
        encoding (dict): Values used to represent fluid and solid (default: 1/0)
        order (str): Flattening order: "x-major", "y-major", "z-major"

    Returns:
        List[bool]: Interpreted mask with True for fluid, False for solid
    """
    nx, ny, nz = shape
    total = nx * ny * nz
    if len(flat_mask) != total:
        raise ValueError(f"❌ Mask length {len(flat_mask)} does not match expected shape {shape} = {total}")

    interpreted = []

    if order == "x-major":
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    index = i + j * nx + k * nx * ny
                    val = flat_mask[index]
                    interpreted.append(val == encoding["fluid"])
    elif order == "y-major":
        for k in range(nz):
            for i in range(nx):
                for j in range(ny):
                    index = j + i * ny + k * nx * ny
                    val = flat_mask[index]
                    interpreted.append(val == encoding["fluid"])
    elif order == "z-major":
        for j in range(ny):
            for i in range(nx):
                for k in range(nz):
                    index = k + i * nz + j * nx * nz
                    val = flat_mask[index]
                    interpreted.append(val == encoding["fluid"])
    else:
        raise ValueError(f"❌ Unsupported flattening order: {order}")

    if debug:
        print(f"[MASK] Decoded mask → fluid cells: {sum(interpreted)} / {len(interpreted)}")

    return interpreted



