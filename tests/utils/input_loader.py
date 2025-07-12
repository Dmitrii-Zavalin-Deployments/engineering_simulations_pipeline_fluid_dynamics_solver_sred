# tests/utils/input_loader.py
# 🧪 Utility to load simulation input and decode geometry mask for test assertions

import json
from pathlib import Path
from src.utils.mask_interpreter import decode_geometry_mask_flat

def load_geometry_mask_bool(
    input_path: str = "data/testing-input-output/fluid_simulation_input.json"
) -> list[bool]:
    """
    Loads the simulation input JSON and returns the decoded geometry mask as a bool list.

    Args:
        input_path (str): Path to fluid_simulation_input.json

    Returns:
        List[bool]: True for fluid cells, False for solid cells
    """
    input_file = Path(input_path)
    if not input_file.is_file():
        raise FileNotFoundError(f"❌ Input file not found: {input_path}")

    with input_file.open() as f:
        config = json.load(f)

    geometry = config.get("geometry_definition")
    if not geometry:
        raise ValueError("❌ Missing 'geometry_definition' in simulation input")

    flat = geometry["geometry_mask_flat"]
    shape = geometry["geometry_mask_shape"]
    encoding = geometry.get("mask_encoding", {"fluid": 1, "solid": 0})
    order = geometry.get("flattening_order", "x-major")

    return decode_geometry_mask_flat(flat, shape, encoding, order)



