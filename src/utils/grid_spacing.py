# src/utils/grid_spacing.py

def compute_grid_spacing(domain: dict) -> tuple[float, float, float]:
    """
    Computes dx, dy, dz spacing from domain definition.
    """
    dx = (domain["max_x"] - domain["min_x"]) / domain["nx"]
    dy = (domain["max_y"] - domain["min_y"]) / domain["ny"]
    dz = (domain["max_z"] - domain["min_z"]) / domain["nz"]
    return dx, dy, dz
