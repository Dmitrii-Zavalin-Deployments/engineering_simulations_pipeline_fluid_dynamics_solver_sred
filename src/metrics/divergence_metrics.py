# src/metrics/divergence_metrics.py
# 📊 Divergence Metrics — computes ∇ · u across fluid cells using central difference in 3D
# 📌 This module enforces continuity diagnostics for reflex scoring and projection validation.
# It excludes only cells explicitly marked fluid_mask=False.
# It does NOT skip cells based on adjacency, boundary proximity, or velocity anomalies.

from src.grid_modules.cell import Cell
from typing import List, Tuple, Dict

# ✅ Centralized debug flag for GitHub Actions logging
debug = True

def compute_max_divergence(grid: List[Cell], domain: Dict) -> float:
    """
    Computes the maximum divergence across the simulation grid using central difference.
    Ignores ghost and solid cells, and only includes fluid cells with valid neighbors.

    Roadmap Alignment:
    Continuity Enforcement:
    - Measures ∇ · u before and after projection
    - Supports reflex scoring and projection diagnostics

    Diagnostic Role:
    - Tags cells with divergence values
    - Enables post-projection divergence tracking

    Returns:
        float: Maximum divergence value detected
    """
    if not grid or not domain:
        return 0.0

    dx = (domain.get("max_x", 1.0) - domain.get("min_x", 0.0)) / domain.get("nx", 1)
    dy = (domain.get("max_y", 1.0) - domain.get("min_y", 0.0)) / domain.get("ny", 1)
    dz = (domain.get("max_z", 1.0) - domain.get("min_z", 0.0)) / domain.get("nz", 1)

    grid_lookup: Dict[Tuple[float, float, float], Cell] = {
        (cell.x, cell.y, cell.z): cell for cell in grid
    }

    def safe_velocity(c: Cell) -> List[float]:
        return c.velocity if isinstance(c.velocity, list) and len(c.velocity) == 3 else [0.0, 0.0, 0.0]

    divergence_values = []

    for cell in grid:
        if not cell.fluid_mask:
            continue  # ❌ Explicit exclusion: solid or ghost cell

        coords = (cell.x, cell.y, cell.z)

        def neighbor_diff(axis_index: int, offset: float) -> float:
            coord_plus = list(coords)
            coord_minus = list(coords)
            coord_plus[axis_index] += offset
            coord_minus[axis_index] -= offset

            plus_cell = grid_lookup.get(tuple(coord_plus))
            minus_cell = grid_lookup.get(tuple(coord_minus))

            if plus_cell and plus_cell.fluid_mask and minus_cell and minus_cell.fluid_mask:
                v_plus = safe_velocity(plus_cell)[axis_index]
                v_minus = safe_velocity(minus_cell)[axis_index]
                spacing = [dx, dy, dz][axis_index]
                return (v_plus - v_minus) / (2.0 * spacing)
            return 0.0

        div_x = neighbor_diff(0, dx)
        div_y = neighbor_diff(1, dy)
        div_z = neighbor_diff(2, dz)

        divergence = div_x + div_y + div_z
        cell.divergence = round(divergence, 6)
        divergence_values.append(abs(divergence))

        if debug:
            print(f"[DIVERGENCE] Cell @ ({cell.x:.2f}, {cell.y:.2f}, {cell.z:.2f}) → ∇·u = {cell.divergence:.6f}")

    max_div = round(max(divergence_values), 5) if divergence_values else 0.0

    if debug:
        print(f"[DIVERGENCE] Max divergence across fluid cells: {max_div:.5f}")

    return max_div



