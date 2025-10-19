# src/physics/divergence.py
# 📈 Divergence Operator — ghost-aware ∇ · u computation for continuity enforcement

from src.grid_modules.cell import Cell
from typing import List, Set
from src.physics.divergence_methods.central import compute_central_divergence

def compute_divergence(grid: List[Cell],
                       config: dict = {},
                       ghost_registry: Set[int] = set(),
                       verbose: bool = False) -> List[float]:
    """
    Computes divergence values for valid fluid cells using central-difference approximation,
    excluding ghost cells.

    Roadmap Alignment:
    Governing Equation:
        Continuity: ∇ · u = ∂u/∂x + ∂v/∂y + ∂w/∂z

    Purpose:
    - Quantify incompressibility violation
    - Feed pressure Poisson solver: ∇²P = ∇ · u
    - Support reflex scoring and mutation diagnostics
    - Exclude ghost cells to preserve physical fidelity at boundaries

    Strategy:
    1. Filter out ghost cells and malformed fluid cells
    2. Apply central differencing via compute_central_divergence
    3. Optionally log per-cell divergence values

    Args:
        grid (List[Cell]): Grid of Cell objects
        config (dict): Domain configuration including spacing and resolution
        ghost_registry (Set[int]): Set of ghost cell IDs to exclude
        verbose (bool): If True, logs per-cell divergence values

    Returns:
        List[float]: Divergence values for fluid cells (order matches input)
    """
    # 🧼 Step 1: Downgrade malformed fluid cells and exclude ghosts
    safe_grid = [
        Cell(
            x=cell.x,
            y=cell.y,
            z=cell.z,
            velocity=cell.velocity if cell.fluid_mask and isinstance(cell.velocity, list) else None,
            pressure=cell.pressure if cell.fluid_mask and isinstance(cell.velocity, list) else None,
            fluid_mask=cell.fluid_mask if cell.fluid_mask and isinstance(cell.velocity, list) else False
        )
        for cell in grid
        if id(cell) not in ghost_registry
    ]

    # 🧪 Step 2: Compute divergence
    divergence_values = compute_central_divergence(safe_grid, config)

    # 📊 Optional logging of results
    if verbose:
        for i, value in enumerate(divergence_values):
            cell = safe_grid[i]
            coord = (cell.x, cell.y, cell.z)
            print(f"🧭 Divergence at {coord}: {value:.6e}")

    if divergence_values:
        max_div = max(abs(v) for v in divergence_values)
        print(f"📈 Max divergence (excluding ghosts): {max_div:.6e}")
    else:
        print("⚠️ Divergence computation returned empty list")

    return divergence_values



