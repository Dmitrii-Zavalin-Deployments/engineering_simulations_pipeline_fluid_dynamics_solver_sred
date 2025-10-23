# src/physics/divergence.py
# 📈 Divergence Operator — ghost-aware ∇ · u computation for continuity enforcement
# 📌 This module computes divergence across fluid cells using central difference.
# It excludes ghost cells and cells explicitly marked fluid_mask=False.
# It does NOT skip based on adjacency or boundary proximity.

from src.grid_modules.cell import Cell
from typing import List, Set
from src.physics.divergence_methods.central import compute_central_divergence

# ✅ Centralized debug flag for GitHub Actions logging
debug = True

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
    safe_grid = []
    for i, cell in enumerate(grid):
        if id(cell) in ghost_registry:
            if debug:
                print(f"[DIVERGENCE] ⛔️ Skipping ghost cell[{i}] @ ({cell.x:.2f}, {cell.y:.2f}, {cell.z:.2f})")
            continue
        has_valid_velocity = cell.fluid_mask and isinstance(cell.velocity, list)
        if debug and not has_valid_velocity:
            print(f"[DIVERGENCE] ⚠️ Downgrading cell[{i}] @ ({cell.x:.2f}, {cell.y:.2f}, {cell.z:.2f}) — invalid velocity or fluid_mask")
        safe_cell = Cell(
            x=cell.x,
            y=cell.y,
            z=cell.z,
            velocity=cell.velocity if has_valid_velocity else None,
            pressure=None,  # ⛔️ Pressure excluded from divergence logic
            fluid_mask=has_valid_velocity
        )
        safe_grid.append(safe_cell)

    if debug:
        print(f"[DIVERGENCE] ✅ Safe grid assembled → {len(safe_grid)} cells")

    # 🧪 Step 2: Compute divergence
    divergence_values = compute_central_divergence(safe_grid, config)

    # 📊 Optional logging of results
    if debug and verbose:
        for i, value in enumerate(divergence_values):
            cell = safe_grid[i]
            coord = (cell.x, cell.y, cell.z)
            print(f"[DIVERGENCE] ∇·u @ {coord}: {value:.6e}")

    if debug:
        if divergence_values:
            max_div = max(abs(v) for v in divergence_values)
            print(f"[DIVERGENCE] 📈 Max divergence (excluding ghosts): {max_div:.6e}")
        else:
            print("[DIVERGENCE] ⚠️ Divergence computation returned empty list")

    return divergence_values



