# src/solvers/pressure_solver.py
# 🔧 Pressure solver — enforces incompressibility via divergence correction

from typing import List, Tuple, Dict
from src.grid_modules.cell import Cell
from src.physics.divergence import compute_divergence
from src.physics.pressure_projection import solve_pressure_poisson

def apply_pressure_correction(grid: List[Cell], input_data: dict, step: int) -> Tuple[List[Cell], bool, int, Dict]:
    """
    Applies pressure correction to enforce incompressible flow.

    Args:
        grid (List[Cell]): Grid of simulation cells
        input_data (dict): Full simulation config
        step (int): Current simulation step index

    Returns:
        Tuple containing:
        - List[Cell]: Grid with updated pressure and velocity values
        - bool: Whether any pressure values changed
        - int: Number of projection iterations or passes
        - dict: Metadata about the correction process, including mutated_cells
    """
    # 🧼 Step 0: Downgrade malformed fluid cells to solid (invalid velocity structure)
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
    ]

    # 🔍 Step 1: Compute divergence of velocity field for valid fluid cells
    divergence = compute_divergence(safe_grid)
    max_div = max(abs(d) for d in divergence) if divergence else 0.0
    print(f"📊 Step {step}: Max divergence = {max_div:.6e}")

    # ⚡ Step 2: Solve pressure Poisson equation based on divergence
    grid_with_pressure, pressure_mutated = solve_pressure_poisson(safe_grid, divergence, input_data)

    # 🧪 Step 2.5: Mutation diagnostics and pressure delta tracking
    mutation_count = 0
    mutated_cells = []
    for old, updated in zip(safe_grid, grid_with_pressure):
        if updated.fluid_mask:
            initial = old.pressure if isinstance(old.pressure, float) else 0.0
            final = updated.pressure if isinstance(updated.pressure, float) else 0.0
            if abs(final - initial) > 1e-8:
                mutation_count += 1
                mutated_cells.append((updated.x, updated.y, updated.z))
                source = "ghost" if getattr(updated, "influenced_by_ghost", False) else "solver"
                print(f"[DEBUG] Pressure updated @ ({updated.x:.2f}, {updated.y:.2f}, {updated.z:.2f}) ← source: {source}")

    if mutation_count == 0:
        print(f"⚠️ Step {step}: Pressure solver ran but no pressure values changed.")
    else:
        print(f"✅ Step {step}: Pressure correction modified {mutation_count} fluid cells.")

    # 📦 Step 3: Prepare solver metadata
    metadata = {
        "max_divergence": max_div,
        "pressure_mutation_count": mutation_count,
        "pressure_solver_passes": 1,  # Placeholder: adapt if iteration count available
        "mutated_cells": mutated_cells
    }

    # 📤 Step 4: Return all expected outputs
    return grid_with_pressure, pressure_mutated, metadata["pressure_solver_passes"], metadata



