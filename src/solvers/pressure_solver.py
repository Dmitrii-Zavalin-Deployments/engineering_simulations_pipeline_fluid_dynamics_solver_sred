# src/solvers/pressure_solver.py
# 🔧 Pressure solver — enforces incompressibility via divergence correction

from typing import List
from src.grid_modules.cell import Cell
from src.physics.divergence import compute_divergence
from src.physics.pressure_projection import solve_pressure_poisson

def apply_pressure_correction(grid: List[Cell], input_data: dict, step: int) -> List[Cell]:
    """
    Applies pressure correction to enforce incompressible flow.

    Args:
        grid (List[Cell]): Grid of simulation cells
        input_data (dict): Full simulation config
        step (int): Current simulation step index

    Returns:
        List[Cell]: Grid with updated pressure values (fluid cells only)
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

    # ⚡ Step 2: Solve pressure Poisson equation based on divergence
    grid_with_pressure, pressure_mutated = solve_pressure_poisson(safe_grid, divergence, input_data)

    # 🧪 Step 2.5: Optional mutation diagnostics (non-functional logging only)
    mutation_count = 0
    for old, updated in zip(safe_grid, grid_with_pressure):
        if updated.fluid_mask:
            initial = old.pressure if isinstance(old.pressure, float) else 0.0
            final = updated.pressure if isinstance(updated.pressure, float) else 0.0
            if abs(final - initial) > 1e-6:
                mutation_count += 1

    if mutation_count == 0:
        print(f"⚠️ Pressure solver ran at step {step}, but no pressure values changed.")
    else:
        print(f"✅ Pressure correction modified {mutation_count} fluid cells at step {step}.")

    # 📤 Step 3: Return grid with updated pressure values (projection of velocity added later)
    return grid_with_pressure



