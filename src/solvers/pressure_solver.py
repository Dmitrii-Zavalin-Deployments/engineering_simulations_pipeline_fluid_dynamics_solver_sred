# src/solvers/pressure_solver.py
# 🔧 Pressure solver — enforces incompressibility via divergence correction

from typing import List, Tuple, Dict
from src.grid_modules.cell import Cell
from src.physics.divergence import compute_divergence
from src.physics.pressure_projection import solve_pressure_poisson
from src.exporters.pressure_delta_map_writer import export_pressure_delta_map

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

    divergence = compute_divergence(safe_grid)
    max_div = max(abs(d) for d in divergence) if divergence else 0.0
    print(f"📊 Step {step}: Max divergence = {max_div:.6e}")

    grid_with_pressure, pressure_mutated = solve_pressure_poisson(safe_grid, divergence, input_data)

    mutation_count = 0
    mutated_cells = []
    pressure_delta_map = {}

    for old, updated in zip(safe_grid, grid_with_pressure):
        if updated.fluid_mask:
            initial = old.pressure if isinstance(old.pressure, float) else 0.0
            final = updated.pressure if isinstance(updated.pressure, float) else 0.0
            delta = abs(final - initial)

            if delta > 1e-8:
                mutation_count += 1
                mutated_cells.append((updated.x, updated.y, updated.z))
                print(f"Pressure updated @ ({updated.x:.2f}, {updated.y:.2f}, {updated.z:.2f}) ← source: solver")  # ✅ Patch applied

            pressure_delta_map[(updated.x, updated.y, updated.z)] = {
                "before": initial,
                "after": final,
                "delta": delta
            }

            # ✅ Optional audit: capture ghost flag if present on mutated cell
            if hasattr(updated, "influenced_by_ghost") and updated.influenced_by_ghost:
                print(f"[TRACE] Step {step}: pressure mutation at ghost-influenced cell ({updated.x:.2f}, {updated.y:.2f}, {updated.z:.2f})")

    if mutation_count == 0:
        print(f"⚠️ Step {step}: Pressure solver ran but no pressure values changed.")
    else:
        print(f"✅ Step {step}: Pressure correction modified {mutation_count} fluid cells.")

    metadata = {
        "max_divergence": max_div,
        "pressure_mutation_count": mutation_count,
        "pressure_solver_passes": 1,
        "mutated_cells": mutated_cells
    }

    export_pressure_delta_map(pressure_delta_map, step_index=step, output_dir="data/snapshots")

    return grid_with_pressure, pressure_mutated, metadata["pressure_solver_passes"], metadata



