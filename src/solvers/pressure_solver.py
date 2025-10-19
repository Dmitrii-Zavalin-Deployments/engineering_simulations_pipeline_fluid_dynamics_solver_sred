# src/solvers/pressure_solver.py
# 💧 Pressure Solver — enforces incompressibility via pressure correction and reflex diagnostics

from typing import List, Tuple, Dict
from src.grid_modules.cell import Cell
from src.physics.pressure_projection import solve_pressure_poisson
from src.exporters.pressure_delta_map_writer import export_pressure_delta_map
from src.diagnostics.mutation_threshold_advisor import get_delta_threshold
from src.physics.divergence_tracker import compute_divergence_stats  # ✅ Modular integration

def apply_pressure_correction(grid: List[Cell], input_data: dict, step: int) -> Tuple[List[Cell], bool, int, Dict]:
    """
    Applies pressure correction to enforce incompressible flow.

    Roadmap Alignment:
    Governing Equation:
        Continuity: ∇ · u = 0

    Modular Enforcement:
    - Divergence computation → divergence_tracker.py
    - Pressure solve: ∇²P = ∇ · u → pressure_projection.py
    - Mutation threshold logic → mutation_threshold_advisor.py
    - Delta map export → pressure_delta_map_writer.py

    Purpose:
    - Enforces incompressibility via pressure correction
    - Anchors continuity enforcement in reflex scoring
    - Tracks mutation causality and ghost influence

    Strategy:
    1. Compute divergence of velocity field
    2. Solve pressure Poisson equation to reduce divergence
    3. Update pressure field and track mutations
    4. Export diagnostics and mutation map

    Args:
        grid (List[Cell]): Grid of simulation cells
        input_data (dict): Full simulation config
        step (int): Current simulation step index

    Returns:
        Tuple containing:
        - List[Cell]: Grid with updated pressure values
        - bool: Whether any pressure values changed
        - int: Number of projection iterations or passes
        - dict: Metadata about the correction process, including mutated_cells
    """
    # ✅ Filter valid fluid cells for pressure solve
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

    # 📊 Step 1: Compute divergence using modular tracker
    domain = input_data["domain_definition"]
    dx = (domain["max_x"] - domain["min_x"]) / domain["nx"]
    dy = (domain["max_y"] - domain["min_y"]) / domain["ny"]
    dz = (domain["max_z"] - domain["min_z"]) / domain["nz"]
    spacing = (dx, dy, dz)

    output_folder = "data/testing-input-output/navier_stokes_output"
    div_stats = compute_divergence_stats(
        safe_grid, spacing,
        label="post-pressure", step_index=step,
        output_folder=output_folder, config=input_data
    )
    divergence = div_stats["divergence"]
    max_div = div_stats["max"]

    # 💧 Step 2: Solve pressure Poisson equation ∇²P = ∇ · u
    grid_with_pressure, pressure_mutated = solve_pressure_poisson(safe_grid, divergence, input_data)

    # 🧠 Step 3: Analyze pressure mutations
    mutation_count = 0
    mutated_cells = []
    pressure_delta_map = {}

    for old, updated in zip(safe_grid, grid_with_pressure):
        if updated.fluid_mask:
            initial = old.pressure if isinstance(old.pressure, float) else 0.0
            final = updated.pressure if isinstance(updated.pressure, float) else 0.0
            delta = abs(final - initial)

            context = {
                "resolution": input_data.get("grid_resolution", "normal"),
                "divergence": divergence[safe_grid.index(old)] if safe_grid.index(old) < len(divergence) else 0.0,
                "time_step": input_data.get("simulation_parameters", {}).get("time_step", 0.05)
            }

            threshold = get_delta_threshold(updated, context)
            print(f"[DEBUG] Step {step}: cell ({updated.x:.2f}, {updated.y:.2f}, {updated.z:.2f}) → threshold = {threshold:.2e}, delta = {delta:.2e}")
            
            if delta > threshold:
                mutation_count += 1
                mutated_cells.append((updated.x, updated.y, updated.z))
                print(f"Pressure updated @ ({updated.x:.2f}, {updated.y:.2f}, {updated.z:.2f}) ← source: solver")

            pressure_delta_map[(updated.x, updated.y, updated.z)] = {
                "before": initial,
                "after": final,
                "delta": delta
            }

            if hasattr(updated, "influenced_by_ghost") and updated.influenced_by_ghost:
                print(f"[TRACE] Step {step}: pressure mutation at ghost-influenced cell ({updated.x:.2f}, {updated.y:.2f}, {updated.z:.2f})")

    # 📋 Step 4: Log mutation summary
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

    # 🗂️ Step 5: Export pressure delta map
    export_pressure_delta_map(pressure_delta_map, step_index=step, output_dir="data/snapshots")

    return grid_with_pressure, pressure_mutated, metadata["pressure_solver_passes"], metadata



