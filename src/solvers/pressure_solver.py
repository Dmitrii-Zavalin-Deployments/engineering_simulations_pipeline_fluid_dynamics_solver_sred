# src/solvers/pressure_solver.py
# 💧 Pressure Solver — enforces incompressibility via pressure correction and reflex diagnostics

from typing import List, Tuple, Dict
from src.grid_modules.cell import Cell
from src.physics.pressure_projection import solve_pressure_poisson
from src.exporters.pressure_delta_map_writer import export_pressure_delta_map
from src.diagnostics.mutation_threshold_advisor import get_delta_threshold
from src.physics.divergence_tracker import compute_divergence_stats
from src.reflex.reflex_pathway_logger import log_reflex_pathway
from src.initialization.fluid_mask_initializer import build_simulation_grid
from src.config.config_validator import validate_config

# 🛠️ Toggle debug logging
DEBUG = True  # Set to True to enable verbose diagnostics

def apply_pressure_correction(grid: List[Cell], input_data: dict, step: int) -> Tuple[List[Cell], bool, int, Dict]:
    validate_config(input_data)
    if DEBUG: print(f"[DEBUG] Step {step}: Config validated")

    if not grid:
        grid = build_simulation_grid(input_data)
        if DEBUG: print(f"[DEBUG] Step {step}: Grid built from input_data")

    safe_grid = []
    for i, cell in enumerate(grid):
        new_cell = Cell(
            x=cell.x,
            y=cell.y,
            z=cell.z,
            velocity=cell.velocity if isinstance(cell.velocity, list) else None,
            pressure=cell.pressure if isinstance(cell.pressure, float) else 0.0,
            fluid_mask=cell.fluid_mask
        )
        new_cell.boundary_type = getattr(cell, "boundary_type", None)
        new_cell.influenced_by_ghost = getattr(cell, "influenced_by_ghost", False)
        safe_grid.append(new_cell)
        if DEBUG and not new_cell.fluid_mask:
            print(f"[DEBUG] ⚠️ Downgrading cell[{i}] @ ({cell.x:.2f}, {cell.y:.2f}, {cell.z:.2f}) — invalid velocity or fluid_mask")

    domain = input_data["domain_definition"]
    dx = (domain["max_x"] - domain["min_x"]) / domain["nx"]
    dy = (domain["max_y"] - domain["min_y"]) / domain["ny"]
    dz = (domain["max_z"] - domain["min_z"]) / domain["nz"]
    spacing = (dx, dy, dz)
    if DEBUG: print(f"[DEBUG] Step {step}: Grid spacing computed → dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}")

    output_folder = "data/testing-input-output/navier_stokes_output"
    div_stats = compute_divergence_stats(
        safe_grid, spacing,
        label="post-pressure", step_index=step,
        output_folder=output_folder, config=input_data
    )
    divergence = div_stats["divergence"]
    max_div = div_stats["max"]
    if DEBUG: print(f"[DEBUG] Step {step}: Divergence computed → max={max_div:.6e}")

    grid_with_pressure, pressure_mutated, ghost_registry = solve_pressure_poisson(safe_grid, divergence, input_data)
    if DEBUG: print(f"[DEBUG] Step {step}: Pressure Poisson solve completed")

    for old, new in zip(safe_grid, grid_with_pressure):
        new.boundary_type = getattr(old, "boundary_type", None)
        new.influenced_by_ghost = getattr(old, "influenced_by_ghost", False)

    mutation_count = 0
    mutated_cells = []
    pressure_delta_map = []

    for old, updated in zip(safe_grid, grid_with_pressure):
        if updated.fluid_mask:
            if getattr(updated, "boundary_type", None) in {"outlet", "wall"}:
                if DEBUG:
                    print(f"[DEBUG] Step {step}: Skipping cell @ ({updated.x:.2f}, {updated.y:.2f}, {updated.z:.2f}) due to boundary type")
                continue

            initial = old.pressure if isinstance(old.pressure, float) else 0.0
            final = updated.pressure if isinstance(updated.pressure, float) else 0.0
            delta = abs(final - initial)

            context = {
                "resolution": input_data.get("grid_resolution", "normal"),
                "divergence": divergence[safe_grid.index(old)] if safe_grid.index(old) < len(divergence) else 0.0,
                "time_step": input_data.get("simulation_parameters", {}).get("time_step", 0.05)
            }

            threshold = get_delta_threshold(updated, context)
            if DEBUG:
                print(f"[DEBUG] Step {step}: cell ({updated.x:.2f}, {updated.y:.2f}, {updated.z:.2f}) → threshold = {threshold:.2e}, delta = {delta:.2e}")

            if delta > threshold:
                mutation_count += 1
                mutated_cells.append(updated)
                updated.pressure_mutated = True
                updated.mutation_source = "pressure_solver"
                updated.mutation_step = step
                if DEBUG:
                    print(f"[DEBUG] ✅ Pressure updated @ ({updated.x:.2f}, {updated.y:.2f}, {updated.z:.2f}) ← source: solver")

            pressure_delta_map.append({
                "x": updated.x,
                "y": updated.y,
                "z": updated.z,
                "before": initial,
                "after": final,
                "delta": delta
            })

            if getattr(updated, "influenced_by_ghost", False):
                updated.mutation_triggered_by = "ghost_influence"
                if DEBUG:
                    print(f"[TRACE] Step {step}: pressure mutation at ghost-influenced cell ({updated.x:.2f}, {updated.y:.2f}, {updated.z:.2f})")

    if DEBUG:
        if mutation_count == 0:
            print(f"⚠️ Step {step}: Pressure solver ran but no pressure values changed.")
        else:
            print(f"✅ Step {step}: Pressure correction modified {mutation_count} fluid cells.")

    ghost_trigger_chain = input_data.get("ghost_trigger_chain", [])
    log_reflex_pathway(
        step_index=step,
        mutated_cells=mutated_cells,
        ghost_trigger_chain=ghost_trigger_chain
    )
    if DEBUG: print(f"[DEBUG] Step {step}: Reflex pathway logged")

    metadata = {
        "max_divergence": max_div,
        "pressure_mutation_count": mutation_count,
        "pressure_solver_passes": 1,
        "mutated_cells": [(c.x, c.y, c.z) for c in mutated_cells],
        "ghost_registry": ghost_registry
    }
    if DEBUG: print(f"[DEBUG] Step {step}: Metadata assembled")

    export_pressure_delta_map(pressure_delta_map, step_index=step, output_dir="data/snapshots")
    if DEBUG: print(f"[DEBUG] Step {step}: Pressure delta map exported")

    return grid_with_pressure, pressure_mutated, metadata["pressure_solver_passes"], metadata



