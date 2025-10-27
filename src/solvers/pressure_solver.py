# src/solvers/pressure_solver.py
# 💧 Pressure Solver — enforces incompressibility via pressure correction and reflex diagnostics
# 📌 This module solves ∇²P = ∇ · u and tags pressure mutations for reflex overlays.
# It excludes only cells explicitly marked fluid_mask=False.
# It does NOT skip based on adjacency or ghost proximity — all logic is geometry-mask-driven.

from typing import List, Tuple, Dict
from src.grid_modules.cell import Cell
from src.physics.pressure_projection import solve_pressure_poisson
from src.exporters.pressure_delta_map_writer import export_pressure_delta_map
from src.diagnostics.mutation_threshold_advisor import get_delta_threshold
from src.physics.divergence_tracker import compute_divergence_stats
from src.reflex.reflex_pathway_logger import log_reflex_pathway
from src.initialization.fluid_mask_initializer import build_simulation_grid
from src.config.config_validator import validate_config
from src.diagnostics.navier_stokes_verifier import run_verification_if_triggered

# ✅ Centralized debug flag for GitHub Actions logging
debug = True

def apply_pressure_correction(grid: List[Cell], input_data: dict, step: int) -> Tuple[List[Cell], bool, int, Dict]:
    # FIX: Architectural Correction - Extract I/O suppression flag from input_data to prevent FileNotFoundError in tests.
    # If this flag is True, we skip writing data to disk.
    disable_io_for_testing = input_data.get("simulation_parameters", {}).get("disable_io_for_testing", False)
    
    validate_config(input_data)
    if debug: print(f"[PRESSURE] Step {step}: Config validated")

    if not grid:
        grid = build_simulation_grid(input_data)
        if debug: print(f"[PRESSURE] Step {step}: Grid built from input_data")

    # --- Setup Safe Grid (Input for Poisson Solver) ---
    safe_grid = []
    for i, cell in enumerate(grid):
        new_cell = Cell(
            x=cell.x,
            y=cell.y,
            z=cell.z,
            # Ensure velocity and pressure are sanitized for solver use, but fluid_mask is preserved
            velocity=cell.velocity if isinstance(cell.velocity, list) else None,
            pressure=cell.pressure if isinstance(cell.pressure, float) else 0.0,
            fluid_mask=cell.fluid_mask
        )
        new_cell.boundary_type = getattr(cell, "boundary_type", None)
        new_cell.influenced_by_ghost = getattr(cell, "influenced_by_ghost", False)
        safe_grid.append(new_cell)

        if debug and not new_cell.fluid_mask:
            reason = []
            if not isinstance(cell.velocity, list):
                reason.append("missing or malformed velocity")
            if not cell.fluid_mask:
                reason.append("fluid_mask=False")
            hint = " | ".join(reason) if reason else "unknown reason"
            print(f"[PRESSURE] ⚠️ Downgrading cell[{i}] @ ({cell.x:.2f}, {cell.y:.2f}, {cell.z:.2f}) — {hint}")

    domain = input_data["domain_definition"]
    dx = (domain["max_x"] - domain["min_x"]) / domain["nx"]
    dy = (domain["max_y"] - domain["min_y"]) / domain["ny"]
    dz = (domain["max_z"] - domain["min_z"]) / domain["nz"]
    spacing = (dx, dy, dz)
    if debug: print(f"[PRESSURE] Step {step}: Grid spacing → dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}")

    output_folder = "data/testing-input-output/navier_stokes_output"
    
    # compute_divergence_stats is now guarded by patches in the test file.
    div_stats = compute_divergence_stats(
        safe_grid, spacing,
        label="post-pressure", step_index=step,
        output_folder=output_folder, 
        config=input_data
    )
    divergence = div_stats["divergence"]
    max_div = div_stats["max"]
    if debug: print(f"[PRESSURE] Step {step}: Divergence computed → max={max_div:.6e}")

    # --- Step 1: Solve Poisson Equation for Pressure ---
    grid_with_pressure, pressure_mutated, ghost_registry = solve_pressure_poisson(safe_grid, divergence, input_data)
    if debug: print(f"[PRESSURE] Step {step}: Pressure Poisson solve completed")

    # --- CRITICAL FIX: Re-enforce non-physical state after pressure solve ---
    for old, new in zip(safe_grid, grid_with_pressure):
        new.boundary_type = getattr(old, "boundary_type", None)
        new.influenced_by_ghost = getattr(old, "influenced_by_ghost", False)

        # FIX: Explicitly copy fluid_mask and original velocity (if non-fluid)
        new.fluid_mask = old.fluid_mask
        if not old.fluid_mask:
            new.velocity = old.velocity

    # --- Step 2: Mutation and Diagnostic Check ---
    mutation_count = 0
    mutated_cells = []
    pressure_delta_map = []
    # (Mutation logic remains the same...)

    for old, updated in zip(safe_grid, grid_with_pressure):
        if updated.fluid_mask:
            if getattr(updated, "boundary_type", None) in {"outlet", "wall"}:
                if debug:
                    print(f"[PRESSURE] Step {step}: Skipping cell @ ({updated.x:.2f}, {updated.y:.2f}, {updated.z:.2f}) due to boundary type")
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
            if debug:
                print(f"[PRESSURE] Step {step}: cell ({updated.x:.2f}, {updated.y:.2f}, {updated.z:.2f}) → threshold = {threshold:.2e}, delta = {delta:.2e}")

            if delta > threshold:
                mutation_count += 1
                mutated_cells.append(updated)
                updated.pressure_mutated = True
                updated.mutation_source = "pressure_solver"
                updated.mutation_step = step
                if debug:
                    print(f"[PRESSURE] ✅ Pressure updated @ ({updated.x:.2f}, {updated.y:.2f}, {updated.z:.2f})")

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
                if debug:
                    print(f"[TRACE] Step {step}: ghost-influenced mutation @ ({updated.x:.2f}, {updated.y:.2f}, {updated.z:.2f})")

    if debug:
        if mutation_count == 0:
            print(f"[PRESSURE] ⚠️ Step {step}: No pressure mutations detected.")
        else:
            print(f"[PRESSURE] ✅ Step {step}: {mutation_count} fluid cells mutated.")

    ghost_trigger_chain = input_data.get("ghost_trigger_chain", [])
    
    # FIX APPLIED HERE: Guard the log_reflex_pathway call
    if not disable_io_for_testing:
        log_reflex_pathway(
            step_index=step,
            mutated_cells=mutated_cells,
            ghost_trigger_chain=ghost_trigger_chain
        )
        if debug: print(f"[PRESSURE] Step {step}: Reflex pathway logged")
    else:
        if debug: print(f"[PRESSURE] Step {step}: Reflex pathway logging skipped (testing mode)")


    metadata = {
        "max_divergence": max_div,
        "pressure_mutation_count": mutation_count,
        "pressure_solver_passes": 1,
        "mutated_cells": [(c.x, c.y, c.z) for c in mutated_cells],
        "ghost_registry": ghost_registry
    }
    if debug: print(f"[PRESSURE] Step {step}: Metadata assembled")

    # This was the first fix location (already guarded)
    if not disable_io_for_testing:
        export_pressure_delta_map(pressure_delta_map, step_index=step, output_dir="data/snapshots")
        if debug: print(f"[PRESSURE] Step {step}: Pressure delta map exported")
    else:
        if debug: print(f"[PRESSURE] Step {step}: Pressure delta map export skipped (testing mode)")

    triggered_flags = []
    if mutation_count == 0:
        triggered_flags.append("no_pressure_mutation")
    if not divergence:
        triggered_flags.append("empty_divergence")
    if any(not isinstance(c.velocity, list) or not c.fluid_mask for c in grid):
        triggered_flags.append("downgraded_cells")

    # This function is successfully mocked in the test file, so we leave it as is.
    run_verification_if_triggered(grid_with_pressure, spacing, step, output_folder, triggered_flags)

    return grid_with_pressure, pressure_mutated, metadata["pressure_solver_passes"], metadata



