# src/step_controller.py
# 🚀 Simulation Step Controller — executes Navier-Stokes evolution per timestep

import logging
from typing import List, Tuple, Optional
from src.grid_modules.cell import Cell
from src.physics.ghost_cell_generator import generate_ghost_cells
from src.physics.boundary_condition_solver import apply_boundary_conditions
from src.physics.ghost_influence_applier import apply_ghost_influence
from src.solvers.navier_stokes_solver import solve_navier_stokes_step  # ✅ Integrated centralized solver
from src.reflex.reflex_controller import apply_reflex
from src.utils.ghost_diagnostics import log_ghost_summary, inject_diagnostics
from src.utils.divergence_tracker import compute_divergence_stats
from src.adaptive.timestep_controller import suggest_timestep

def evolve_step(
    grid: List[Cell],
    input_data: dict,
    step: int,
    config: Optional[dict] = None,
    reflex_score: Optional[int] = None
) -> Tuple[List[Cell], dict]:
    """
    Evolves the fluid grid by one simulation step using the full Navier-Stokes formulation:

    Governing Equations:
    - Continuity: ∇ · u = 0
    - Momentum: ρ(∂u/∂t + u · ∇u) = -∇P + μ∇²u + F

    Evolution Sequence:
    1. Ghost cell padding and boundary enforcement
    2. Ghost influence propagation (velocity/pressure transfer)
    3. Navier-Stokes solver (momentum + pressure + projection)
    4. Reflex logic and diagnostics

    Returns:
        Tuple[List[Cell], dict]: Updated grid and reflex metadata for snapshot
    """
    logging.info(f"🌀 [evolve_step] Step {step}: Starting evolution")

    # 📐 Spatial discretization
    domain = input_data["domain_definition"]
    dx = (domain["max_x"] - domain["min_x"]) / domain["nx"]
    dy = (domain["max_y"] - domain["min_y"]) / domain["ny"]
    dz = (domain["max_z"] - domain["min_z"]) / domain["nz"]
    spacing = (dx, dy, dz)

    output_folder = "data/testing-input-output/navier_stokes_output"

    # 🧱 Step 1: Ghost cell generation
    padded_grid, ghost_registry = generate_ghost_cells(grid, input_data)
    logging.debug(f"🧱 Generated {len(ghost_registry)} ghost cells")
    log_ghost_summary(ghost_registry)

    # 🚧 Step 2: Apply boundary conditions to ghost cells and adjacent fluid
    boundary_tagged_grid = apply_boundary_conditions(padded_grid, ghost_registry, input_data)
    boundary_applied = True

    # 👣 Step 3: Ghost influence propagation
    influence_count = apply_ghost_influence(
        boundary_tagged_grid,
        spacing,
        verbose=(config or {}).get("reflex_verbosity", "") == "high",
        radius=(config or {}).get("ghost_adjacency_depth", 1)
    )
    logging.debug(f"👣 Ghost influence applied to {influence_count} fluid cells")

    # 📊 Step 4: Divergence check before pressure solve
    stats_before = compute_divergence_stats(
        boundary_tagged_grid, spacing,
        label="before projection", step_index=step,
        output_folder=output_folder, config=config
    )
    stats_before["max"]

    # ⏱️ Step 5: Adaptive timestep suggestion
    base_dt = input_data.get("default_timestep", 0.01)
    delta_path = f"data/snapshots/pressure_delta_map_step_{step:04d}.json"
    trace_path = "data/testing-input-output/navier_stokes_output/mutation_pathways_log.json"
    dt = suggest_timestep(delta_path, trace_path, base_dt=base_dt, reflex_score=reflex_score)

    # 🧠 Step 6: Navier-Stokes solver (momentum + pressure + projection)
    velocity_projected_grid, ns_metadata = solve_navier_stokes_step(boundary_tagged_grid, input_data, step)

    # 📊 Step 7: Divergence check after projection
    stats_after = compute_divergence_stats(
        velocity_projected_grid, spacing,
        label="after projection", step_index=step,
        output_folder=output_folder, config=config
    )
    divergence_after = stats_after["max"]

    # 🧠 Step 8: Reflex logic and metadata injection
    reflex_metadata = apply_reflex(
        velocity_projected_grid,
        input_data,
        step,
        ghost_influence_count=influence_count,
        config=config,
        pressure_solver_invoked=True,
        pressure_mutated=ns_metadata.get("pressure_mutated", False),
        post_projection_divergence=divergence_after
    )
    print(f"[DEBUG] 📋 Reflex Flags: {reflex_metadata}")

    # 🧾 Metadata enrichment
    reflex_metadata["ghost_influence_count"] = influence_count
    reflex_metadata["ghost_registry"] = ghost_registry
    reflex_metadata["boundary_condition_applied"] = boundary_applied
    reflex_metadata["projection_passes"] = ns_metadata.get("projection_passes", 1)
    reflex_metadata["adaptive_timestep"] = dt
    reflex_metadata.update(ns_metadata)

    reflex_metadata = inject_diagnostics(
        reflex_metadata, ghost_registry,
        grid=velocity_projected_grid,
        spacing=spacing
    )

    logging.info(f"✅ [evolve_step] Step {step}: Evolution complete")
    return velocity_projected_grid, reflex_metadata



