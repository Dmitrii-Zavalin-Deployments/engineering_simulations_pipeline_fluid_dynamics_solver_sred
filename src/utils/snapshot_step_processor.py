# src/utils/snapshot_step_processor.py
# ⚙️ Snapshot Step Processor — handles per-step diagnostics, scoring, and export

import os
from src.output.snapshot_writer import export_influence_flags
from src.output.mutation_pathways_logger import log_mutation_pathway
from src.reflex.reflex_pathway_logger import log_reflex_pathway
from src.visualization.influence_overlay import render_influence_overlay
from src.visualization.reflex_overlay_mapper import render_reflex_overlay
from src.utils.snapshot_summary_writer import write_step_summary
from src.adaptive.grid_refiner import propose_refinement_zones
from src.utils.ghost_diagnostics import inject_diagnostics
from src.utils.ghost_registry import build_ghost_registry, extract_ghost_coordinates
from src.reflex.spatial_tagging.ghost_face_mapper import tag_ghost_adjacency
from src.reflex.spatial_tagging.suppression_zones import detect_suppression_zones, extract_mutated_coordinates
from src.initialization.fluid_mask_initializer import build_simulation_grid
from src.config.config_validator import validate_config

# 🛠️ Toggle debug logging
DEBUG = False  # Set to True to enable verbose diagnostics

def process_snapshot_step(
    step: int,
    grid,
    reflex: dict,
    spacing: tuple,
    config: dict,
    expected_size: int,
    output_folder: str
) -> tuple:
    validate_config(config)
    grid = build_simulation_grid(config)

    fluid_cells = [c for c in grid if getattr(c, "fluid_mask", False)]
    ghost_cells = [c for c in grid if not getattr(c, "fluid_mask", True)]

    if DEBUG:
        print(f"[DEBUG] reflex {reflex}")
        print(f"[DEBUG] Step {step} → fluid cells: {len(fluid_cells)}, ghost cells: {len(ghost_cells)}, total: {len(grid)}")
        if len(fluid_cells) != expected_size:
            print(f"[DEBUG] ⚠️ Unexpected fluid cell count → expected: {expected_size}, found: {len(fluid_cells)}")

    export_influence_flags(grid, step_index=step, output_folder=output_folder, config=config)

    ghost_registry = reflex.get("ghost_registry") or build_ghost_registry(grid)
    ghost_coords = extract_ghost_coordinates(ghost_registry)

    if "adjacency_zones" not in reflex:
        reflex["adjacency_zones"] = tag_ghost_adjacency(grid, ghost_coords, spacing)

    if "suppression_zones" not in reflex:
        mutated_coords = extract_mutated_coordinates(reflex.get("mutated_cells", []))
        reflex["suppression_zones"] = detect_suppression_zones(grid, ghost_coords, mutated_coords, spacing)

    influence_log = {
        "step_score": reflex.get("reflex_score", 0.0),
        "adjacency_zones": reflex.get("adjacency_zones", []),
        "suppression_zones": reflex.get("suppression_zones", [])
    }
    overlay_path = os.path.join(output_folder, "overlays", f"step_{step:03d}.png")
    render_influence_overlay(influence_log, overlay_path)

    mutation_causes = []
    if reflex.get("ghost_influence_count", 0) > 0:
        mutation_causes.append("ghost_influence")
    if reflex.get("boundary_condition_applied", False):
        mutation_causes.append("boundary_override")

    mutated_cells_raw = reflex.get("mutated_cells", [])
    if DEBUG:
        print(f"[DEBUG] mutated_cells (step {step}): {[type(c) for c in mutated_cells_raw[:3]]}")

    raw_pm = reflex.get("pressure_mutated", False)
    pressure_mutated = (
        True if isinstance(raw_pm, dict) or hasattr(raw_pm, "__dict__")
        else bool(raw_pm)
    )

    log_mutation_pathway(
        step_index=step,
        pressure_mutated=pressure_mutated,
        triggered_by=mutation_causes,
        output_folder=output_folder,
        triggered_cells=[
            (c.x, c.y, c.z) for c in mutated_cells_raw
            if hasattr(c, "x") and hasattr(c, "y") and hasattr(c, "z")
        ],
        ghost_trigger_chain=reflex.get("ghost_trigger_chain", [])
    )

    log_reflex_pathway(
        step_index=step,
        mutated_cells=mutated_cells_raw,
        ghost_trigger_chain=reflex.get("ghost_trigger_chain", [])
    )

    summary_path = os.path.join(output_folder, "step_summary.txt")
    with open(summary_path, "a") as f:
        divergence_value = reflex.get("max_divergence")
        divergence_str = f"{divergence_value:.6e}" if isinstance(divergence_value, (int, float)) else "?"

        adjacent_count = len(reflex.get("adjacency_zones", []))
        ghost_count = len(ghost_registry) if isinstance(ghost_registry, dict) else "?"
        influence_applied = reflex.get("ghost_influence_count", "?")
        projection_attempted = reflex.get("pressure_solver_invoked", "?")
        projection_skipped = reflex.get("projection_skipped", "?")

        reflex_score_val = reflex.get("reflex_score")
        reflex_score_str = f"{reflex_score_val:.2f}" if isinstance(reflex_score_val, (int, float)) else "?"

        mutated_cells_count = len(reflex.get("mutated_cells", []))
        adaptive_timestep_val = reflex.get("adaptive_timestep")
        adaptive_timestep_str = f"{adaptive_timestep_val:.3f}" if isinstance(adaptive_timestep_val, (int, float)) else "?"

        f.write(f"""[🔄 Step {step} Summary]
    • Ghosts: {ghost_count}
    • Fluid–ghost adjacents: {adjacent_count}
    • Influence applied: {influence_applied}
    • Max divergence: {divergence_str}
    • Projection attempted: {projection_attempted}
    • Projection skipped: {projection_skipped}
    • Pressure mutated: {pressure_mutated}
    • Reflex score: {reflex_score_str}
    • Mutated cells: {mutated_cells_count}
    • Adaptive timestep: {adaptive_timestep_str}

    """)

    serialized_grid = []
    for cell in grid:
        fluid = getattr(cell, "fluid_mask", True)
        serialized_grid.append({
            "x": cell.x,
            "y": cell.y,
            "z": cell.z,
            "fluid_mask": fluid,
            "velocity": cell.velocity if fluid else None,
            "pressure": cell.pressure if fluid else None
        })

    score = reflex.get("reflex_score")
    reflex_clean = {k: v for k, v in reflex.items() if k not in ["pressure_mutated", "velocity_projected", "reflex_score"]}
    snapshot = {
        "step_index": step,
        "grid": serialized_grid,
        "pressure_mutated": pressure_mutated,
        "velocity_projected": reflex.get("velocity_projected", True),
        **reflex_clean,
        "reflex_score": float(score) if isinstance(score, (int, float)) else 0.0
    }

    snapshot = inject_diagnostics(snapshot, ghost_registry, grid, spacing=spacing)
    write_step_summary(step, snapshot, output_folder="data/summaries")

    delta_path = f"data/snapshots/pressure_delta_map_step_{step:04d}.json"
    propose_refinement_zones(delta_path, spacing, step_index=step)

    reflex_score = snapshot.get("reflex_score", 0.0)
    mutation_coords = [
        (cell["x"], cell["y"]) if isinstance(cell, dict)
        else cell if isinstance(cell, tuple) and len(cell) == 2
        else (0, 0)
        for cell in snapshot.get("mutated_cells", [])
    ]
    adjacency_coords = snapshot.get("adjacency_zones", [])
    suppression_coords = snapshot.get("suppression_zones", [])
    mutation_density = snapshot.get("mutation_density", 0.0)

    overlay_reflex_path = os.path.join("data", "overlays", f"reflex_overlay_step_{step:03d}.png")
    render_reflex_overlay(
        step_index=step,
        reflex_score=reflex_score,
        mutation_coords=mutation_coords,
        adjacency_coords=adjacency_coords,
        suppression_coords=suppression_coords,
        output_path=overlay_reflex_path,
        mutation_density=mutation_density
    )

    return grid, snapshot
