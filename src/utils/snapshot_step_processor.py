# src/utils/snapshot_step_processor.py
# ⚙️ Snapshot Step Processor — handles per-step diagnostics, scoring, and export

import os
import json
from src.output.snapshot_writer import export_influence_flags
from src.output.mutation_pathways_logger import log_mutation_pathway
from src.visualization.influence_overlay import render_influence_overlay
from src.visualization.reflex_overlay_mapper import render_reflex_overlay
from src.utils.snapshot_summary_writer import write_step_summary
from src.adaptive.grid_refiner import propose_refinement_zones
from src.utils.ghost_diagnostics import inject_diagnostics

def process_snapshot_step(
    step: int,
    grid,
    reflex: dict,
    spacing: tuple,
    config: dict,
    expected_size: int,
    output_folder: str
) -> tuple:
    fluid_cells = [c for c in grid if getattr(c, "fluid_mask", False)]
    ghost_cells = [c for c in grid if not getattr(c, "fluid_mask", True)]

    print(f"[DEBUG] reflex {reflex}")
    print(f"[DEBUG] Step {step} → fluid cells: {len(fluid_cells)}, ghost cells: {len(ghost_cells)}, total: {len(grid)}")
    if len(fluid_cells) != expected_size:
        print(f"[DEBUG] ⚠️ Unexpected fluid cell count → expected: {expected_size}, found: {len(fluid_cells)}")

    export_influence_flags(grid, step_index=step, output_folder=output_folder, config=config)

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
    print(f"[DEBUG] mutated_cells (step {step}): {[type(c) for c in mutated_cells_raw[:3]]}")

    raw_pm = reflex.get("pressure_mutated", False)
    if isinstance(raw_pm, bool):
        pressure_mutated = raw_pm
    elif isinstance(raw_pm, dict) or hasattr(raw_pm, "__dict__"):
        print("[WARNING] pressure_mutated was unexpectedly a complex object — coercing to True")
        pressure_mutated = True
    else:
        pressure_mutated = bool(raw_pm)

    log_mutation_pathway(
        step_index=step,
        pressure_mutated=pressure_mutated,
        triggered_by=mutation_causes,
        output_folder=output_folder,
        triggered_cells=[
            (c.x, c.y, c.z) for c in mutated_cells_raw
            if hasattr(c, "x") and hasattr(c, "y") and hasattr(c, "z")
        ]
    )

    print(f"[DEBUG] reflex data for step summary {reflex}")
    print(f"[DEBUG] reflex data testing {reflex["fluid_cells_adjacent_to_ghosts"]}")

    summary_path = os.path.join(output_folder, "step_summary.txt")
    with open(summary_path, "a") as f:
        try:
            divergence_value = reflex["max_divergence"]
            divergence_str = f"{divergence_value:.6e}"
        except (KeyError, TypeError):
            divergence_str = "?"

        f.write(f"""[🔄 Step {step} Summary]
    • Ghosts: {len(reflex["ghost_registry"])}
    • Fluid–ghost adjacents: {reflex["fluid_cells_adjacent_to_ghosts"]}
    • Influence applied: {reflex["ghost_influence_count"]}
    • Max divergence: {divergence_str}
    • Projection attempted: {reflex["pressure_solver_invoked"]}
    • Projection skipped: {reflex["projection_skipped"]}
    • Pressure mutated: {pressure_mutated}

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

    ghost_registry = reflex.get("ghost_registry") or {
        id(c): {"coordinate": (c.x, c.y, c.z)}
        for c in grid if not getattr(c, "fluid_mask", True)
    }

    # ✅ Reflex score injection preserved correctly, avoiding overwrite
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
    print(f"[VERIFY] Injected reflex_score: {snapshot['reflex_score']} ({type(snapshot['reflex_score'])})")
    print(f"[VERIFY] is instance true: {isinstance(score, (int, float))}")
    print(f"[VERIFY] score is: {score}")

    snapshot = inject_diagnostics(snapshot, ghost_registry, grid, spacing=spacing)
    write_step_summary(step, snapshot, output_folder="data/summaries")

    delta_path = f"data/snapshots/pressure_delta_map_step_{step:04d}.json"
    propose_refinement_zones(delta_path, spacing, step_index=step)

    reflex_score = snapshot.get("reflex_score", 0.0)
    if not isinstance(reflex_score, (int, float)):
        reflex_score = 0.0

    mutation_coords = [
        (cell["x"], cell["y"]) if isinstance(cell, dict)
        else cell if isinstance(cell, tuple) and len(cell) == 2
        else (0, 0)
        for cell in snapshot.get("mutated_cells", [])
    ]
    adjacency_coords = reflex.get("adjacency_zones", [])
    suppression_coords = reflex.get("suppression_zones", [])

    overlay_reflex_path = os.path.join("data", "overlays", f"reflex_overlay_step_{step:03d}.png")
    render_reflex_overlay(
        step_index=step,
        reflex_score=reflex_score,
        mutation_coords=mutation_coords,
        adjacency_coords=adjacency_coords,
        suppression_coords=suppression_coords,
        output_path=overlay_reflex_path
    )

    return grid, snapshot



