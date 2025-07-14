# src/snapshot_manager.py
# 🧾 Snapshot Manager — handles step-wise simulation evolution and snapshot construction

import os
import json
import logging
from src.step_controller import evolve_step
from src.utils.ghost_diagnostics import inject_diagnostics
from src.output.snapshot_writer import export_influence_flags
from src.output.mutation_pathways_logger import log_mutation_pathway

def generate_snapshots(input_data: dict, scenario_name: str, config: dict) -> list:
    time_step = input_data["simulation_parameters"]["time_step"]
    total_time = input_data["simulation_parameters"]["total_time"]
    output_interval = input_data["simulation_parameters"].get("output_interval", 1)
    if output_interval <= 0:
        logging.warning(f"⚠️ output_interval was set to {output_interval}. Using fallback of 1.")
        output_interval = 1

    domain = input_data["domain_definition"]
    initial_conditions = input_data["initial_conditions"]
    geometry = input_data.get("geometry_definition")

    print(f"🧩 Domain resolution: {domain['nx']}×{domain['ny']}×{domain['nz']}")
    print(f"⚙️  Output interval: {output_interval}")

    if geometry:
        from src.grid_generator import generate_grid_with_mask
        grid = generate_grid_with_mask(domain, initial_conditions, geometry)
        mask_flat = geometry.get("geometry_mask_flat", [])
        fluid_code = geometry.get("mask_encoding", {}).get("fluid", 1)
        expected_size = mask_flat.count(fluid_code)
    else:
        from src.grid_generator import generate_grid
        grid = generate_grid(domain, initial_conditions)
        expected_size = domain["nx"] * domain["ny"] * domain["nz"]

    dx = (domain["max_x"] - domain["min_x"]) / domain["nx"]
    dy = (domain["max_y"] - domain["min_y"]) / domain["ny"]
    dz = (domain["max_z"] - domain["min_z"]) / domain["nz"]
    spacing = (dx, dy, dz)
    print(f"[DEBUG] Grid spacing → dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}")

    num_steps = int(total_time / time_step)
    snapshots = []
    mutation_report = {
        "pressure_mutated": 0,
        "velocity_projected": 0,
        "projection_skipped": 0
    }

    output_folder = os.path.join("data", "testing-input-output", "navier_stokes_output")
    summary_path = os.path.join(output_folder, "step_summary.txt")
    os.makedirs(output_folder, exist_ok=True)

    for step in range(num_steps + 1):
        grid, reflex = evolve_step(grid, input_data, step, config=config)

        fluid_cells = [c for c in grid if getattr(c, "fluid_mask", False)]
        ghost_cells = [c for c in grid if not getattr(c, "fluid_mask", True)]

        print(f"[DEBUG] Step {step} → fluid cells: {len(fluid_cells)}, ghost cells: {len(ghost_cells)}, total: {len(grid)}")
        if len(fluid_cells) != expected_size:
            print(f"[DEBUG] ⚠️ Unexpected fluid cell count → expected: {expected_size}, found: {len(fluid_cells)}")

        export_influence_flags(grid, step_index=step, output_folder=output_folder, config=config)

        mutation_causes = []
        if reflex.get("ghost_influence_count", 0) > 0:
            mutation_causes.append("ghost_influence")
        if reflex.get("boundary_condition_applied", False):
            mutation_causes.append("boundary_override")

        mutated_cells_raw = reflex.get("mutated_cells", [])
        print(f"[DEBUG] mutated_cells (step {step}): {[type(c) for c in mutated_cells_raw[:3]]}")

        # ✅ Coerce pressure_mutated to boolean to avoid Cell leakage
        raw_pm = reflex.get("pressure_mutated", False)
        if isinstance(raw_pm, bool):
            pressure_mutated = raw_pm
        elif isinstance(raw_pm, dict) or hasattr(raw_pm, "__dict__"):
            print("[WARNING] pressure_mutated was unexpectedly a complex object — coercing to True")
            pressure_mutated = True
        else:
            print(f"[WARNING] pressure_mutated had unexpected type {type(raw_pm)} — coercing to bool")
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

        with open(summary_path, "a") as f:
            f.write(f"""[🔄 Step {step} Summary]
• Ghosts: {len(reflex.get("ghost_registry", []))}
• Fluid–ghost adjacents: {reflex.get("fluid_cells_adjacent_to_ghosts", "?")}
• Influence applied: {reflex.get("ghost_influence_count", 0)}
• Max divergence: {reflex.get("max_divergence", "?"):.6e}
• Projection attempted: {reflex.get("pressure_solver_invoked", False)}
• Projection skipped: {reflex.get("projection_skipped", False)}
• Pressure mutated: {pressure_mutated}

""")

        if pressure_mutated:
            mutation_report["pressure_mutated"] += 1
        if reflex.get("velocity_projected", True):
            mutation_report["velocity_projected"] += 1
        if reflex.get("projection_skipped", False):
            mutation_report["projection_skipped"] += 1

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

        snapshot = {
            "step_index": step,
            "grid": serialized_grid,
            "pressure_mutated": pressure_mutated,
            "velocity_projected": reflex.get("velocity_projected", True),
            **{k: v for k, v in reflex.items() if k not in ["pressure_mutated", "velocity_projected"]}
        }

        snapshot = inject_diagnostics(snapshot, ghost_registry, grid, spacing=spacing)

        if step % output_interval == 0:
            snapshots.append((step, snapshot))

    print("🧾 Final Simulation Summary:")
    print(f"   Pressure mutated steps   → {mutation_report['pressure_mutated']}")
    print(f"   Velocity projected steps → {mutation_report['velocity_projected']}")
    print(f"   Projection skipped steps → {mutation_report['projection_skipped']}")

    return snapshots



