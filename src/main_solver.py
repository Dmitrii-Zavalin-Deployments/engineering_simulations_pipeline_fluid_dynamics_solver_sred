# src/main_solver.py
# 🧠 Top-level driver for fluid simulation — orchestrates snapshots using step evolution

import os
import sys
import json
import logging

# ✅ Add path adjustment for module resolution in CI or direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.input_reader import load_simulation_input
from src.grid_generator import generate_grid, generate_grid_with_mask
from src.step_controller import evolve_step
from src.utils.ghost_diagnostics import inject_diagnostics
from src.output.snapshot_writer import export_influence_flags  # ✅ NEW import

def generate_snapshots(input_data: dict, scenario_name: str) -> list:
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
        grid = generate_grid_with_mask(domain, initial_conditions, geometry)
        mask_flat = geometry.get("geometry_mask_flat", [])
        fluid_code = geometry.get("mask_encoding", {}).get("fluid", 1)
        expected_size = mask_flat.count(fluid_code)
    else:
        grid = generate_grid(domain, initial_conditions)
        expected_size = domain["nx"] * domain["ny"] * domain["nz"]

    dx = (domain["max_x"] - domain["min_x"]) / domain["nx"]
    dy = (domain["max_y"] - domain["min_y"]) / domain["ny"]
    dz = (domain["max_z"] - domain["min_z"]) / domain["nz"]
    spacing = (dx, dy, dz)
    print(f"[DEBUG] Grid spacing → dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}")

    num_steps = int(total_time / time_step)
    snapshots = []
    mutation_report = {"pressure_mutated": 0, "velocity_projected": 0, "projection_skipped": 0}

    output_folder = os.path.join("data", "testing-input-output", "navier_stokes_output")
    summary_path = os.path.join(output_folder, "step_summary.txt")
    os.makedirs(output_folder, exist_ok=True)

    for step in range(num_steps + 1):
        grid, reflex_metadata = evolve_step(grid, input_data, step)

        fluid_cells = [c for c in grid if getattr(c, "fluid_mask", False)]
        ghost_cells = [c for c in grid if not getattr(c, "fluid_mask", True)]

        print(f"[DEBUG] Step {step} → fluid cells: {len(fluid_cells)}, ghost cells: {len(ghost_cells)}, total: {len(grid)}")
        if len(fluid_cells) != expected_size:
            print(f"[DEBUG] ⚠️ Unexpected fluid cell count at step {step} → expected: {expected_size}, found: {len(fluid_cells)}")
        for i, cell in enumerate(fluid_cells[:3]):
            print(f"[DEBUG] Fluid[{i}] @ ({cell.x:.2f}, {cell.y:.2f}, {cell.z:.2f})")
        for i, cell in enumerate(ghost_cells[:3]):
            print(f"[DEBUG] Ghost[{i}] @ ({cell.x:.2f}, {cell.y:.2f}, {cell.z:.2f})")

        # ✅ Append per-step summary to step_summary.txt
        with open(summary_path, "a") as f:
            f.write(f"""[🔄 Step {step} Summary]
• Ghosts: {len(reflex_metadata['ghost_registry'])}
• Fluid–ghost adjacents: {reflex_metadata.get("fluid_cells_adjacent_to_ghosts", "?")}
• Influence applied: {reflex_metadata.get("ghost_influence_count", 0)}
• Max divergence: {reflex_metadata.get("max_divergence", "?"):.6e}
• Projection skipped: {reflex_metadata.get("projection_skipped", False)}

""")

        # ✅ Export ghost influence flag map per step
        export_influence_flags(grid, step_index=step, output_folder=output_folder)

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

        if reflex_metadata.get("pressure_mutated", False):
            mutation_report["pressure_mutated"] += 1
        if reflex_metadata.get("velocity_projected", True):
            mutation_report["velocity_projected"] += 1
        if reflex_metadata.get("projection_skipped", False):
            mutation_report["projection_skipped"] += 1

        snapshot = {
            "step_index": step,
            "grid": serialized_grid,
            "pressure_mutated": reflex_metadata.get("pressure_mutated", False),
            "velocity_projected": reflex_metadata.get("velocity_projected", True),
            **{k: v for k, v in reflex_metadata.items() if k not in ["velocity_projected", "pressure_mutated"]}
        }

        ghost_registry = reflex_metadata.get("ghost_registry") or {
            id(c): {"coordinate": (c.x, c.y, c.z)} for c in grid if not getattr(c, "fluid_mask", True)
        }

        snapshot = inject_diagnostics(snapshot, ghost_registry, grid, spacing=spacing)

        if step % output_interval == 0:
            snapshots.append((step, snapshot))

    print("🧾 Final Simulation Summary:")
    print(f"   Pressure mutated steps   → {mutation_report['pressure_mutated']}")
    print(f"   Velocity projected steps → {mutation_report['velocity_projected']}")
    print(f"   Projection skipped steps → {mutation_report['projection_skipped']}")

    return snapshots

def run_solver(input_path: str):
    scenario_name = os.path.splitext(os.path.basename(input_path))[0]
    input_data = load_simulation_input(input_path)
    output_folder = os.path.join("data", "testing-input-output", "navier_stokes_output")
    os.makedirs(output_folder, exist_ok=True)

    print(f"🧠 [main_solver] Starting simulation for: {scenario_name}")
    print(f"📄 Input path: {input_path}")
    print(f"📁 Output folder: {output_folder}")

    snapshots = generate_snapshots(input_data, scenario_name)

    for step, snapshot in snapshots:
        formatted_step = f"{step:04d}"
        filename = f"{scenario_name}_step_{formatted_step}.json"
        path = os.path.join(output_folder, filename)
        with open(path, "w") as f:
            json.dump(snapshot, f, indent=2)
        print(f"🔄 Step {formatted_step} written → {filename}")

    print(f"✅ Simulation complete. Total snapshots: {len(snapshots)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️ Please provide an input file path.")
        print("   Example: python src/main_solver.py data/testing-input-output/fluid_simulation_input.json")
        sys.exit(1)

    run_solver(sys.argv[1])



