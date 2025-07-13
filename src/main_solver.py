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

def generate_snapshots(input_data: dict, scenario_name: str) -> list:
    """
    Runs the simulation over time using persistent evolving grid.
    Each snapshot reflects updated fluid state and reflex diagnostics.
    Ensures all cells — fluid, solid, and ghost — have consistent serialization.
    """
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

    # ✅ Initialize grid with or without embedded fluid_mask
    if geometry:
        grid = generate_grid_with_mask(domain, initial_conditions, geometry)
    else:
        grid = generate_grid(domain, initial_conditions)

    num_steps = int(total_time / time_step)
    snapshots = []

    for step in range(num_steps + 1):
        # ✅ Evolve grid and collect reflex diagnostics
        grid, reflex_metadata = evolve_step(grid, input_data, step)

        # ✅ Clean and serialize each cell
        serialized_grid = []
        for cell in grid:
            serialized_grid.append({
                "x": cell.x,
                "y": cell.y,
                "z": cell.z,
                "fluid_mask": getattr(cell, "fluid_mask", True),
                "velocity": cell.velocity if getattr(cell, "fluid_mask", True) else None,
                "pressure": cell.pressure if getattr(cell, "fluid_mask", True) else None
            })

        # ✅ Assemble snapshot with step index and flat reflex metrics
        snapshot = {
            "step_index": step,
            "grid": serialized_grid,
            **reflex_metadata
        }

        if step % output_interval == 0:
            snapshots.append((step, snapshot))

    return snapshots

def run_solver(input_path: str):
    """
    Executes simulation using the specified input file.
    Writes interval-based snapshots into the output folder.
    """
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



