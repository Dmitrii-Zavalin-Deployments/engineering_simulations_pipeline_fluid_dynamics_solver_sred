# src/main_solver.py

import os
import json
import sys
from src.input_reader import load_simulation_input
from src.grid_generator import generate_grid
from src.metrics.velocity_metrics import compute_max_velocity
from src.metrics.divergence_metrics import compute_max_divergence
from src.metrics.cfl_controller import compute_global_cfl
from src.metrics.overflow_monitor import detect_overflow
from src.metrics.damping_manager import should_dampen
from src.metrics.projection_evaluator import calculate_projection_passes

def generate_snapshots(input_data: dict, scenario_name: str) -> list:
    """
    Generates a list of snapshots based on simulation settings.
    Each snapshot includes metrics calculated from a freshly generated grid.
    """
    time_step = input_data["simulation_parameters"]["time_step"]
    total_time = input_data["simulation_parameters"]["total_time"]
    output_interval = input_data["simulation_parameters"]["output_interval"]

    domain = input_data["domain_definition"]
    initial_conditions = input_data["initial_conditions"]

    num_steps = int(total_time / time_step)
    snapshots = []

    for step in range(num_steps + 1):
        grid = generate_grid(domain, initial_conditions)

        snapshot = {
            "step": step,
            "grid": grid,
            "max_velocity": compute_max_velocity(grid),
            "max_divergence": compute_max_divergence(grid),
            "global_cfl": compute_global_cfl(grid, time_step),
            "overflow_detected": detect_overflow(grid),
            "damping_enabled": should_dampen(grid, time_step),
            "projection_passes": calculate_projection_passes(grid)
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



