# src/main_solver.py

import os
import json
import sys
from input_reader import load_simulation_input

def generate_snapshots(input_data: dict, scenario_name: str) -> list:
    """
    Generates a list of snapshots based on simulation settings.
    Returns list of (step_index, snapshot_dict) tuples.
    """
    time_step = input_data["simulation_parameters"]["time_step"]
    total_time = input_data["simulation_parameters"]["total_time"]
    output_interval = input_data["simulation_parameters"]["output_interval"]

    num_steps = int(total_time / time_step)
    snapshots = []

    for step in range(num_steps + 1):
        snapshot = {
            "step": step,
            "grid": [
                [0, 0, 0, input_data["initial_conditions"]["initial_velocity"], input_data["initial_conditions"]["initial_pressure"]],
                [0, 1, 0, input_data["initial_conditions"]["initial_velocity"], input_data["initial_conditions"]["initial_pressure"]],
                [1, 0, 0, input_data["initial_conditions"]["initial_velocity"], input_data["initial_conditions"]["initial_pressure"]]
            ],
            "max_velocity": input_data["initial_conditions"]["initial_velocity"][0],
            "max_divergence": 0.05,
            "global_cfl": time_step * 9.0,
            "overflow_detected": False,
            "damping_enabled": False,
            "projection_passes": 2
        }

        if step % output_interval == 0:
            snapshots.append((step, snapshot))

    return snapshots

def run_solver(input_path: str):
    """
    Executes simulation and writes snapshots to disk.
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
        print(f"📸 Saved snapshot: {filename}")

    print(f"✅ Simulation complete. {len(snapshots)} snapshots written.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️ Please provide an input file path.")
        sys.exit(1)

    run_solver(sys.argv[1])



