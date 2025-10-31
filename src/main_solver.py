# src/main_solver.py
# 🚀 Modular Navier-Stokes Simulation Orchestrator
# 📌 Executes a 4-step pipeline: parse input, formulate system, solve equations, write output

import os
import sys
import json

from src.step1_input_validation.input_reader import load_simulation_input
from src.step1_input_validation.config_validator import validate_config
# from src.initialization.fluid_mask_initializer import build_simulation_grid
# from src.snapshot_manager import generate_snapshots
# from src.output.snapshot_writer import write_snapshot
# from src.upload_to_dropbox import upload_to_dropbox

# ✅ Centralized debug flag
debug = True


def step_1_parse_and_validate(input_path: str) -> dict:
    config = load_simulation_input(input_path)
    validate_config(config)
    if debug:
        print("✅ [Step 1] Parsed and validated input schema.")
        print(json.dumps(config, indent=2)[:1000])
    return config


def step_2_formulate_system(config: dict) -> dict:
    build_simulation_grid(config)

    navier_stokes_system = {
        "domain": config.get("domain_definition"),
        "fluid_properties": config.get("fluid_properties"),
        "initial_conditions": config.get("initial_conditions"),
        "boundary_conditions": config.get("boundary_conditions"),
        "simulation_parameters": config.get("simulation_parameters"),
        "geometry_definition": config.get("geometry_definition"),
        "grid": config.get("grid"),  # Assumes grid was injected by build_simulation_grid
    }

    if debug:
        print("✅ [Step 2] Navier-Stokes system formulated.")
        print(json.dumps(navier_stokes_system, indent=2)[:1000])

    return navier_stokes_system


def step_3_solve_system(navier_stokes_system: dict):
    snapshots = generate_snapshots(sim_config=navier_stokes_system)
    if debug:
        print(f"✅ [Step 3] Generated {len(snapshots)} snapshots.")
        print(f"📦 First snapshot preview:\n{json.dumps(snapshots[0][1], indent=2)[:1000]}")
    return snapshots


def step_4_write_output(snapshots, scenario_name: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for step, snapshot in snapshots:
        filename = f"{scenario_name}_step_{step:04d}.json"
        path = os.path.join(output_dir, filename)
        write_snapshot(snapshot, path)
        if debug:
            print(f"📝 [Step 4] Snapshot {step:04d} written → {filename}")
    if os.getenv("UPLOAD_TO_DROPBOX", "false").lower() == "true":
        upload_to_dropbox(output_dir)
        if debug:
            print("☁️ Output uploaded to Dropbox.")


def run_simulation(input_path: str, output_dir: str | None = None):
    scenario_name = os.path.splitext(os.path.basename(input_path))[0]
    output_folder = output_dir or os.path.join("data", "testing-input-output", "navier_stokes_output")

    config = step_1_parse_and_validate(input_path)
    config["step_index"] = 0

    navier_stokes_system = step_2_formulate_system(config)
    snapshots = step_3_solve_system(navier_stokes_system)
    step_4_write_output(snapshots, scenario_name, output_folder)

    if debug:
        print("✅ Simulation complete.")


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else None
    output_dir = os.getenv("OUTPUT_RESULTS_BASE_DIR", None)

    if not input_file:
        print("❌ Error: No input file provided.")
        sys.exit(1)

    run_simulation(input_file, output_dir)
