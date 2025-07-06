# src/main_solver.py

import sys
import os

# Ensure project root is in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from simulation import Simulation
from utils.simulation_output_manager import setup_simulation_output_directory


def cli_entrypoint():
    if len(sys.argv) != 3:
        print("Usage: python src/main_solver.py <input_json_filepath> <output_directory>", file=sys.stderr)
        sys.exit(1)

    input_file = sys.argv[1]
    base_output_dir = sys.argv[2]
    output_dir = os.path.join(base_output_dir, "navier_stokes_output")

    try:
        sim = Simulation(input_file, output_dir)
        setup_simulation_output_directory(sim, output_dir)
        sim.run()
        print("✅ Main Navier-Stokes simulation executed successfully.")
    except Exception as e:
        print(f"Error: Simulation failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli_entrypoint()



