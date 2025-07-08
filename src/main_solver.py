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
from stability_utils import run_stability_checks  # ✅ Imported from shared utility module

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

        # 🔍 Optional: Run stability diagnostics post-simulation
        passed, metrics = run_stability_checks(
            velocity_field=sim.velocity_field,
            pressure_field=sim.pressure_field,
            divergence_field=sim.divergence_field,
            step=sim.step_count,
            expected_velocity_shape=sim.velocity_field.shape,
            expected_pressure_shape=sim.pressure_field.shape,
            expected_divergence_shape=sim.divergence_field.shape,
            divergence_mode=sim.divergence_mode,
            max_allowed_divergence=sim.max_allowed_divergence,
            velocity_limit=sim.velocity_limit,
            spike_factor=sim.divergence_spike_factor
        )

        if not passed:
            print("⚠️ Stability check failed after solver run.")
            # Optional: exit or trigger recovery

        print("✅ Main Navier-Stokes simulation executed successfully.")
    except Exception as e:
        print(f"Error: Simulation failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    cli_entrypoint()



