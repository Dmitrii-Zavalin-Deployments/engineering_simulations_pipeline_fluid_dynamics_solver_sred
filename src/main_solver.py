# src/main_solver.py

import numpy as np
import sys
import os
import math
from datetime import datetime

# --- REFINED FIX FOR ImportError ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- END REFINED FIX ---

from numerical_methods.explicit_solver import ExplicitSolver
from physics.boundary_conditions_applicator import apply_boundary_conditions
from solver.initialization import (
    load_input_data,
    initialize_simulation_parameters,
    initialize_grid,
    initialize_fields,
    print_initial_setup
)
from solver.results_handler import save_field_snapshot
from utils.simulation_output_manager import setup_simulation_output_directory


class Simulation:
    """
    Main simulation class for the Navier-Stokes solver.
    This class orchestrates the simulation process.
    """
    def __init__(self, input_file_path, output_dir):
        print("--- Simulation Setup ---")
        self.input_file_path = input_file_path
        self.output_dir = output_dir
        self.start_time = datetime.now().isoformat()

        self.input_data = load_input_data(self.input_file_path)
        initialize_simulation_parameters(self, self.input_data)
        initialize_grid(self, self.input_data)  # sets self.mesh_info

        # Defensive check: boundary_conditions should have 'cell_indices'
        if 'boundary_conditions' in self.mesh_info:
            for name, bc_data in self.mesh_info['boundary_conditions'].items():
                if 'cell_indices' not in bc_data:
                    raise ValueError(
                        f"❌ Boundary condition '{name}' is missing 'cell_indices'. "
                        "Did you forget to run preprocessing with 'identify_boundary_nodes()'?"
                    )
                bc_data['cell_indices'] = np.array(bc_data['cell_indices'], dtype=int)

        self.grid_shape = self.mesh_info['grid_shape']
        initialize_fields(self, self.input_data)
        self.velocity_field = np.stack((self.u, self.v, self.w), axis=-1)

        fluid_properties = {'density': self.rho, 'viscosity': self.nu}
        self.time_stepper = ExplicitSolver(fluid_properties, self.mesh_info, self.time_step)

        print_initial_setup(self)

    def run(self):
        print("--- Running Simulation ---")
        self.current_time = 0.0
        self.step_count = 0
        fields_dir = os.path.join(self.output_dir, "fields")
        num_steps = int(round(self.total_time / self.time_step))

        print(f"Total desired simulation time: {self.total_time} s")
        print(f"Time step per iteration: {self.time_step} s")
        print(f"Calculated number of simulation steps: {num_steps}")

        try:
            save_field_snapshot(self.step_count, self.velocity_field, self.p, fields_dir)

            for i in range(num_steps):
                self.velocity_field, self.p = self.time_stepper.step(self.velocity_field, self.p)
                self.step_count += 1
                self.current_time = self.step_count * self.time_step

                if (self.step_count % self.output_frequency_steps == 0) or \
                   (self.step_count == num_steps and self.step_count != 0):
                    save_field_snapshot(self.step_count, self.velocity_field, self.p, fields_dir)

                if self.step_count % self.output_frequency_steps == 0:
                    print(f"Step {self.step_count}: Time = {self.current_time:.4f} s")

            print("--- Simulation Finished ---")
            print(f"Final time: {self.current_time:.4f} s, Total steps: {self.step_count}")

        except Exception as e:
            print(f"An unexpected error occurred during simulation: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python src/main_solver.py <input_json_filepath> <output_directory>")
        print("Example: python src/main_solver.py temp/solver_input.json data/testing-output-run")
        sys.exit(1)

    input_file = sys.argv[1]
    base_output_dir = sys.argv[2]
    output_dir = os.path.join(base_output_dir, "navier_stokes_output")

    try:
        simulation_instance = Simulation(input_file, output_dir)
        setup_simulation_output_directory(simulation_instance, output_dir)
        simulation_instance.run()
        print("✅ Main Navier-Stokes simulation executed successfully.")

    except Exception as e:
        print(f"Error: Process completed with exit code 1. An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)



