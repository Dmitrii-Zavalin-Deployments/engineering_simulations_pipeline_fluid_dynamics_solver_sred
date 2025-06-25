# src/main_solver.py

import numpy as np
import sys
import os

# --- FIX FOR ImportError: attempted relative import with no known parent package ---
# Get the directory where the current script resides (e.g., /home/runner/work/.../src).
script_dir = os.path.dirname(os.path.abspath(__file__))
# Add this directory to sys.path so that we can import from `physics` and `solver`.
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
# --- END FIX ---

# Import functions from the new modular files
from physics.solver_utils import compute_velocity_field, update_pressure_field, apply_boundary_conditions
from solver.initialization import (
    load_input_data,
    initialize_simulation_parameters,
    initialize_grid,
    initialize_fields,
    print_initial_setup
)
from solver.results_handler import save_simulation_results


class Simulation:
    """
    Main simulation class for the Navier-Stokes solver.
    This class orchestrates the simulation process.
    """
    def __init__(self, input_file_path):
        """
        Initializes the simulation by loading pre-processed input data
        and setting up the grid and fields.
        """
        print("--- Simulation Setup ---")
        self.input_file_path = input_file_path
        
        # Use external function to load input data
        self.input_data = load_input_data(self.input_file_path)
        
        # Use external functions for initialization
        initialize_simulation_parameters(self, self.input_data)
        initialize_grid(self, self.input_data)
        initialize_fields(self, self.input_data)
        
        # Use external function to print setup details
        print_initial_setup(self)

    def run(self):
        """Main simulation loop."""
        print("--- Running Simulation ---")
        current_time = 0.0
        step_count = 0
        
        try:
            while current_time < self.total_time:
                # Apply boundary conditions at the start of each step
                apply_boundary_conditions(self.u, self.v, self.w, self.p, self.boundary_conditions, self.mesh_info)
                
                # Compute the new velocity field
                self.u, self.v, self.w = compute_velocity_field(
                    self.u, self.v, self.w, self.p,
                    self.dx, self.dy, self.dz, self.time_step, self.rho, self.nu
                )

                # Update the pressure field
                self.p = update_pressure_field(
                    self.u, self.v, self.w, self.p,
                    self.dx, self.dy, self.dz, self.time_step, self.rho,
                    self.boundary_conditions, self.mesh_info
                )

                # Update time and step count
                current_time += self.time_step
                step_count += 1

                if step_count % self.output_frequency_steps == 0:
                    print(f"Step {step_count}: Time = {current_time:.4f} s")

            print("--- Simulation Finished ---")
            print(f"Final time: {current_time:.4f} s, Total steps: {step_count}")

        except Exception as e:
            print(f"An unexpected error occurred during simulation: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Note: The `save_results` method has been moved to results_handler.py

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python src/main_solver.py <input_json_filepath> <output_json_filepath>")
        print("Example: python src/main_solver.py temp/solver_input.json results/solution.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        # Create a simulation instance
        simulation_instance = Simulation(input_file)
        
        # Run the simulation
        simulation_instance.run()
        
        # Use the external function to save results
        save_simulation_results(simulation_instance, output_file)
        
    except Exception as e:
        print(f"Error: Process completed with exit code 1. An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)