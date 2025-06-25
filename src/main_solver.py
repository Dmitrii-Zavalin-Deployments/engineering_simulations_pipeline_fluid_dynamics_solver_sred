# src/main_solver.py

import numpy as np
import sys
import os
import math
from datetime import datetime # Import datetime to get the current time

# --- REFINED FIX FOR ImportError ---
# Get the directory of the current script (e.g., /path/to/project/src).
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory of the script's directory (e.g., /path/to/project).
# This is the root of your source package structure.
project_root = os.path.dirname(script_dir)
# Add the project root to sys.path, allowing absolute imports like 'src.physics' to work.
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- END REFINED FIX ---

# --- UPDATED IMPORTS ---
# Import the new ExplicitSolver class.
from numerical_methods.explicit_solver import ExplicitSolver

# Import boundary condition functions from their new, specific modules.
# Note: apply_boundary_conditions is expected to be used internally by ExplicitSolver,
# not directly by the Simulation class here.
from physics.boundary_conditions_applicator import apply_boundary_conditions 

from solver.initialization import (
    load_input_data,
    initialize_simulation_parameters,
    initialize_grid,
    initialize_fields,
    print_initial_setup
)
# Import functions from the refactored results_handler for saving snapshots and final summary
from solver.results_handler import save_field_snapshot, save_final_summary
# Import the new output manager
from utils.simulation_output_manager import setup_simulation_output_directory


class Simulation:
    """
    Main simulation class for the Navier-Stokes solver.
    This class orchestrates the simulation process.
    """
    def __init__(self, input_file_path, output_dir):
        """
        Initalizes the simulation by loading pre-processed input data
        and setting up the grid and fields.
        """
        print("--- Simulation Setup ---")
        self.input_file_path = input_file_path
        self.output_dir = output_dir # Store the output directory
        
        # --- FIX: Record the start time of the simulation ---
        self.start_time = datetime.now().isoformat()
        
        # Use external function to load input data
        self.input_data = load_input_data(self.input_file_path)
        
        # Use external functions for initialization
        initialize_simulation_parameters(self, self.input_data)
        initialize_grid(self, self.input_data) # This populates self.mesh_info
        
        # --- FIX: Convert cell_indices lists to NumPy arrays after loading ---
        if 'boundary_conditions' in self.mesh_info:
            print("DEBUG (Simulation.__init__): Converting 'cell_indices' lists to NumPy arrays.")
            for bc_data in self.mesh_info['boundary_conditions'].values():
                if 'cell_indices' in bc_data:
                    bc_data['cell_indices'] = np.array(bc_data['cell_indices'], dtype=int)
        # --- END FIX ---

        # --- FIX: Expose grid_shape as a top-level attribute for easy access ---
        self.grid_shape = self.mesh_info['grid_shape']
        # --- END FIX ---

        initialize_fields(self, self.input_data)
        
        # After initialization, combine the separate velocity components into a single array
        # This aligns with the new solver's input requirements.
        self.velocity_field = np.stack((self.u, self.v, self.w), axis=-1)
        
        # Instantiate the explicit solver object.
        # This object will manage the time stepping logic.
        fluid_properties = {'density': self.rho, 'viscosity': self.nu}
        # The ExplicitSolver expects mesh_info, which now has the 'grid_shape' and 'boundary_conditions'
        self.time_stepper = ExplicitSolver(fluid_properties, self.mesh_info, self.time_step)

        # Use external function to print setup details
        print_initial_setup(self)

    def run(self):
        """Main simulation loop."""
        print("--- Running Simulation ---")
        self.current_time = 0.0
        self.step_count = 0
        
        # Determine the subdirectory for field snapshots
        fields_dir = os.path.join(self.output_dir, "fields")
        
        # Calculate the total number of steps based on total_time and time_step
        num_steps = int(round(self.total_time / self.time_step))
        
        print(f"Total desired simulation time: {self.total_time} s")
        print(f"Time step per iteration: {self.time_step} s")
        print(f"Calculated number of simulation steps: {num_steps}")

        try:
            # Save the initial state (step 0) before the loop starts.
            # This is always saved as per the requirement.
            save_field_snapshot(self.step_count, self.velocity_field, self.p, fields_dir)
            
            # Use a for loop for predictable iteration count
            for i in range(num_steps):
                # The ExplicitSolver's step method handles advection, diffusion, pressure
                # projection, and boundary conditions in one step.
                self.velocity_field, self.p = self.time_stepper.step(self.velocity_field, self.p)

                # Update time and step count
                self.step_count += 1
                self.current_time = self.step_count * self.time_step 
                
                # --- FIX: Save snapshot only if it's a multiple of output_frequency_steps
                # OR if it's the very last step. Step 0 is already saved above. ---
                if (self.step_count % self.output_frequency_steps == 0) or \
                   (self.step_count == num_steps and self.step_count != 0): # Ensure we don't double save step 0
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
    
    # --- FIX: Modify output_dir to include the 'navier_stokes_output' subdirectory ---
    input_file = sys.argv[1]
    base_output_dir = sys.argv[2]
    output_dir = os.path.join(base_output_dir, "navier_stokes_output")
    
    try:
        # Create a simulation instance
        simulation_instance = Simulation(input_file, output_dir)
        
        # --- NEW: Setup output directory and save initial metadata BEFORE running the simulation ---
        setup_simulation_output_directory(simulation_instance, output_dir)
        
        # Run the simulation
        simulation_instance.run()
        
        # --- NEW: Save a final summary after the simulation is done ---
        # Note: save_final_summary will now only generate its output if called.
        # Based on previous request to remove final_summary.json, this call might be removed
        # but is kept here for now as you only asked about its size.
        # If you still want to remove it, this line should be commented/removed.
        save_final_summary(simulation_instance, output_dir)
        
    except Exception as e:
        print(f"Error: Process completed with exit code 1. An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)