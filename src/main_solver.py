# src/main_solver.py

import numpy as np
import time
import os
import json
import sys
from datetime import datetime # Added for datetime.now() if used in print_initial_setup or elsewhere

# Add the project root directory to sys.path to enable 'src' module imports
# This is crucial when running the script from a subdirectory or via GitHub Actions
script_dir = os.path.dirname(__file__) # Directory of the current script (src)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir)) # Parent directory of src
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Corrected imports based on your provided older main_solver.py structure and ls -R
# Note: ExplicitSolver is not used in the current explicit step-by-step implementation,
# but keeping it commented out if you plan to reintroduce it.
# from numerical_methods.explicit_solver import ExplicitSolver
from physics.boundary_conditions_applicator import apply_boundary_conditions
from numerical_methods.advection import compute_advection_diffusion
from numerical_methods.diffusion import compute_diffusion_term
from numerical_methods.poisson_solver import solve_poisson_for_phi
from numerical_methods.pressure_correction import apply_pressure_correction

# Imports from src/solver/initialization.py as per your older snippet
from solver.initialization import (
    load_input_data,
    initialize_simulation_parameters,
    initialize_grid,
    initialize_fields,
    print_initial_setup
)
# Assuming save_field_snapshot and save_config_and_mesh are in src/utils/io.py
# as per your current main_solver.py. If they are in results_handler, this needs adjustment.
from utils.io import save_field_snapshot, save_config_and_mesh
from utils.metrics_calculator import calculate_total_kinetic_energy, calculate_max_velocity_magnitude, calculate_pressure_range, calculate_mean_pressure, calculate_std_dev_pressure, calculate_divergence, calculate_max_cfl
# from utils.simulation_output_manager import setup_simulation_output_directory # If this is explicitly called

class NavierStokesSolver: # Kept class name as NavierStokesSolver for consistency with current code
    def __init__(self, config_path):
        print("--- Simulation Setup ---")
        self.config_path = config_path # Store config_path for load_input_data
        self.output_base_dir = os.environ.get("OUTPUT_RESULTS_BASE_DIR", "data/testing-output-run/navier_stokes_output")
        os.makedirs(os.path.join(self.output_base_dir, "fields"), exist_ok=True)

        # Use functions from solver.initialization as per the old structure
        self.input_data = load_input_data(self.config_path)
        initialize_simulation_parameters(self, self.input_data) # This should set self.time_step, self.total_time, self.output_frequency_steps, self.rho, self.nu
        initialize_grid(self, self.input_data) # This should set self.mesh_info, self.dx, self.dy, self.dz, self.nx, self.ny, self.nz

        # Ensure boundary condition indices are NumPy arrays (as in old snippet)
        bc_dict = self.mesh_info.get("boundary_conditions", {})
        for name, bc_data in bc_dict.items():
            if "cell_indices" not in bc_data or "ghost_indices" not in bc_data:
                raise ValueError(f"❌ BC '{name}' missing cell or ghost indices. Was pre-processing successful?")
            bc_data["cell_indices"] = np.array(bc_data["cell_indices"], dtype=int)
            bc_data["ghost_indices"] = np.array(bc_data["ghost_indices"], dtype=int)
        self.mesh_info["boundary_conditions"] = bc_dict # Update mesh_info with processed BCs

        initialize_fields(self, self.input_data) # This should set self.u, self.v, self.w, self.p
        self.velocity_field = np.stack((self.u, self.v, self.w), axis=-1) # Create combined velocity_field
        self.pressure_field = self.p # Alias p to pressure_field for consistency with functions

        self.fluid_properties = {"density": self.rho, "viscosity": self.nu}

        # Apply boundary conditions for the initial state (step 0)
        self.velocity_field, self.pressure_field = apply_boundary_conditions(
            self.velocity_field, self.pressure_field,
            self.fluid_properties, self.mesh_info, is_tentative_step=False,
            step_number=0, output_frequency_steps=self.output_frequency_steps # Pass for initial state logging
        )

        self.current_time = 0.0
        self.step_number = 0 # Renamed from step_count to step_number for consistency

        print_initial_setup(self) # Call the print function from solver.initialization

        print("-------------------------")
        save_config_and_mesh(self.input_data, self.mesh_info, self.output_base_dir) # Use self.input_data for config

    # The calculate_max_cfl method was part of the Simulation class in your old snippet.
    # I'm keeping it here, but it's also in utils.metrics_calculator, so we could remove this
    # class method and just call the utility function directly. For now, keeping it as a method.
    def calculate_max_cfl(self, velocity_field):
        """
        Calculates the maximum CFL number in the domain.
        CFL = max(|u|*dt/dx, |v|*dt/dy, |w|*dt/dz)
        """
        # Consider only interior cells for CFL calculation
        u_interior = velocity_field[1:-1, 1:-1, 1:-1, 0]
        v_interior = velocity_field[1:-1, 1:-1, 1:-1, 1]
        w_interior = velocity_field[1:-1, 1:-1, 1:-1, 2]

        max_u = np.max(np.abs(u_interior)) if u_interior.size > 0 else 0.0
        max_v = np.max(np.abs(v_interior)) if v_interior.size > 0 else 0.0
        max_w = np.max(np.abs(w_interior)) if w_interior.size > 0 else 0.0

        cfl_x = (max_u * self.time_step) / self.dx if self.dx > 0 else 0.0
        cfl_y = (max_v * self.time_step) / self.dy if self.dy > 0 else 0.0
        cfl_z = (max_w * self.time_step) / self.dz if self.dz > 0 else 0.0

        return max(cfl_x, cfl_y, cfl_z)


    def run(self):
        print("\n--- Running Simulation ---")
        num_steps = int(round(self.total_time / self.time_step))

        print(f"Total desired simulation time: {self.total_time} s")
        print(f"Time step per iteration: {self.time_step} s")
        print(f"Calculated number of simulation steps: {num_steps}")

        # Save initial state (step 0)
        save_field_snapshot(self.velocity_field, self.pressure_field, self.step_number, self.output_base_dir)

        try:
            for _ in range(num_steps):
                self.step_number += 1
                self.current_time = self.step_number * self.time_step

                # Only print detailed step header if it's an output step
                if self.step_number % self.output_frequency_steps == 0:
                    print(f"\n--- Starting Explicit Time Step (Step {self.step_number}) ---")
                    print(f"  Current Time: {self.current_time:.6f} s")

                # 1. Compute advection and diffusion terms (explicit part of velocity update)
                velocity_star = compute_advection_diffusion(
                    self.velocity_field, self.fluid_viscosity, self.time_step, self.mesh_info,
                    self.step_number, self.output_frequency_steps
                )

                # 2. Apply boundary conditions to the tentative velocity field (u*)
                if self.step_number % self.output_frequency_steps == 0:
                    print("  2. Applying boundary conditions to the tentative velocity field (u*)...")
                velocity_star, _ = apply_boundary_conditions(
                    velocity_star, self.pressure_field, # pressure_field is not modified here
                    self.fluid_properties, self.mesh_info, is_tentative_step=True,
                    step_number=self.step_number, output_frequency_steps=self.output_frequency_steps
                )

                # 3. Solve Poisson equation for pressure correction
                if self.step_number % self.output_frequency_steps == 0:
                    print("  3. Solving Poisson equation for pressure correction...")
                max_divergence_before_correction = calculate_divergence(velocity_star, self.mesh_info)
                if self.step_number % self.output_frequency_steps == 0:
                    print(f"    - Max divergence before correction: {max_divergence_before_correction:.6e}")

                phi = solve_poisson_for_phi(
                    velocity_star, self.mesh_info, self.time_step, self.fluid_properties["density"],
                    self.pressure_field,
                    step_number=self.step_number, output_frequency_steps=self.output_frequency_steps
                )

                # 4. Apply pressure correction to velocity and update pressure
                if self.step_number % self.output_frequency_steps == 0:
                    print("  4. Applying pressure correction to velocity and updating pressure...")
                self.velocity_field, self.pressure_field = apply_pressure_correction(
                    velocity_star, self.pressure_field, phi, self.mesh_info, self.time_step, self.fluid_properties["density"],
                    step_number=self.step_number, output_frequency_steps=self.output_frequency_steps
                )

                # 5. Apply final boundary conditions to the corrected fields
                if self.step_number % self.output_frequency_steps == 0:
                    print("  5. Applying final boundary conditions to the corrected fields...")
                self.velocity_field, self.pressure_field = apply_boundary_conditions(
                    self.velocity_field, self.pressure_field,
                    self.fluid_properties, self.mesh_info, is_tentative_step=False,
                    step_number=self.step_number, output_frequency_steps=self.output_frequency_steps
                )

                # Output and logging frequency
                if (self.step_number % self.output_frequency_steps == 0) or \
                   (self.step_number == num_steps and self.step_number != 0):

                    print(f"\n📊 Step {self.step_number} @ t = {self.current_time:.4f}s")
                    total_kinetic_energy = calculate_total_kinetic_energy(self.velocity_field)
                    max_velocity_magnitude = calculate_max_velocity_magnitude(self.velocity_field)
                    pressure_range_val = calculate_pressure_range(self.pressure_field)
                    mean_pressure_val = calculate_mean_pressure(self.pressure_field)
                    std_dev_pressure_val = calculate_std_dev_pressure(self.pressure_field)
                    divergence_val = calculate_divergence(self.velocity_field, self.mesh_info)

                    print(f"   • Total Kinetic Energy     : {total_kinetic_energy:.4e}")
                    print(f"   • Max Velocity Magnitude   : {max_velocity_magnitude:.4e}")
                    print(f"   • Pressure Range (interior): [{pressure_range_val[0]:.4e}, {pressure_range_val[1]:.4e}]")
                    print(f"   • Mean Pressure (interior) : {mean_pressure_val:.4e}")
                    print(f"   • Std Dev Pressure         : {std_dev_pressure_val:.4e}")
                    print(f"   • Divergence ∇·u           : Max = {divergence_val[0]:.4e}, Mean = {divergence_val[1]:.4e}")
                    print("--- Explicit Time Step Complete ---")

                    save_field_snapshot(self.velocity_field, self.pressure_field, self.step_number, self.output_base_dir)

                    # Calculate and print CFL number (always print if outputting step)
                    cfl_number = self.calculate_max_cfl(self.velocity_field) # Using class method
                    print(f"    CFL Number: {cfl_number:.4f}")
                    print(f"Step {self.step_number}: Time = {self.current_time:.4f} s, ||u|| = {np.linalg.norm(self.velocity_field):.6e}")

            print("\n--- Simulation Complete ---")
            print(f"Final time: {self.current_time:.4f} s, Total steps: {self.step_number}")

        except Exception as e:
            print(f"An unexpected error occurred during simulation: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            raise


def cli_entrypoint():
    if len(sys.argv) != 3:
        print("Usage: python src/main_solver.py <input_json_filepath> <output_directory>", file=sys.stderr)
        sys.exit(1)

    input_file = sys.argv[1]
    base_output_dir = sys.argv[2]
    # The output directory logic is now handled within the NavierStokesSolver __init__
    # to ensure consistency with the setup_simulation_output_directory if it were used.
    # For now, we'll pass the base_output_dir and let the class construct the full path.

    try:
        sim = NavierStokesSolver(input_file) # Pass only input_file, output_dir is derived
        # If setup_simulation_output_directory is needed, it would be called here.
        # setup_simulation_output_directory(sim, base_output_dir) # Uncomment if this function is used
        sim.run()
        print("✅ Main Navier-Stokes simulation executed successfully.")

    except Exception as e:
        print(f"Error: Simulation failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    cli_entrypoint()



