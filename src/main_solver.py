# src/main_solver.py

import numpy as np
import time
import os
import json
import sys

# Add the project root directory to sys.path to enable 'src' module imports
# This is crucial when running the script from a subdirectory or via GitHub Actions
script_dir = os.path.dirname(__file__) # Directory of the current script (src)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir)) # Parent directory of src
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Corrected imports based on your ls -R output
from src.grid.grid_generator import GridGenerator
from src.physics.initialization import InitialConditionsApplier # Corrected path
from src.numerical_methods.advection import compute_advection_diffusion # Corrected path
from src.numerical_methods.diffusion import compute_diffusion_term # Corrected path
from src.physics.boundary_conditions_applicator import apply_boundary_conditions
from src.numerical_methods.poisson_solver import solve_poisson_for_phi
from src.numerical_methods.pressure_correction import apply_pressure_correction
from src.utils.io import save_field_snapshot, save_config_and_mesh # Corrected path
from src.utils.metrics_calculator import calculate_total_kinetic_energy, calculate_max_velocity_magnitude, calculate_pressure_range, calculate_mean_pressure, calculate_std_dev_pressure, calculate_divergence, calculate_max_cfl

class NavierStokesSolver:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.grid_generator = GridGenerator(self.config["domain_definition"])
        self.initial_conditions_applier = InitialConditionsApplier(self.config["initial_conditions"])

        self.nx = self.config["domain_definition"]["nx"]
        self.ny = self.config["domain_definition"]["ny"]
        self.nz = self.config["domain_definition"]["nz"]
        self.dx = self.config["domain_definition"]["max_x"] / self.nx
        self.dy = self.config["domain_definition"]["max_y"] / self.ny
        self.dz = self.config["domain_definition"]["max_z"] / self.nz
        self.mesh_info = self.grid_generator.generate_grid()

        # Add grid spacing to mesh_info for easy access in other modules
        self.mesh_info["dx"] = self.dx
        self.mesh_info["dy"] = self.dy
        self.mesh_info["dz"] = self.dz

        # Initialize fields with ghost cells
        # Velocity field: (nx+2, ny+2, nz+2, 3) for u, v, w components
        self.velocity_field = np.zeros((self.nx + 2, self.ny + 2, self.nz + 2, 3), dtype=np.float64)
        # Pressure field: (nx+2, ny+2, nz+2)
        self.pressure_field = np.zeros((self.nx + 2, self.ny + 2, self.nz + 2), dtype=np.float64)

        # Apply initial conditions to interior cells
        self.velocity_field, self.pressure_field = self.initial_conditions_applier.apply_initial_conditions(
            self.velocity_field, self.pressure_field, self.mesh_info
        )

        # Apply boundary conditions for the initial state
        self.velocity_field, self.pressure_field = apply_boundary_conditions(
            self.velocity_field, self.pressure_field,
            self.config["fluid_properties"], self.mesh_info, is_tentative_step=False,
            step_number=0, output_frequency_steps=self.output_frequency_steps # Pass for initial state logging
        )

        self.time_step = self.config["simulation_parameters"]["time_step"]
        self.total_time = self.config["simulation_parameters"]["total_time"]
        self.output_frequency_steps = self.config["simulation_parameters"].get("output_frequency_steps", 1)
        self.current_time = 0.0
        self.step_number = 0

        self.fluid_density = self.config["fluid_properties"]["density"]
        self.fluid_viscosity = self.config["fluid_properties"]["viscosity"]

        # Output directory setup
        self.output_base_dir = os.environ.get("OUTPUT_RESULTS_BASE_DIR", "data/testing-output-run/navier_stokes_output")
        os.makedirs(os.path.join(self.output_base_dir, "fields"), exist_ok=True)

        print("--- Simulation Setup ---")
        print(f"Loading pre-processed input from: {config_path}")
        print(f"Grid dimensions: {self.nx}x{self.ny}x{self.nz} cells")
        print(f"Grid spacing: dx={self.dx:.4f}, dy={self.dy:.4f}, dz={self.dz:.4f}")
        print(f"Total time: {self.total_time:.2f} s, Time step: {self.time_step:.2e} s") # Changed to .2e for small time steps
        print(f"Fluid: rho={self.fluid_density}, nu={self.fluid_viscosity}")
        print(f"Solver: {self.config['simulation_parameters']['solver']}")
        print("-------------------------")

        save_config_and_mesh(self.config, self.mesh_info, self.output_base_dir)

    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return json.load(f)

    def run(self):
        print("\n--- Running Simulation ---")
        print(f"Total desired simulation time: {self.total_time} s")
        print(f"Time step per iteration: {self.time_step} s")
        num_steps = int(self.total_time / self.time_step)
        print(f"Calculated number of simulation steps: {num_steps}")

        # Save initial state (step 0)
        save_field_snapshot(self.velocity_field, self.pressure_field, self.step_number, self.output_base_dir)

        while self.current_time < self.total_time - sys.float_info.epsilon: # Use epsilon for float comparison
            self.step_number += 1
            self.current_time += self.time_step

            # Only print detailed step header if it's an output step
            if self.step_number % self.output_frequency_steps == 0:
                print(f"\n--- Starting Explicit Time Step (Step {self.step_number}) ---") # Combined for clarity
                print(f"  Current Time: {self.current_time:.6f} s")

            # 1. Compute advection and diffusion terms (explicit part of velocity update)
            # Pass step_number and output_frequency_steps to control internal debug prints
            velocity_star = compute_advection_diffusion(
                self.velocity_field, self.fluid_viscosity, self.time_step, self.mesh_info,
                self.step_number, self.output_frequency_steps
            )

            # 2. Apply boundary conditions to the tentative velocity field (u*)
            # Pressure BCs are NOT applied at this stage
            if self.step_number % self.output_frequency_steps == 0:
                print("  2. Applying boundary conditions to the tentative velocity field (u*)...")
            velocity_star, _ = apply_boundary_conditions(
                velocity_star, self.pressure_field, # pressure_field is not modified here
                self.config["fluid_properties"], self.mesh_info, is_tentative_step=True,
                step_number=self.step_number, output_frequency_steps=self.output_frequency_steps
            )

            # 3. Solve Poisson equation for pressure correction
            if self.step_number % self.output_frequency_steps == 0:
                print("  3. Solving Poisson equation for pressure correction...")
            max_divergence_before_correction = calculate_divergence(velocity_star, self.mesh_info)
            if self.step_number % self.output_frequency_steps == 0:
                print(f"    - Max divergence before correction: {max_divergence_before_correction:.6e}")

            phi = solve_poisson_for_phi(
                velocity_star, self.mesh_info, self.time_step, self.fluid_density,
                self.pressure_field, # Pass current pressure field for BC initialization in Poisson solver
                step_number=self.step_number, output_frequency_steps=self.output_frequency_steps
            )

            # 4. Apply pressure correction to velocity and update pressure
            if self.step_number % self.output_frequency_steps == 0:
                print("  4. Applying pressure correction to velocity and updating pressure...")
            self.velocity_field, self.pressure_field = apply_pressure_correction(
                velocity_star, self.pressure_field, phi, self.mesh_info, self.time_step, self.fluid_density,
                step_number=self.step_number, output_frequency_steps=self.output_frequency_steps
            )

            # 5. Apply final boundary conditions to the corrected fields
            # Pressure BCs ARE applied at this stage
            if self.step_number % self.output_frequency_steps == 0:
                print("  5. Applying final boundary conditions to the corrected fields...")
            self.velocity_field, self.pressure_field = apply_boundary_conditions(
                self.velocity_field, self.pressure_field,
                self.config["fluid_properties"], self.mesh_info, is_tentative_step=False,
                step_number=self.step_number, output_frequency_steps=self.output_frequency_steps
            )

            # Calculate and print metrics, and save snapshot only if it's an output step
            if self.step_number % self.output_frequency_steps == 0:
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
                cfl_number = calculate_max_cfl(self.velocity_field, self.time_step, self.mesh_info)
                print(f"    CFL Number: {cfl_number:.4f}")
                print(f"Step {self.step_number}: Time = {self.current_time:.4f} s, ||u|| = {np.linalg.norm(self.velocity_field):.6e}")

        print("\n--- Simulation Complete ---")



