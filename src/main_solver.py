# src/main_solver.py

import numpy as np
import os
import sys

# Add the project root to the sys.path
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

from src.mesh_generator import create_mesh
from src.initial_conditions_generator import create_initial_conditions
from src.numerical_methods.explicit_solver import ExplicitSolver
from src.utils.visualization import plot_velocity_field, plot_pressure_field
from src.utils.io_utils import save_data, load_data

class MainSolver:
    """
    Main solver class for incompressible fluid simulation.
    Orchestrates mesh generation, initial conditions, time stepping, and output.
    """
    def __init__(self, config: dict):
        self.config = config
        self.fluid_properties = config["fluid_properties"]
        self.simulation_params = config["simulation_parameters"]
        self.output_params = config["output_parameters"]
        self.mesh_info = create_mesh(config["mesh_parameters"])

        self.dt = self.simulation_params["dt"]
        self.total_steps = self.simulation_params["total_steps"]
        # Default to 100 if not specified, ensuring logging every 100 steps
        self.output_frequency_steps = self.output_params.get("output_frequency_steps", 100)
        # Default to 0 (no saving by default)
        self.save_frequency_steps = self.output_params.get("save_frequency_steps", 0)
        # Default to 0 (no plotting by default)
        self.plot_frequency_steps = self.output_params.get("plot_frequency_steps", 0)
        # Default to True for now, can be controlled via config
        self.verbose_logging = self.output_params.get("verbose_logging", True)

        self.velocity_field, self.pressure_field = create_initial_conditions(
            self.mesh_info, self.fluid_properties
        )

        self.solver = ExplicitSolver(
            fluid_properties=self.fluid_properties,
            mesh_info=self.mesh_info,
            dt=self.dt
        )
        self.step_number = 0 # Initialize step_number
        print("Solver initialized successfully.")

    def solve(self):
        """
        Runs the main simulation loop.
        """
        print(f"Starting simulation for {self.total_steps} steps with dt = {self.dt}...")

        for self.step_number in range(self.total_steps):
            current_time = self.step_number * self.dt

            # Determine if verbose logging should occur for this specific step
            # Log verbosely on step 0 and every output_frequency_steps, if verbose_logging is globally enabled
            should_log_verbose = self.verbose_logging and \
                                 (self.step_number % self.output_frequency_steps == 0 or self.step_number == 0)

            if should_log_verbose: # Only print this general update if verbose
                print(f"\n--- Time Step {self.step_number + 1}/{self.total_steps} (Time: {current_time:.4f}s) ---")

            # Perform one explicit time step, passing the logging flag
            self.velocity_field, self.pressure_field = self.solver.step(
                self.velocity_field,
                self.pressure_field,
                should_log_verbose=should_log_verbose # Pass the logging flag to the solver
            )

            # Save data if frequency dictates and saving is enabled
            # (self.step_number + 1) because step_number is 0-indexed for range, but we want 1-indexed for output
            if self.save_frequency_steps > 0 and \
               (self.step_number + 1) % self.save_frequency_steps == 0:
                step_output_dir = os.path.join(self.output_params["output_dir"], f"step_{self.step_number + 1}")
                os.makedirs(step_output_dir, exist_ok=True)
                save_data(self.velocity_field, "velocity", step_output_dir, self.step_number + 1)
                save_data(self.pressure_field, "pressure", step_output_dir, self.step_number + 1)
                if should_log_verbose: # Only print this saving message if verbose
                    print(f"Data saved for step {self.step_number + 1} to {step_output_dir}")


            # Plotting if frequency dictates and plotting is enabled
            if self.plot_frequency_steps > 0 and \
               (self.step_number + 1) % self.plot_frequency_steps == 0:
                plot_output_dir = os.path.join(self.output_params["output_dir"], "plots")
                os.makedirs(plot_output_dir, exist_ok=True)
                if should_log_verbose: # Only print this plotting message if verbose
                    print(f"Generating plots for step {self.step_number + 1}...")
                plot_velocity_field(self.velocity_field, self.mesh_info, plot_output_dir, self.step_number + 1)
                plot_pressure_field(self.pressure_field, self.mesh_info, plot_output_dir, self.step_number + 1)
                if should_log_verbose:
                    print(f"Plots saved to {plot_output_dir}")

        print(f"\nSimulation finished after {self.total_steps} steps.")
        print(f"Final simulation time: {self.total_steps * self.dt:.4f}s")

        # Optionally save final state regardless of output frequency
        if self.output_params.get("save_final_state", True): # Default to True
            final_output_dir = os.path.join(self.output_params["output_dir"], "final_state")
            os.makedirs(final_output_dir, exist_ok=True)
            save_data(self.velocity_field, "velocity_final", final_output_dir)
            save_data(self.pressure_field, "pressure_final", final_output_dir)
            print(f"Final state saved to {final_output_dir}")