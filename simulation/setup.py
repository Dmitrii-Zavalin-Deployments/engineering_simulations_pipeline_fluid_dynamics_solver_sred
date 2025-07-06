# simulation/setup.py

import numpy as np
import sys
import os
from datetime import datetime

from numerical_methods.explicit_solver import ExplicitSolver
from numerical_methods.implicit_solver import ImplicitSolver
from physics.boundary_conditions_applicator import apply_boundary_conditions
from solver.initialization import (
    load_input_data,
    initialize_simulation_parameters,
    initialize_grid,
    initialize_fields,
    print_initial_setup
)

class Simulation:
    def __init__(self, input_file_path: str, output_dir: str):
        print("--- Simulation Setup ---")
        self.input_file_path = input_file_path
        self.output_dir = output_dir
        self.start_time = datetime.now().isoformat()

        # Load and parse input JSON
        self.input_data = load_input_data(self.input_file_path)
        initialize_simulation_parameters(self, self.input_data)
        initialize_grid(self, self.input_data)

        # Preprocess boundary condition indices into NumPy arrays
        bc_dict = self.input_data["mesh_info"].get("boundary_conditions", {})
        for name, bc_data in bc_dict.items():
            if "cell_indices" not in bc_data or "ghost_indices" not in bc_data:
                raise ValueError(f"❌ BC '{name}' missing cell or ghost indices. Was pre-processing successful?")
            bc_data["cell_indices"] = np.array(bc_data["cell_indices"], dtype=int)
            bc_data["ghost_indices"] = np.array(bc_data["ghost_indices"], dtype=int)

        # Extract mesh info and dimensions
        self.mesh_info = self.input_data["mesh_info"]
        self.grid_shape = self.mesh_info["grid_shape"]
        self.dx = self.mesh_info["dx"]
        self.dy = self.mesh_info["dy"]
        self.dz = self.mesh_info["dz"]

        # Field initialization
        initialize_fields(self, self.input_data)
        self.velocity_field = np.stack((self.u, self.v, self.w), axis=-1)

        self.fluid_properties = {
            "density": self.rho,
            "viscosity": self.nu,
            "pressure_projection_passes": self.input_data["simulation_parameters"].get("projection_passes", 1)
        }
        self.boundary_conditions = bc_dict

        # Additional tunable parameters
        self.max_allowed_divergence = self.input_data["simulation_parameters"].get("max_allowed_divergence", 3e-2)
        self.divergence_mode = self.input_data["simulation_parameters"].get("divergence_mode", "log")

        # Solver selection: explicit, implicit, or future variants
        solver_type = self.input_data["simulation_parameters"].get("solver", "explicit").lower()
        if solver_type == "explicit":
            self.time_stepper = ExplicitSolver(self.fluid_properties, self.mesh_info, self.time_step)
            print("Using Explicit Solver.")
        elif solver_type == "implicit":
            self.time_stepper = ImplicitSolver(self.fluid_properties, self.mesh_info, self.time_step)
            print("Using Implicit (Semi-Implicit) Solver.")
        else:
            raise ValueError(f"Unknown solver type specified: '{solver_type}'. Must be 'explicit' or 'implicit'.")

        print_initial_setup(self)



