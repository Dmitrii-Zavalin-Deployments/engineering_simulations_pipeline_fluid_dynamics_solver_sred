# simulation/setup.py

import numpy as np
import sys
import os
import json
from datetime import datetime
from pathlib import Path

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
from simulation.adaptive_scheduler import AdaptiveScheduler  # 🧠 Add adaptive scheduler

def load_config_overrides(config_path: str = "src/config.json"):
    if Path(config_path).exists():
        with open(config_path) as f:
            config_data = json.load(f)
        print(f"✅ Loaded override config from: {config_path}")
        return config_data.get("simulation_parameters", {})
    return {}

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

        # Optional override from config.json
        override_params = load_config_overrides()
        sim_params = self.input_data.get("simulation_parameters", {})
        sim_params.update(override_params)  # config.json takes precedence

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

        # Extract simulation-level tuning parameters
        self.max_allowed_divergence = sim_params.get("max_allowed_divergence", 3e-2)
        self.divergence_spike_factor = sim_params.get("divergence_spike_factor", 100.0)
        self.divergence_mode = sim_params.get("divergence_mode", "log")
        self.num_projection_passes = sim_params.get("projection_passes", 1)
        self.projection_passes_max = sim_params.get("projection_passes_max", self.num_projection_passes)
        self.residual_kill_threshold = sim_params.get("residual_kill_threshold", 1e4)
        self.smoother_sweeps = sim_params.get("smoother_sweeps", 5)
        self.fallback_solver = sim_params.get("fallback_solver", "direct")
        self.adaptive_dt_enabled = sim_params.get("adaptive_dt_enabled", True)

        # 🚦 Additional adaptive controls
        self.damping_enabled = sim_params.get("damping_enabled", True)
        self.damping_factor = sim_params.get("damping_factor", 0.05)
        self.enable_projection_ramp = sim_params.get("enable_projection_ramp", True)
        self.projection_pass_decay = sim_params.get("projection_pass_decay", False)
        self.energy_threshold = sim_params.get("energy_threshold", 1e6)
        self.smoother_adaptive_enabled = sim_params.get("smoother_adaptive_enabled", True)
        self.stabilization_window = sim_params.get("stabilization_window", 5)

        # Fluid properties + projection passes
        self.fluid_properties = {
            "density": self.rho,
            "viscosity": self.nu,
            "pressure_projection_passes": self.num_projection_passes
        }

        self.boundary_conditions = bc_dict

        # 📌 Instantiate AdaptiveScheduler
        self.scheduler = AdaptiveScheduler(sim_params)

        # Solver selection
        solver_type = sim_params.get("solver", "explicit").lower()
        if solver_type == "explicit":
            self.time_stepper = ExplicitSolver(self.fluid_properties, self.mesh_info, self.time_step)
            print("Using Explicit Solver.")
        elif solver_type == "implicit":
            self.time_stepper = ImplicitSolver(self.fluid_properties, self.mesh_info, self.time_step)
            print("Using Implicit (Semi-Implicit) Solver.")
        else:
            raise ValueError(f"Unknown solver type specified: '{solver_type}'. Must be 'explicit' or 'implicit'.")

        print_initial_setup(self)



