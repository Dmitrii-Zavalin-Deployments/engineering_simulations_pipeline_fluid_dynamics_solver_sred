# src/main_solver.py

import numpy as np
import sys
import os
from datetime import datetime

# --- REFINED FIX FOR ImportError ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- END REFINED FIX ---

from numerical_methods.explicit_solver import ExplicitSolver
from numerical_methods.implicit_solver import ImplicitSolver # Import ImplicitSolver
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
from utils.log_utils import log_flow_metrics # Import log_flow_metrics here
from numerical_methods.pressure_divergence import compute_pressure_divergence # Needed for initial divergence


class Simulation:
    def __init__(self, input_file_path, output_dir):
        print("--- Simulation Setup ---")
        self.input_file_path = input_file_path
        self.output_dir = output_dir # This will be the full 'base_output_dir/navier_stokes_output'
        self.start_time = datetime.now().isoformat()

        self.input_data = load_input_data(self.input_file_path)
        initialize_simulation_parameters(self, self.input_data)
        initialize_grid(self, self.input_data)

        bc_dict = self.input_data["mesh_info"].get("boundary_conditions", {})
        for name, bc_data in bc_dict.items():
            if "cell_indices" not in bc_data or "ghost_indices" not in bc_data:
                raise ValueError(f"❌ BC '{name}' missing cell or ghost indices. Was pre-processing successful?")
            bc_data["cell_indices"] = np.array(bc_data["cell_indices"], dtype=int)
            bc_data["ghost_indices"] = np.array(bc_data["ghost_indices"], dtype=int)

        self.mesh_info = self.input_data["mesh_info"]
        self.grid_shape = self.mesh_info["grid_shape"] # This is (nx, ny, nz) for interior
        # The actual field arrays will have (nx+2, ny+2, nz+2) due to ghost cells
        self.dx = self.mesh_info["dx"]
        self.dy = self.mesh_info["dy"]
        self.dz = self.mesh_info["dz"]

        initialize_fields(self, self.input_data)
        self.velocity_field = np.stack((self.u, self.v, self.w), axis=-1)

        self.fluid_properties = {"density": self.rho, "viscosity": self.nu}
        self.boundary_conditions = bc_dict
        
        # --- NEW: Select solver based on input data ---
        solver_type = self.input_data["simulation_parameters"].get("solver", "explicit").lower()
        if solver_type == "explicit":
            self.time_stepper = ExplicitSolver(self.fluid_properties, self.mesh_info, self.time_step)
            print("Using Explicit Solver.")
        elif solver_type == "implicit":
            # THIS IS THE CRUCIAL FIX for the "takes no arguments" error:
            self.time_stepper = ImplicitSolver(self.fluid_properties, self.mesh_info, self.time_step)
            print("Using Implicit (Semi-Implicit) Solver.")
        else:
            raise ValueError(f"Unknown solver type specified: '{solver_type}'. Must be 'explicit' or 'implicit'.")
        # --- END NEW ---

        print_initial_setup(self)

    def calculate_max_cfl(self, velocity_field):
        """
        Calculates the maximum CFL number in the domain.
        CFL = max(|u|*dt/dx, |v|*dt/dy, |w|*dt/dz)
        """
        # Consider only interior cells for CFL calculation
        u_interior = velocity_field[1:-1, 1:-1, 1:-1, 0]
        v_interior = velocity_field[1:-1, 1:-1, 1:-1, 1]
        w_interior = velocity_field[1:-1, 1:-1, 1:-1, 2]

        # Handle empty arrays if dimensions are 0 (e.g., for nx, ny, nz = 1 cases, which are not typical for CFD)
        max_u = np.max(np.abs(u_interior)) if u_interior.size > 0 else 0.0
        max_v = np.max(np.abs(v_interior)) if v_interior.size > 0 else 0.0
        max_w = np.max(np.abs(w_interior)) if w_interior.size > 0 else 0.0

        cfl_x = (max_u * self.time_step) / self.dx if self.dx > 0 else 0.0
        cfl_y = (max_v * self.time_step) / self.dy if self.dy > 0 else 0.0
        cfl_z = (max_w * self.time_step) / self.dz if self.dz > 0 else 0.0

        return max(cfl_x, cfl_y, cfl_z)


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
            # Calculate initial divergence (at step 0) for logging
            initial_divergence_field = compute_pressure_divergence(self.velocity_field, self.mesh_info)
            
            save_field_snapshot(self.step_count, self.velocity_field, self.p, fields_dir)
            
            # Initial logging of metrics for step 0
            log_flow_metrics(
                velocity_field=self.velocity_field,
                pressure_field=self.p,
                divergence_field=initial_divergence_field, # Use computed divergence at step 0
                fluid_density=self.fluid_properties["density"],
                step_count=self.step_count,
                current_time=self.current_time,
                output_frequency_steps=self.output_frequency_steps,
                num_steps=num_steps
            )

            for _ in range(num_steps):
                self.step_count += 1 # Increment step_count here, before the step calculation
                self.current_time = self.step_count * self.time_step

                # --- Debugging: Check fields before time step ---
                print(f"[DEBUG @ Step {self.step_count}] Velocity BEFORE step: min={np.nanmin(self.velocity_field):.4e}, max={np.nanmax(self.velocity_field):.4e}, has_nan={np.any(np.isnan(self.velocity_field))}, has_inf={np.any(np.isinf(self.velocity_field))}")
                print(f"[DEBUG @ Step {self.step_count}] Pressure BEFORE step: min={np.nanmin(self.p):.4e}, max={np.nanmax(self.p):.4e}, has_nan={np.any(np.isnan(self.p))}, has_inf={np.any(np.isinf(self.p))}")


                # --- Update: Call the selected time_stepper and get divergence ---
                # Both ExplicitSolver and ImplicitSolver.step are expected to return (velocity, pressure, divergence_field)
                self.velocity_field, self.p, divergence_at_step_field = self.time_stepper.step(
                    self.velocity_field, self.p
                )

                # --- Debugging: Check fields AFTER time step ---
                print(f"[DEBUG @ Step {self.step_count}] Velocity AFTER step: min={np.nanmin(self.velocity_field):.4e}, max={np.nanmax(self.velocity_field):.4e}, has_nan={np.any(np.isnan(self.velocity_field))}, has_inf={np.any(np.isinf(self.velocity_field))}")
                print(f"[DEBUG @ Step {self.step_count}] Pressure AFTER step: min={np.nanmin(self.p):.4e}, max={np.nanmax(self.p):.4e}, has_nan={np.any(np.isnan(self.p))}, has_inf={np.any(np.isinf(self.p))}")
                print(f"[DEBUG @ Step {self.step_count}] Divergence AFTER step: min={np.nanmin(divergence_at_step_field):.4e}, max={np.nanmax(divergence_at_step_field):.4e}, has_nan={np.any(np.isnan(divergence_at_step_field))}, has_inf={np.any(np.isinf(divergence_at_step_field))}")


                # ✅ Re-apply final boundary conditions
                # This is crucial after the pressure correction step.
                # Note: The 'step' method of the solver *should* already apply BCs internally
                # to its intermediate velocity and pressure fields. This final call here
                # ensures that the BCs are rigorously enforced on the final fields
                # before saving or further computations, providing a consistent state.
                apply_boundary_conditions(
                    velocity_field=self.velocity_field,
                    pressure_field=self.p,
                    fluid_properties=self.fluid_properties,
                    mesh_info=self.mesh_info,
                    is_tentative_step=False
                )
                # --- Debugging: Check fields AFTER final BCs for the step ---
                print(f"[DEBUG @ Step {self.step_count}] Velocity AFTER final BCs: min={np.nanmin(self.velocity_field):.4e}, max={np.nanmax(self.velocity_field):.4e}, has_nan={np.any(np.isnan(self.velocity_field))}, has_inf={np.any(np.isinf(self.velocity_field))}")
                print(f"[DEBUG @ Step {self.step_count}] Pressure AFTER final BCs: min={np.nanmin(self.p):.4e}, max={np.nanmax(self.p):.4e}, has_nan={np.any(np.isnan(self.p))}, has_inf={np.any(np.isinf(self.p))}")


                # Output and logging frequency for snapshots
                if (self.step_count % self.output_frequency_steps == 0) or \
                   (self.step_count == num_steps and self.step_count != 0):
                    save_field_snapshot(self.step_count, self.velocity_field, self.p, fields_dir)
                    
                    # Calculate and print CFL number
                    max_cfl = self.calculate_max_cfl(self.velocity_field)
                    print(f"    CFL Number: {max_cfl:.4f}")
                    
                # Always call log_flow_metrics, which will internally decide whether to print detailed metrics
                log_flow_metrics(
                    velocity_field=self.velocity_field,
                    pressure_field=self.p,
                    divergence_field=divergence_at_step_field, # Use the divergence from the current step
                    fluid_density=self.fluid_properties["density"],
                    step_count=self.step_count,
                    current_time=self.current_time,
                    output_frequency_steps=self.output_frequency_steps,
                    num_steps=num_steps
                )

                # Check for numerical stability (NaN/Inf) at the end of each step
                if np.any(np.isnan(self.velocity_field)) or np.any(np.isinf(self.velocity_field)) or \
                   np.any(np.isnan(self.p)) or np.any(np.isinf(self.p)):
                    raise RuntimeError(f"NaN or Inf detected in fields at step {self.step_count}, time {self.current_time:.4e}s. Simulation aborted.")


            print("--- Simulation Finished ---")
            print(f"Final time: {self.current_time:.4f} s, Total steps: {self.step_count}")

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
    # Reverted this line to your original structure
    output_dir = os.path.join(base_output_dir, "navier_stokes_output") 

    try:
        sim = Simulation(input_file, output_dir)
        # Reverted this line to your original call signature
        setup_simulation_output_directory(sim, output_dir) 
        sim.run()
        print("✅ Main Navier-Stokes simulation executed successfully.")

    except Exception as e:
        print(f"Error: Simulation failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    cli_entrypoint()



