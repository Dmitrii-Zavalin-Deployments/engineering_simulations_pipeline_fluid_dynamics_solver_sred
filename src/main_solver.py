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
    def __init__(self, input_file_path, output_dir):
        print("--- Simulation Setup ---")
        self.input_file_path = input_file_path
        self.output_dir = output_dir
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
        self.time_stepper = ExplicitSolver(self.fluid_properties, self.mesh_info, self.time_step)

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
            save_field_snapshot(self.step_count, self.velocity_field, self.p, fields_dir)

            for _ in range(num_steps):
                self.velocity_field, self.p = self.time_stepper.step(self.velocity_field, self.p)

                # ✅ Re-apply final boundary conditions
                # This is crucial after the pressure correction step
                apply_boundary_conditions(
                    velocity_field=self.velocity_field,
                    pressure_field=self.p,
                    fluid_properties=self.fluid_properties,
                    mesh_info=self.mesh_info,
                    is_tentative_step=False
                )

                self.step_count += 1
                self.current_time = self.step_count * self.time_step

                # Output and logging frequency
                if (self.step_count % self.output_frequency_steps == 0) or \
                   (self.step_count == num_steps and self.step_count != 0):
                    save_field_snapshot(self.step_count, self.velocity_field, self.p, fields_dir)
                    
                    # Calculate and print CFL number
                    max_cfl = self.calculate_max_cfl(self.velocity_field)
                    print(f"    CFL Number: {max_cfl:.4f}")
                    
                    # Also print flow metrics as before
                    # Ensure log_flow_metrics also handles potential NaNs/Infs gracefully
                    # (it should already be doing so based on previous updates)
                    vel_mag = np.linalg.norm(self.velocity_field[1:-1, 1:-1, 1:-1, :])
                    print(f"Step {self.step_count}: Time = {self.current_time:.4f} s, ||u|| = {vel_mag:.6e}")


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
    output_dir = os.path.join(base_output_dir, "navier_stokes_output")

    try:
        sim = Simulation(input_file, output_dir)
        setup_simulation_output_directory(sim, output_dir)
        sim.run()
        print("✅ Main Navier-Stokes simulation executed successfully.")

    except Exception as e:
        print(f"Error: Simulation failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    cli_entrypoint()




