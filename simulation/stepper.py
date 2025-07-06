# simulation/stepper.py

import os
import numpy as np
import sys

from physics.boundary_conditions_applicator import apply_boundary_conditions
from numerical_methods.pressure_divergence import compute_pressure_divergence
from solver.results_handler import save_field_snapshot
from utils.log_utils import log_flow_metrics


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
        initial_divergence_field = compute_pressure_divergence(self.velocity_field, self.mesh_info)
        save_field_snapshot(self.step_count, self.velocity_field, self.p, fields_dir)

        log_flow_metrics(
            velocity_field=self.velocity_field,
            pressure_field=self.p,
            divergence_field=initial_divergence_field,
            fluid_density=self.fluid_properties["density"],
            step_count=self.step_count,
            current_time=self.current_time,
            output_frequency_steps=self.output_frequency_steps,
            num_steps=num_steps
        )

        for _ in range(num_steps):
            self.step_count += 1
            self.current_time = self.step_count * self.time_step

            print(f"[DEBUG @ Step {self.step_count}] Velocity BEFORE step: min={np.nanmin(self.velocity_field):.4e}, max={np.nanmax(self.velocity_field):.4e}")
            print(f"[DEBUG @ Step {self.step_count}] Pressure BEFORE step: min={np.nanmin(self.p):.4e}, max={np.nanmax(self.p):.4e}")

            self.velocity_field, self.p, divergence_at_step_field = self.time_stepper.step(self.velocity_field, self.p)

            print(f"[DEBUG @ Step {self.step_count}] Velocity AFTER step: min={np.nanmin(self.velocity_field):.4e}, max={np.nanmax(self.velocity_field):.4e}")
            print(f"[DEBUG @ Step {self.step_count}] Pressure AFTER step: min={np.nanmin(self.p):.4e}, max={np.nanmax(self.p):.4e}")
            print(f"[DEBUG @ Step {self.step_count}] Divergence AFTER step: min={np.nanmin(divergence_at_step_field):.4e}, max={np.nanmax(divergence_at_step_field):.4e}")

            apply_boundary_conditions(
                velocity_field=self.velocity_field,
                pressure_field=self.p,
                fluid_properties=self.fluid_properties,
                mesh_info=self.mesh_info,
                is_tentative_step=False
            )

            if (self.step_count % self.output_frequency_steps == 0) or \
               (self.step_count == num_steps and self.step_count != 0):
                save_field_snapshot(self.step_count, self.velocity_field, self.p, fields_dir)
                max_cfl = self.calculate_max_cfl(self.velocity_field)
                print(f"    CFL Number: {max_cfl:.4f}")

            log_flow_metrics(
                velocity_field=self.velocity_field,
                pressure_field=self.p,
                divergence_field=divergence_at_step_field,
                fluid_density=self.fluid_properties["density"],
                step_count=self.step_count,
                current_time=self.current_time,
                output_frequency_steps=self.output_frequency_steps,
                num_steps=num_steps
            )

            if np.any(np.isnan(self.velocity_field)) or np.any(np.isinf(self.velocity_field)) or \
               np.any(np.isnan(self.p)) or np.any(np.isinf(self.p)):
                raise RuntimeError(f"NaN or Inf detected in fields at step {self.step_count}, t={self.current_time:.4e}s")

        print("--- Simulation Finished ---")
        print(f"Final time: {self.current_time:.4f} s, Total steps: {self.step_count}")

    except Exception as e:
        print(f"An unexpected error occurred during simulation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise



