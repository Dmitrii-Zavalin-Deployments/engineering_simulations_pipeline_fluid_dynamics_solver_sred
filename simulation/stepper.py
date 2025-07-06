# simulation/stepper.py

import os
import numpy as np
import sys

from physics.boundary_conditions_applicator import apply_boundary_conditions
from numerical_methods.pressure_divergence import compute_pressure_divergence
from solver.results_handler import save_field_snapshot
from utils.log_utils import log_flow_metrics
from utils.simulation_output_manager import log_divergence_snapshot
from tests.stability_tests import run_stability_checks
from simulation.cfl_utils import calculate_max_cfl  # ✅ fix call context

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

        log_divergence_snapshot(initial_divergence_field, self.step_count, self.output_dir)

        for _ in range(num_steps):
            self.step_count += 1
            self.current_time = self.step_count * self.time_step

            print(f"[DEBUG @ Step {self.step_count}] Velocity BEFORE step: min={np.nanmin(self.velocity_field):.4e}, max={np.nanmax(self.velocity_field):.4e}")
            print(f"[DEBUG @ Step {self.step_count}] Pressure BEFORE step: min={np.nanmin(self.p):.4e}, max={np.nanmax(self.p):.4e}")

            # --- Pressure Correction Pass ---
            projection_passes = getattr(self, "num_projection_passes", 1)
            for pass_num in range(projection_passes):
                self.velocity_field, self.p, divergence_at_step_field = self.time_stepper.step(
                    self.velocity_field,
                    self.p
                )
                print(f"🔁 Pressure Projection Pass {pass_num + 1}")

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

            print(f"[DEBUG @ Step {self.step_count}] Velocity AFTER final BCs: min={np.nanmin(self.velocity_field):.4e}, max={np.nanmax(self.velocity_field):.4e}")
            print(f"[DEBUG @ Step {self.step_count}] Pressure AFTER final BCs: min={np.nanmin(self.p):.4e}, max={np.nanmax(self.p):.4e}")

            run_stability_checks(
                velocity_field=self.velocity_field,
                pressure_field=self.p,
                divergence_field=divergence_at_step_field,
                step=self.step_count,
                expected_velocity_shape=self.velocity_field.shape,
                expected_pressure_shape=self.p.shape,
                expected_divergence_shape=divergence_at_step_field.shape,
                divergence_mode=getattr(self, "divergence_mode", "log"),
                max_allowed_divergence=getattr(self, "max_allowed_divergence", 3e-2)
            )

            log_divergence_snapshot(divergence_at_step_field, self.step_count, self.output_dir)

            if (self.step_count % self.output_frequency_steps == 0) or \
               (self.step_count == num_steps and self.step_count != 0):
                save_field_snapshot(self.step_count, self.velocity_field, self.p, fields_dir)
                max_cfl = calculate_max_cfl(self)  # ✅ correct call to utility
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



