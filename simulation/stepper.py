# simulation/stepper.py

import os
import numpy as np
import sys

from physics.boundary_conditions_applicator import apply_boundary_conditions
from numerical_methods.pressure_divergence import compute_pressure_divergence
from solver.results_handler import save_field_snapshot
from utils.log_utils import log_flow_metrics
from utils.simulation_output_manager import log_divergence_snapshot
from stability_utils import run_stability_checks  # ✅ Updated import from shared utility
from simulation.cfl_utils import calculate_max_cfl

def step_health_monitor(step, max_div, residual, kill_threshold, divergence_tolerance, sim_instance):
    """
    Emergency handler if solver exceeds safety thresholds.
    Reduces dt and clamps projection depth to attempt recovery.
    """
    if residual > kill_threshold or max_div > 100 * divergence_tolerance:
        print(f"⚠️ Stability compromised @ step {step}. dt halved, projection_passes clamped.")
        sim_instance.time_step *= sim_instance.dt_reduction_factor
        sim_instance.time_step = max(sim_instance.dt_min, sim_instance.time_step)
        sim_instance.num_projection_passes = 1
        return True
    return False

def run(self):
    print("--- Running Simulation ---")
    self.current_time = 0.0
    self.step_count = 0
    fields_dir = os.path.join(self.output_dir, "fields")
    num_steps = int(round(self.total_time / self.time_step))

    print(f"Total simulation time: {self.total_time} s")
    print(f"Initial time step: {self.time_step} s → {num_steps} steps")

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
            num_steps=num_steps,
            dt=self.time_step
        )

        log_divergence_snapshot(initial_divergence_field, self.step_count, self.output_dir)

        for _ in range(num_steps):
            self.step_count += 1
            self.current_time = self.step_count * self.time_step

            self.scheduler.check_energy_and_dampen(self)

            projection_passes = getattr(self, "num_projection_passes", 1)
            for pass_num in range(projection_passes):
                smoother_depth = self.scheduler.get_smoother_iterations(
                    getattr(self.time_stepper, "last_pressure_residual", 0.0)
                )
                self.velocity_field, self.p, divergence_at_step_field = self.time_stepper.step(
                    self.velocity_field,
                    self.p,
                    smoother_iterations=smoother_depth
                )

            apply_boundary_conditions(
                velocity_field=self.velocity_field,
                pressure_field=self.p,
                fluid_properties=self.fluid_properties,
                mesh_info=self.mesh_info,
                is_tentative_step=False
            )

            passed, div_metrics = run_stability_checks(
                velocity_field=self.velocity_field,
                pressure_field=self.p,  # ✅ Correct reference, not .pressure_field
                divergence_field=divergence_at_step_field,
                step=self.step_count,
                expected_velocity_shape=self.velocity_field.shape,
                expected_pressure_shape=self.p.shape,
                expected_divergence_shape=divergence_at_step_field.shape,
                divergence_mode=getattr(self, "divergence_mode", "log"),
                max_allowed_divergence=getattr(self, "max_allowed_divergence", 3e-2),
                velocity_limit=getattr(self, "velocity_limit", 10.0),
                spike_factor=getattr(self, "divergence_spike_factor", 100.0)
            )

            max_div = div_metrics["max"]
            residual = getattr(self.time_stepper, "last_pressure_residual", 0.0)
            recovery_triggered = step_health_monitor(
                step=self.step_count,
                max_div=max_div,
                residual=residual,
                kill_threshold=self.residual_kill_threshold,
                divergence_tolerance=self.max_allowed_divergence,
                sim_instance=self
            )

            self.scheduler.update_projection_passes(self, residual, max_div)
            self.scheduler.monitor_divergence_and_escalate(self, divergence_at_step_field, self.step_count)

            log_divergence_snapshot(
                divergence_at_step_field,
                self.step_count,
                self.output_dir,
                additional_meta={
                    "max_divergence": max_div,
                    "pressure_residual": residual,
                    "adaptive_recovery": recovery_triggered,
                    "dt": self.time_step,
                    "projection_passes": self.num_projection_passes,
                    "divergence_delta": div_metrics.get("delta"),
                    "divergence_slope": div_metrics.get("slope"),
                    "divergence_spike": div_metrics.get("spike_triggered")
                }
            )

            if (self.step_count % self.output_frequency_steps == 0) or \
               (self.step_count == num_steps and self.step_count != 0):
                save_field_snapshot(self.step_count, self.velocity_field, self.p, fields_dir)
                cfl_metrics = calculate_max_cfl(self)
                print(f"    CFL @ step {self.step_count}: {cfl_metrics['global_max']:.4f}")

            log_flow_metrics(
                velocity_field=self.velocity_field,
                pressure_field=self.p,
                divergence_field=divergence_at_step_field,
                fluid_density=self.fluid_properties["density"],
                step_count=self.step_count,
                current_time=self.current_time,
                output_frequency_steps=self.output_frequency_steps,
                num_steps=num_steps,
                dt=self.time_step,
                residual_divergence=residual,
                divergence_delta=div_metrics.get("delta"),
                divergence_slope=div_metrics.get("slope"),
                effectiveness_score=getattr(self.time_stepper, "effectiveness_score", None),
                event_tag="adaptive_recovery" if recovery_triggered else None,
                recovery_triggered=recovery_triggered,
                projection_passes=self.num_projection_passes,
                damping_applied=self.damping_enabled,
                smoother_iterations=smoother_depth,
                vcycle_residuals=None
            )

            if np.any(np.isnan(self.velocity_field)) or np.any(np.isinf(self.velocity_field)) or \
               np.any(np.isnan(self.p)) or np.any(np.isinf(self.p)):
                raise RuntimeError(f"❌ NaN or Inf detected at step {self.step_count}, t={self.current_time:.4e}s")

        print("--- Simulation Finished ---")
        print(f"Final time: {self.current_time:.4f} s, Total steps: {self.step_count}")

    except Exception as e:
        print(f"Unhandled error during simulation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise



