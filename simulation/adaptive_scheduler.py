# simulation/adaptive_scheduler.py

"""
Adaptive Scheduler:
Manages projection depth, smoother tuning, energy damping, and trend escalation
based on solver health and runtime instability detection.
"""

import numpy as np

class AdaptiveScheduler:
    def __init__(self, config):
        self.enabled = config.get("enable_projection_ramp", True)
        self.max_passes = config.get("projection_passes_max", 3)
        self.decay_enabled = config.get("projection_pass_decay", False)
        self.smoother_adaptive_enabled = config.get("smoother_adaptive_enabled", True)
        self.energy_threshold = config.get("energy_threshold", 1e6)
        self.damping_enabled = config.get("damping_enabled", True)
        self.damping_factor = config.get("damping_factor", 0.05)
        self.stabilization_window = config.get("stabilization_window", 5)
        self.max_consecutive_failures = config.get("max_consecutive_failures", 3)
        self.dt_reduction_factor = config.get("dt_reduction_factor", 0.5)
        self.dt_min = config.get("dt_min", 1e-5)

        # Internal tracking state
        self.stability_counter = 0
        self.last_divergence_max = None
        self.consecutive_spike_count = 0

    def update_projection_passes(self, sim_instance, current_residual, current_divergence):
        if not self.enabled:
            return

        ramp = sim_instance.num_projection_passes
        spike_threshold = sim_instance.divergence_spike_factor * sim_instance.max_allowed_divergence

        if (current_residual is not None and current_residual > sim_instance.residual_kill_threshold) or \
           (current_divergence > spike_threshold):
            ramp = min(ramp + 1, self.max_passes)
            self.stability_counter = 0
            print(f"🔧 Projection depth ramped → {ramp}")
        else:
            self.stability_counter += 1
            if self.decay_enabled and self.stability_counter >= self.stabilization_window:
                ramp = max(1, ramp - 1)
                print(f"🧹 Projection depth decayed → {ramp}")
                self.stability_counter = 0

        sim_instance.num_projection_passes = ramp

    def get_smoother_iterations(self, last_residual):
        if not self.smoother_adaptive_enabled or last_residual is None:
            return 3
        if last_residual > 5e4:
            return 8
        elif last_residual > 1e3:
            return 5
        else:
            return 3

    def apply_velocity_damping(self, velocity_field):
        if not self.damping_enabled:
            return velocity_field
        damped = velocity_field * (1 - self.damping_factor)
        print(f"🛑 Velocity damping applied with factor {self.damping_factor:.2f}")
        return damped

    def check_energy_and_dampen(self, sim_instance):
        interior_v = sim_instance.velocity_field[1:-1, 1:-1, 1:-1, :]
        velocity_mag = np.linalg.norm(np.nan_to_num(interior_v), axis=-1)
        kinetic_energy = 0.5 * sim_instance.fluid_properties["density"] * np.sum(velocity_mag**2)

        if kinetic_energy > self.energy_threshold:
            print(f"🚨 Energy spike detected: KE={kinetic_energy:.2e} > threshold={self.energy_threshold:.2e}")
            sim_instance.velocity_field = self.apply_velocity_damping(sim_instance.velocity_field)

    def monitor_divergence_and_escalate(self, sim_instance, divergence_field, step):
        """
        Tracks divergence trend slope and escalates if explosive growth detected.
        """
        interior = divergence_field[1:-1, 1:-1, 1:-1]
        interior = np.nan_to_num(interior, nan=0.0, posinf=0.0, neginf=0.0)
        current_max = float(np.max(np.abs(interior)))
        spike_threshold = sim_instance.divergence_spike_factor * sim_instance.max_allowed_divergence

        print(f"📡 ∇·u monitor @ Step {step} → max = {current_max:.4e}")

        if self.last_divergence_max is not None:
            delta = current_max - self.last_divergence_max
            slope = delta / max(1, step)
            print(f"📉 ∇·u slope: Δ={delta:.4e}, Δ/step={slope:.4e}")

            if current_max > spike_threshold:
                self.consecutive_spike_count += 1
                print(f"🚨 Spike count: {self.consecutive_spike_count}")
                if self.consecutive_spike_count >= self.max_consecutive_failures:
                    self.auto_escalate(sim_instance)
            else:
                self.consecutive_spike_count = 0

        self.last_divergence_max = current_max

    def auto_escalate(self, sim_instance):
        """
        Escalates correction strategy upon persistent divergence spikes.
        """
        if sim_instance.num_projection_passes < self.max_passes:
            sim_instance.num_projection_passes += 1
            print(f"⚙️ Auto-escalation: projection_passes → {sim_instance.num_projection_passes}")

        sim_instance.velocity_field = self.apply_velocity_damping(sim_instance.velocity_field)

        old_dt = sim_instance.time_step
        new_dt = max(self.dt_min, old_dt * self.dt_reduction_factor)
        if new_dt < old_dt:
            sim_instance.time_step = new_dt
            print(f"⏬ Timestep reduced → {new_dt:.4e}")

        self.consecutive_spike_count = 0



