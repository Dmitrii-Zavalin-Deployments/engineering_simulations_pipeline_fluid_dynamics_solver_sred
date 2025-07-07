# simulation/adaptive_scheduler.py

"""
Adaptive Scheduler:
Manages projection pass depth and smoother iterations based on solver health and runtime metrics.
Designed to help the fluid solver adapt to instability, spikes, and recovery phases intelligently.
"""

import numpy as np

class AdaptiveScheduler:
    def __init__(self, config):
        self.enabled = config.get("enable_projection_ramp", True)
        self.max_passes = config.get("projection_passes_max", 3)
        self.decay_enabled = config.get("projection_pass_decay", False)
        self.smoother_adaptive_enabled = config.get("smoother_adaptive_enabled", True)
        self.energy_threshold = config.get("energy_threshold", 1e+06)
        self.damping_enabled = config.get("damping_enabled", True)
        self.damping_factor = config.get("damping_factor", 0.05)
        self.stabilization_window = config.get("stabilization_window", 5)

        # Internal state
        self.stability_counter = 0
        self.last_divergence = None

    def update_projection_passes(self, sim_instance, current_residual, current_divergence):
        if not self.enabled:
            return

        ramp = sim_instance.num_projection_passes

        if current_residual is not None and current_residual > sim_instance.residual_kill_threshold or \
           current_divergence > sim_instance.divergence_spike_factor * sim_instance.max_allowed_divergence:
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
        if last_residual > 5e+04:
            return 8
        elif last_residual > 1e+03:
            return 5
        else:
            return 3

    def apply_velocity_damping(self, velocity_field):
        if not self.damping_enabled:
            return velocity_field
        damped = velocity_field * (1 - self.damping_factor)
        print(f"🛑 Energy damping applied with factor {self.damping_factor:.2f}")
        return damped

    def check_energy_and_dampen(self, sim_instance):
        interior_v = sim_instance.velocity_field[1:-1, 1:-1, 1:-1, :]
        velocity_mag = np.linalg.norm(np.nan_to_num(interior_v), axis=-1)
        kinetic_energy = 0.5 * sim_instance.fluid_properties["density"] * np.sum(velocity_mag**2)

        if kinetic_energy > self.energy_threshold:
            print(f"🚨 Energy spike detected: KE={kinetic_energy:.2e} > threshold={self.energy_threshold:.2e}")
            sim_instance.velocity_field = self.apply_velocity_damping(sim_instance.velocity_field)



