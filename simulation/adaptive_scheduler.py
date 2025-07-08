# adaptive_scheduler.py

class AdaptiveScheduler:
    def __init__(self, config):
        self.config = config
        # Reflex configuration parameters
        self.damping_enabled = config.get("damping_enabled", True)
        self.damping_factor = config.get("damping_factor", 0.1)
        self.divergence_spike_factor = config.get("divergence_spike_factor", 100.0)
        self.abort_divergence_threshold = config.get("abort_divergence_threshold", 1e6)
        self.abort_velocity_threshold = config.get("abort_velocity_threshold", 1e6)
        self.abort_cfl_threshold = config.get("abort_cfl_threshold", 1e6)
        self.projection_passes_max = config.get("projection_passes_max", 4)
        self.max_consecutive_failures = config.get("max_consecutive_failures", 3)
        self.consecutive_spike_count = 0

    def apply_velocity_damping(self, velocity_field):
        if not self.damping_enabled:
            return velocity_field
        return velocity_field * (1.0 - self.damping_factor)

    def check_extreme_instability(self, sim, max_divergence, max_velocity, global_cfl):
        if (
            max_divergence > self.abort_divergence_threshold or
            max_velocity > self.abort_velocity_threshold or
            global_cfl > self.abort_cfl_threshold
        ):
            raise RuntimeError("Simulation terminated due to extreme instability.")

    def monitor_divergence_and_escalate(self, sim, divergence_field, step):
        current_max = float(np.max(divergence_field))
        threshold = self.divergence_spike_factor * getattr(sim, "max_allowed_divergence", 1.0)
        if current_max > threshold:
            self.consecutive_spike_count += 1
        else:
            self.consecutive_spike_count = 0

    def update_projection_passes(self, sim, residual, max_div):
        if getattr(sim, "num_projection_passes", 1) < self.projection_passes_max:
            sim.num_projection_passes += 1
        elif sim.num_projection_passes > self.projection_passes_max:
            sim.num_projection_passes = self.projection_passes_max

    def get_current_reflex_status(self):
        return {
            "projection_passes": self.projection_passes_max,
            "damping_enabled": self.damping_enabled,
            "spike_count": self.consecutive_spike_count
        }



