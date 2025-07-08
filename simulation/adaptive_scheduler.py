# simulation/adaptive_scheduler.py

import numpy as np  # ✅ Required for metric evaluations
import warnings
from src.stability_utils import get_threshold  # ✅ Centralized threshold accessor

class AdaptiveScheduler:
    def __init__(self, config):
        self.config = config
        # Reflex configuration parameters using safe accessor
        self.damping_enabled = get_threshold(config, "damping_enabled", True)
        self.damping_factor = get_threshold(config, "damping_factor", 0.1)
        self.divergence_spike_factor = get_threshold(config, "divergence_spike_factor", 100.0)
        self.abort_divergence_threshold = get_threshold(config, "abort_divergence_threshold", 1e6)
        self.abort_velocity_threshold = get_threshold(config, "abort_velocity_threshold", 1e6)
        self.abort_cfl_threshold = get_threshold(config, "abort_cfl_threshold", 1e6)
        self.projection_passes_max = get_threshold(config, "projection_passes_max", 4)
        self.max_consecutive_failures = get_threshold(config, "max_consecutive_failures", 3)
        self.consecutive_spike_count = 0

    def apply_velocity_damping(self, velocity_field):
        """
        Applies a damping factor to the velocity field if enabled.
        """
        if not self.damping_enabled:
            return velocity_field
        return velocity_field * (1.0 - self.damping_factor)

    def check_extreme_instability(self, sim, max_divergence, max_velocity, global_cfl):
        """
        Aborts the simulation if any metrics exceed defined thresholds.
        """
        if (
            max_divergence > self.abort_divergence_threshold or
            max_velocity > self.abort_velocity_threshold or
            global_cfl > self.abort_cfl_threshold
        ):
            raise RuntimeError("Simulation terminated due to extreme instability.")

    def monitor_divergence_and_escalate(self, sim, divergence_field, step):
        """
        Tracks divergence spikes and increments escalation counter.
        """
        current_max = float(np.max(divergence_field))
        threshold = self.divergence_spike_factor * getattr(sim, "max_allowed_divergence", 1.0)
        if current_max > threshold:
            self.consecutive_spike_count += 1
        else:
            self.consecutive_spike_count = 0

    def update_projection_passes(self, sim, residual, max_div):
        """
        Escalates or clamps the number of projection passes based on failure severity.
        """
        if getattr(sim, "num_projection_passes", 1) < self.projection_passes_max:
            sim.num_projection_passes += 1
        elif sim.num_projection_passes > self.projection_passes_max:
            sim.num_projection_passes = self.projection_passes_max

    def get_current_reflex_status(self):
        """
        Returns a snapshot of current reflex settings and escalation state.
        """
        return {
            "projection_passes": self.projection_passes_max,
            "damping_enabled": self.damping_enabled,
            "spike_count": self.consecutive_spike_count
        }



