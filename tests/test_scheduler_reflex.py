# tests/test_scheduler_reflex.py

import pytest
import numpy as np
from simulation.adaptive_scheduler import AdaptiveScheduler
from stability_utils import validate_threshold_config

REQUIRED_KEYS = [
    "damping_enabled",
    "damping_factor",
    "divergence_spike_factor",
    "abort_divergence_threshold",
    "abort_velocity_threshold",
    "abort_cfl_threshold",
    "projection_passes_max",
    "max_consecutive_failures"
]

class DummySim:
    def __init__(self):
        self.num_projection_passes = 1
        self.time_step = 0.01
        self.dt_min = 1e-5
        self.dt_reduction_factor = 0.5
        self.velocity_field = np.ones((6, 6, 6, 3)) * 100.0
        self.output_dir = "./logs/"
        self.fluid_properties = {"density": 1.0}
        self.divergence_mode = "log"
        self.max_allowed_divergence = 0.03
        self.velocity_limit = 10.0
        self.damping_enabled = True
        self.scheduler = None

def test_check_extreme_instability_triggers_abort(complete_reflex_config):
    assert validate_threshold_config(complete_reflex_config, REQUIRED_KEYS)
    sim = DummySim()
    scheduler = AdaptiveScheduler(complete_reflex_config)
    with pytest.raises(RuntimeError):
        scheduler.check_extreme_instability(
            sim,
            max_divergence=5e6,
            max_velocity=1e6 + 1,
            global_cfl=2e6
        )

def test_apply_velocity_damping_with_enabled_flag(complete_reflex_config):
    assert validate_threshold_config(complete_reflex_config, REQUIRED_KEYS)
    scheduler = AdaptiveScheduler(complete_reflex_config)
    field = np.ones((4, 4, 4, 3)) * 10.0
    damped_field = scheduler.apply_velocity_damping(field)
    expected_field = field * (1.0 - complete_reflex_config["damping_factor"])
    assert np.allclose(damped_field, expected_field)

def test_apply_velocity_damping_skipped_when_disabled(complete_reflex_config):
    config = complete_reflex_config.copy()
    config["damping_enabled"] = False
    assert validate_threshold_config(config, REQUIRED_KEYS)
    scheduler = AdaptiveScheduler(config)
    field = np.ones((4, 4, 4, 3)) * 8.0
    damped_field = scheduler.apply_velocity_damping(field)
    assert np.array_equal(damped_field, field)

def test_monitor_divergence_and_escalate_spike_triggers_counter(complete_reflex_config):
    assert validate_threshold_config(complete_reflex_config, REQUIRED_KEYS)
    sim = DummySim()
    scheduler = AdaptiveScheduler(complete_reflex_config)
    divergence_field = np.ones((8, 8, 8)) * 50.0
    scheduler.monitor_divergence_and_escalate(sim, divergence_field, step=2)
    assert scheduler.consecutive_spike_count >= 1

def test_update_projection_passes_escalates_depth(complete_reflex_config):
    assert validate_threshold_config(complete_reflex_config, REQUIRED_KEYS)
    sim = DummySim()
    scheduler = AdaptiveScheduler(complete_reflex_config)
    scheduler.update_projection_passes(sim, residual=5000.0, max_div=100.0)
    assert sim.num_projection_passes >= 2

def test_update_projection_passes_clamps_depth(complete_reflex_config):
    config = complete_reflex_config.copy()
    config["projection_passes_max"] = 3
    assert validate_threshold_config(config, REQUIRED_KEYS)
    sim = DummySim()
    sim.num_projection_passes = 5
    scheduler = AdaptiveScheduler(config)
    scheduler.update_projection_passes(sim, residual=10.0, max_div=0.01)
    assert sim.num_projection_passes <= config["projection_passes_max"]

def test_get_current_reflex_status_snapshot(complete_reflex_config):
    assert validate_threshold_config(complete_reflex_config, REQUIRED_KEYS)
    sim = DummySim()
    scheduler = AdaptiveScheduler(complete_reflex_config)
    divergence_field = np.ones((6, 6, 6)) * 100.0
    scheduler.monitor_divergence_and_escalate(sim, divergence_field, step=3)
    status = scheduler.get_current_reflex_status()
    assert isinstance(status, dict)
    assert status["projection_passes"] == complete_reflex_config["projection_passes_max"]
    assert status["damping_enabled"] is True
    assert status["spike_count"] >= 1



