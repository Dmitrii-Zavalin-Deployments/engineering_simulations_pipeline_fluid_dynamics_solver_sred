import pytest
from src.reflex.reflex_logic import should_dampen, adjust_time_step

class MockCell:
    def __init__(self, x, y, z, velocity=None, fluid_mask=True,
                 pressure_mutated=False, damping_triggered=False, transport_triggered=False):
        self.x = x
        self.y = y
        self.z = z
        self.velocity = velocity
        self.fluid_mask = fluid_mask
        self.pressure_mutated = pressure_mutated
        self.damping_triggered = damping_triggered
        self.transport_triggered = transport_triggered

def test_should_dampen_triggers_on_high_volatility():
    grid = [
        MockCell(0, 0, 0, velocity=[1.0, 0.0, 0.0]),
        MockCell(1, 0, 0, velocity=[10.0, 0.0, 0.0])
    ]
    assert should_dampen(grid, volatility_threshold=0.5) is True

def test_should_dampen_returns_false_on_low_volatility():
    grid = [
        MockCell(0, 0, 0, velocity=[1.0, 0.0, 0.0]),
        MockCell(1, 0, 0, velocity=[1.1, 0.0, 0.0])
    ]
    assert should_dampen(grid, volatility_threshold=0.5) is False

def test_should_dampen_skips_nonfluid_cells():
    grid = [
        MockCell(0, 0, 0, velocity=[5.0, 0.0, 0.0], fluid_mask=False),
        MockCell(1, 0, 0, velocity=[5.0, 0.0, 0.0], fluid_mask=False)
    ]
    assert should_dampen(grid) is False

def test_adjust_time_step_reduces_dt_on_high_cfl():
    grid = [
        MockCell(0, 0, 0, velocity=[10.1, 0.0, 0.0]),  # ✅ slightly above threshold
        MockCell(1, 0, 0, velocity=[10.1, 0.0, 0.0])
    ]
    config = {
        "simulation_parameters": {"time_step": 0.1},
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0, "nx": 1
        }
    }
    adjusted = adjust_time_step(grid, config, cfl_limit=1.0)
    assert adjusted < 0.1

def test_adjust_time_step_scales_dt_on_high_mutation_ratio():
    grid = [
        MockCell(0, 0, 0, velocity=[1.0, 0.0, 0.0], pressure_mutated=True),
        MockCell(1, 0, 0, velocity=[1.0, 0.0, 0.0], damping_triggered=True),
        MockCell(2, 0, 0, velocity=[1.0, 0.0, 0.0], transport_triggered=True),
        MockCell(3, 0, 0, velocity=[1.0, 0.0, 0.0])
    ]
    config = {
        "simulation_parameters": {"time_step": 0.1},
        "domain_definition": {
            "min_x": 0.0, "max_x": 4.0, "nx": 4
        }
    }
    adjusted = adjust_time_step(grid, config)
    assert adjusted < 0.1

def test_adjust_time_step_returns_original_dt_when_stable():
    grid = [
        MockCell(0, 0, 0, velocity=[0.1, 0.0, 0.0]),
        MockCell(1, 0, 0, velocity=[0.1, 0.0, 0.0])
    ]
    config = {
        "simulation_parameters": {"time_step": 0.1},
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0, "nx": 1
        }
    }
    adjusted = adjust_time_step(grid, config)
    assert adjusted == 0.1



