# ✅ Unit Test Suite — Step Controller (Fully Patched)
# 📄 Full Path: tests/test_step_controller.py

import pytest
from dataclasses import dataclass
from src.step_controller import evolve_step

# Minimal Cell implementation for isolated testing
@dataclass
class DummyCell:
    x: int
    y: int
    z: int
    velocity: tuple[float, float, float]  # ✅ Patched to match unpacking expectations
    pressure: float
    fluid_mask: bool

@pytest.fixture
def minimal_input():
    return {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0, "nx": 1,
            "min_y": 0.0, "max_y": 1.0, "ny": 1,
            "min_z": 0.0, "max_z": 1.0, "nz": 1
        },
        "initial_conditions": {
            "initial_velocity": [0.0, 0.0, 0.0],
            "initial_pressure": 0.0
        },
        "boundary_conditions": {
            "apply_to": ["x-min"],
            "velocity": [0.0, 0.0, 0.0],
            "pressure": 0.0,
            "no_slip": True
        },
        "fluid_properties": {
            "density": 1.0,
            "viscosity": 0.01
        },
        "pressure_solver": {
            "method": "jacobi",
            "tolerance": 1e-6
        },
        "default_timestep": 0.01
    }

def test_evolve_step_outputs_valid_grid_and_metadata(minimal_input):
    grid = [DummyCell(x=0, y=0, z=0, velocity=(1.0, 0.0, 0.0), pressure=0.1, fluid_mask=True)]  # ✅ Patched
    updated, meta = evolve_step(grid, minimal_input, step=0)
    assert isinstance(updated, list)
    assert isinstance(meta, dict)
    assert "ghost_influence_count" in meta
    assert "adaptive_timestep" in meta
    assert "projection_passes" in meta
    assert "reflex_score" in meta

def test_evolve_step_applies_config_and_score(minimal_input):
    grid = [DummyCell(x=0, y=0, z=0, velocity=(0.5, 0.0, 0.0), pressure=0.05, fluid_mask=True)]  # ✅ Patched
    config = {
        "ghost_adjacency_depth": 2,
        "reflex_verbosity": "high"
    }
    reflex_score = 9
    _, meta = evolve_step(grid, minimal_input, step=1, config=config, reflex_score=reflex_score)
    assert meta["adaptive_timestep"] > 0
    assert isinstance(meta["ghost_influence_count"], int)
    assert meta["boundary_condition_applied"] is True

def test_evolve_step_pressure_metadata_injection(minimal_input):
    grid = [DummyCell(x=0, y=0, z=0, velocity=(0.0, 0.0, 0.0), pressure=0.0, fluid_mask=True)]  # ✅ Patched
    _, meta = evolve_step(grid, minimal_input, step=2)
    assert "pressure_mutated" in meta
    assert isinstance(meta.get("pressure_mutated"), bool)



