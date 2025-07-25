# src/adaptive/timestep_controller.py
# 🔄 Timestep Controller — proposes dynamic timestep adjustment based on mutation intensity

import os
import json
from typing import Optional

def load_pressure_delta(path: str) -> dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def load_mutation_trace(path: str) -> list:
    try:
        with open(path, "r") as f:
            trace = json.load(f)
            return trace if isinstance(trace, list) else []
    except Exception:
        return []

def compute_mutation_density(pressure_delta_map: dict) -> float:
    total_cells = len(pressure_delta_map)
    mutated = sum(1 for cell in pressure_delta_map.values() if cell.get("delta", 0.0) > 0.0)
    if total_cells == 0:
        return 0.0
    return mutated / total_cells

def compute_mutation_frequency(mutation_trace: list, recent_steps: int = 5) -> float:
    recent = mutation_trace[-recent_steps:]
    if not recent:
        return 0.0
    mutated_steps = sum(1 for step in recent if step.get("pressure_mutated", False))
    return mutated_steps / len(recent)

def suggest_timestep(
    pressure_delta_path: str,
    mutation_trace_path: str,
    base_dt: float = 0.01,
    reflex_score: Optional[int] = None,
    min_score: int = 4
) -> float:
    delta_map = load_pressure_delta(pressure_delta_path)
    mutation_trace = load_mutation_trace(mutation_trace_path)

    mutation_density = compute_mutation_density(delta_map)
    mutation_frequency = compute_mutation_frequency(mutation_trace)

    score_ok = reflex_score is None or reflex_score >= min_score
    if not score_ok:
        print(f"[TIMESTEP] Reflex score below threshold → maintaining base_dt={base_dt:.6f}")
        return base_dt

    print(f"[TIMESTEP] Mutation density={mutation_density:.4f}, frequency={mutation_frequency:.2f}")

    if mutation_density > 0.20 and mutation_frequency >= 0.8:
        new_dt = base_dt * 0.5
        print(f"[TIMESTEP] 🔻 High mutation → reducing timestep to {new_dt:.6f}")
        return new_dt
    elif mutation_density < 0.05 and mutation_frequency <= 0.2:
        new_dt = base_dt * 1.5
        print(f"[TIMESTEP] 🔺 Low mutation → increasing timestep to {new_dt:.6f}")
        return new_dt
    else:
        print(f"[TIMESTEP] ⚖️ Timestep unchanged → {base_dt:.6f}")
        return base_dt



