# src/adaptive/grid_refiner.py
# 🧭 Grid Refiner — identifies mutation clusters for spatial refinement

import os
import json
from typing import List, Tuple
from collections import Counter

def load_delta_map(path: str) -> List[Tuple[float, float, float]]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        return []

    return [
        tuple(map(float, key.strip("()").split(",")))
        for key, info in data.items()
        if abs(info.get("delta", 0.0)) > 0.0
    ]

def detect_mutation_clusters(
    coords: List[Tuple[float, float, float]],
    spacing: Tuple[float, float, float],
    radius: int = 1,
    threshold: int = 5  # ✅ Made configurable
) -> List[Tuple[float, float, float]]:
    """
    Returns coordinates with high mutation neighbor density.
    """
    dx, dy, dz = spacing
    counter = Counter()

    for base in coords:
        for other in coords:
            if base == other:
                continue
            if abs(base[0] - other[0]) <= dx * radius and \
               abs(base[1] - other[1]) <= dy * radius and \
               abs(base[2] - other[2]) <= dz * radius:
                counter[base] += 1

    return [coord for coord, count in counter.items() if count >= threshold]

def propose_refinement_zones(
    delta_map_path: str,
    spacing: Tuple[float, float, float],
    step_index: int,
    output_folder: str = "data/refinement"
) -> List[Tuple[float, float, float]]:
    os.makedirs(output_folder, exist_ok=True)
    active_coords = load_delta_map(delta_map_path)

    if not active_coords:
        print(f"[REFINER] ⚠️ No pressure deltas found → skipping step {step_index}")
        return []

    clusters = detect_mutation_clusters(active_coords, spacing)  # Can now pass `threshold` if needed

    if clusters:
        output_path = os.path.join(output_folder, f"refinement_step_{step_index:04d}.json")
        with open(output_path, "w") as f:
            json.dump({"refinement_zones": clusters}, f, indent=2)
        print(f"[REFINER] 🧭 Proposed {len(clusters)} refinement zones → {output_path}")
    else:
        print(f"[REFINER] 🚫 No clusters detected in step {step_index}")

    return clusters



