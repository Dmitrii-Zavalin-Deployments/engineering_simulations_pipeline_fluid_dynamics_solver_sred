# src/adaptive/grid_refiner.py
# 🧭 Grid Refiner — identifies mutation clusters for spatial refinement
# 📌 This module operates on exported pressure delta maps.
# It does NOT interact with fluid_mask or geometry masking logic.
# It is NOT responsible for solver inclusion/exclusion decisions.

import os
import json
from typing import List, Tuple
from collections import Counter

# ✅ Centralized debug flag for GitHub Actions logging
debug = True


def load_delta_map(path: str) -> List[Tuple[float, float, float]]:
    """
    Loads a pressure delta map and extracts coordinates with non-zero delta.
    These represent candidate locations for refinement.
    """
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
    threshold: int = 5
) -> List[Tuple[float, float, float]]:
    """
    Identifies clusters of mutation points based on spatial proximity.
    Returns coordinates with high mutation neighbor density.
    """
    dx, dy, dz = spacing
    counter = Counter()

    for base in coords:
        for other in coords:
            if base == other:
                continue
            if (
                abs(base[0] - other[0]) <= dx * radius
                and abs(base[1] - other[1]) <= dy * radius
                and abs(base[2] - other[2]) <= dz * radius
            ):
                counter[base] += 1

    return [coord for coord, count in counter.items() if count >= threshold]


def propose_refinement_zones(
    delta_map_path: str,
    spacing: Tuple[float, float, float],
    step_index: int,
    output_folder: str = "data/refinement",
    threshold: int = 5
) -> List[Tuple[float, float, float]]:
    """
    Loads a delta map and proposes refinement zones based on mutation clustering.
    Outputs a JSON file if clusters are found.
    """
    os.makedirs(output_folder, exist_ok=True)
    active_coords = load_delta_map(delta_map_path)

    if not active_coords:
        if debug:
            print(
                f"[REFINER] ⚠️ No pressure deltas found → skipping step "
                f"{step_index}"
            )
        return []

    clusters = detect_mutation_clusters(
        active_coords, spacing, threshold=threshold
    )

    if clusters:
        output_path = os.path.join(
            output_folder,
            f"refinement_step_{step_index:04d}.json"
        )
        with open(output_path, "w") as f:
            json.dump({"refinement_zones": clusters}, f, indent=2)
        if debug:
            print(
                f"[REFINER] 🧭 Proposed {len(clusters)} refinement zones →\n"
                f"          {output_path}"
            )
    else:
        if debug:
            print(
                f"[REFINER] 🚫 No clusters detected in step {step_index}"
            )

    return clusters
