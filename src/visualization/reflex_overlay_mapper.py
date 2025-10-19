# src/visualization/reflex_overlay_mapper.py
# 🖼️ Reflex Overlay Mapper — renders annotated spatial diagnostics if score is sufficient

import os
import matplotlib.pyplot as plt
from typing import List, Tuple

def render_reflex_overlay(
    step_index: int,
    reflex_score: float,
    mutation_coords: List[Tuple[float, float]],
    adjacency_coords: List[Tuple[float, float]],
    suppression_coords: List[Tuple[float, float]],
    output_path: str,
    score_threshold: float = 4.0,
    mutation_density: float = 0.0
):
    """
    Renders reflex map of ghost adjacency and pressure mutation zones.

    Roadmap Alignment:
    Reflex Scoring:
    - Visualizes mutation zones, suppression fallback, and ghost adjacency
    - Annotates reflex score and mutation density for overlays

    Args:
        step_index (int): Simulation step index
        reflex_score (float): Reflex integrity score
        mutation_coords (List[Tuple]): Coordinates of mutated pressure cells
        adjacency_coords (List[Tuple]): Ghost-adjacent fluid zones
        suppression_coords (List[Tuple]): Influence suppression zones
        output_path (str): Path to write PNG image
        score_threshold (float): Minimum reflex score to trigger rendering
        mutation_density (float): Ratio of mutated cells to fluid cells
    """
    # ✅ Defensive fallback patch
    if not isinstance(reflex_score, (int, float)):
        reflex_score = 0.0
    if reflex_score < score_threshold:
        print(f"[OVERLAY] 🛑 Skipping overlay: score {reflex_score} below threshold {score_threshold}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(
        f"🖼️ Reflex Overlay — Step {step_index}\n"
        f"Score={reflex_score:.2f} | Mutation Density={mutation_density:.2%}"
    )

    def plot_zone(coords, marker, label, color):
        if coords:
            x = [c[0] for c in coords]
            y = [c[1] for c in coords]
            ax.scatter(x, y, marker=marker, label=f"{label} ({len(coords)})", c=color, alpha=0.7)

    plot_zone(mutation_coords, marker="s", label="Pressure Mutation", color="red")
    plot_zone(adjacency_coords, marker="o", label="Ghost Adjacency", color="blue")
    plot_zone(suppression_coords, marker="x", label="Suppressed Influence", color="gray")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[OVERLAY] ✅ Step {step_index} overlay written → {output_path}")



