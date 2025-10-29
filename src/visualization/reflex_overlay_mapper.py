# src/visualization/reflex_overlay_mapper.py
# 🖼️ Reflex Overlay Mapper — renders annotated spatial diagnostics if score is sufficient
# 📌 This module visualizes ghost adjacency, suppression fallback, pressure mutation zones, and boundary context.
# It excludes only cells explicitly marked fluid_mask=False.
# It does NOT skip based on adjacency or ghost proximity — all logic is geometry-mask-driven.

import os
import matplotlib.pyplot as plt
from typing import List, Tuple

# ✅ Centralized debug flag for GitHub Actions logging
debug = False

def render_reflex_overlay(
    step_index: int,
    reflex_score: float,
    mutation_coords: List[Tuple[float, float]],
    adjacency_coords: List[Tuple[float, float]],
    suppression_coords: List[Tuple[float, float]],
    output_path: str,
    score_threshold: float = 4.0,
    mutation_density: float = 0.0,
    boundary_coords: List[Tuple[float, float]] = None
):
    """
    Renders reflex map of ghost adjacency and pressure mutation zones.

    Roadmap Alignment:
    Reflex Scoring:
    - Visualizes mutation zones, suppression fallback, and ghost adjacency
    - Annotates reflex score and mutation density for overlays
    - Highlights boundary proximity for spatial context

    Integration:
    - adjacency_coords from ghost_face_mapper.tag_ghost_adjacency()
    - mutation_coords from pressure mutation tagging
    - suppression_coords from suppression_zones.detect_suppression_zones()
    - boundary_coords from is_boundary tagging logic
    """
    if not isinstance(reflex_score, (int, float)):
        reflex_score = 0.0
    if reflex_score < score_threshold:
        if debug:
            print(f"[OVERLAY] 🛑 Skipping overlay: score {reflex_score} below threshold {score_threshold}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(
        f"🖼️ Reflex Overlay — Step {step_index}\n"
        f"Score={reflex_score:.2f} | Mutation Density={mutation_density:.2%}"
    )

    def plot_zone(coords, marker, label, color, zorder=1, edgecolor=None, alpha=0.75):
        if coords:
            x = [c[0] for c in coords]
            y = [c[1] for c in coords]
            ax.scatter(
                x, y,
                marker=marker,
                label=f"{label} ({len(coords)})",
                c=color,
                edgecolors=edgecolor if edgecolor else color,
                alpha=alpha,
                zorder=zorder
            )

    # 🔴 Pressure mutation zones
    plot_zone(mutation_coords, marker="s", label="Pressure Mutation", color="red", zorder=4)

    # 🔵 Ghost adjacency zones
    plot_zone(adjacency_coords, marker="o", label="Ghost Adjacency", color="blue", zorder=3)

    # ⚪ Suppression fallback zones
    plot_zone(suppression_coords, marker="x", label="Suppressed Influence", color="gray", zorder=2)

    # 🟢 Boundary cells (subtle tint)
    if boundary_coords:
        plot_zone(boundary_coords, marker=".", label="Boundary Cells", color="green", zorder=1, alpha=0.3)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    if debug:
        print(f"[OVERLAY] ✅ Step {step_index} overlay written → {output_path}")



