# src/visualization/influence_overlay.py
# 🌀 Influence Overlay — renders reflex-complete adjacency and suppression zones for visual diagnostics
# 📌 This module visualizes ghost adjacency, suppression fallback, and boundary context.
# It excludes only cells explicitly marked fluid_mask=False.
# It does NOT skip based on adjacency or ghost proximity — all logic is geometry-mask-driven.

import matplotlib.pyplot as plt
import os

# ✅ Centralized debug flag for GitHub Actions logging
debug = True

def render_influence_overlay(influence_log, output_path, score_threshold=0.85):
    """
    Renders adjacency and suppression zones from influence logs.
    Triggered only if the reflex-complete step score meets threshold.

    Parameters:
    - influence_log: dict containing zone coordinates and classification
    - output_path: string path to save the generated visual
    - score_threshold: float, reflex-complete threshold
    """
    score = influence_log.get("step_score")
    if not isinstance(score, (int, float)):
        score = 0.0

    if score < score_threshold:
        if debug:
            print(f"[OVERLAY] 🛑 Skipping overlay: score {score} below threshold {score_threshold}")
        return

    adjacency = influence_log.get("adjacency_zones", [])
    suppression = influence_log.get("suppression_zones", [])
    boundary = influence_log.get("boundary_cells", [])  # ✅ Optional input

    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.set_title("🌀 Influence Overlay: Reflex-Complete Zones")
    ax.set_aspect('equal')

    # Render boundary cells (subtle green dots)
    if boundary:
        x_b = [pt["x"] for pt in boundary]
        y_b = [pt["y"] for pt in boundary]
        ax.scatter(x_b, y_b, c="green", s=10, alpha=0.2, label="Boundary Cells", zorder=0)

    # Render adjacency zones
    for zone in adjacency:
        x, y, r = zone["x"], zone["y"], zone["radius"]
        circle = plt.Circle((x, y), r, color='blue', alpha=0.4, label="Adjacency")
        ax.add_patch(circle)

    # Render suppression zones
    for zone in suppression:
        x, y, r = zone["x"], zone["y"], zone["radius"]
        circle = plt.Circle((x, y), r, color='red', alpha=0.3, label="Suppression")
        ax.add_patch(circle)

    ax.legend(loc='upper right')
    ax.grid(True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    if debug:
        print(f"[OVERLAY] ✅ Influence overlay saved to {output_path}")



