# src/visualization/overlay_integrity_panel.py
# 🧭 Overlay Integrity Panel — renders overlay thumbnails and flags reflex anomalies for audit dashboards

import os
import json
from typing import List, Dict
from PIL import Image, ImageDraw, ImageFont

def load_overlay_metadata(snapshot_dir: str) -> List[Dict]:
    """
    Loads reflex metadata from snapshot summaries.

    Args:
        snapshot_dir (str): Directory containing snapshot JSON files

    Returns:
        List[Dict]: List of metadata entries per step
    """
    entries = []
    for fname in sorted(os.listdir(snapshot_dir)):
        if fname.endswith(".json") and "step_" in fname:
            path = os.path.join(snapshot_dir, fname)
            try:
                with open(path) as f:
                    data = json.load(f)
                    entries.append({
                        "step": data.get("step_index"),
                        "score": data.get("reflex_score", 0.0),
                        "mutation_density": data.get("mutation_density", 0.0),
                        "suppression_zones": len(data.get("suppression_zones", [])),
                        "overlay_path": os.path.join("data", "overlays", f"reflex_overlay_step_{data.get('step_index'):03d}.png")
                    })
            except Exception as e:
                print(f"[ERROR] Failed to load {fname}: {e}")
    return entries

def flag_anomalies(entry: Dict) -> List[str]:
    """
    Flags anomalies based on reflex metrics.

    Args:
        entry (Dict): Snapshot metadata

    Returns:
        List[str]: List of anomaly flags
    """
    flags = []
    if entry["score"] >= 4.0:
        flags.append("⚠️ High Reflex Score")
    if entry["mutation_density"] > 0.15:
        flags.append("🧬 Dense Mutation")
    if entry["suppression_zones"] > 10:
        flags.append("🛑 Suppression Spike")
    return flags

def render_integrity_panel(snapshot_dir: str, output_path: str = "data/diagnostics/integrity_panel.png"):
    """
    Renders a panel of overlay thumbnails with anomaly flags.

    Args:
        snapshot_dir (str): Directory containing snapshot JSON files
        output_path (str): Path to save the panel image
    """
    entries = load_overlay_metadata(snapshot_dir)
    if not entries:
        print("[INTEGRITY] No snapshot entries found.")
        return

    thumb_size = (160, 160)
    padding = 20
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    try:
        font = ImageFont.truetype(font_path, 14)
    except:
        font = ImageFont.load_default()

    cols = 5
    rows = (len(entries) + cols - 1) // cols
    panel_width = cols * (thumb_size[0] + padding)
    panel_height = rows * (thumb_size[1] + 2 * padding)

    panel = Image.new("RGB", (panel_width, panel_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(panel)

    for idx, entry in enumerate(entries):
        x = (idx % cols) * (thumb_size[0] + padding)
        y = (idx // cols) * (thumb_size[1] + 2 * padding)

        try:
            thumb = Image.open(entry["overlay_path"]).resize(thumb_size)
            panel.paste(thumb, (x, y))
        except Exception as e:
            print(f"[ERROR] Missing overlay for step {entry['step']}: {e}")
            continue

        label = f"Step {entry['step']:03d}"
        draw.text((x, y + thumb_size[1] + 5), label, fill=(0, 0, 0), font=font)

        flags = flag_anomalies(entry)
        for i, flag in enumerate(flags):
            draw.text((x, y + thumb_size[1] + 25 + i * 15), flag, fill=(255, 0, 0), font=font)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    panel.save(output_path)
    print(f"[INTEGRITY] Overlay panel saved → {output_path}")



