import os
import json
import pytest
from PIL import Image
from src.visualization.overlay_integrity_panel import (
    load_overlay_metadata,
    flag_anomalies,
    render_integrity_panel
)

@pytest.fixture
def snapshot_dir(tmp_path):
    dir_path = tmp_path / "snapshots"
    os.makedirs(dir_path, exist_ok=True)

    # Valid snapshot with overlay
    with open(dir_path / "step_001.json", "w") as f:
        json.dump({
            "step_index": 1,
            "reflex_score": 4.2,
            "mutation_density": 0.2,
            "boundary_mutation_ratio": 0.35,
            "suppression_zones": [{"x": 0, "y": 0, "radius": 1.0}] * 11
        }, f)

    overlay_path = tmp_path / "data" / "overlays"
    os.makedirs(overlay_path, exist_ok=True)
    Image.new("RGB", (160, 160), color=(200, 200, 200)).save(overlay_path / "reflex_overlay_step_001.png")

    # Invalid JSON file
    with open(dir_path / "step_002.json", "w") as f:
        f.write("{ invalid json }")

    # Snapshot with missing overlay
    with open(dir_path / "step_003.json", "w") as f:
        json.dump({
            "step_index": 3,
            "reflex_score": 0.5,
            "mutation_density": 0.05,
            "boundary_mutation_ratio": 0.1,
            "suppression_zones": []
        }, f)

    return str(dir_path)

def test_load_overlay_metadata(snapshot_dir):
    entries = load_overlay_metadata(snapshot_dir)
    assert isinstance(entries, list)
    assert any(e["step"] == 1 for e in entries)
    assert all("overlay_path" in e for e in entries)

def test_flag_anomalies_detects_all():
    entry = {
        "score": 4.5,
        "mutation_density": 0.2,
        "suppression_zones": 12,
        "boundary_mutation_ratio": 0.4
    }
    flags = flag_anomalies(entry)
    assert "⚠️ High Reflex Score" in flags
    assert "🧬 Dense Mutation" in flags
    assert "🛑 Suppression Spike" in flags
    assert "🧱 Boundary Leakage" in flags

def test_flag_anomalies_detects_none():
    entry = {
        "score": 0.5,
        "mutation_density": 0.05,
        "suppression_zones": 2,
        "boundary_mutation_ratio": 0.1
    }
    flags = flag_anomalies(entry)
    assert flags == []

def test_render_integrity_panel_creates_output(snapshot_dir, tmp_path):
    output_path = tmp_path / "diagnostics" / "integrity_panel.png"
    render_integrity_panel(snapshot_dir, str(output_path))
    assert output_path.exists()

def test_render_integrity_panel_handles_missing_overlay(snapshot_dir, tmp_path, capsys):
    output_path = tmp_path / "diagnostics" / "integrity_panel.png"
    render_integrity_panel(snapshot_dir, str(output_path))
    output = capsys.readouterr().out
    assert "Missing overlay for step 3" in output



