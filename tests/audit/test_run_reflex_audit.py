# tests/audit/test_run_reflex_audit.py
# ✅ Validation suite for src/audit/run_reflex_audit.py

import os
import json
import shutil
import tempfile
import pytest
from unittest import mock

from src.audit.run_reflex_audit import load_snapshots, run_reflex_audit

@pytest.fixture
def snapshot_dir_with_valid_json(tmp_path):
    for i in range(3):
        data = {
            "step_index": i,
            "mutated_cells": i * 10,
            "pathway_recorded": i % 2 == 0,
            "has_projection": i % 2 == 1,
            "reflex_score": 100 - i * 10
        }
        fname = tmp_path / f"step_{i:04d}.json"
        with open(fname, "w") as f:
            json.dump(data, f)
    return str(tmp_path)

@pytest.fixture
def snapshot_dir_with_invalid_json(tmp_path):
    valid = tmp_path / "step_0000.json"
    with open(valid, "w") as f:
        json.dump({"step_index": 0}, f)
    broken = tmp_path / "step_0001.json"
    broken.write_text("{ invalid json }")
    return str(tmp_path)

@pytest.fixture
def empty_snapshot_dir(tmp_path):
    return str(tmp_path)

@pytest.fixture
def mock_batch_evaluate_trace():
    with mock.patch("src.audit.run_reflex_audit.batch_evaluate_trace") as mock_eval:
        mock_eval.return_value = [
            {
                "step_index": 0,
                "mutated_cells": 5,
                "pathway_recorded": True,
                "has_projection": True,
                "reflex_score": 92
            }
        ]
        yield mock_eval

@pytest.fixture
def mock_render_integrity_panel():
    with mock.patch("src.audit.run_reflex_audit.render_integrity_panel") as mock_render:
        yield mock_render

def test_load_snapshots_valid(snapshot_dir_with_valid_json):
    snapshots = load_snapshots(snapshot_dir_with_valid_json)
    assert len(snapshots) == 3
    assert all(isinstance(s, dict) for s in snapshots)
    assert snapshots[0]["step_index"] == 0

def test_load_snapshots_with_invalid_json(snapshot_dir_with_invalid_json, capsys):
    snapshots = load_snapshots(snapshot_dir_with_invalid_json)
    captured = capsys.readouterr()
    assert len(snapshots) == 1
    assert "[ERROR] Failed to load" in captured.out

def test_load_snapshots_empty(empty_snapshot_dir):
    snapshots = load_snapshots(empty_snapshot_dir)
    assert snapshots == []

def test_run_reflex_audit_full_flow(snapshot_dir_with_valid_json, tmp_path, mock_batch_evaluate_trace, mock_render_integrity_panel, capsys):
    output_dir = tmp_path / "diagnostics"
    log_path = output_dir / "mutation_pathways_log.json"

    run_reflex_audit(
        snapshot_dir=snapshot_dir_with_valid_json,
        output_folder=str(output_dir),
        pathway_log=str(log_path)
    )

    captured = capsys.readouterr()
    assert "📋 Starting Reflex Audit..." in captured.out
    assert "📊 Reflex Snapshot Summary:" in captured.out
    assert "✅ Reflex Audit Complete" in captured.out
    mock_batch_evaluate_trace.assert_called_once()
    mock_render_integrity_panel.assert_called_once()
    assert os.path.exists(output_dir)

def test_run_reflex_audit_no_snapshots(empty_snapshot_dir, tmp_path, capsys):
    output_dir = tmp_path / "diagnostics"
    run_reflex_audit(
        snapshot_dir=empty_snapshot_dir,
        output_folder=str(output_dir),
        pathway_log=str(output_dir / "log.json")
    )
    captured = capsys.readouterr()
    assert "[AUDIT] No snapshots found." in captured.out



