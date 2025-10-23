# tests/diagnostics/test_navier_stokes_verifier.py
# ✅ Verifier Activation Test — confirms that navier_stokes_verifier runs when diagnostic flags are triggered

import os
import json

def test_verifier_triggered_on_empty_divergence(tmp_path):
    from src.diagnostics.navier_stokes_verifier import run_verification_if_triggered
    from src.grid_modules.cell import Cell

    # 🧱 Setup: minimal downgraded grid
    grid = [Cell(x=0.0, y=0.0, z=0.0, velocity=None, pressure=None, fluid_mask=False)]
    spacing = (0.1, 0.1, 0.1)
    output_folder = str(tmp_path)
    step_index = 99

    # 🚨 Trigger verification flags
    flags = ["empty_divergence", "downgraded_cells", "no_pressure_mutation"]

    # 🧠 Run verifier
    run_verification_if_triggered(grid, spacing, step_index, output_folder, flags)

    # ✅ Assert expected outputs
    continuity_path = os.path.join(output_folder, "continuity_verification_step_0099.json")
    pressure_path = os.path.join(output_folder, "pressure_verification_step_0099.json")
    downgrade_path = os.path.join(output_folder, "downgrade_verification_step_0099.json")

    assert os.path.exists(continuity_path), "Continuity verification file missing"
    assert os.path.exists(pressure_path), "Pressure verification file missing"
    assert os.path.exists(downgrade_path), "Expected downgrade verification file even if empty"

    # ✅ Validate downgrade file structure
    with open(downgrade_path) as f:
        data = json.load(f)
        assert "downgraded_cells" in data
        assert isinstance(data["downgraded_cells"], list)



