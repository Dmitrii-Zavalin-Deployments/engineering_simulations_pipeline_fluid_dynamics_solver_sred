# src/solver/results_handler.py

import os
import json
import numpy as np

def save_field_snapshot(step_count, velocity_field, pressure_field, fields_dir):
    """
    Saves the current velocity and pressure fields to a JSON file.

    Raises:
        OSError, PermissionError, or IOError if the write fails.
    """
    try:
        os.makedirs(fields_dir, exist_ok=True)

        filename = f"step_{step_count:04d}.json"
        filepath = os.path.join(fields_dir, filename)

        field_data = {
            "step": step_count,
            "velocity": velocity_field.tolist(),
            "pressure": pressure_field.tolist()
        }

        with open(filepath, 'w') as f:
            json.dump(field_data, f, indent=2)

        print(f"Saved snapshot step {step_count} → {filepath}")

    except (OSError, PermissionError, IOError) as e:
        print(f"❌ Failed to save snapshot step {step_count} to {fields_dir}: {e}")
        raise  # Re-raise so calling context (e.g., tests or CLI) can handle appropriately



