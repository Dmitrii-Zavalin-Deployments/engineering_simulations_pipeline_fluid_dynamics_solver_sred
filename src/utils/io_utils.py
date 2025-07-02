# src/utils/io_utils.py

import json
import sys
import numpy as np

def load_json_schema(filepath):
    """Loads a JSON schema from a file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Schema file not found at {filepath}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON schema file at {filepath}", file=sys.stderr)
        sys.exit(1)

def convert_numpy_to_list(obj):
    """
    Recursively converts NumPy arrays within a dictionary or list to standard Python lists.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(elem) for elem in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def save_json(data, filepath):
    """Saves data to a JSON file, handling non-serializable types."""
    try:
        cleaned_data = convert_numpy_to_list(data)
        with open(filepath, 'w') as f:
            json.dump(cleaned_data, f, indent=2)
    except Exception as e:
        print(f"Error saving JSON to {filepath}: {e}", file=sys.stderr)
        sys.exit(1)

def apply_config_defaults(config):
    """
    Fills in default values for optional fields in the simulation config.
    Raises KeyError for truly required fields.
    """
    config.setdefault("grid", {})
    config["grid"].setdefault("dx", 1.0)

    config.setdefault("time", {})
    config["time"].setdefault("time_step", 0.01)

    config.setdefault("solver", {})
    if not config["solver"].get("method"):
        config["solver"]["method"] = "explicit"

    config.setdefault("initial_conditions", {})
    config["initial_conditions"].setdefault("velocity_magnitude", 1.0)

    # Required check: density must be present
    if "fluid" not in config or "density" not in config["fluid"]:
        raise KeyError("Missing required field: fluid.density")

    return config



