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
        return obj.tolist() # This is the key conversion
    else:
        return obj

def save_json(data, filepath):
    """Saves data to a JSON file, handling non-serializable types."""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving JSON to {filepath}: {e}", file=sys.stderr)
        sys.exit(1)