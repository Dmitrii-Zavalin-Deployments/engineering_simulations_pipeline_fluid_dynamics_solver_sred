import os
import json

def load_json(filename):
    """
    Loads fluid simulation data from JSON file.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"❌ Error: Input file not found at {filename}")
    with open(filename, "r") as file:
        return json.load(file)