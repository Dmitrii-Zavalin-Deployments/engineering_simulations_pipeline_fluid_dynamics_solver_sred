# src/utils/config_loader.py

import json
import os
import warnings

def get_flat_scheduler_config(path, section="damping_tests", silent=False):
    """
    Loads a configuration file and extracts the specified section as a flat dictionary.
    Intended for scheduler threshold loading from runtime or test config files.

    Parameters:
        path (str): Path to the JSON config file
        section (str): Section key to extract (default: 'damping_tests')
        silent (bool): If True, suppress warning if section is missing

    Returns:
        dict: Flattened dictionary from the specified section
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[ERROR] Config file not found at path: {path}")

    with open(path, "r") as f:
        config_data = json.load(f)

    if section not in config_data:
        if not silent:
            warnings.warn(f"[CONFIG LOADER] Section '{section}' missing in config. Returning empty dict.")
        return {}

    section_data = config_data.get(section, {})
    return section_data



