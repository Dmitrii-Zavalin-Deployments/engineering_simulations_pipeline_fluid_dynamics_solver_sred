import os
import re

OUTPUT_DIR = "navier_stokes_output"
FIELDS_DIR = os.path.join(OUTPUT_DIR, "fields")


def test_output_files_exist_and_are_nonempty():
    # Check top-level files
    for filename in ["config.json", "mesh.json"]:
        path = os.path.join(OUTPUT_DIR, filename)
        assert os.path.isfile(path), f"{filename} not found"
        assert os.path.getsize(path) > 0, f"{filename} is empty"

    # Check fields directory exists and has files
    assert os.path.isdir(FIELDS_DIR), "'fields/' directory missing"

    step_files = [
        f for f in os.listdir(FIELDS_DIR)
        if re.match(r"step_\d{4,}\.json$", f)
    ]
    assert step_files, "No step_XXXX.json files found in 'fields/'"
    
    for f in step_files:
        path = os.path.join(FIELDS_DIR, f)
        assert os.path.getsize(path) > 0, f"{f} in 'fields/' is empty"



