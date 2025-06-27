import os
import json
import numpy as np
import re
import pytest

OUTPUT_DIR = os.environ.get("ACTUAL_OUTPUT_DIR", "data/testing-output-run/navier_stokes_output")
FIELDS_DIR = os.path.join(OUTPUT_DIR, "fields")
REL_TOL = 1e-6
ABS_TOL = 1e-9


def list_step_files():
    if not os.path.isdir(FIELDS_DIR):
        return []
    step_files = [
        f for f in os.listdir(FIELDS_DIR)
        if re.match(r"step_\d+\.json$", f)
    ]
    return sorted(step_files, key=lambda f: int(re.search(r"(\d+)", f).group(1)))


def load_velocity(path):
    with open(path, "r") as f:
        data = json.load(f)
    velocity = np.array(data["velocity"])
    return velocity


def compute_total_kinetic_energy(velocity):
    KE = 0.5 * np.sum(np.linalg.norm(velocity, axis=-1)**2)
    return KE


@pytest.mark.skipif(not os.path.isdir(FIELDS_DIR), reason="Fields directory not found")
def test_kinetic_energy_monotonic_decay():
    step_files = list_step_files()
    if len(step_files) < 2:
        pytest.skip("Not enough output frames to assess energy decay")

    energies = []
    for step_file in step_files:
        path = os.path.join(FIELDS_DIR, step_file)
        velocity = load_velocity(path)
        KE = compute_total_kinetic_energy(velocity)
        energies.append(KE)

    energies = np.array(energies)
    energy_diffs = np.diff(energies)

    # Allow for minor numerical fluctuation, but not growth
    assert np.all(energy_diffs <= ABS_TOL), (
        "Kinetic energy increased between steps, violating monotonic decay."
    )



