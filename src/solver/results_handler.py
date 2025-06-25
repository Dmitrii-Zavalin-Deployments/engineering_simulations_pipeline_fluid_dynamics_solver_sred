# src/solver/results_handler.py

import json
import os
import numpy as np

def save_simulation_metadata(sim_instance, output_dir):
    """
    Saves static simulation metadata and grid configuration.
    - config.json stores simulation parameters
    - mesh.json stores coordinate layout and BC mapping
    - readme.txt describes file schema and indexing contract
    """
    os.makedirs(output_dir, exist_ok=True)

    # === CONFIG ===
    config_path = os.path.join(output_dir, 'config.json')
    config_data = {
        "time_step": sim_instance.time_step,
        "total_time": sim_instance.total_time,
        "grid_shape": sim_instance.grid_shape,
        "variables": ["velocity", "pressure"],
        "fluid_density": sim_instance.rho,
        "fluid_viscosity": sim_instance.nu
    }
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    print(f"Saved simulation config to: {config_path}")

    # === MESH ===
    mesh_path = os.path.join(output_dir, 'mesh.json')
    mesh_data = {
        "grid_shape": sim_instance.grid_shape,
        "dx": sim_instance.mesh_info["dx"],
        "dy": sim_instance.mesh_info["dy"],
        "dz": sim_instance.mesh_info["dz"],
        "node_coordinates": sim_instance.mesh_info.get("node_coordinates", [])
    }

    # Include boundary conditions mapped to face indices and values
    bc_section = {}
    for bc_name, bc in sim_instance.mesh_info.get("boundary_conditions", {}).items():
        bc_value = bc.get("value", bc.get("pressure", None))
        bc_section[bc_name] = {
            "type": bc["type"],
            "cell_indices": (
                bc["cell_indices"].tolist()
                if isinstance(bc.get("cell_indices"), np.ndarray)
                else bc.get("cell_indices")
            ),
            "value": bc_value
        }
    mesh_data["boundary_conditions"] = bc_section

    with open(mesh_path, 'w') as f:
        json.dump(mesh_data, f, indent=2)
    print(f"Saved mesh definition to: {mesh_path}")

    # === README ===
    readme_path = os.path.join(output_dir, 'readme.txt')
    with open(readme_path, 'w') as f:
        f.write(
            "Simulation File Structure:\n"
            "- config.json: timestep, total time, grid size, physical parameters\n"
            "- mesh.json: spatial resolution, coordinates, and boundary mappings\n"
            "- fields/: one file per sampled timestep (e.g. step_0025.json)\n\n"
            "Indexing Policy:\n"
            "All field arrays are aligned with node_coordinates from mesh.json.\n"
            "Index i in velocity.x or pressure matches i-th node in mesh.\n"
        )
    print(f"Created readme: {readme_path}")

def save_field_snapshot(step_number, velocity_field, pressure_field, fields_dir):
    """
    Saves velocity and pressure fields at a given timestep into fields/ directory.
    """
    os.makedirs(fields_dir, exist_ok=True)

    # File format: fields/step_0025.json
    filename = f"step_{step_number:04d}.json"
    filepath = os.path.join(fields_dir, filename)

    snapshot = {
        "fields": {
            "velocity": {
                "x": velocity_field[..., 0].tolist(),
                "y": velocity_field[..., 1].tolist(),
                "z": velocity_field[..., 2].tolist()
            },
            "pressure": pressure_field.tolist()
        }
    }

    with open(filepath, 'w') as f:
        json.dump(snapshot, f, indent=2)
    print(f"Saved snapshot step {step_number} → {filepath}")

def save_final_summary(sim_instance, output_dir):
    """
    Saves final metadata after simulation completes.
    """
    summary = {
        "final_time": sim_instance.current_time,
        "steps_completed": sim_instance.step_count
    }
    path = os.path.join(output_dir, 'final_summary.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved final summary to: {path}")
