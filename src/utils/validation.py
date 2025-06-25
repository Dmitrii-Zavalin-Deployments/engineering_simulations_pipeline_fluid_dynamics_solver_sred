# src/utils/validation.py

import sys

def validate_json_schema(data):
    """
    Validates JSON data against a simplified schema.
    """
    # Check for top-level keys
    required_keys = ["fluid_properties", "simulation_parameters", "mesh", "boundary_conditions", "initial_conditions"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Input JSON missing top-level key: '{key}'")

    # Basic mesh structure check
    if "mesh" not in data or "boundary_faces" not in data["mesh"]:
        raise ValueError("Input JSON missing 'mesh.boundary_faces' section.")
    if not isinstance(data["mesh"]["boundary_faces"], list):
        raise ValueError("'mesh.boundary_faces' must be a list.")

    for i, face in enumerate(data["mesh"]["boundary_faces"]):
        if "face_id" not in face or "nodes" not in face:
            raise ValueError(f"Boundary face at index {i} is missing a 'face_id' or 'nodes' key.")

        if not isinstance(face["nodes"], dict):
            raise ValueError(f"Boundary face at index {i} 'nodes' value must be a dictionary (e.g., node_id: [x,y,z]). Found type: {type(face['nodes'])}")

        for node_id, node_coords in face["nodes"].items():
            if not (isinstance(node_coords, list) and len(node_coords) == 3 and all(isinstance(coord, (int, float)) for coord in node_coords)):
                raise ValueError(f"Node '{node_id}' in boundary face {i} must be a 3-element list [x, y, z] of numbers.")

    fluid_props = data.get("fluid_properties", {})
    if not isinstance(fluid_props.get("density"), (int, float)) or fluid_props["density"] <= 0:
        raise ValueError("Fluid density must be a positive number.")
    if not isinstance(fluid_props.get("viscosity"), (int, float)) or fluid_props["viscosity"] < 0:
        raise ValueError("Fluid viscosity must be a non-negative number.")

    sim_params = data.get("simulation_parameters", {})
    if not isinstance(sim_params.get("time_step"), (int, float)) or sim_params["time_step"] <= 0:
        raise ValueError("Simulation time_step must be a positive number.")
    if not isinstance(sim_params.get("total_time"), (int, float)) or sim_params["total_time"] <= 0:
        raise ValueError("Simulation total_time must be a positive number.")
    if sim_params.get("solver") not in ["explicit", "implicit"]:
        raise ValueError("Solver must be 'explicit' or 'implicit'.")

    if not isinstance(data.get("boundary_conditions"), dict):
        raise ValueError("Boundary conditions must be a dictionary.")

    initial_conds = data.get("initial_conditions", {})
    if "initial_velocity" not in initial_conds or not (isinstance(initial_conds["initial_velocity"], list) and len(initial_conds["initial_velocity"]) == 3):
        raise ValueError("initial_conditions must contain 'initial_velocity' as a 3-element list of numbers.")
    if "initial_pressure" not in initial_conds or not isinstance(initial_conds["initial_pressure"], (int, float)):
        raise ValueError("initial_conditions must contain 'initial_pressure' as a number.")

    print("✅ Input JSON passed basic structural validation.")