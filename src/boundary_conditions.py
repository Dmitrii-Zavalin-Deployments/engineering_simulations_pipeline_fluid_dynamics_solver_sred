import json

def read_boundary_conditions(json_path):
    with open(json_path, "r") as file:
        boundary_data = json.load(file)
    return boundary_data

def apply_boundary_conditions(mesh, boundary_data):
    for face_id in boundary_data["inlet_faces"]:
        if face_id not in mesh.faces:
            print(f"❌ Warning: Face ID {face_id} not found in mesh. Skipping.")
            continue
        mesh.faces[face_id]["pressure"] = boundary_data["inlet_boundary"]["pressure"]
        mesh.faces[face_id]["fluid_properties"] = boundary_data["inlet_boundary"]["fluid_properties"]

    for face_id in boundary_data["outlet_faces"]:
        if face_id not in mesh.faces:
            print(f"❌ Warning: Face ID {face_id} not found in mesh. Skipping.")
            continue
        mesh.faces[face_id]["velocity"] = boundary_data["outlet_boundary"].get("velocity", None)

    for face_id in boundary_data["wall_faces"]:
        if face_id not in mesh.faces:
            print(f"❌ Warning: Face ID {face_id} not found in mesh. Skipping.")
            continue
        mesh.faces[face_id]["no_slip"] = boundary_data["wall_boundary"]["no_slip"]
        mesh.faces[face_id]["wall_properties"] = boundary_data["wall_boundary"]["wall_properties"]
        mesh.faces[face_id]["wall_functions"] = boundary_data["wall_boundary"]["wall_functions"]



