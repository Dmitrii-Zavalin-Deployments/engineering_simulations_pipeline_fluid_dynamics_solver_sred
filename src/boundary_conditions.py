import json

def read_boundary_conditions(json_path):
    with open(json_path, "r") as file:
        boundary_data = json.load(file)
    return boundary_data

def apply_boundary_conditions(mesh, boundary_data):
    for face_id in boundary_data["inlet_faces"]:
        face = mesh.faces[face_id]
        face.set_pressure(boundary_data["inlet_boundary"]["pressure"])
        face.set_fluid_properties(boundary_data["inlet_boundary"]["fluid_properties"])

    for face_id in boundary_data["outlet_faces"]:
        face = mesh.faces[face_id]
        face.set_velocity(boundary_data["outlet_boundary"]["velocity"])

    for face_id in boundary_data["wall_faces"]:
        face = mesh.faces[face_id]
        face.apply_no_slip()
        face.set_wall_properties(boundary_data["wall_boundary"]["wall_properties"])



