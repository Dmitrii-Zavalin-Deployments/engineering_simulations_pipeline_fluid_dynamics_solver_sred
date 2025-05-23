import json

class Mesh:
    def __init__(self, json_filename):
        self.faces = {}  # Dictionary mapping face IDs to properties
        self.boundaries = {}  # Stores boundary types and conditions

        self.load_mesh_from_json(json_filename)

    def load_mesh_from_json(self, json_filename):
        """ Reads mesh data from JSON and assigns structured properties dynamically. """
        try:
            with open(json_filename, "r") as file:
                mesh_data = json.load(file)

            # Extract boundary faces
            self.boundaries["inlet_faces"] = mesh_data.get("inlet_faces", [])
            self.boundaries["outlet_faces"] = mesh_data.get("outlet_faces", [])
            self.boundaries["wall_faces"] = mesh_data.get("wall_faces", [])

            all_faces = (self.boundaries["inlet_faces"] +
                         self.boundaries["outlet_faces"] +
                         self.boundaries["wall_faces"])

            # Assign properties dynamically for each face based on its boundary type
            for face_id in self.boundaries["inlet_faces"]:
                self.faces[face_id] = {
                    "id": face_id,
                    "type": mesh_data["inlet_boundary"].get("type", None),
                    "pressure": mesh_data["inlet_boundary"].get("pressure", None),
                    "fluid_properties": mesh_data["inlet_boundary"].get("fluid_properties", None)
                }

            for face_id in self.boundaries["outlet_faces"]:
                self.faces[face_id] = {
                    "id": face_id,
                    "type": mesh_data["outlet_boundary"].get("type", None),
                    "velocity": mesh_data["outlet_boundary"].get("velocity", None)
                }

            for face_id in self.boundaries["wall_faces"]:
                self.faces[face_id] = {
                    "id": face_id,
                    "type": "wall",
                    "no_slip": mesh_data["wall_boundary"].get("no_slip", None),
                    "wall_properties": mesh_data["wall_boundary"].get("wall_properties", None),
                    "wall_functions": mesh_data["wall_boundary"].get("wall_functions", None)
                }

            print(f"✅ Assigned structured properties to {len(self.faces)} faces.")

        except FileNotFoundError:
            print(f"❌ Error: Mesh JSON file '{json_filename}' not found.")
            raise

        except json.JSONDecodeError:
            print(f"❌ Error: Mesh JSON file '{json_filename}' is not valid JSON.")
            raise

        except Exception as e:
            print(f"❌ Unexpected error loading mesh file: {e}")
            raise



