import json
import os

class Mesh:
    def __init__(self, json_filename):
        self.cells = []  # Placeholder for future cell-based mesh initialization
        self.faces = {}  # Dictionary mapping face IDs to properties
        self.boundaries = {}  # Tracks boundary types and conditions

        self.load_mesh_from_json(json_filename)

    def load_mesh_from_json(self, json_filename):
        try:
            with open(json_filename, "r") as file:
                mesh_data = json.load(file)

            # Extract boundary faces
            self.boundaries["inlet_faces"] = mesh_data.get("inlet_faces", [])
            self.boundaries["outlet_faces"] = mesh_data.get("outlet_faces", [])
            self.boundaries["wall_faces"] = mesh_data.get("wall_faces", [])

            # Initialize face properties (assuming unique face IDs)
            for face_id in self.boundaries["inlet_faces"]:
                self.faces[face_id] = {
                    "type": "inlet",
                    "pressure": mesh_data["inlet_boundary"]["pressure"],
                    "fluid_properties": mesh_data["inlet_boundary"]["fluid_properties"]
                }

            for face_id in self.boundaries["outlet_faces"]:
                self.faces[face_id] = {
                    "type": "outlet",
                    "velocity": mesh_data["outlet_boundary"].get("velocity", None)
                }

            for face_id in self.boundaries["wall_faces"]:
                self.faces[face_id] = {
                    "type": "wall",
                    "no_slip": mesh_data["wall_boundary"]["no_slip"],
                    "wall_properties": mesh_data["wall_boundary"]["wall_properties"],
                    "wall_functions": mesh_data["wall_boundary"]["wall_functions"]
                }

            print(f"✅ Mesh successfully initialized from {json_filename}")
        
        except FileNotFoundError:
            print(f"❌ Error: Mesh JSON file '{json_filename}' not found.")
            raise
        
        except json.JSONDecodeError:
            print(f"❌ Error: Mesh JSON file '{json_filename}' is not valid JSON.")
            raise
        
        except Exception as e:
            print(f"❌ Unexpected error loading mesh file: {e}")
            raise

    def get_boundary_faces(self, boundary_type):
        """ Returns face IDs for a given boundary type ('inlet', 'outlet', 'wall') """
        return self.boundaries.get(f"{boundary_type}_faces", [])



