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

            # Compute `face_to_cells` dynamically for structured meshes
            face_to_cells = {}  # Initialize empty mapping
            for face_id in all_faces:
                connected_cells = []

                # Assume faces are shared between two neighboring cells in a structured grid
                cell_left = face_id - 1 if face_id > 0 else None  # Left cell
                cell_right = face_id + 1 if face_id < len(all_faces) - 1 else None  # Right cell

                if cell_left is not None:
                    connected_cells.append(cell_left)
                if cell_right is not None:
                    connected_cells.append(cell_right)

                face_to_cells[face_id] = connected_cells  # ✅ Assign dynamically

            # Assign properties dynamically for each face based on its boundary type
            for face_id in all_faces:
                owner_cell = face_to_cells.get(face_id, [None])[0]  # ✅ Assign owner dynamically

                # Compute normal dynamically if missing in JSON
                if "face_normals" in mesh_data:
                    normal = mesh_data["face_normals"].get(face_id, [0.0, 0.0, 0.0])  # Use JSON data if available
                else:
                    # Estimate normal direction using connected cells
                    if len(face_to_cells[face_id]) == 2:
                        normal = [c2 - c1 for c1, c2 in zip(face_to_cells[face_id][0], face_to_cells[face_id][1])]
                    else:
                        normal = [0.0, 0.0, 1.0]  # Default normal for isolated faces

                self.faces[face_id] = {
                    "id": face_id,
                    "owner": owner_cell,  # ✅ Ensure valid owner assignment
                    "connected_cells": face_to_cells[face_id],  # ✅ Store computed adjacency
                    "normal": normal,  # ✅ Assign computed or default normal
                    "area": mesh_data["face_areas"].get(face_id, 1.0),  # Default if missing
                }

                # Add additional properties based on boundary type
                if face_id in self.boundaries["inlet_faces"]:
                    self.faces[face_id].update({
                        "type": mesh_data["inlet_boundary"].get("type", None),
                        "pressure": mesh_data["inlet_boundary"].get("pressure", None),
                        "fluid_properties": mesh_data["inlet_boundary"].get("fluid_properties", None)
                    })

                elif face_id in self.boundaries["outlet_faces"]:
                    self.faces[face_id].update({
                        "type": mesh_data["outlet_boundary"].get("type", None),
                        "velocity": mesh_data["outlet_boundary"].get("velocity", None)
                    })

                elif face_id in self.boundaries["wall_faces"]:
                    self.faces[face_id].update({
                        "type": "wall",
                        "no_slip": mesh_data["wall_boundary"].get("no_slip", None),
                        "wall_properties": mesh_data["wall_boundary"].get("wall_properties", None),
                        "wall_functions": mesh_data["wall_boundary"].get("wall_functions", None)
                    })

            print(f"✅ Assigned structured properties to {len(self.faces)} faces, including computed `face_to_cells` and normals.")

        except FileNotFoundError:
            print(f"❌ Error: Mesh JSON file '{json_filename}' not found.")
            raise

        except json.JSONDecodeError:
            print(f"❌ Error: Mesh JSON file '{json_filename}' is not valid JSON.")
            raise

        except Exception as e:
            print(f"❌ Unexpected error loading mesh file: {e}")
            raise
