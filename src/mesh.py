import json  # Assuming the mesh file is JSON; adjust accordingly

class Mesh:
    def __init__(self, filename):
        self.cells, self.faces, self.boundaries = self.load_mesh(filename)

    def load_mesh(self, filename):
        try:
            with open(filename, "r") as file:
                mesh_data = json.load(file)

            # Ensure mesh_data contains required elements
            cells = mesh_data.get("cells", [])
            faces = mesh_data.get("faces", [])
            boundaries = mesh_data.get("boundaries", [])

            return cells, faces, boundaries

        except FileNotFoundError:
            print(f"❌ Error: Mesh file '{filename}' not found.")
            return [], [], []

        except json.JSONDecodeError:
            print(f"❌ Error: Mesh file '{filename}' is not valid JSON.")
            return [], [], []

        except Exception as e:
            print(f"❌ Unexpected error loading mesh file: {e}")
            return [], [], []



