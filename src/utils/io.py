# src/utils/io.py

import os
import json
import numpy as np
import base64
import struct
import gzip # Import gzip for potential .gz JSON output

def load_json(filename: str) -> dict:
    """
    Loads fluid simulation data from a JSON file.

    Args:
        filename (str): The path to the JSON file.

    Returns:
        dict: The loaded data as a dictionary.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If the file content is not valid JSON.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"❌ Error: Input file not found at {filename}")
    with open(filename, "r") as file:
        return json.load(file)

def save_json(data: dict, filename: str, use_gzip: bool = False):
    """
    Saves data to a JSON file. Optionally compresses it with gzip.

    Args:
        data (dict): The dictionary to save.
        filename (str): The path to the output JSON file.
        use_gzip (bool): If True, saves as a .gz compressed JSON file.
    """
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if use_gzip:
        if not filename.endswith('.gz'):
            filename += '.gz'
        with gzip.open(filename, 'wt', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"Data successfully saved to '{filename}' (gzipped JSON).")
    else:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"Data successfully saved to '{filename}' (JSON).")


def write_output_to_vtk(
    velocity_field: np.ndarray,
    pressure_field: np.ndarray,
    x_coords_grid_lines: np.ndarray,
    y_coords_grid_lines: np.ndarray,
    z_coords_grid_lines: np.ndarray,
    output_filepath: str
):
    """
    Writes velocity and pressure fields to a VTK Image Data (.vti) file.
    The data is stored in appended, base64-encoded binary format.
    This format is ideal for visualization in tools like ParaView.

    Args:
        velocity_field (np.ndarray): The velocity field (nx, ny, nz, 3) in float64.
        pressure_field (np.ndarray): The pressure field (nx, ny, nz) in float64.
        x_coords_grid_lines (np.ndarray): 1D array of x-coordinates for grid lines (edges).
        y_coords_grid_lines (np.ndarray): 1D array of y-coordinates for grid lines (edges).
        z_coords_grid_lines (np.ndarray): 1D array of z-coordinates for grid lines (edges).
        output_filepath (str): Full path to the output .vti file (e.g., "results/output_step_X.vti").
    """
    # Ensure all data is float32 for VTK for smaller file sizes and common rendering
    velocity_field = velocity_field.astype(np.float32)
    pressure_field = pressure_field.astype(np.float32)

    # Determine grid dimensions (number of cells)
    # VTK's ImageData Extent is 0 to (N-1) for N cells.
    nx, ny, nz = velocity_field.shape[:3]

    # Determine origin (min_x, min_y, min_z)
    origin_x = float(x_coords_grid_lines[0])
    origin_y = float(y_coords_grid_lines[0])
    origin_z = float(z_coords_grid_lines[0])

    # Determine spacing (dx, dy, dz)
    # Calculate spacing from grid lines. Handle cases with single grid line for a dimension.
    spacing_x = (x_coords_grid_lines[-1] - x_coords_grid_lines[0]) / (nx) if nx > 0 else 1.0
    spacing_y = (y_coords_grid_lines[-1] - y_coords_grid_lines[0]) / (ny) if ny > 0 else 1.0
    spacing_z = (z_coords_grid_lines[-1] - z_coords_grid_lines[0]) / (nz) if nz > 0 else 1.0
    
    # Ensure spacing is never zero for VTK's ImageData representation, even if it's a collapsed dimension
    # (i.e. min_val == max_val and nx=1).
    if spacing_x == 0.0: spacing_x = 1.0
    if spacing_y == 0.0: spacing_y = 1.0
    if spacing_z == 0.0: spacing_z = 1.0

    # Ensure output directory exists
    output_dir = os.path.dirname(output_filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Flatten velocity and pressure data for writing
    # Velocity is (nx*ny*nz, 3)
    # Pressure is (nx*ny*nz)
    flattened_velocity = velocity_field.reshape(-1, 3)
    flattened_pressure = pressure_field.flatten()

    # Pack binary data
    # VTK expects data to be contiguous in memory.
    # Convert to bytes
    velocity_bytes = flattened_velocity.tobytes(order='C')
    pressure_bytes = flattened_pressure.tobytes(order='C')

    # Calculate offsets for appended data
    # VTK's appended data format includes an 8-byte header (UInt64) for the size of the block.
    # The first block starts at offset 0 after the initial underscore.
    velocity_offset = 0
    # The pressure data starts after the velocity data PLUS its 8-byte size header
    pressure_offset = len(velocity_bytes) + 8 

    # Prepare appended data block with size headers ('>Q' for big-endian unsigned long long)
    appended_data = b''
    appended_data += struct.pack('>Q', len(velocity_bytes)) + velocity_bytes
    appended_data += struct.pack('>Q', len(pressure_bytes)) + pressure_bytes
    
    # Base64 encode the appended data (without the leading underscore as VTKFile adds it)
    encoded_appended_data = base64.b64encode(appended_data).decode('ascii')

    # Construct the VTK XML string
    vtk_xml_content = f"""<?xml version="1.0"?>
<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" header_type="UInt64">
  <ImageData WholeExtent="0 {nx-1} 0 {ny-1} 0 {nz-1}"
             Origin="{origin_x} {origin_y} {origin_z}"
             Spacing="{spacing_x} {spacing_y} {spacing_z}">
    <Piece Extent="0 {nx-1} 0 {ny-1} 0 {nz-1}">
      <CellData>
        <DataArray type="Float32" Name="Velocity" format="appended" offset="{velocity_offset}" NumberOfComponents="3"/>
        <DataArray type="Float32" Name="Pressure" format="appended" offset="{pressure_offset}" NumberOfComponents="1"/>
      </CellData>
    </Piece>
  </ImageData>
  <AppendedData encoding="base64">
    _{encoded_appended_data}
  </AppendedData>
</VTKFile>"""

    # Write the XML content to the .vti file
    try:
        with open(output_filepath, 'w') as f:
            f.write(vtk_xml_content)
        print(f"VTK output successfully saved to '{output_filepath}'") # Commented for brevity in loop
    except IOError as e:
        print(f"Error writing VTK file '{output_filepath}': {e}", file=sys.stderr)
        raise

def save_checkpoint(
    filepath: str, 
    velocity_field: np.ndarray, 
    pressure_field: np.ndarray, 
    current_time: float
):
    """
    Saves the current simulation state (velocity, pressure, and time) to a .npz file.
    This allows for restarting simulations or detailed plotting using `plotter.py`.

    Args:
        filepath (str): Full path to the output .npz file.
        velocity_field (np.ndarray): Current velocity field.
        pressure_field (np.ndarray): Current pressure field.
        current_time (float): The current simulation time.
    """
    output_dir = os.path.dirname(filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        np.savez_compressed(
            filepath,
            velocity=velocity_field,
            pressure=pressure_field,
            current_time=current_time
        )
        print(f"Checkpoint successfully saved to '{filepath}'") # Commented for brevity in loop
    except Exception as e:
        print(f"Error saving checkpoint to '{filepath}': {e}", file=sys.stderr)
        raise