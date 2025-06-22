import numpy as np
import matplotlib.pyplot as plt
import os

def plot_2d_slice(field, title, dim='z', slice_idx=0, component=None, vmin=None, vmax=None, cmap='viridis'):
    """
    Plots a 2D slice of a 3D field using matplotlib.

    Args:
        field (np.ndarray): The 3D NumPy array (e.g., pressure or a velocity component).
                            Can be (nx, ny, nz) for scalar, or (nx, ny, nz, 3) for vector.
        title (str): Title for the plot.
        dim (str): The dimension to slice along ('x', 'y', or 'z').
        slice_idx (int): The index of the slice along the specified dimension.
        component (int, optional): For vector fields (e.g., velocity), specifies which component
                                   to plot (0 for Vx, 1 for Vy, 2 for Vz). None for scalar fields.
        vmin (float, optional): Minimum value for the colorbar.
        vmax (float, optional): Maximum value for the colorbar.
        cmap (str): Colormap to use (e.g., 'viridis', 'jet', 'coolwarm').
    """
    if field.ndim == 4 and component is not None: # It's a vector field, plot a component
        data_to_plot = field[..., component]
        title = f"{title} ({['Vx', 'Vy', 'Vz'][component]} Component)"
    elif field.ndim == 3 and component is None: # It's a scalar field
        data_to_plot = field
    else:
        raise ValueError("Field dimension and component mismatch. "
                         "Provide component for 4D field, or None for 3D field.")

    nx, ny, nz = data_to_plot.shape

    if dim == 'x':
        if not (0 <= slice_idx < nx):
            print(f"Warning: slice_idx {slice_idx} out of bounds for x-dimension (0 to {nx-1}). Setting to 0.")
            slice_idx = 0
        slice_data = data_to_plot[slice_idx, :, :]
        xlabel, ylabel = 'Y', 'Z'
    elif dim == 'y':
        if not (0 <= slice_idx < ny):
            print(f"Warning: slice_idx {slice_idx} out of bounds for y-dimension (0 to {ny-1}). Setting to 0.")
            slice_idx = 0
        slice_data = data_to_plot[:, slice_idx, :]
        xlabel, ylabel = 'X', 'Z'
    elif dim == 'z':
        if not (0 <= slice_idx < nz):
            print(f"Warning: slice_idx {slice_idx} out of bounds for z-dimension (0 to {nz-1}). Setting to 0.")
            slice_idx = 0
        slice_data = data_to_plot[:, :, slice_idx]
        xlabel, ylabel = 'X', 'Y'
    else:
        raise ValueError("Invalid dimension. Must be 'x', 'y', or 'z'.")

    plt.figure(figsize=(8, 6))
    plt.imshow(slice_data.T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto') # Transpose for correct orientation
    plt.colorbar(label=f"{title} Value")
    plt.title(f"{title}\nSlice at {dim}={slice_idx}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

def load_and_plot_checkpoint(filepath, dim='z', slice_idx=0):
    """
    Loads a .npz checkpoint file and plots slices of velocity magnitude,
    and individual velocity components (Vx, Vy, Vz), and pressure.

    Args:
        filepath (str): Path to the .npz checkpoint file.
        dim (str): Dimension to slice along ('x', 'y', 'z').
        slice_idx (int): Index of the slice.
    """
    if not os.path.exists(filepath):
        print(f"Error: Checkpoint file not found at {filepath}")
        return

    print(f"Loading data from {filepath}")
    data = np.load(filepath)
    velocity = data['velocity']
    pressure = data['pressure']
    current_time = data['current_time'] if 'current_time' in data else 'N/A'

    print(f"Data loaded: Velocity shape {velocity.shape}, Pressure shape {pressure.shape}, Time: {current_time:.2f}s")

    # Plot Velocity Magnitude
    velocity_magnitude = np.linalg.norm(velocity, axis=-1)
    plot_2d_slice(velocity_magnitude, f"Velocity Magnitude (Time: {current_time:.2f}s)",
                  dim=dim, slice_idx=slice_idx, cmap='jet')

    # Plot individual velocity components
    plot_2d_slice(velocity, f"X-Velocity (Time: {current_time:.2f}s)",
                  dim=dim, slice_idx=slice_idx, component=0, cmap='coolwarm')
    plot_2d_slice(velocity, f"Y-Velocity (Time: {current_time:.2f}s)",
                  dim=dim, slice_idx=slice_idx, component=1, cmap='coolwarm')
    plot_2d_slice(velocity, f"Z-Velocity (Time: {current_time:.2f}s)",
                  dim=dim, slice_idx=slice_idx, component=2, cmap='coolwarm')

    # Plot Pressure
    plot_2d_slice(pressure, f"Pressure (Time: {current_time:.2f}s)",
                  dim=dim, slice_idx=slice_idx, cmap='viridis')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize simulation checkpoint data.")
    parser.add_argument("filepath", help="Path to the .npz checkpoint file.")
    parser.add_argument("--dim", type=str, default='z', choices=['x', 'y', 'z'],
                        help="Dimension to slice along (x, y, or z). Default: z")
    parser.add_argument("--slice-idx", type=int, default=0,
                        help="Index of the slice along the chosen dimension. Default: 0")
    args = parser.parse_args()

    load_and_plot_checkpoint(args.filepath, args.dim, args.slice_idx)