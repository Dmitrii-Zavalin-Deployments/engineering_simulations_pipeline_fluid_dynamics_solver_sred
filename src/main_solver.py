# src/main_solver.py

import json
import numpy as np
import time
import sys
import os

# --- Import Simulation Modules ---
# This block ensures all necessary components are available before execution.
try:
    # Utils
    from src.utils.grid import create_structured_grid_info
    from src.utils.io import load_json, write_output_to_vtk, save_checkpoint
    
    # Physics
    from src.physics.initialization import initialize_fields
    from src.physics.boundary_conditions import apply_boundary_conditions, identify_boundary_nodes
    
    # Numerical Methods (Solvers)
    from src.numerical_methods.explicit_solver import ExplicitSolver
    from src.numerical_methods.implicit_solver import ImplicitSolver
    
except ImportError as e:
    print(f"Error importing essential simulation modules: {e}", file=sys.stderr)
    print("Please ensure all modules and functions are correctly defined and that your PYTHONPATH is configured to include the project root.", file=sys.stderr)
    sys.exit(1)


class Simulation:
    """
    Orchestrates the entire fluid simulation lifecycle, from setup to execution and output.
    
    This class handles:
    1. Parsing input configuration.
    2. Initializing the structured grid and fluid fields.
    3. Processing boundary conditions.
    4. Managing the time-stepping loop.
    5. Saving results in VTK and checkpoint formats.
    """

    def __init__(self, config_filepath: str):
        """
        Initializes the simulation environment by loading configuration and setting up the grid.

        Args:
            config_filepath (str): Path to the main input JSON file.
        """
        print("--- Simulation Setup ---")
        self.config = self._load_and_validate_config(config_filepath)
        
        self.simulation_settings = self.config['simulation_parameters']
        self.fluid_properties = self.config['fluid_properties']
        self.domain_definition = self.config['domain_settings'] # From pre-processed input

        # 1. Create a structured grid information dictionary
        self.mesh_info = create_structured_grid_info(self.domain_definition)
        self.grid_shape = self.mesh_info['grid_shape']
        print(f"Grid Dimensions: {self.grid_shape[0]}x{self.grid_shape[1]}x{self.grid_shape[2]} cells")
        print(f"Grid Spacing (dx, dy, dz): ({self.mesh_info['dx']:.4e}, {self.mesh_info['dy']:.4e}, {self.mesh_info['dz']:.4e})")

        # 2. Identify and process boundary conditions
        all_mesh_boundary_faces = self.config.get('mesh', {}).get('boundary_faces', [])
        processed_bcs = identify_boundary_nodes(
            self.config['boundary_conditions'],
            all_mesh_boundary_faces,
            self.mesh_info
        )
        self.mesh_info['boundary_conditions'] = processed_bcs
        print(f"Processed {len(processed_bcs)} boundary condition groups.")
        
        # 3. Initialize velocity and pressure fields
        num_cells = self.mesh_info['num_cells']
        initial_velocity = self.config['initial_conditions']['velocity']
        initial_pressure = self.config['initial_conditions']['pressure']
        
        # Initialize fields as a flat array, then reshape to grid shape
        velocity_flat, pressure_flat = initialize_fields(num_cells, initial_velocity, initial_pressure)
        self.velocity_field = velocity_flat.reshape(self.grid_shape + (3,))
        self.pressure_field = pressure_flat.reshape(self.grid_shape)

        # 4. Apply initial boundary conditions to the fields
        apply_boundary_conditions(
            self.velocity_field, 
            self.pressure_field, 
            self.fluid_properties, 
            self.mesh_info,             
            is_tentative_step=False
        )
        print("Initialized fields and applied initial boundary conditions.")
        
        # 5. Instantiate the selected solver
        self.solver = self._instantiate_solver()
        
        self.current_time = 0.0
        self.step_count = 0
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory created at '{self.output_dir}'")
        
        print("--- Setup Complete ---")

    def _load_and_validate_config(self, filepath: str) -> dict:
        """Loads and validates the JSON configuration file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Input file not found: '{filepath}'")
        
        config = load_json(filepath)
        
        # Basic validation of required sections
        required_sections = ['domain_settings', 'fluid_properties', 'simulation_parameters', 'initial_conditions', 'boundary_conditions']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: '{section}' in input JSON.")
                
        return config

    def _instantiate_solver(self):
        """Instantiates the correct solver class based on configuration."""
        solver_type = self.simulation_settings['solver']
        
        if solver_type == "explicit":
            print(f"Solver: Explicit (Fractional Step)")
            return ExplicitSolver(
                self.fluid_properties,
                self.mesh_info,
                self.simulation_settings['time_step']
            )
        elif solver_type == "implicit":
            print(f"Solver: Implicit (Conceptual Placeholder)")
            return ImplicitSolver(
                self.fluid_properties,
                self.mesh_info,
                self.simulation_settings['time_step']
            )
        else:
            raise ValueError(f"Unsupported solver type specified in 'simulation_parameters': '{solver_type}'. Choose 'explicit' or 'implicit'.")

    def run(self):
        """
        Runs the main time-stepping loop of the simulation.
        """
        total_time = self.simulation_settings['total_time']
        time_step = self.simulation_settings['time_step']
        output_interval = self.simulation_settings.get('output_interval', 1)
        checkpoint_interval = self.simulation_settings.get('checkpoint_interval', 100) # New setting for checkpoints
        num_time_steps = int(total_time / time_step)

        print(f"\n--- Starting Fluid Simulation ---")
        print(f"Time Step: {time_step} s, Total Time: {total_time} s, Total Steps: {num_time_steps}")
        print(f"Outputting results every {output_interval} steps.")
        print(f"Saving checkpoints every {checkpoint_interval} steps.")
        print("---------------------------------\n")

        start_sim_time = time.time()
        
        # Pre-calculate grid lines for VTK output, as they are not stored in mesh_info
        min_x, max_x = self.mesh_info['min_x'], self.mesh_info['max_x']
        min_y, max_y = self.mesh_info['min_y'], self.mesh_info['max_y']
        min_z, max_z = self.mesh_info['min_z'], self.mesh_info['max_z']
        nx, ny, nz = self.mesh_info['nx'], self.mesh_info['ny'], self.mesh_info['nz']
        
        x_coords_grid_lines = np.linspace(min_x, max_x, nx + 1)
        y_coords_grid_lines = np.linspace(min_y, max_y, ny + 1)
        z_coords_grid_lines = np.linspace(min_z, max_z, nz + 1)

        # Main time loop
        while self.current_time < total_time:
            self.step_count += 1
            
            try:
                # The solver's step method returns the updated fields
                self.velocity_field, self.pressure_field = self.solver.step(
                    self.velocity_field, 
                    self.pressure_field
                )
            except Exception as e:
                print(f"An error occurred during time step {self.step_count}: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                sys.exit(1) # Exit immediately on solver error
            
            self.current_time += time_step

            # --- I/O Operations ---
            if self.step_count % output_interval == 0:
                print(f"Time: {self.current_time:.4f}s | Step: {self.step_count}/{num_time_steps}")
                
                # Write VTK output
                vtk_filename = f"navier_stokes_step_{self.step_count}.vti"
                vtk_filepath = os.path.join(self.output_dir, vtk_filename)
                write_output_to_vtk(
                    self.velocity_field,
                    self.pressure_field,
                    x_coords_grid_lines, y_coords_grid_lines, z_coords_grid_lines,
                    vtk_filepath
                )
            
            # Save checkpoint (e.g., for restarting or plotting)
            if self.step_count % checkpoint_interval == 0:
                checkpoint_filename = f"checkpoint_step_{self.step_count}.npz"
                checkpoint_filepath = os.path.join(self.output_dir, checkpoint_filename)
                # Note: save_checkpoint is a new function we will add to io.py
                save_checkpoint(
                    checkpoint_filepath,
                    self.velocity_field,
                    self.pressure_field,
                    self.current_time
                )


        end_sim_time = time.time()
        print(f"\n--- Simulation Complete ---")
        print(f"Total simulation time: {end_sim_time - start_sim_time:.2f} seconds.")
        print(f"---------------------------\n")

    def get_results_for_schema(self) -> dict:
        """
        Formats the final simulation results into a dictionary that conforms to a schema.
        """
        # Note: This is an example to match the original output format.
        # Storing the entire history would require more memory.
        return {
            "time_points": [self.current_time],
            "velocity_history": [self.velocity_field.reshape(-1, 3).tolist()],
            "pressure_history": [self.pressure_field.flatten().tolist()],
            "mesh_info": {
                "nodes": self.mesh_info['num_cells'],
                "nodes_coords": self.mesh_info['all_cell_centers_flat'].tolist(),
                "grid_shape": list(self.grid_shape),
                "dx": self.mesh_info['dx'],
                "dy": self.mesh_info['dy'],
                "dz": self.mesh_info['dz']
            }
        }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python src/main_solver.py <input_json_filepath> <output_results_json_filepath>")
        print("Example: python src/main_solver.py temp/solver_input.json results/navier_stokes_output.json")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        # Instantiate the simulation class and run it
        simulation_instance = Simulation(input_file)
        simulation_instance.run()

        # Save the final summary results to a JSON file
        final_results = simulation_instance.get_results_for_schema()
        output_dir_json = os.path.dirname(output_file)
        if output_dir_json and not os.path.exists(output_dir_json):
            os.makedirs(output_dir_json)

        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        print(f"Summary simulation results successfully saved to '{output_file}'")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in '{input_file}': {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Configuration or Logic Error: {e}", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f"File I/O Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during simulation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)