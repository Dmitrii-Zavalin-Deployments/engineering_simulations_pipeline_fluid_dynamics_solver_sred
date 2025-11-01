# src/step1_input_validation/input_reader.py
# 📥 Input Reader — parses and validates structured Navier-Stokes simulation input
# 📌 This module anchors schema alignment for reflex scoring, mutation overlays, and diagnostic traceability.

import os
import json
import argparse

# ✅ Centralized debug flag for GitHub Actions logging
debug = False

def load_simulation_input(filepath: str) -> dict:
    if debug:
        print(f"📁 Loading input file: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Input file not found: {filepath}")

    with open(filepath, "r") as f:
        try:
            data = json.load(f)
            if debug:
                print(f"📦 JSON loaded successfully. Top-level keys: {list(data.keys())}")
        except json.JSONDecodeError as e:
            raise ValueError(f"❌ Failed to parse JSON: {e}")

    required_sections = [
        "domain_definition",
        "fluid_properties",
        "initial_conditions",
        "simulation_parameters",
        "boundary_conditions"
    ]
    for section in required_sections:
        if section not in data:
            raise KeyError(f"❌ Missing required section: '{section}'")
        if debug:
            print(f"✅ Section '{section}' found.")

    domain = data["domain_definition"]
    if debug:
        print(f"📂 Domain keys: {list(domain.keys())}")
    for key in ["nx", "ny", "nz", "x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]:
        if key not in domain:
            raise KeyError(f"❌ Missing domain key: '{key}'")
        if debug:
            print(f"✅ Domain key '{key}' = {domain[key]}")
    if debug:
        print(f"🧩 Domain resolution: {domain['nx']}×{domain['ny']}×{domain['nz']}")
        print(f"📐 Domain bounds: x={domain['x_min']}→{domain['x_max']}, y={domain['y_min']}→{domain['y_max']}, z={domain['z_min']}→{domain['z_max']}")

    fluid = data["fluid_properties"]
    if debug:
        print(f"📂 Fluid keys: {list(fluid.keys())}")
    for key in ["density", "viscosity"]:
        if key not in fluid:
            raise KeyError(f"❌ Missing fluid property: '{key}'")
        if debug:
            print(f"✅ Fluid key '{key}' = {fluid[key]}")
    if debug:
        print(f"🌊 Fluid density (ρ): {fluid['density']}")
        print(f"🌊 Fluid viscosity (μ): {fluid['viscosity']}")

    init = data["initial_conditions"]
    if debug:
        print(f"📂 Initial condition keys: {list(init.keys())}")
    for key in ["initial_velocity", "initial_pressure"]:
        if key not in init:
            raise KeyError(f"❌ Missing initial condition: '{key}'")
        if debug:
            print(f"✅ Initial key '{key}' = {init[key]}")
    if debug:
        print(f"🌀 Initial velocity: {init['initial_velocity']}")
        print(f"🌀 Initial pressure: {init['initial_pressure']}")

    sim = data["simulation_parameters"]
    if debug:
        print(f"📂 Simulation keys: {list(sim.keys())}")
    for key in ["time_step", "total_time", "output_interval"]:
        if key not in sim:
            raise KeyError(f"❌ Missing simulation parameter: '{key}'")
        if debug:
            print(f"✅ Simulation key '{key}' = {sim[key]}")
    if debug:
        print(f"⏱️ Time step (Δt): {sim['time_step']}")
        print(f"⏱️ Total time (T): {sim['total_time']}")
        print(f"⚙️ Output interval: {sim['output_interval']}")

    if "pressure_solver" in data:
        pressure_cfg = data["pressure_solver"]
        if debug:
            print(f"📂 Pressure solver keys: {list(pressure_cfg.keys())}")
        for key in ["method", "tolerance"]:
            if key not in pressure_cfg:
                raise KeyError(f"❌ Missing pressure solver key: '{key}'")
            if debug:
                print(f"✅ Pressure solver key '{key}' = {pressure_cfg[key]}")
        if debug:
            print(f"💧 Pressure Solver → Method: {pressure_cfg['method']}, Tolerance: {pressure_cfg['tolerance']}")

    bc_list = data["boundary_conditions"]
    if debug:
        print(f"📂 Boundary conditions count: {len(bc_list)}")
    if not isinstance(bc_list, list):
        raise TypeError("❌ 'boundary_conditions' must be a list.")
    for i, bc in enumerate(bc_list):
        if not isinstance(bc, dict):
            raise TypeError(f"❌ boundary_conditions[{i}] must be a dictionary.")
        if debug:
            print(f"📂 boundary_conditions[{i}] keys: {list(bc.keys())}")
        for key in ["role", "type", "apply_to", "apply_faces"]:
            if key not in bc:
                raise KeyError(f"❌ boundary_conditions[{i}] missing required key: '{key}'")
            if debug:
                print(f"✅ boundary_conditions[{i}] key '{key}' = {bc[key]}")
        if debug:
            print(f"🚧 Boundary Role: {bc['role']}")
            print(f"   Type: {bc['type']}")
            print(f"   Apply To: {bc['apply_to']}")
            print(f"   Apply Faces: {bc['apply_faces']}")
            print(f"   Velocity: {bc['velocity']}" if "velocity" in bc else "   Velocity: —")
            print(f"   Pressure: {bc['pressure']}" if "pressure" in bc else "   Pressure: —")
            print(f"   No-Slip: {bc['no_slip']}" if "no_slip" in bc else "   No-Slip: —")

    if "geometry_definition" in data:
        geometry = data["geometry_definition"]
        if debug:
            print(f"📂 Geometry keys: {list(geometry.keys())}")
        for key in ["geometry_mask_flat", "geometry_mask_shape", "mask_encoding", "flattening_order"]:
            if key not in geometry:
                raise KeyError(f"❌ Missing geometry key: '{key}'")
            if debug:
                print(f"✅ Geometry key '{key}' = {geometry[key]}")
        if debug:
            print(f"🧱 Geometry mask shape: {geometry['geometry_mask_shape']}")
            print(f"🧱 Mask encoding: fluid={geometry['mask_encoding']['fluid']}, solid={geometry['mask_encoding']['solid']}")
            print(f"🧱 Flattening order: {geometry['flattening_order']}")
            print(f"🧱 Mask preview: {geometry['geometry_mask_flat']}")
            print(f"🧮 Mask length: {len(geometry['geometry_mask_flat'])}")
            expected_len = geometry["geometry_mask_shape"][0] * geometry["geometry_mask_shape"][1] * geometry["geometry_mask_shape"][2]
            print(f"🧮 Expected mask length from shape: {expected_len}")
            if len(geometry["geometry_mask_flat"]) != expected_len:
                raise ValueError(f"❌ geometry_mask_flat length mismatch: expected {expected_len}, got {len(geometry['geometry_mask_flat'])}")

    return data

def main():
    parser = argparse.ArgumentParser(description="Input Reader CLI for Navier-Stokes simulation input")
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    args = parser.parse_args()

    result = load_simulation_input(args.input)

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()



