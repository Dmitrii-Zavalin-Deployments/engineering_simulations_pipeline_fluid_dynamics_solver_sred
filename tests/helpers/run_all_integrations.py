# tests/helpers/run_all_integrations.py

import subprocess
import json
import sys
from pathlib import Path

tests = [
    {
        "module": "config_validator",
        "path": "src/step1_input_data_parsing/config_validator.py",
        "expected": "tests/test_models/config_validator_output.json"
    },
    {
        "module": "input_reader",
        "path": "src/step1_input_data_parsing/input_reader.py",
        "expected": "tests/test_models/input_reader_output.json"
    },
    {
        "module": "fluid_mask_initializer",
        "path": "src/step2_creating_navier_stokes_equations/fluid_mask_initializer.py",
        "expected": "tests/test_models/fluid_mask_output.json"
    },
    {
        "module": "initial_field_assigner",
        "path": "src/step2_creating_navier_stokes_equations/initial_field_assigner.py",
        "expected": "tests/test_models/initial_field_output.json"
    }
    # Add more modules here
]

input_file = "tests/test_models/test_model_input.json"
runner_script = "tests/helpers/integration_tests_runner.py"
compare_script = "tests/helpers/compare_json.py"

def run_test(module_name, module_path, expected_path):
    temp_output = f"tests/test_models/{module_name}_cli_output.json"

    print(f"\n--- Running test: {module_name}.py ---")

    if not Path(input_file).is_file() or not Path(expected_path).is_file():
        print(f"❌ Fatal Error: Missing input or expected output for {module_name}")
        sys.exit(1)

    subprocess.run([
        "python3", runner_script,
        "--input", input_file,
        "--module", module_path,
        "--output", temp_output
    ], check=True)

    subprocess.run([
        "python3", compare_script,
        expected_path,
        temp_output
    ], check=True)

    print(f"✅ Integration test completed for {module_name}.py")

def main():
    for test in tests:
        run_test(test["module"], test["path"], test["expected"])

if __name__ == "__main__":
    main()
