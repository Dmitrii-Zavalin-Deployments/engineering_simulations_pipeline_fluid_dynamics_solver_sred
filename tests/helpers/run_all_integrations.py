# tests/helpers/run_all_integrations.py

import subprocess
import sys
from pathlib import Path

tests = [
    {
        "module": "config_validator",
        "path": "src/step_0_input_data_parsing/config_validator.py",
        "expected": "tests/test_models/config_validator_output.json",
        "input": "tests/test_models/test_model_input.json"
    },
    {
        "module": "input_reader",
        "path": "src/step_0_input_data_parsing/input_reader.py",
        "expected": "tests/test_models/input_reader_output.json",
        "input": "tests/test_models/test_model_input.json"
    }#,
    # {
    #     "module": "fluid_mask_initializer",
    #     "path": "src/step2_creating_navier_stokes_equations/fluid_mask_initializer.py",
    #     "expected": "tests/test_models/fluid_mask_initializer_output.json",
    #     "input": "tests/test_models/test_step1_output.json"
    # }
]

compare_script = "tests/helpers/compare_json.py"

def run_test(module_name, module_path, expected_path, input_path):
    temp_output = f"tests/test_models/{module_name}_cli_output.json"

    print(f"\n--- Running test: {module_name}.py ---")

    if not Path(input_path).is_file() or not Path(expected_path).is_file():
        print(f"❌ Fatal Error: Missing input or expected output for {module_name}")
        sys.exit(1)

    try:
        subprocess.run([
            "python3", module_path,
            "--input", input_path,
            "--output", temp_output
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Execution failed for {module_name}.py: {e}")
        sys.exit(1)

    try:
        subprocess.run([
            "python3", compare_script,
            expected_path,
            temp_output
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Comparison failed for {module_name}.py: {e}")
        sys.exit(1)

    print(f"✅ Integration test completed for {module_name}.py")

def main():
    for test in tests:
        run_test(test["module"], test["path"], test["expected"], test["input"])

if __name__ == "__main__":
    main()
