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
    },
    {
        "module": "cell_builder",
        "path": "src/step_1_solver_initialization/cell_builder.py",
        "expected": "tests/test_models/test_step_1_output.json",
        "input": "tests/test_models/test_step_0_output.json"
    }
]

compare_script = "tests/helpers/compare_json.py"

def run_test(module_name, module_path, expected_path, input_path):
    temp_output = f"tests/test_models/{module_name}_cli_output.json"

    print(f"\n--- Running test: {module_name}.py ---")
    print(f"📂 Input:    {input_path}")
    print(f"📂 Expected: {expected_path}")
    print(f"📂 TempOut:  {temp_output}")

    if not Path(input_path).is_file() or not Path(expected_path).is_file():
        print(f"❌ Fatal Error: Missing input or expected output for {module_name}")
        sys.exit(1)

    # Run module CLI
    cmd = ["python3", module_path, "--input", input_path, "--output", temp_output]
    print(f"▶️ Executing: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout.strip():
            print(f"--- {module_name} stdout ---\n{result.stdout}")
        if result.stderr.strip():
            print(f"--- {module_name} stderr ---\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Execution failed for {module_name}.py")
        print(f"Exit code: {e.returncode}")
        print(f"--- stdout ---\n{e.stdout}")
        print(f"--- stderr ---\n{e.stderr}")
        sys.exit(1)

    # Compare output
    cmd = ["python3", compare_script, expected_path, temp_output]
    print(f"▶️ Comparing: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout.strip():
            print(f"--- compare_json stdout ---\n{result.stdout}")
        if result.stderr.strip():
            print(f"--- compare_json stderr ---\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Comparison failed for {module_name}.py")
        print(f"Exit code: {e.returncode}")
        print(f"--- stdout ---\n{e.stdout}")
        print(f"--- stderr ---\n{e.stderr}")
        # Show preview of generated output file
        if Path(temp_output).is_file():
            print(f"📄 Generated output preview:")
            print(Path(temp_output).read_text()[:500])
        sys.exit(1)

    print(f"✅ Integration test completed for {module_name}.py")

def main():
    for test in tests:
        run_test(test["module"], test["path"], test["expected"], test["input"])

if __name__ == "__main__":
    main()



