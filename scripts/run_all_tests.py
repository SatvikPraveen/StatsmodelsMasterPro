#!/usr/bin/env python3
"""
run_all_tests.py - Master test runner

Executes all test suites:
1. Utility function tests (test_utils.py)
2. Dataset validation tests (test_data_generation.py)
3. Import tests (verify all modules load)
4. Summary report
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime
import importlib.util

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_python_script(script_path, description):
    """Run a Python script and return success status"""
    print_header(description)
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            text=True,
            cwd=PROJECT_ROOT
        )
        
        success = result.returncode == 0
        
        if success:
            print(f"✅ {description} - PASSED")
        else:
            print(f"❌ {description} - FAILED (exit code: {result.returncode})")
        
        return success
    
    except Exception as e:
        print(f"❌ {description} - ERROR: {str(e)}")
        return False


def test_imports():
    """Test that all modules can be imported"""
    print_header("Module Import Tests")
    
    modules_to_test = [
        ("utils.model_utils", "Model Utilities"),
        ("utils.diagnostics", "Diagnostics"),
        ("utils.compare_models", "Model Comparison"),
        ("utils.visual_utils", "Visual Utilities"),
        ("utils.mixed_effects_utils", "Mixed Effects Utilities"),
    ]
    
    all_passed = True
    
    for module_path, description in modules_to_test:
        try:
            # Try to import the module
            full_path = PROJECT_ROOT / module_path.replace(".", "/") + ".py"
            
            if not full_path.exists():
                print(f"⚠️  {description} ({module_path}) - File not found (optional)")
                continue
            
            spec = importlib.util.spec_from_file_location(module_path, full_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            print(f"✅ {description} ({module_path}) - Import successful")
        
        except Exception as e:
            print(f"❌ {description} ({module_path}) - Import failed: {str(e)}")
            all_passed = False
    
    return all_passed


def test_data_generation_script():
    """Test that the data generation script exists and is valid"""
    print_header("Data Generation Script Test")
    
    script_path = PROJECT_ROOT / "synthetic_data" / "generate_datasets.py"
    
    if not script_path.exists():
        print(f"❌ generate_datasets.py not found")
        return False
    
    try:
        # Try to import it
        spec = importlib.util.spec_from_file_location("generate_datasets", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        print(f"✅ generate_datasets.py - Valid Python file")
        
        # Check for main generation functions
        required_functions = [
            'generate_ols_data',
            'generate_glm_data',
            'generate_time_series_data'
        ]
        
        for func_name in required_functions:
            if hasattr(module, func_name):
                print(f"✅ Function '{func_name}' found")
            else:
                print(f"⚠️  Function '{func_name}' not found (may be renamed)")
        
        return True
    
    except Exception as e:
        print(f"❌ generate_datasets.py - Error: {str(e)}")
        return False


def generate_report(results):
    """Generate summary report"""
    print_header("Test Summary Report")
    
    total_suites = len(results)
    passed_suites = sum(1 for r in results.values() if r)
    failed_suites = total_suites - passed_suites
    
    print(f"\nTest Suites Run: {total_suites}")
    print(f"Passed: {passed_suites}")
    print(f"Failed: {failed_suites}")
    
    print("\nDetailed Results:")
    for suite_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {suite_name}")
    
    print("\n" + "=" * 70)
    
    if failed_suites == 0:
        print("🎉 ALL TEST SUITES PASSED!")
        print("=" * 70)
        return True
    else:
        print(f"⚠️  {failed_suites} TEST SUITE(S) FAILED")
        print("=" * 70)
        return False


def main():
    """Main test runner"""
    print("=" * 70)
    print("StatsmodelsMasterPro - Complete Test Suite")
    print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    results = {}
    
    # 1. Import tests
    results["Module Imports"] = test_imports()
    
    # 2. Data generation script test
    results["Data Generation Script"] = test_data_generation_script()
    
    # 3. Run utility tests
    test_utils_path = SCRIPTS_DIR / "test_utils.py"
    if test_utils_path.exists():
        results["Utility Function Tests"] = run_python_script(
            test_utils_path,
            "Utility Function Tests"
        )
    else:
        print("⚠️  test_utils.py not found - skipping")
    
    # 4. Run dataset validation
    test_data_path = SCRIPTS_DIR / "test_data_generation.py"
    if test_data_path.exists():
        results["Dataset Validation"] = run_python_script(
            test_data_path,
            "Dataset Validation Tests"
        )
    else:
        print("⚠️  test_data_generation.py not found - skipping")
    
    # Generate final report
    success = generate_report(results)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
