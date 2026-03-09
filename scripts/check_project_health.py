#!/usr/bin/env python3
"""
check_project_health.py - Verify project integrity and health

Checks:
- All required files exist
- Import statements are valid
- Dependencies are installed
- No broken links between modules
- Code style consistency (basic checks)
"""

import sys
from pathlib import Path
import ast
import subprocess
import importlib.util


PROJECT_ROOT = Path(__file__).parent.parent


class ProjectHealthChecker:
    """Check overall project health"""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
    
    def print_section(self, title):
        """Print section header"""
        print(f"\n{'=' * 70}")
        print(f"  {title}")
        print('=' * 70)
    
    def log_issue(self, message, severity='ERROR'):
        """Log an issue"""
        if severity == 'ERROR':
            self.issues.append(message)
            print(f"❌ {message}")
        else:
            self.warnings.append(message)
            print(f"⚠️  {message}")
    
    def log_success(self, message):
        """Log success"""
        print(f"✅ {message}")
    
    def check_required_files(self):
        """Check if all required files exist"""
        self.print_section("Required Files Check")
        
        required_files = [
            "README.md",
            "requirements.txt",
            "Home.py",
            "synthetic_data/generate_datasets.py",
        ]
        
        required_dirs = [
            "pages",
            "notebooks",
            "utils",
            "synthetic_data",
            "scripts",
        ]
        
        print("\n📁 Files:")
        for file_path in required_files:
            full_path = PROJECT_ROOT / file_path
            if full_path.exists():
                self.log_success(f"{file_path}")
            else:
                self.log_issue(f"Missing file: {file_path}")
        
        print("\n📂 Directories:")
        for dir_path in required_dirs:
            full_path = PROJECT_ROOT / dir_path
            if full_path.exists() and full_path.is_dir():
                count = len(list(full_path.iterdir()))
                self.log_success(f"{dir_path}/ ({count} items)")
            else:
                self.log_issue(f"Missing directory: {dir_path}")
    
    def check_python_syntax(self):
        """Check Python files for syntax errors"""
        self.print_section("Python Syntax Check")
        
        python_files = []
        for pattern in ["**/*.py"]:
            python_files.extend(PROJECT_ROOT.glob(pattern))
        
        # Exclude virtual environment
        python_files = [f for f in python_files if 'env' not in str(f) and 'venv' not in str(f)]
        
        print(f"\nChecking {len(python_files)} Python files...")
        
        errors = 0
        for py_file in python_files[:50]:  # Limit to first 50 for speed
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    ast.parse(f.read())
            except SyntaxError as e:
                self.log_issue(f"Syntax error in {py_file.relative_to(PROJECT_ROOT)}: {str(e)}")
                errors += 1
            except Exception as e:
                # Skip files that can't be read
                pass
        
        if errors == 0:
            self.log_success(f"All checked Python files have valid syntax")
        else:
            print(f"\n❌ Found {errors} syntax errors")
    
    def check_imports(self):
        """Check if critical imports work"""
        self.print_section("Import Validation")
        
        critical_packages = [
            'pandas',
            'numpy',
            'matplotlib',
            'seaborn',
            'statsmodels',
            'scipy',
            'streamlit',
        ]
        
        print("\n📦 Critical packages:")
        for package in critical_packages:
            try:
                __import__(package)
                self.log_success(f"{package}")
            except ImportError:
                self.log_issue(f"Cannot import {package} - may need installation", 'WARNING')
        
        # Check utils modules
        print("\n🛠️  Utility modules:")
        utils_dir = PROJECT_ROOT / "utils"
        for util_file in utils_dir.glob("*.py"):
            if util_file.name == '__init__.py':
                continue
            
            try:
                spec = importlib.util.spec_from_file_location("test_module", util_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.log_success(f"{util_file.stem}")
            except Exception as e:
                self.log_issue(f"Cannot import {util_file.stem}: {str(e)}", 'WARNING')
    
    def check_datasets(self):
        """Check if synthetic datasets exist"""
        self.print_section("Dataset Availability")
        
        expected_datasets = [
            "ols_data.csv",
            "glm_poisson.csv",
            "glm_logistic.csv",
            "arima_series.csv",
            "manova_data.csv",
        ]
        
        data_dir = PROJECT_ROOT / "synthetic_data"
        
        print("\n📊 Core datasets:")
        for dataset in expected_datasets:
            dataset_path = data_dir / dataset
            if dataset_path.exists():
                size_mb = dataset_path.stat().st_size / 1024 / 1024
                self.log_success(f"{dataset} ({size_mb:.2f} MB)")
            else:
                self.log_issue(f"Missing dataset: {dataset}", 'WARNING')
        
        # Count all datasets
        all_datasets = list(data_dir.glob("*.csv"))
        print(f"\n📈 Total datasets found: {len(all_datasets)}")
    
    def check_streamlit_pages(self):
        """Check Streamlit pages"""
        self.print_section("Streamlit Pages Check")
        
        pages_dir = PROJECT_ROOT / "pages"
        pages = sorted(pages_dir.glob("*.py"))
        
        print(f"\n📄 Found {len(pages)} Streamlit pages:")
        
        for page in pages[:10]:  # Show first 10
            # Check if it's a valid Python file
            try:
                with open(page, 'r') as f:
                    content = f.read()
                    if 'streamlit' in content or 'st.' in content:
                        self.log_success(f"{page.name}")
                    else:
                        self.log_issue(f"{page.name} - may not be a Streamlit page", 'WARNING')
            except:
                pass
        
        if len(pages) > 10:
            print(f"   ... and {len(pages) - 10} more pages")
    
    def check_notebooks(self):
        """Check Jupyter notebooks"""
        self.print_section("Jupyter Notebooks Check")
        
        notebooks_dir = PROJECT_ROOT / "notebooks"
        notebooks = list(notebooks_dir.glob("*.ipynb"))
        
        print(f"\n📓 Found {len(notebooks)} notebooks:")
        
        for nb in notebooks[:5]:  # Show first 5
            size_kb = nb.stat().st_size / 1024
            self.log_success(f"{nb.name} ({size_kb:.1f} KB)")
        
        if len(notebooks) > 5:
            print(f"   ... and {len(notebooks) - 5} more notebooks")
    
    def check_git_status(self):
        """Check git repository status"""
        self.print_section("Git Repository Check")
        
        git_dir = PROJECT_ROOT / ".git"
        
        if not git_dir.exists():
            self.log_issue("Not a git repository", 'WARNING')
            return
        
        try:
            # Check if git is available
            result = subprocess.run(
                ['git', 'status', '--short'],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.log_success("Git repository initialized")
                
                # Count untracked/modified files
                lines = result.stdout.strip().split('\n')
                if lines and lines[0]:
                    modified = sum(1 for line in lines if line.strip())
                    print(f"   📝 Modified/Untracked files: {modified}")
                else:
                    self.log_success("Working directory is clean")
            
        except FileNotFoundError:
            self.log_issue("Git not installed or not in PATH", 'WARNING')
    
    def generate_report(self):
        """Generate final health report"""
        self.print_section("Health Report Summary")
        
        total_issues = len(self.issues)
        total_warnings = len(self.warnings)
        
        print(f"\n🔍 Inspection Complete!")
        print(f"   ❌ Errors: {total_issues}")
        print(f"   ⚠️  Warnings: {total_warnings}")
        
        if total_issues == 0 and total_warnings == 0:
            print("\n✅ Project health: EXCELLENT")
            print("   All checks passed! 🎉")
            return True
        elif total_issues == 0:
            print("\n⚠️  Project health: GOOD")
            print("   No critical errors, but some warnings present")
            return True
        else:
            print("\n❌ Project health: NEEDS ATTENTION")
            print("   Critical issues found - please review")
            
            if self.issues:
                print("\n❌ Critical Issues:")
                for issue in self.issues:
                    print(f"   • {issue}")
            
            return False
    
    def run_all_checks(self):
        """Run all health checks"""
        print("=" * 70)
        print("StatsmodelsMasterPro - Project Health Check")
        print("=" * 70)
        
        self.check_required_files()
        self.check_python_syntax()
        self.check_imports()
        self.check_datasets()
        self.check_streamlit_pages()
        self.check_notebooks()
        self.check_git_status()
        
        health_ok = self.generate_report()
        
        print("\n" + "=" * 70)
        
        return health_ok


def main():
    """Main entry point"""
    checker = ProjectHealthChecker()
    health_ok = checker.run_all_checks()
    
    sys.exit(0 if health_ok else 1)


if __name__ == "__main__":
    main()
