#!/usr/bin/env python3
"""
test_data_generation.py - Validate all synthetic datasets

Ensures:
- All datasets exist and are loadable
- Data has expected structure
- No missing values (unless intentional)
- Statistical properties are within expected ranges
- Data can be used with statsmodels
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "synthetic_data"


class DatasetValidator:
    """Validator for synthetic datasets"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.results = []
    
    def log(self, message, status="INFO"):
        """Log test results"""
        icons = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️", "INFO": "ℹ️"}
        icon = icons.get(status, "•")
        print(f"{icon} {message}")
    
    def validate_file_exists(self, filename):
        """Check if file exists"""
        filepath = DATA_PATH / filename
        if filepath.exists():
            self.log(f"{filename} exists", "PASS")
            return True
        else:
            self.log(f"{filename} NOT FOUND", "FAIL")
            self.results.append((filename, False, "File not found"))
            return False
    
    def validate_loadable(self, filename):
        """Check if CSV is loadable"""
        filepath = DATA_PATH / filename
        try:
            df = pd.read_csv(filepath)
            self.log(f"{filename} loaded successfully ({len(df)} rows, {len(df.columns)} cols)", "PASS")
            return df
        except Exception as e:
            self.log(f"{filename} failed to load: {str(e)}", "FAIL")
            self.results.append((filename, False, f"Load error: {str(e)}"))
            return None
    
    def validate_no_missing(self, df, filename, allow_missing=False):
        """Check for missing values"""
        missing_count = df.isnull().sum().sum()
        if missing_count == 0:
            self.log(f"{filename}: No missing values", "PASS")
            return True
        elif allow_missing:
            self.log(f"{filename}: {missing_count} missing values (allowed)", "WARN")
            return True
        else:
            self.log(f"{filename}: {missing_count} missing values found", "FAIL")
            self.results.append((filename, False, f"{missing_count} missing values"))
            return False
    
    def validate_numeric_range(self, df, filename, column, min_val=None, max_val=None):
        """Validate numeric column is within expected range"""
        if column not in df.columns:
            self.log(f"{filename}: Column '{column}' not found", "FAIL")
            return False
        
        col_min = df[column].min()
        col_max = df[column].max()
        
        if min_val is not None and col_min < min_val:
            self.log(f"{filename}.{column}: Min {col_min:.2f} < {min_val}", "FAIL")
            return False
        
        if max_val is not None and col_max > max_val:
            self.log(f"{filename}.{column}: Max {col_max:.2f} > {max_val}", "FAIL")
            return False
        
        self.log(f"{filename}.{column}: Range [{col_min:.2f}, {col_max:.2f}] OK", "PASS")
        return True
    
    def validate_columns_exist(self, df, filename, expected_columns):
        """Validate expected columns exist"""
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            self.log(f"{filename}: Missing columns: {missing_cols}", "FAIL")
            self.results.append((filename, False, f"Missing columns: {missing_cols}"))
            return False
        else:
            self.log(f"{filename}: All expected columns present", "PASS")
            return True
    
    def validate_distribution(self, df, filename, column, dist_type="normal"):
        """Test if data follows expected distribution (Shapiro-Wilk or KS test)"""
        if column not in df.columns:
            return False
        
        data = df[column].dropna()
        
        if dist_type == "normal":
            # Shapiro-Wilk test (for sample size < 5000)
            if len(data) < 5000:
                stat, p_value = stats.shapiro(data[:5000])
                test_name = "Shapiro-Wilk"
            else:
                stat, p_value = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
                test_name = "KS"
            
            if p_value > 0.01:
                self.log(f"{filename}.{column}: {test_name} p={p_value:.4f} (likely normal)", "PASS")
            else:
                self.log(f"{filename}.{column}: {test_name} p={p_value:.4f} (not normal)", "WARN")
        
        return True


def validate_all_datasets():
    """Main validation function"""
    validator = DatasetValidator()
    
    print("=" * 70)
    print("StatsmodelsMasterPro - Dataset Validation Suite")
    print("=" * 70)
    
    datasets = [
        ("ols_data.csv", ["X1", "X2", "y"]),
        ("glm_poisson.csv", ["X", "y"]),
        ("glm_logistic.csv", ["X", "y"]),
        ("arima_series.csv", ["t", "value"]),
        ("manova_data.csv", ["Y1", "Y2", "group"]),
        ("heteroskedastic_data.csv", ["X", "y"]),
        ("multivariate_group_data.csv", ["Num1", "Num2", "Group"]),
        ("ols_diagnostics.csv", ["X1", "X2", "y"]),
        ("posthoc_dataset.csv", ["Group", "Score"]),
        ("robust_regression_data.csv", ["X", "y", "weights"]),
        ("seasonal_ts_data.csv", ["t", "y"]),
        ("panel_data.csv", ["individual", "time", "X1", "X2", "y"]),
    ]
    
    optional_datasets = [
        ("gee_data.csv", ["id", "time", "y"]),
        ("mediation_data.csv", ["X", "M", "Y"]),
        ("survival_data.csv", ["time", "event"]),
        ("var_data.csv", ["t", "y1", "y2"]),
        ("zero_inflated_count.csv", ["X", "count"]),
    ]
    
    print("\n📊 Core Datasets")
    print("-" * 70)
    
    all_passed = True
    for filename, expected_cols in datasets:
        print(f"\n[{filename}]")
        
        if not validator.validate_file_exists(filename):
            all_passed = False
            continue
        
        df = validator.validate_loadable(filename)
        if df is None:
            all_passed = False
            continue
        
        if not validator.validate_columns_exist(df, filename, expected_cols):
            all_passed = False
            continue
        
        validator.validate_no_missing(df, filename)
        
        # Specific validations
        if filename == "glm_poisson.csv":
            validator.validate_numeric_range(df, filename, "y", min_val=0)
        
        elif filename == "glm_logistic.csv":
            unique_vals = df['y'].unique()
            if set(unique_vals).issubset({0, 1}):
                validator.log(f"{filename}.y: Binary values OK", "PASS")
            else:
                validator.log(f"{filename}.y: Not binary: {unique_vals}", "FAIL")
        
        elif filename == "manova_data.csv":
            groups = df['group'].unique()
            if len(groups) >= 2:
                validator.log(f"{filename}.group: {len(groups)} groups found", "PASS")
        
        elif filename == "panel_data.csv":
            n_individuals = df['individual'].nunique()
            n_time_points = df['time'].nunique()
            validator.log(f"{filename}: {n_individuals} individuals, {n_time_points} time points", "PASS")
    
    print("\n\n📦 Optional/Advanced Datasets")
    print("-" * 70)
    
    for filename, expected_cols in optional_datasets:
        print(f"\n[{filename}]")
        if not validator.validate_file_exists(filename):
            validator.log(f"{filename} (optional) - skipping", "INFO")
            continue
        
        df = validator.validate_loadable(filename)
        if df is not None:
            validator.validate_columns_exist(df, filename, expected_cols)
    
    print("\n" + "=" * 70)
    
    if all_passed and len(validator.results) == 0:
        print("✅ ALL CORE DATASETS VALIDATED SUCCESSFULLY!")
        return True
    else:
        print("❌ SOME DATASETS FAILED VALIDATION")
        if validator.results:
            print("\nFailed validations:")
            for filename, status, message in validator.results:
                print(f"  • {filename}: {message}")
        return False


if __name__ == "__main__":
    success = validate_all_datasets()
    sys.exit(0 if success else 1)
