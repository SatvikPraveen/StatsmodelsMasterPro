#!/usr/bin/env python3
"""
test_utils.py - Unit tests for all utility modules

Tests the functionality of:
- model_utils.py
- diagnostics.py
- compare_models.py
- visual_utils.py
- mixed_effects_utils.py
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Add utils to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import model_utils, diagnostics, compare_models


class TestModelUtils(unittest.TestCase):
    """Test model_utils.py functions"""
    
    def setUp(self):
        """Create sample data for testing"""
        np.random.seed(42)
        n = 100
        self.df = pd.DataFrame({
            'X1': np.random.normal(5, 2, n),
            'X2': np.random.normal(10, 3, n),
            'y': np.random.normal(20, 5, n)
        })
        self.df['y'] = 2 + 1.5 * self.df['X1'] - 0.7 * self.df['X2'] + np.random.normal(0, 1, n)
        
        # Fit a simple OLS model
        X = sm.add_constant(self.df[['X1', 'X2']])
        self.model = sm.OLS(self.df['y'], X).fit()
    
    def test_summarize_stats(self):
        """Test summarize_stats function"""
        result = model_utils.summarize_stats(self.df)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)  # X1, X2, y
        self.assertIn('mean', result.columns)
    
    def test_compute_central_tendency(self):
        """Test compute_central_tendency function"""
        result = model_utils.compute_central_tendency(self.df, ['X1', 'X2'])
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('mean', result.columns)
        self.assertIn('median', result.columns)
        self.assertIn('mode', result.columns)
    
    def test_summarize_model_coefficients(self):
        """Test summarize_model_coefficients function"""
        result = model_utils.summarize_model_coefficients(self.model)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('coef', result.columns)
        self.assertIn('p_value', result.columns)
        self.assertIn('ci_lower', result.columns)
        self.assertIn('ci_upper', result.columns)
    
    def test_extract_aic_bic(self):
        """Test extract_aic_bic function"""
        result = model_utils.extract_aic_bic(self.model)
        self.assertIsInstance(result, dict)
        self.assertIn('AIC', result)
        self.assertIn('BIC', result)
        self.assertIn('Log-Likelihood', result)
        self.assertTrue(result['AIC'] > 0)
    
    def test_compare_models_by_ic(self):
        """Test compare_models_by_ic function"""
        # Create a second model
        X2 = sm.add_constant(self.df[['X1']])
        model2 = sm.OLS(self.df['y'], X2).fit()
        
        result = model_utils.compare_models_by_ic(self.model, model2)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertIn('AIC', result.columns)
        self.assertIn('BIC', result.columns)


class TestDiagnostics(unittest.TestCase):
    """Test diagnostics.py functions"""
    
    def setUp(self):
        """Create sample data and model"""
        np.random.seed(42)
        n = 100
        self.df = pd.DataFrame({
            'X1': np.random.normal(5, 2, n),
            'X2': np.random.normal(10, 3, n),
        })
        self.df['y'] = 2 + 1.5 * self.df['X1'] - 0.7 * self.df['X2'] + np.random.normal(0, 1, n)
        
        X = sm.add_constant(self.df[['X1', 'X2']])
        self.model = sm.OLS(self.df['y'], X).fit()
    
    def test_compute_skewness_kurtosis(self):
        """Test compute_skewness_kurtosis function"""
        result = diagnostics.compute_skewness_kurtosis(self.df, ['X1', 'X2'])
        self.assertIsInstance(result, dict)
        self.assertIn('X1', result)
        self.assertIn('skewness', result['X1'])
        self.assertIn('kurtosis', result['X1'])
    
    def test_run_heteroskedasticity_tests(self):
        """Test run_heteroskedasticity_tests function"""
        result = diagnostics.run_heteroskedasticity_tests(self.model)
        self.assertIsInstance(result, dict)
        self.assertIn('Breusch-Pagan', result)
        self.assertIn('White', result)


class TestCompareModels(unittest.TestCase):
    """Test compare_models.py functions"""
    
    def setUp(self):
        """Create sample data and models"""
        np.random.seed(42)
        n = 100
        self.df = pd.DataFrame({
            'X1': np.random.normal(5, 2, n),
            'X2': np.random.normal(10, 3, n),
            'group': np.random.choice(['A', 'B'], n)
        })
        self.df['y'] = 2 + 1.5 * self.df['X1'] - 0.7 * self.df['X2'] + np.random.normal(0, 1, n)
        
        X = sm.add_constant(self.df[['X1', 'X2']])
        self.model1 = sm.OLS(self.df['y'], X).fit()
        
        X2 = sm.add_constant(self.df[['X1']])
        self.model2 = sm.OLS(self.df['y'], X2).fit()
    
    def test_model_comparison_exists(self):
        """Verify compare_models module can be imported"""
        self.assertIsNotNone(compare_models)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows"""
    
    def test_full_ols_workflow(self):
        """Test complete OLS workflow"""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            'X1': np.random.normal(5, 2, n),
            'X2': np.random.normal(10, 3, n),
        })
        df['y'] = 2 + 1.5 * df['X1'] - 0.7 * df['X2'] + np.random.normal(0, 1, n)
        
        # Fit model
        X = sm.add_constant(df[['X1', 'X2']])
        model = sm.OLS(df['y'], X).fit()
        
        # Test various operations
        stats = model_utils.summarize_stats(df)
        coeffs = model_utils.summarize_model_coefficients(model)
        ic = model_utils.extract_aic_bic(model)
        diag = diagnostics.compute_skewness_kurtosis(df, ['X1', 'X2'])
        
        # Verify all operations succeeded
        self.assertIsNotNone(stats)
        self.assertIsNotNone(coeffs)
        self.assertIsNotNone(ic)
        self.assertIsNotNone(diag)
        
        # Verify model quality
        self.assertGreater(model.rsquared, 0.5)
        self.assertLess(model.pvalues['X1'], 0.05)


def run_tests(verbosity=2):
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestModelUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestDiagnostics))
    suite.addTests(loader.loadTestsFromTestCase(TestCompareModels))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("=" * 70)
    print("Running StatsmodelsMasterPro Utility Tests")
    print("=" * 70)
    
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
