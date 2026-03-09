#!/usr/bin/env python3
"""
validate_datasets.py - Comprehensive dataset validation

Checks:
- Statistical properties
- Assumption validity (normality, homoscedasticity)
- Outlier detection
- Multicollinearity
- Data quality issues
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, kstest, levene
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "synthetic_data"


class DatasetValidator:
    """Comprehensive dataset validation"""
    
    def __init__(self):
        self.results = {}
    
    def print_section(self, title):
        """Print formatted section header"""
        print(f"\n{'=' * 70}")
        print(f"  {title}")
        print('=' * 70)
    
    def validate_ols_assumptions(self):
        """Validate OLS regression assumptions"""
        self.print_section("OLS Regression - Assumption Validation")
        
        try:
            df = pd.read_csv(DATA_PATH / "ols_data.csv")
            
            # 1. Check for missing values
            missing = df.isnull().sum().sum()
            print(f"\n1️⃣  Missing Values: {missing}")
            if missing == 0:
                print("   ✅ No missing values")
            else:
                print(f"   ❌ Found {missing} missing values")
            
            # 2. Linearity - correlation check
            print(f"\n2️⃣  Linearity Check (Correlations):")
            corr_x1_y = df['X1'].corr(df['y'])
            corr_x2_y = df['X2'].corr(df['y'])
            print(f"   X1 ~ y: {corr_x1_y:.4f}")
            print(f"   X2 ~ y: {corr_x2_y:.4f}")
            
            # 3. Multicollinearity - VIF
            print(f"\n3️⃣  Multicollinearity (VIF):")
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            import statsmodels.api as sm
            
            X = df[['X1', 'X2']]
            X_with_const = sm.add_constant(X)
            
            vif_data = pd.DataFrame()
            vif_data["Variable"] = X.columns
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
            
            print(vif_data.to_string(index=False))
            
            if (vif_data["VIF"] < 10).all():
                print("   ✅ No severe multicollinearity (VIF < 10)")
            else:
                print("   ⚠️  High multicollinearity detected (VIF > 10)")
            
            # 4. Normality of residuals
            print(f"\n4️⃣  Normality of Residuals:")
            model = sm.OLS(df['y'], X_with_const).fit()
            
            shapiro_stat, shapiro_p = shapiro(model.resid)
            print(f"   Shapiro-Wilk test: W={shapiro_stat:.4f}, p={shapiro_p:.4f}")
            
            if shapiro_p > 0.05:
                print("   ✅ Residuals appear normal (p > 0.05)")
            else:
                print("   ⚠️  Residuals may not be normal (p < 0.05)")
            
            # 5. Homoscedasticity
            print(f"\n5️⃣  Homoscedasticity (Constant Variance):")
            from statsmodels.stats.diagnostic import het_breuschpagan
            
            bp_test = het_breuschpagan(model.resid, model.model.exog)
            print(f"   Breusch-Pagan test: LM={bp_test[0]:.4f}, p={bp_test[1]:.4f}")
            
            if bp_test[1] > 0.05:
                print("   ✅ No heteroscedasticity detected (p > 0.05)")
            else:
                print("   ⚠️  Heteroscedasticity present (p < 0.05)")
            
            # 6. Outliers
            print(f"\n6️⃣  Outlier Detection:")
            from statsmodels.stats.outliers_influence import OLSInfluence
            
            influence = OLSInfluence(model)
            cooks_d = influence.cooks_distance[0]
            
            threshold = 4 / len(df)
            outliers = np.sum(cooks_d > threshold)
            print(f"   Cook's Distance threshold: {threshold:.4f}")
            print(f"   Observations exceeding threshold: {outliers}")
            
            if outliers == 0:
                print("   ✅ No influential outliers detected")
            else:
                print(f"   ⚠️  {outliers} potential outliers found")
            
            # 7. Independence (Durbin-Watson)
            print(f"\n7️⃣  Independence of Errors:")
            from statsmodels.stats.stattools import durbin_watson
            
            dw_stat = durbin_watson(model.resid)
            print(f"   Durbin-Watson statistic: {dw_stat:.4f}")
            
            if 1.5 < dw_stat < 2.5:
                print("   ✅ No autocorrelation detected (1.5 < DW < 2.5)")
            else:
                print("   ⚠️  Possible autocorrelation")
            
            self.results['OLS'] = 'VALIDATED'
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            self.results['OLS'] = f'FAILED: {str(e)}'
    
    def validate_glm_assumptions(self):
        """Validate GLM assumptions"""
        self.print_section("GLM - Assumption Validation")
        
        # Poisson GLM
        try:
            print("\n📊 Poisson GLM Validation:")
            df = pd.read_csv(DATA_PATH / "glm_poisson.csv")
            
            # Check: y should be non-negative integers
            print(f"\n1️⃣  Data Type Check:")
            is_count = (df['y'] >= 0).all() and (df['y'] % 1 == 0).all()
            print(f"   Non-negative integers: {'✅ Yes' if is_count else '❌ No'}")
            
            # Check: Mean-variance relationship
            print(f"\n2️⃣  Mean-Variance Relationship:")
            mean_y = df['y'].mean()
            var_y = df['y'].var()
            print(f"   Mean: {mean_y:.2f}")
            print(f"   Variance: {var_y:.2f}")
            print(f"   Ratio (Var/Mean): {var_y/mean_y:.2f}")
            
            if 0.8 < var_y/mean_y < 1.2:
                print("   ✅ Variance ≈ Mean (good for Poisson)")
            elif var_y/mean_y > 1.2:
                print("   ⚠️  Overdispersion (Var > Mean) - consider Negative Binomial")
            else:
                print("   ⚠️  Underdispersion (Var < Mean)")
            
            self.results['GLM_Poisson'] = 'VALIDATED'
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            self.results['GLM_Poisson'] = f'FAILED: {str(e)}'
        
        # Logistic GLM
        try:
            print("\n\n📊 Logistic GLM Validation:")
            df = pd.read_csv(DATA_PATH / "glm_logistic.csv")
            
            print(f"\n1️⃣  Binary Outcome Check:")
            unique_vals = sorted(df['y'].unique())
            is_binary = set(unique_vals).issubset({0, 1})
            print(f"   Unique values: {unique_vals}")
            print(f"   Binary (0/1): {'✅ Yes' if is_binary else '❌ No'}")
            
            print(f"\n2️⃣  Class Balance:")
            value_counts = df['y'].value_counts()
            print(value_counts)
            
            balance_ratio = value_counts.min() / value_counts.max()
            if balance_ratio > 0.3:
                print(f"   ✅ Reasonably balanced (ratio: {balance_ratio:.2f})")
            else:
                print(f"   ⚠️  Imbalanced classes (ratio: {balance_ratio:.2f})")
            
            self.results['GLM_Logistic'] = 'VALIDATED'
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            self.results['GLM_Logistic'] = f'FAILED: {str(e)}'
    
    def validate_time_series(self):
        """Validate time series properties"""
        self.print_section("Time Series - Validation")
        
        try:
            df = pd.read_csv(DATA_PATH / "arima_series.csv")
            series = df['value'].values
            
            print(f"\n1️⃣  Basic Properties:")
            print(f"   Length: {len(series)}")
            print(f"   Mean: {series.mean():.4f}")
            print(f"   Std Dev: {series.std():.4f}")
            
            # Stationarity test (ADF)
            print(f"\n2️⃣  Stationarity Test (ADF):")
            from statsmodels.tsa.stattools import adfuller
            
            adf_result = adfuller(series)
            print(f"   ADF Statistic: {adf_result[0]:.4f}")
            print(f"   p-value: {adf_result[1]:.4f}")
            print(f"   Critical Values:")
            for key, value in adf_result[4].items():
                print(f"      {key}: {value:.4f}")
            
            if adf_result[1] < 0.05:
                print("   ✅ Series is stationary (p < 0.05)")
            else:
                print("   ⚠️  Series may be non-stationary (p > 0.05)")
            
            # Autocorrelation
            print(f"\n3️⃣  Autocorrelation:")
            from statsmodels.stats.diagnostic import acorr_ljungbox
            
            lb_test = acorr_ljungbox(series, lags=[10], return_df=True)
            print(f"   Ljung-Box test (lag 10): p={lb_test['lb_pvalue'].values[0]:.4f}")
            
            if lb_test['lb_pvalue'].values[0] < 0.05:
                print("   ✅ Significant autocorrelation present")
            else:
                print("   ⚠️  No significant autocorrelation")
            
            self.results['TimeSeries'] = 'VALIDATED'
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            self.results['TimeSeries'] = f'FAILED: {str(e)}'
    
    def validate_panel_data(self):
        """Validate panel data structure"""
        self.print_section("Panel Data - Validation")
        
        try:
            if (DATA_PATH / "panel_data.csv").exists():
                df = pd.read_csv(DATA_PATH / "panel_data.csv")
                
                print(f"\n1️⃣  Panel Structure:")
                n_individuals = df['individual'].nunique()
                n_time_points = df['time'].nunique()
                expected_rows = n_individuals * n_time_points
                
                print(f"   Individuals: {n_individuals}")
                print(f"   Time points: {n_time_points}")
                print(f"   Expected rows: {expected_rows}")
                print(f"   Actual rows: {len(df)}")
                
                if len(df) == expected_rows:
                    print("   ✅ Balanced panel")
                else:
                    print("   ⚠️  Unbalanced panel")
                
                print(f"\n2️⃣  Missing Values:")
                missing_by_col = df.isnull().sum()
                print(missing_by_col)
                
                if missing_by_col.sum() == 0:
                    print("   ✅ No missing values")
                
                self.results['PanelData'] = 'VALIDATED'
            else:
                print("⚠️  panel_data.csv not found")
                self.results['PanelData'] = 'SKIPPED'
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            self.results['PanelData'] = f'FAILED: {str(e)}'
    
    def run_all_validations(self):
        """Run all validations"""
        print("=" * 70)
        print("StatsmodelsMasterPro - Dataset Validation Suite")
        print("=" * 70)
        
        self.validate_ols_assumptions()
        self.validate_glm_assumptions()
        self.validate_time_series()
        self.validate_panel_data()
        
        # Summary
        print("\n" + "=" * 70)
        print("Validation Summary")
        print("=" * 70)
        
        for dataset, status in self.results.items():
            icon = "✅" if "VALIDATED" in status else ("⏭️" if "SKIPPED" in status else "❌")
            print(f"{icon} {dataset}: {status}")
        
        print("=" * 70)


if __name__ == "__main__":
    validator = DatasetValidator()
    validator.run_all_validations()
