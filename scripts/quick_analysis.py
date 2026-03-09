#!/usr/bin/env python3
"""
quick_analysis.py - Run quick statistical analysis from command line

Usage:
    python quick_analysis.py                    # Interactive mode
    python quick_analysis.py ols                # Run OLS on default data
    python quick_analysis.py glm poisson        # Run Poisson GLM
    python quick_analysis.py arima              # Run ARIMA
    python quick_analysis.py --data mydata.csv  # Use custom data
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "synthetic_data"


class QuickAnalysis:
    """Quick statistical analysis runner"""
    
    def __init__(self):
        self.data = None
        self.model = None
    
    def load_data(self, filename):
        """Load data from file"""
        if Path(filename).exists():
            data_path = Path(filename)
        else:
            data_path = DATA_PATH / filename
        
        if not data_path.exists():
            print(f"❌ Data file not found: {filename}")
            return False
        
        try:
            self.data = pd.read_csv(data_path)
            print(f"✅ Loaded data: {data_path}")
            print(f"   Shape: {self.data.shape}")
            print(f"   Columns: {list(self.data.columns)}")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False
    
    def run_ols(self, y_col='y', x_cols=None):
        """Run OLS regression"""
        print("\n" + "=" * 70)
        print("📊 OLS Regression Analysis")
        print("=" * 70)
        
        if self.data is None:
            if not self.load_data('ols_data.csv'):
                return
        
        # Auto-detect columns if not specified
        if x_cols is None:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if y_col in numeric_cols:
                numeric_cols.remove(y_col)
            x_cols = numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols
        
        print(f"\n📋 Model Specification:")
        print(f"   Response: {y_col}")
        print(f"   Predictors: {x_cols}")
        
        try:
            X = sm.add_constant(self.data[x_cols])
            self.model = sm.OLS(self.data[y_col], X).fit()
            
            # Display results
            print(f"\n📈 Model Summary:")
            print(f"   R-squared: {self.model.rsquared:.4f}")
            print(f"   Adj. R-squared: {self.model.rsquared_adj:.4f}")
            print(f"   AIC: {self.model.aic:.2f}")
            print(f"   BIC: {self.model.bic:.2f}")
            print(f"   F-statistic: {self.model.fvalue:.2f} (p={self.model.f_pvalue:.4f})")
            
            print(f"\n📊 Coefficients:")
            coef_df = pd.DataFrame({
                'Coefficient': self.model.params,
                'Std Error': self.model.bse,
                't-value': self.model.tvalues,
                'p-value': self.model.pvalues,
                'CI Lower': self.model.conf_int()[0],
                'CI Upper': self.model.conf_int()[1]
            })
            print(coef_df.to_string())
            
            print(f"\n✅ OLS analysis complete!")
            
        except Exception as e:
            print(f"❌ Error running OLS: {str(e)}")
    
    def run_glm_poisson(self, y_col='y', x_col='X'):
        """Run Poisson GLM"""
        print("\n" + "=" * 70)
        print("📊 Poisson GLM Analysis")
        print("=" * 70)
        
        if self.data is None:
            if not self.load_data('glm_poisson.csv'):
                return
        
        print(f"\n📋 Model Specification:")
        print(f"   Response: {y_col} (count)")
        print(f"   Predictor: {x_col}")
        
        try:
            formula = f'{y_col} ~ {x_col}'
            self.model = smf.glm(formula, data=self.data, 
                                family=sm.families.Poisson()).fit()
            
            print(f"\n📈 Model Summary:")
            print(f"   AIC: {self.model.aic:.2f}")
            print(f"   BIC: {self.model.bic:.2f}")
            print(f"   Deviance: {self.model.deviance:.2f}")
            print(f"   Pearson Chi2: {self.model.pearson_chi2:.2f}")
            
            print(f"\n📊 Coefficients:")
            coef_df = pd.DataFrame({
                'Coefficient': self.model.params,
                'Std Error': self.model.bse,
                'p-value': self.model.pvalues
            })
            print(coef_df.to_string())
            
            print(f"\n✅ Poisson GLM analysis complete!")
            
        except Exception as e:
            print(f"❌ Error running Poisson GLM: {str(e)}")
    
    def run_glm_logistic(self, y_col='y', x_col='X'):
        """Run Logistic GLM"""
        print("\n" + "=" * 70)
        print("📊 Logistic Regression Analysis")
        print("=" * 70)
        
        if self.data is None:
            if not self.load_data('glm_logistic.csv'):
                return
        
        print(f"\n📋 Model Specification:")
        print(f"   Response: {y_col} (binary)")
        print(f"   Predictor: {x_col}")
        
        try:
            formula = f'{y_col} ~ {x_col}'
            self.model = smf.glm(formula, data=self.data, 
                                family=sm.families.Binomial()).fit()
            
            print(f"\n📈 Model Summary:")
            print(f"   AIC: {self.model.aic:.2f}")
            print(f"   BIC: {self.model.bic:.2f}")
            print(f"   Null Deviance: {self.model.null_deviance:.2f}")
            print(f"   Deviance: {self.model.deviance:.2f}")
            
            print(f"\n📊 Coefficients:")
            coef_df = pd.DataFrame({
                'Coefficient': self.model.params,
                'Odds Ratio': np.exp(self.model.params),
                'Std Error': self.model.bse,
                'p-value': self.model.pvalues
            })
            print(coef_df.to_string())
            
            print(f"\n✅ Logistic regression analysis complete!")
            
        except Exception as e:
            print(f"❌ Error running Logistic GLM: {str(e)}")
    
    def run_arima(self, value_col='value', order=(1, 0, 1)):
        """Run ARIMA analysis"""
        print("\n" + "=" * 70)
        print(f"📊 ARIMA{order} Analysis")
        print("=" * 70)
        
        if self.data is None:
            if not self.load_data('arima_series.csv'):
                return
        
        try:
            from statsmodels.tsa.arima.model import ARIMA
            
            series = self.data[value_col].values
            print(f"\n📋 Series Info:")
            print(f"   Length: {len(series)}")
            print(f"   Mean: {series.mean():.4f}")
            print(f"   Std Dev: {series.std():.4f}")
            
            self.model = ARIMA(series, order=order).fit()
            
            print(f"\n📈 Model Summary:")
            print(f"   AIC: {self.model.aic:.2f}")
            print(f"   BIC: {self.model.bic:.2f}")
            print(f"   Log-Likelihood: {self.model.llf:.2f}")
            
            print(f"\n📊 Parameters:")
            param_df = pd.DataFrame({
                'Parameter': self.model.params,
                'p-value': self.model.pvalues
            })
            print(param_df.to_string())
            
            # Forecast
            forecast = self.model.forecast(steps=10)
            print(f"\n🔮 10-Step Forecast:")
            for i, val in enumerate(forecast[:5], 1):
                print(f"   Step {i}: {val:.4f}")
            print(f"   ...")
            
            print(f"\n✅ ARIMA analysis complete!")
            
        except Exception as e:
            print(f"❌ Error running ARIMA: {str(e)}")
    
    def interactive_mode(self):
        """Run in interactive mode"""
        print("=" * 70)
        print("StatsmodelsMasterPro - Quick Analysis (Interactive Mode)")
        print("=" * 70)
        
        print("\nAvailable analyses:")
        print("  1. OLS Regression")
        print("  2. Poisson GLM")
        print("  3. Logistic GLM")
        print("  4. ARIMA Time Series")
        print("  5. Exit")
        
        choice = input("\nSelect analysis (1-5): ").strip()
        
        if choice == '1':
            self.run_ols()
        elif choice == '2':
            self.run_glm_poisson()
        elif choice == '3':
            self.run_glm_logistic()
        elif choice == '4':
            self.run_arima()
        elif choice == '5':
            print("Goodbye!")
        else:
            print("Invalid choice.")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Quick statistical analysis runner"
    )
    
    parser.add_argument('analysis', nargs='?', 
                       choices=['ols', 'glm', 'arima'],
                       help='Type of analysis to run')
    parser.add_argument('subtype', nargs='?',
                       choices=['poisson', 'logistic'],
                       help='GLM subtype (for glm analysis)')
    parser.add_argument('--data', help='Custom data file')
    
    args = parser.parse_args()
    
    analyzer = QuickAnalysis()
    
    # Load custom data if specified
    if args.data:
        if not analyzer.load_data(args.data):
            sys.exit(1)
    
    # Run requested analysis
    if args.analysis == 'ols':
        analyzer.run_ols()
    elif args.analysis == 'glm':
        if args.subtype == 'poisson':
            analyzer.run_glm_poisson()
        elif args.subtype == 'logistic':
            analyzer.run_glm_logistic()
        else:
            print("For GLM, specify subtype: poisson or logistic")
            sys.exit(1)
    elif args.analysis == 'arima':
        analyzer.run_arima()
    else:
        # Interactive mode
        analyzer.interactive_mode()


if __name__ == "__main__":
    main()
