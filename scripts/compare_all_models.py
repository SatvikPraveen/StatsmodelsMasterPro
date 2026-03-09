#!/usr/bin/env python3
"""
compare_all_models.py - Automated model comparison across datasets

Compares:
- Different model types (OLS, GLM, robust)
- Nested models (likelihood ratio tests)
- Information criteria (AIC, BIC)
- Cross-validation performance
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "synthetic_data"
OUTPUT_PATH = PROJECT_ROOT / "exports" / "comparisons"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


class ModelComparator:
    """Compare multiple statistical models"""
    
    def __init__(self):
        self.comparisons = []
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def compare_ols_models(self):
        """Compare nested OLS models"""
        print("\n📊 Comparing OLS Models...")
        
        try:
            df = pd.read_csv(DATA_PATH / "ols_data.csv")
            
            # Model 1: Only X1
            X1 = sm.add_constant(df[['X1']])
            model1 = sm.OLS(df['y'], X1).fit()
            
            # Model 2: X1 + X2
            X2 = sm.add_constant(df[['X1', 'X2']])
            model2 = sm.OLS(df['y'], X2).fit()
            
            # Compare
            comparison = pd.DataFrame({
                'Model': ['OLS (X1 only)', 'OLS (X1 + X2)'],
                'R_squared': [model1.rsquared, model2.rsquared],
                'Adj_R_squared': [model1.rsquared_adj, model2.rsquared_adj],
                'AIC': [model1.aic, model2.aic],
                'BIC': [model1.bic, model2.bic],
                'MSE': [model1.mse_resid, model2.mse_resid],
                'N_Params': [len(model1.params), len(model2.params)]
            })
            
            # Likelihood Ratio Test (for nested models)
            lr_stat = -2 * (model1.llf - model2.llf)
            df_diff = len(model2.params) - len(model1.params)
            
            from scipy.stats import chi2
            p_value = 1 - chi2.cdf(lr_stat, df_diff)
            
            comparison['LR_Test_p'] = [None, p_value]
            
            print(comparison.to_string(index=False))
            
            if p_value < 0.05:
                print(f"\n✅ Model 2 (X1+X2) is significantly better (LR p={p_value:.4f})")
            else:
                print(f"\n⚠️  No significant improvement with X2 (LR p={p_value:.4f})")
            
            self.comparisons.append(('OLS_Comparison', comparison))
            
            # Save
            output_path = OUTPUT_PATH / f"ols_comparison_{self.timestamp}.csv"
            comparison.to_csv(output_path, index=False)
            print(f"📁 Saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    def compare_glm_families(self):
        """Compare different GLM families on Poisson data"""
        print("\n📊 Comparing GLM Families...")
        
        try:
            df = pd.read_csv(DATA_PATH / "glm_poisson.csv")
            
            # Poisson family (correct)
            model_poisson = smf.glm('y ~ X', data=df, family=sm.families.Poisson()).fit()
            
            # Gaussian family (incorrect for count data)
            model_gaussian = smf.glm('y ~ X', data=df, family=sm.families.Gaussian()).fit()
            
            # Negative Binomial (alternative for overdispersed counts)
            try:
                from statsmodels.discrete.discrete_model import NegativeBinomial
                model_nb = smf.negativebinomial('y ~ X', data=df).fit(disp=False)
                nb_available = True
            except:
                nb_available = False
                model_nb = None
            
            comparison = pd.DataFrame({
                'Model': ['Poisson GLM', 'Gaussian GLM'] + (['Negative Binomial'] if nb_available else []),
                'AIC': [model_poisson.aic, model_gaussian.aic] + ([model_nb.aic] if nb_available else []),
                'BIC': [model_poisson.bic, model_gaussian.bic] + ([model_nb.bic] if nb_available else []),
                'Deviance': [model_poisson.deviance, model_gaussian.deviance] + ([model_nb.deviance] if nb_available else []),
                'LogLikelihood': [model_poisson.llf, model_gaussian.llf] + ([model_nb.llf] if nb_available else [])
            })
            
            print(comparison.to_string(index=False))
            
            best_model = comparison.loc[comparison['AIC'].idxmin(), 'Model']
            print(f"\n✅ Best model by AIC: {best_model}")
            
            self.comparisons.append(('GLM_Family_Comparison', comparison))
            
            output_path = OUTPUT_PATH / f"glm_family_comparison_{self.timestamp}.csv"
            comparison.to_csv(output_path, index=False)
            print(f"📁 Saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    def compare_robust_methods(self):
        """Compare OLS vs Robust regression"""
        print("\n📊 Comparing Robust Regression Methods...")
        
        try:
            # Check if robust data exists
            if not (DATA_PATH / "robust_regression_data.csv").exists():
                print("⚠️  Robust regression data not found, using OLS data instead")
                df = pd.read_csv(DATA_PATH / "ols_data.csv")
                X_cols = ['X1', 'X2']
            else:
                df = pd.read_csv(DATA_PATH / "robust_regression_data.csv")
                X_cols = ['X']
            
            X = sm.add_constant(df[X_cols])
            
            # Standard OLS
            model_ols = sm.OLS(df['y'], X).fit()
            
            # Robust Linear Model (M-estimation)
            model_rlm = sm.RLM(df['y'], X).fit()
            
            # Weighted Least Squares (if weights available)
            if 'weights' in df.columns:
                model_wls = sm.WLS(df['y'], X, weights=df['weights']).fit()
                wls_available = True
            else:
                wls_available = False
                model_wls = None
            
            # Compare coefficients
            coef_comparison = pd.DataFrame({
                'Variable': model_ols.params.index,
                'OLS_Coef': model_ols.params.values,
                'RLM_Coef': model_rlm.params.values,
            })
            
            if wls_available:
                coef_comparison['WLS_Coef'] = model_wls.params.values
            
            print("\nCoefficient Comparison:")
            print(coef_comparison.to_string(index=False))
            
            # Model statistics
            stats_comparison = pd.DataFrame({
                'Model': ['OLS', 'RLM'] + (['WLS'] if wls_available else []),
                'MSE': [
                    model_ols.mse_resid,
                    np.mean(model_rlm.resid**2)
                ] + ([model_wls.mse_resid] if wls_available else []),
                'MAE': [
                    np.mean(np.abs(model_ols.resid)),
                    np.mean(np.abs(model_rlm.resid))
                ] + ([np.mean(np.abs(model_wls.resid))] if wls_available else [])
            })
            
            print("\nModel Statistics:")
            print(stats_comparison.to_string(index=False))
            
            self.comparisons.append(('Robust_Comparison', coef_comparison))
            self.comparisons.append(('Robust_Stats', stats_comparison))
            
            output_path = OUTPUT_PATH / f"robust_comparison_{self.timestamp}.csv"
            coef_comparison.to_csv(output_path, index=False)
            print(f"📁 Saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    def compare_time_series_orders(self):
        """Compare ARIMA models with different orders"""
        print("\n📊 Comparing Time Series Model Orders...")
        
        try:
            df = pd.read_csv(DATA_PATH / "arima_series.csv")
            series = df['value'].values
            
            from statsmodels.tsa.arima.model import ARIMA
            
            # Try different ARIMA orders
            orders = [
                (0, 0, 0),  # White noise
                (1, 0, 0),  # AR(1)
                (0, 0, 1),  # MA(1)
                (1, 0, 1),  # ARMA(1,1)
                (2, 0, 2),  # ARMA(2,2)
            ]
            
            results = []
            for order in orders:
                try:
                    model = ARIMA(series, order=order).fit()
                    results.append({
                        'Model': f'ARIMA{order}',
                        'AIC': model.aic,
                        'BIC': model.bic,
                        'LogLikelihood': model.llf,
                        'N_Params': len(model.params)
                    })
                except:
                    pass
            
            comparison = pd.DataFrame(results)
            comparison = comparison.sort_values('AIC')
            
            print(comparison.to_string(index=False))
            
            best_model = comparison.iloc[0]['Model']
            print(f"\n✅ Best model by AIC: {best_model}")
            
            self.comparisons.append(('ARIMA_Comparison', comparison))
            
            output_path = OUTPUT_PATH / f"arima_comparison_{self.timestamp}.csv"
            comparison.to_csv(output_path, index=False)
            print(f"📁 Saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    def run_all_comparisons(self):
        """Run all model comparisons"""
        print("=" * 70)
        print("StatsmodelsMasterPro - Model Comparison Suite")
        print("=" * 70)
        
        self.compare_ols_models()
        self.compare_glm_families()
        self.compare_robust_methods()
        self.compare_time_series_orders()
        
        # Summary
        print("\n" + "=" * 70)
        print(f"✅ Completed {len(self.comparisons)} model comparisons")
        print(f"📁 Results saved to: {OUTPUT_PATH}")
        print("=" * 70)


if __name__ == "__main__":
    comparator = ModelComparator()
    comparator.run_all_comparisons()
