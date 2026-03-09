#!/usr/bin/env python3
"""
export_all_results.py - Batch export all plots and tables

Exports:
- All diagnostic plots (residuals, Q-Q, influence)
- Model comparison tables
- Coefficient tables
- Summary statistics
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "synthetic_data"
PLOT_PATH = PROJECT_ROOT / "exports" / "plots"
TABLE_PATH = PROJECT_ROOT / "exports" / "tables"

PLOT_PATH.mkdir(parents=True, exist_ok=True)
TABLE_PATH.mkdir(parents=True, exist_ok=True)


class ResultExporter:
    """Export all model results, plots, and tables"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exported_files = []
    
    def export_ols_results(self):
        """Export OLS model results"""
        print("\n📊 Exporting OLS Results...")
        
        try:
            df = pd.read_csv(DATA_PATH / "ols_data.csv")
            X = sm.add_constant(df[['X1', 'X2']])
            model = sm.OLS(df['y'], X).fit()
            
            # Export coefficient table
            coef_df = pd.DataFrame({
                'Coefficient': model.params,
                'Std_Error': model.bse,
                'p_value': model.pvalues,
                'CI_Lower': model.conf_int()[0],
                'CI_Upper': model.conf_int()[1]
            })
            coef_path = TABLE_PATH / f"ols_coefficients_{self.timestamp}.csv"
            coef_df.to_csv(coef_path)
            self.exported_files.append(coef_path)
            
            # Export summary statistics
            summary_path = TABLE_PATH / f"ols_summary_stats_{self.timestamp}.csv"
            df.describe().to_csv(summary_path)
            self.exported_files.append(summary_path)
            
            # Plot: Residuals vs Fitted
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Residuals vs Fitted
            axes[0, 0].scatter(model.fittedvalues, model.resid, alpha=0.6)
            axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=1)
            axes[0, 0].set_xlabel('Fitted Values')
            axes[0, 0].set_ylabel('Residuals')
            axes[0, 0].set_title('Residuals vs Fitted')
            
            # Q-Q Plot
            sm.qqplot(model.resid, line='45', ax=axes[0, 1])
            axes[0, 1].set_title('Q-Q Plot')
            
            # Histogram of residuals
            axes[1, 0].hist(model.resid, bins=30, edgecolor='black', alpha=0.7)
            axes[1, 0].set_xlabel('Residuals')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Residual Distribution')
            
            # Actual vs Predicted
            axes[1, 1].scatter(df['y'], model.fittedvalues, alpha=0.6)
            axes[1, 1].plot([df['y'].min(), df['y'].max()], 
                           [df['y'].min(), df['y'].max()], 
                           'r--', linewidth=2)
            axes[1, 1].set_xlabel('Actual Values')
            axes[1, 1].set_ylabel('Predicted Values')
            axes[1, 1].set_title('Actual vs Predicted')
            
            plt.tight_layout()
            plot_path = PLOT_PATH / f"ols_diagnostics_{self.timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.exported_files.append(plot_path)
            
            print(f"  ✅ OLS results exported ({len([f for f in self.exported_files if 'ols' in str(f)])} files)")
            
        except Exception as e:
            print(f"  ❌ OLS export failed: {str(e)}")
    
    def export_glm_results(self):
        """Export GLM results"""
        print("\n📊 Exporting GLM Results...")
        
        # Poisson GLM
        try:
            df = pd.read_csv(DATA_PATH / "glm_poisson.csv")
            model = smf.glm('y ~ X', data=df, family=sm.families.Poisson()).fit()
            
            # Coefficient table
            coef_df = pd.DataFrame({
                'Coefficient': model.params,
                'Std_Error': model.bse,
                'p_value': model.pvalues
            })
            coef_path = TABLE_PATH / f"glm_poisson_coefficients_{self.timestamp}.csv"
            coef_df.to_csv(coef_path)
            self.exported_files.append(coef_path)
            
            # Plot: Deviance Residuals
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Residuals
            ax1.scatter(model.fittedvalues, model.resid_deviance, alpha=0.6)
            ax1.axhline(0, color='red', linestyle='--')
            ax1.set_xlabel('Fitted Values')
            ax1.set_ylabel('Deviance Residuals')
            ax1.set_title('Poisson GLM: Residuals vs Fitted')
            
            # Actual vs Predicted
            ax2.scatter(df['y'], model.fittedvalues, alpha=0.6)
            ax2.set_xlabel('Actual Count')
            ax2.set_ylabel('Predicted Count')
            ax2.set_title('Actual vs Predicted')
            
            plt.tight_layout()
            plot_path = PLOT_PATH / f"glm_poisson_diagnostics_{self.timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.exported_files.append(plot_path)
            
            print(f"  ✅ GLM Poisson results exported")
            
        except Exception as e:
            print(f"  ❌ GLM Poisson export failed: {str(e)}")
        
        # Logistic GLM
        try:
            df = pd.read_csv(DATA_PATH / "glm_logistic.csv")
            model = smf.glm('y ~ X', data=df, family=sm.families.Binomial()).fit()
            
            coef_df = pd.DataFrame({
                'Coefficient': model.params,
                'Odds_Ratio': np.exp(model.params),
                'p_value': model.pvalues
            })
            coef_path = TABLE_PATH / f"glm_logistic_coefficients_{self.timestamp}.csv"
            coef_df.to_csv(coef_path)
            self.exported_files.append(coef_path)
            
            print(f"  ✅ GLM Logistic results exported")
            
        except Exception as e:
            print(f"  ❌ GLM Logistic export failed: {str(e)}")
    
    def export_time_series_results(self):
        """Export time series results"""
        print("\n📊 Exporting Time Series Results...")
        
        try:
            df = pd.read_csv(DATA_PATH / "arima_series.csv")
            series = df['value'].values
            
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
            
            model = ARIMA(series, order=(1, 0, 1)).fit()
            
            # Export forecast
            forecast = model.forecast(steps=30)
            forecast_df = pd.DataFrame({
                'Step': range(1, 31),
                'Forecast': forecast
            })
            forecast_path = TABLE_PATH / f"arima_forecast_{self.timestamp}.csv"
            forecast_df.to_csv(forecast_path, index=False)
            self.exported_files.append(forecast_path)
            
            # Plot: ACF and PACF
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Original series
            axes[0, 0].plot(series)
            axes[0, 0].set_title('Original Time Series')
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('Value')
            
            # ACF
            plot_acf(series, lags=40, ax=axes[0, 1])
            axes[0, 1].set_title('Autocorrelation Function')
            
            # PACF
            plot_pacf(series, lags=40, ax=axes[1, 0], method='ywm')
            axes[1, 0].set_title('Partial Autocorrelation Function')
            
            # Residuals
            axes[1, 1].plot(model.resid)
            axes[1, 1].axhline(0, color='red', linestyle='--')
            axes[1, 1].set_title('Model Residuals')
            axes[1, 1].set_xlabel('Time')
            axes[1, 1].set_ylabel('Residual')
            
            plt.tight_layout()
            plot_path = PLOT_PATH / f"arima_diagnostics_{self.timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.exported_files.append(plot_path)
            
            print(f"  ✅ Time series results exported")
            
        except Exception as e:
            print(f"  ❌ Time series export failed: {str(e)}")
    
    def export_correlation_matrix(self):
        """Export correlation matrices for numeric datasets"""
        print("\n📊 Exporting Correlation Matrices...")
        
        try:
            df = pd.read_csv(DATA_PATH / "ols_data.csv")
            
            # Correlation matrix
            corr = df.corr()
            corr_path = TABLE_PATH / f"correlation_matrix_{self.timestamp}.csv"
            corr.to_csv(corr_path)
            self.exported_files.append(corr_path)
            
            # Heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8})
            plt.title('Correlation Matrix')
            plt.tight_layout()
            
            plot_path = PLOT_PATH / f"correlation_heatmap_{self.timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.exported_files.append(plot_path)
            
            print(f"  ✅ Correlation analysis exported")
            
        except Exception as e:
            print(f"  ❌ Correlation export failed: {str(e)}")
    
    def export_all(self):
        """Export all results"""
        print("=" * 70)
        print("StatsmodelsMasterPro - Batch Export")
        print("=" * 70)
        
        self.export_ols_results()
        self.export_glm_results()
        self.export_time_series_results()
        self.export_correlation_matrix()
        
        # Summary
        print("\n" + "=" * 70)
        print(f"✅ Export Complete!")
        print(f"📁 Total files exported: {len(self.exported_files)}")
        print(f"📊 Plots: {PLOT_PATH}")
        print(f"📋 Tables: {TABLE_PATH}")
        print("=" * 70)
        
        # List all exported files
        print("\nExported Files:")
        for file in self.exported_files:
            print(f"  • {file.name}")


if __name__ == "__main__":
    exporter = ResultExporter()
    exporter.export_all()
