#!/usr/bin/env python3
"""
generate_all_reports.py - Generate markdown/HTML reports for all analyses

Creates comprehensive reports including:
- Model summaries
- Statistical tables
- Interpretation guides
- Export to markdown and HTML
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
REPORT_PATH = PROJECT_ROOT / "exports" / "reports"
REPORT_PATH.mkdir(parents=True, exist_ok=True)


class ReportGenerator:
    """Generate comprehensive analysis reports"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.report_content = []
    
    def add_header(self, text, level=1):
        """Add markdown header"""
        self.report_content.append(f"\n{'#' * level} {text}\n")
    
    def add_text(self, text):
        """Add regular text"""
        self.report_content.append(f"{text}\n")
    
    def add_table(self, df, title=None):
        """Add DataFrame as markdown table"""
        if title:
            self.add_header(title, level=3)
        self.report_content.append(df.to_markdown())
        self.report_content.append("\n")
    
    def add_code_block(self, code, language="python"):
        """Add code block"""
        self.report_content.append(f"```{language}\n{code}\n```\n")
    
    def generate_ols_report(self):
        """Generate OLS analysis report"""
        self.add_header("OLS Regression Analysis", level=2)
        self.add_text(f"*Generated: {self.timestamp}*")
        
        try:
            df = pd.read_csv(DATA_PATH / "ols_data.csv")
            
            self.add_header("Dataset Overview", level=3)
            self.add_text(f"- **Observations**: {len(df)}")
            self.add_text(f"- **Variables**: {', '.join(df.columns)}")
            
            self.add_header("Descriptive Statistics", level=3)
            self.add_table(df.describe().T)
            
            # Fit model
            X = sm.add_constant(df[['X1', 'X2']])
            model = sm.OLS(df['y'], X).fit()
            
            self.add_header("Model Results", level=3)
            self.add_text(f"- **R-squared**: {model.rsquared:.4f}")
            self.add_text(f"- **Adjusted R-squared**: {model.rsquared_adj:.4f}")
            self.add_text(f"- **AIC**: {model.aic:.2f}")
            self.add_text(f"- **BIC**: {model.bic:.2f}")
            self.add_text(f"- **F-statistic**: {model.fvalue:.2f} (p={model.f_pvalue:.4f})")
            
            # Coefficients
            self.add_header("Coefficients", level=3)
            coef_df = pd.DataFrame({
                'Coefficient': model.params,
                'Std Error': model.bse,
                'p-value': model.pvalues,
                'CI Lower': model.conf_int()[0],
                'CI Upper': model.conf_int()[1]
            })
            self.add_table(coef_df)
            
            self.add_header("Interpretation", level=3)
            self.add_text("- **X1**: Each unit increase in X1 is associated with a "
                         f"{model.params['X1']:.3f} unit change in y.")
            self.add_text("- **X2**: Each unit increase in X2 is associated with a "
                         f"{model.params['X2']:.3f} unit change in y.")
            
            return True
        except Exception as e:
            self.add_text(f"❌ Error generating OLS report: {str(e)}")
            return False
    
    def generate_glm_report(self):
        """Generate GLM analysis report"""
        self.add_header("Generalized Linear Models", level=2)
        
        # Poisson GLM
        try:
            self.add_header("Poisson Regression", level=3)
            df = pd.read_csv(DATA_PATH / "glm_poisson.csv")
            
            model = smf.glm('y ~ X', data=df, family=sm.families.Poisson()).fit()
            
            self.add_text(f"- **AIC**: {model.aic:.2f}")
            self.add_text(f"- **Deviance**: {model.deviance:.2f}")
            self.add_text(f"- **Pearson Chi2**: {model.pearson_chi2:.2f}")
            
            coef_df = pd.DataFrame({
                'Coefficient': model.params,
                'p-value': model.pvalues
            })
            self.add_table(coef_df, "Coefficients")
            
        except Exception as e:
            self.add_text(f"❌ Error: {str(e)}")
        
        # Logistic GLM
        try:
            self.add_header("Logistic Regression", level=3)
            df = pd.read_csv(DATA_PATH / "glm_logistic.csv")
            
            model = smf.glm('y ~ X', data=df, family=sm.families.Binomial()).fit()
            
            self.add_text(f"- **AIC**: {model.aic:.2f}")
            self.add_text(f"- **Null Deviance**: {model.null_deviance:.2f}")
            self.add_text(f"- **Deviance**: {model.deviance:.2f}")
            
            coef_df = pd.DataFrame({
                'Coefficient': model.params,
                'Odds Ratio': np.exp(model.params),
                'p-value': model.pvalues
            })
            self.add_table(coef_df, "Coefficients & Odds Ratios")
            
        except Exception as e:
            self.add_text(f"❌ Error: {str(e)}")
    
    def generate_time_series_report(self):
        """Generate time series analysis report"""
        self.add_header("Time Series Analysis", level=2)
        
        try:
            df = pd.read_csv(DATA_PATH / "arima_series.csv")
            series = df['value'].values
            
            self.add_text(f"- **Observations**: {len(series)}")
            
            # Fit ARIMA
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(series, order=(1, 0, 1)).fit()
            
            self.add_header("ARIMA(1,0,1) Results", level=3)
            self.add_text(f"- **AIC**: {model.aic:.2f}")
            self.add_text(f"- **BIC**: {model.bic:.2f}")
            self.add_text(f"- **Log-Likelihood**: {model.llf:.2f}")
            
            coef_df = pd.DataFrame({
                'Parameter': model.params,
                'p-value': model.pvalues
            })
            self.add_table(coef_df, "Model Parameters")
            
        except Exception as e:
            self.add_text(f"❌ Error: {str(e)}")
    
    def generate_summary_report(self):
        """Generate overall summary"""
        self.add_header("StatsmodelsMasterPro - Analysis Report", level=1)
        self.add_text(f"**Generated**: {self.timestamp}")
        self.add_text(f"**Project**: Statistical Modeling & Analysis Suite")
        
        self.add_header("Overview", level=2)
        self.add_text("This report contains results from all statistical analyses "
                     "performed on synthetic datasets.")
        
        # Generate individual reports
        self.generate_ols_report()
        self.generate_glm_report()
        self.generate_time_series_report()
    
    def save_report(self, filename="analysis_report.md"):
        """Save report to markdown file"""
        report_path = REPORT_PATH / filename
        
        with open(report_path, 'w') as f:
            f.write('\n'.join(self.report_content))
        
        print(f"✅ Report saved to: {report_path}")
        
        # Also try to convert to HTML if pypandoc is available
        try:
            import pypandoc
            html_path = report_path.with_suffix('.html')
            pypandoc.convert_file(str(report_path), 'html', outputfile=str(html_path))
            print(f"✅ HTML report saved to: {html_path}")
        except ImportError:
            print("ℹ️  Install pypandoc for HTML conversion: pip install pypandoc")
        except Exception as e:
            print(f"⚠️  HTML conversion failed: {str(e)}")
        
        return report_path


def main():
    """Generate all reports"""
    print("=" * 70)
    print("StatsmodelsMasterPro - Report Generator")
    print("=" * 70)
    
    generator = ReportGenerator()
    generator.generate_summary_report()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = generator.save_report(f"analysis_report_{timestamp}.md")
    
    print("\n" + "=" * 70)
    print(f"📄 Report generated successfully!")
    print(f"📂 Location: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
