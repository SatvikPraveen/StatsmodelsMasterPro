#!/usr/bin/env python3
"""
benchmark_performance.py - Performance benchmarking

Benchmarks:
- Model fitting time
- Memory usage
- Prediction speed
- Scalability with data size
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import time
import tracemalloc
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "synthetic_data"


class PerformanceBenchmark:
    """Benchmark model performance"""
    
    def __init__(self):
        self.results = []
    
    def print_section(self, title):
        """Print formatted section"""
        print(f"\n{'=' * 70}")
        print(f"  {title}")
        print('=' * 70)
    
    def benchmark_function(self, func, name, iterations=1):
        """Benchmark a function's performance"""
        print(f"\n📊 Benchmarking: {name}")
        
        # Warm-up
        try:
            func()
        except:
            pass
        
        # Memory tracking
        tracemalloc.start()
        
        # Time tracking
        times = []
        for i in range(iterations):
            start_time = time.time()
            try:
                func()
                end_time = time.time()
                times.append(end_time - start_time)
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                return None
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"   ⏱️  Avg Time: {avg_time:.4f}s (±{std_time:.4f}s)")
        print(f"   💾 Memory: Current={current/1024/1024:.2f}MB, Peak={peak/1024/1024:.2f}MB")
        
        result = {
            'Function': name,
            'Avg_Time_s': avg_time,
            'Std_Time_s': std_time,
            'Peak_Memory_MB': peak / 1024 / 1024,
            'Iterations': iterations
        }
        
        self.results.append(result)
        return result
    
    def benchmark_ols(self):
        """Benchmark OLS regression"""
        self.print_section("OLS Regression Benchmark")
        
        df = pd.read_csv(DATA_PATH / "ols_data.csv")
        
        def fit_ols():
            X = sm.add_constant(df[['X1', 'X2']])
            model = sm.OLS(df['y'], X).fit()
            return model
        
        self.benchmark_function(fit_ols, "OLS Fitting", iterations=10)
        
        # Benchmark with different sample sizes
        print(f"\n📈 Scalability Test:")
        sizes = [50, 100, 200, 500, 1000]
        
        for size in sizes:
            df_sample = df.sample(min(size, len(df)), replace=True, random_state=42)
            
            def fit_ols_sized():
                X = sm.add_constant(df_sample[['X1', 'X2']])
                model = sm.OLS(df_sample['y'], X).fit()
                return model
            
            start = time.time()
            fit_ols_sized()
            elapsed = time.time() - start
            
            print(f"   n={size:4d}: {elapsed:.4f}s")
    
    def benchmark_glm(self):
        """Benchmark GLM models"""
        self.print_section("GLM Benchmark")
        
        # Poisson
        df = pd.read_csv(DATA_PATH / "glm_poisson.csv")
        
        def fit_poisson():
            model = smf.glm('y ~ X', data=df, family=sm.families.Poisson()).fit()
            return model
        
        self.benchmark_function(fit_poisson, "Poisson GLM Fitting", iterations=10)
        
        # Logistic
        df_log = pd.read_csv(DATA_PATH / "glm_logistic.csv")
        
        def fit_logistic():
            model = smf.glm('y ~ X', data=df_log, family=sm.families.Binomial()).fit()
            return model
        
        self.benchmark_function(fit_logistic, "Logistic GLM Fitting", iterations=10)
    
    def benchmark_time_series(self):
        """Benchmark time series models"""
        self.print_section("Time Series Benchmark")
        
        df = pd.read_csv(DATA_PATH / "arima_series.csv")
        series = df['value'].values[:100]  # Use subset for speed
        
        from statsmodels.tsa.arima.model import ARIMA
        
        def fit_arima():
            model = ARIMA(series, order=(1, 0, 1)).fit()
            return model
        
        self.benchmark_function(fit_arima, "ARIMA(1,0,1) Fitting", iterations=5)
        
        # Forecast benchmark
        model = ARIMA(series, order=(1, 0, 1)).fit()
        
        def forecast():
            return model.forecast(steps=10)
        
        self.benchmark_function(forecast, "ARIMA Forecasting (10 steps)", iterations=10)
    
    def benchmark_robust_regression(self):
        """Benchmark robust regression"""
        self.print_section("Robust Regression Benchmark")
        
        # Use OLS data if robust data not available
        if (DATA_PATH / "robust_regression_data.csv").exists():
            df = pd.read_csv(DATA_PATH / "robust_regression_data.csv")
            X_cols = ['X']
        else:
            df = pd.read_csv(DATA_PATH / "ols_data.csv")
            X_cols = ['X1', 'X2']
        
        X = sm.add_constant(df[X_cols])
        
        # OLS baseline
        def fit_ols():
            model = sm.OLS(df['y'], X).fit()
            return model
        
        self.benchmark_function(fit_ols, "OLS (baseline)", iterations=10)
        
        # RLM
        def fit_rlm():
            model = sm.RLM(df['y'], X).fit()
            return model
        
        self.benchmark_function(fit_rlm, "Robust Linear Model (RLM)", iterations=10)
    
    def benchmark_multivariate(self):
        """Benchmark multivariate tests"""
        self.print_section("Multivariate Tests Benchmark")
        
        df = pd.read_csv(DATA_PATH / "manova_data.csv")
        
        from statsmodels.multivariate.manova import MANOVA
        
        def fit_manova():
            manova = MANOVA.from_formula('Y1 + Y2 ~ group', data=df)
            result = manova.mv_test()
            return result
        
        self.benchmark_function(fit_manova, "MANOVA", iterations=5)
    
    def generate_report(self):
        """Generate benchmark report"""
        self.print_section("Benchmark Summary")
        
        if self.results:
            df = pd.DataFrame(self.results)
            df = df.sort_values('Avg_Time_s')
            
            print("\n📊 Performance Rankings (Fastest to Slowest):")
            print(df[['Function', 'Avg_Time_s', 'Peak_Memory_MB']].to_string(index=False))
            
            # Save to CSV
            output_path = PROJECT_ROOT / "exports" / "benchmarks"
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_path = output_path / f"benchmark_results_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            
            print(f"\n📁 Results saved to: {csv_path}")
        else:
            print("No benchmark results collected.")
    
    def run_all_benchmarks(self):
        """Run all benchmarks"""
        print("=" * 70)
        print("StatsmodelsMasterPro - Performance Benchmark Suite")
        print("=" * 70)
        print(f"\nSystem: Python {sys.version.split()[0]}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.benchmark_ols()
        self.benchmark_glm()
        self.benchmark_time_series()
        self.benchmark_robust_regression()
        self.benchmark_multivariate()
        
        self.generate_report()
        
        print("\n" + "=" * 70)
        print("✅ Benchmarking Complete!")
        print("=" * 70)


if __name__ == "__main__":
    benchmark = PerformanceBenchmark()
    benchmark.run_all_benchmarks()
