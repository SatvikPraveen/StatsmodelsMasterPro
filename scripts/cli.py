#!/usr/bin/env python3
"""
cli.py - Command-line interface for StatsmodelsMasterPro

Usage:
    python scripts/cli.py --help
    python scripts/cli.py analyze ols --data ols_data.csv
    python scripts/cli.py test all
    python scripts/cli.py export plots
    python scripts/cli.py validate datasets
"""

import sys
import argparse
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


class StatsCLI:
    """Command-line interface for StatsmodelsMasterPro"""
    
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="StatsmodelsMasterPro - Statistical Analysis CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Run all tests
  python cli.py test all
  
  # Run OLS analysis
  python cli.py analyze ols
  
  # Export all results
  python cli.py export all
  
  # Validate datasets
  python cli.py validate
  
  # Generate reports
  python cli.py report
  
  # Run benchmarks
  python cli.py benchmark
            """
        )
        
        # Create subparsers
        subparsers = self.parser.add_subparsers(dest='command', help='Command to execute')
        
        # Test command
        test_parser = subparsers.add_parser('test', help='Run tests')
        test_parser.add_argument('suite', choices=['all', 'utils', 'data'], 
                                help='Test suite to run')
        
        # Analyze command
        analyze_parser = subparsers.add_parser('analyze', help='Run analysis')
        analyze_parser.add_argument('model', 
                                   choices=['ols', 'glm', 'arima', 'all'],
                                   help='Model type to run')
        analyze_parser.add_argument('--data', help='Custom data file (optional)')
        
        # Export command
        export_parser = subparsers.add_parser('export', help='Export results')
        export_parser.add_argument('type', 
                                  choices=['plots', 'tables', 'all'],
                                  help='What to export')
        
        # Validate command
        validate_parser = subparsers.add_parser('validate', help='Validate datasets')
        
        # Report command
        report_parser = subparsers.add_parser('report', help='Generate reports')
        
        # Benchmark command
        benchmark_parser = subparsers.add_parser('benchmark', help='Run performance benchmarks')
        
        # Compare command
        compare_parser = subparsers.add_parser('compare', help='Compare models')
        
        # Info command
        info_parser = subparsers.add_parser('info', help='Show project information')
    
    def run_script(self, script_name, description):
        """Run a Python script"""
        script_path = SCRIPTS_DIR / script_name
        
        if not script_path.exists():
            print(f"❌ Script not found: {script_name}")
            return False
        
        print(f"\n{'=' * 70}")
        print(f"🚀 {description}")
        print('=' * 70)
        
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=PROJECT_ROOT
            )
            return result.returncode == 0
        except Exception as e:
            print(f"❌ Error running {script_name}: {str(e)}")
            return False
    
    def cmd_test(self, args):
        """Handle test command"""
        if args.suite == 'all':
            return self.run_script('run_all_tests.py', 'Running All Tests')
        elif args.suite == 'utils':
            return self.run_script('test_utils.py', 'Running Utility Tests')
        elif args.suite == 'data':
            return self.run_script('test_data_generation.py', 'Running Dataset Validation')
    
    def cmd_analyze(self, args):
        """Handle analyze command"""
        if args.model == 'all':
            return self.run_script('batch_run_all_models.py', 'Running All Models')
        elif args.model == 'ols':
            return self.run_script('quick_analysis.py', f'Running OLS Analysis')
        else:
            print(f"Analysis for {args.model} - use quick_analysis.py")
            return True
    
    def cmd_export(self, args):
        """Handle export command"""
        return self.run_script('export_all_results.py', 'Exporting Results')
    
    def cmd_validate(self, args):
        """Handle validate command"""
        return self.run_script('validate_datasets.py', 'Validating Datasets')
    
    def cmd_report(self, args):
        """Handle report command"""
        return self.run_script('generate_all_reports.py', 'Generating Reports')
    
    def cmd_benchmark(self, args):
        """Handle benchmark command"""
        return self.run_script('benchmark_performance.py', 'Running Benchmarks')
    
    def cmd_compare(self, args):
        """Handle compare command"""
        return self.run_script('compare_all_models.py', 'Comparing Models')
    
    def cmd_info(self, args):
        """Display project information"""
        print("\n" + "=" * 70)
        print("StatsmodelsMasterPro - Project Information")
        print("=" * 70)
        
        # Count files
        pages_count = len(list((PROJECT_ROOT / "pages").glob("*.py")))
        notebooks_count = len(list((PROJECT_ROOT / "notebooks").glob("*.ipynb")))
        datasets_count = len(list((PROJECT_ROOT / "synthetic_data").glob("*.csv")))
        utils_count = len(list((PROJECT_ROOT / "utils").glob("*.py"))) - 1  # Exclude __init__
        scripts_count = len(list(SCRIPTS_DIR.glob("*.py")))
        
        print(f"\n📊 Project Statistics:")
        print(f"  • Streamlit Pages: {pages_count}")
        print(f"  • Jupyter Notebooks: {notebooks_count}")
        print(f"  • Synthetic Datasets: {datasets_count}")
        print(f"  • Utility Modules: {utils_count}")
        print(f"  • Automation Scripts: {scripts_count}")
        
        print(f"\n📁 Directory Structure:")
        print(f"  • Root: {PROJECT_ROOT}")
        print(f"  • Data: {PROJECT_ROOT / 'synthetic_data'}")
        print(f"  • Scripts: {SCRIPTS_DIR}")
        print(f"  • Exports: {PROJECT_ROOT / 'exports'}")
        
        print(f"\n🛠️  Available Commands:")
        print(f"  • test      - Run test suites")
        print(f"  • analyze   - Run statistical analyses")
        print(f"  • export    - Export plots and tables")
        print(f"  • validate  - Validate datasets")
        print(f"  • report    - Generate markdown reports")
        print(f"  • benchmark - Performance benchmarking")
        print(f"  • compare   - Compare multiple models")
        
        print("\n" + "=" * 70)
        return True
    
    def run(self):
        """Parse arguments and run command"""
        args = self.parser.parse_args()
        
        if not args.command:
            self.parser.print_help()
            return 0
        
        # Route to appropriate handler
        handlers = {
            'test': self.cmd_test,
            'analyze': self.cmd_analyze,
            'export': self.cmd_export,
            'validate': self.cmd_validate,
            'report': self.cmd_report,
            'benchmark': self.cmd_benchmark,
            'compare': self.cmd_compare,
            'info': self.cmd_info
        }
        
        handler = handlers.get(args.command)
        if handler:
            success = handler(args)
            return 0 if success else 1
        else:
            print(f"❌ Unknown command: {args.command}")
            return 1


def main():
    """Main entry point"""
    cli = StatsCLI()
    sys.exit(cli.run())


if __name__ == "__main__":
    main()
