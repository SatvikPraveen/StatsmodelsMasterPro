#!/usr/bin/env python3
"""
update_readme_stats.py - Auto-update README with current project statistics

Updates:
- File counts
- Dataset counts
- Line of code statistics
- Last updated timestamp
- Contribution stats
"""

import sys
from pathlib import Path
from datetime import datetime
import subprocess


PROJECT_ROOT = Path(__file__).parent.parent
README_PATH = PROJECT_ROOT / "README.md"


class ReadmeUpdater:
    """Update README with current statistics"""
    
    def __init__(self):
        self.stats = {}
    
    def collect_stats(self):
        """Collect project statistics"""
        print("📊 Collecting project statistics...\n")
        
        # Count Streamlit pages
        pages_dir = PROJECT_ROOT / "pages"
        self.stats['streamlit_pages'] = len(list(pages_dir.glob("*.py"))) if pages_dir.exists() else 0
        print(f"✅ Streamlit pages: {self.stats['streamlit_pages']}")
        
        # Count Jupyter notebooks
        notebooks_dir = PROJECT_ROOT / "notebooks"
        self.stats['notebooks'] = len(list(notebooks_dir.glob("*.ipynb"))) if notebooks_dir.exists() else 0
        print(f"✅ Jupyter notebooks: {self.stats['notebooks']}")
        
        # Count datasets
        data_dir = PROJECT_ROOT / "synthetic_data"
        self.stats['datasets'] = len(list(data_dir.glob("*.csv"))) if data_dir.exists() else 0
        print(f"✅ Datasets: {self.stats['datasets']}")
        
        # Count utility modules
        utils_dir = PROJECT_ROOT / "utils"
        if utils_dir.exists():
            self.stats['utils'] = len([f for f in utils_dir.glob("*.py") if f.name != '__init__.py'])
        else:
            self.stats['utils'] = 0
        print(f"✅ Utility modules: {self.stats['utils']}")
        
        # Count scripts
        scripts_dir = PROJECT_ROOT / "scripts"
        self.stats['scripts'] = len(list(scripts_dir.glob("*.py"))) if scripts_dir.exists() else 0
        print(f"✅ Automation scripts: {self.stats['scripts']}")
        
        # Count lines of code (Python files only)
        self.stats['total_lines'] = self.count_lines_of_code()
        print(f"✅ Total lines of Python code: {self.stats['total_lines']:,}")
        
        # Git stats
        self.collect_git_stats()
    
    def count_lines_of_code(self):
        """Count total lines of Python code"""
        total_lines = 0
        
        for py_file in PROJECT_ROOT.rglob("*.py"):
            # Skip virtual environments and build directories
            if any(skip in str(py_file) for skip in ['env', 'venv', '__pycache__', 'build', 'dist']):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    total_lines += len(f.readlines())
            except:
                pass
        
        return total_lines
    
    def collect_git_stats(self):
        """Collect git repository statistics"""
        try:
            # Check if git is available
            result = subprocess.run(
                ['git', 'rev-list', '--count', 'HEAD'],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.stats['commits'] = int(result.stdout.strip())
                print(f"✅ Git commits: {self.stats['commits']}")
            else:
                self.stats['commits'] = 'N/A'
        except:
            self.stats['commits'] = 'N/A'
    
    def generate_stats_section(self):
        """Generate statistics section for README"""
        section = f"""
## 📊 Project Statistics

*Last updated: {datetime.now().strftime('%B %d, %Y')}*

| Metric | Count |
|--------|------:|
| 📄 Streamlit Pages | {self.stats['streamlit_pages']} |
| 📓 Jupyter Notebooks | {self.stats['notebooks']} |
| 📊 Synthetic Datasets | {self.stats['datasets']} |
| 🛠️ Utility Modules | {self.stats['utils']} |
| 🤖 Automation Scripts | {self.stats['scripts']} |
| 💻 Lines of Code | {self.stats['total_lines']:,} |
| 🎯 Git Commits | {self.stats['commits']} |

"""
        return section
    
    def update_readme(self):
        """Update README with new statistics"""
        print("\n📝 Updating README.md...")
        
        if not README_PATH.exists():
            print("❌ README.md not found!")
            return False
        
        # Read current README
        with open(README_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Generate new stats section
        new_stats = self.generate_stats_section()
        
        # Check if stats section already exists
        if "## 📊 Project Statistics" in content:
            # Replace existing section
            import re
            pattern = r'## 📊 Project Statistics.*?(?=\n## |\Z)'
            updated_content = re.sub(pattern, new_stats.strip(), content, flags=re.DOTALL)
            
            if updated_content != content:
                # Write back
                with open(README_PATH, 'w', encoding='utf-8') as f:
                    f.write(updated_content)
                print("✅ README.md updated (replaced existing stats)")
                return True
            else:
                print("⚠️  No changes made to README.md")
                return False
        else:
            # Append stats section before the last section or at the end
            # Try to insert before "## License" or similar ending sections
            insert_markers = ["## License", "## Contributing", "## Credits"]
            inserted = False
            
            for marker in insert_markers:
                if marker in content:
                    parts = content.split(marker, 1)
                    updated_content = parts[0] + new_stats + "\n" + marker + parts[1]
                    
                    with open(README_PATH, 'w', encoding='utf-8') as f:
                        f.write(updated_content)
                    
                    print(f"✅ README.md updated (inserted before '{marker}')")
                    inserted = True
                    break
            
            if not inserted:
                # Append at the end
                with open(README_PATH, 'a', encoding='utf-8') as f:
                    f.write("\n" + new_stats)
                print("✅ README.md updated (appended at end)")
            
            return True
    
    def run(self):
        """Run the updater"""
        print("=" * 70)
        print("StatsmodelsMasterPro - README Statistics Updater")
        print("=" * 70)
        
        self.collect_stats()
        success = self.update_readme()
        
        print("\n" + "=" * 70)
        if success:
            print("✅ README.md successfully updated with current statistics!")
        else:
            print("⚠️  README.md update incomplete")
        print("=" * 70)
        
        return success


def main():
    """Main entry point"""
    updater = ReadmeUpdater()
    success = updater.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
