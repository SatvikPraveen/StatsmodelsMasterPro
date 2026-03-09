import os
import re
from pathlib import Path

# --- CONFIG ---
NOTEBOOK_DIR = Path("notebooks")
UTILS_DIR = Path("utils")
EXTENSIONS = [".ipynb", ".py"]
IGNORED_FILES = ["__init__.py"]


def extract_imports_from_notebooks():
    imports = set()
    for nb_file in NOTEBOOK_DIR.rglob("*.ipynb"):
        with open(nb_file, "r", encoding="utf-8") as f:
            content = f.read()
            matches = re.findall(r"from\s+utils\.([a-zA-Z_]+)\s+import", content)
            imports.update(matches)
    return imports


def find_all_utils_files():
    return {
        f.stem for f in UTILS_DIR.glob("*.py")
        if f.name not in IGNORED_FILES
    }


if __name__ == "__main__":
    used_utils = extract_imports_from_notebooks()
    all_utils = find_all_utils_files()

    unused = all_utils - used_utils

    print("✅ Used utility modules:", sorted(used_utils))
    print("📂 All utility modules:", sorted(all_utils))

    if unused:
        print("\n⚠️ Unused utility files:")
        for file in sorted(unused):
            print(f" - {file}.py")
    else:
        print("\n🎉 No unused utility files found!")
