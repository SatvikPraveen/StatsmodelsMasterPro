#!/usr/bin/env python3
"""
generate_api_docs.py - Auto-generate API documentation for utility modules

Generates:
- Function signatures
- Docstrings
- Parameter descriptions
- Return types
- Usage examples
"""

import sys
from pathlib import Path
import inspect
import importlib.util
from datetime import datetime


PROJECT_ROOT = Path(__file__).parent.parent
UTILS_DIR = PROJECT_ROOT / "utils"
DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_DIR.mkdir(exist_ok=True)


class APIDocGenerator:
    """Generate API documentation from Python modules"""
    
    def __init__(self):
        self.docs = []
    
    def add_header(self, text, level=1):
        """Add markdown header"""
        self.docs.append(f"\n{'#' * level} {text}\n")
    
    def add_text(self, text):
        """Add plain text"""
        self.docs.append(f"{text}\n")
    
    def add_code_block(self, code, language="python"):
        """Add code block"""
        self.docs.append(f"```{language}\n{code}\n```\n")
    
    def import_module(self, module_path):
        """Import a module from file path"""
        try:
            spec = importlib.util.spec_from_file_location("module", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            print(f"❌ Error importing {module_path}: {str(e)}")
            return None
    
    def document_function(self, func_name, func):
        """Generate documentation for a single function"""
        self.add_header(f"`{func_name}`", level=3)
        
        # Get signature
        try:
            sig = inspect.signature(func)
            self.add_text(f"**Signature:**")
            self.add_code_block(f"{func_name}{sig}")
        except:
            pass
        
        # Get docstring
        doc = inspect.getdoc(func)
        if doc:
            self.add_text(f"**Description:**")
            self.add_text(doc)
        else:
            self.add_text("*No docstring available*")
        
        # Get source code (first few lines)
        try:
            source = inspect.getsource(func)
            lines = source.split('\n')[:10]
            if len(lines) < len(source.split('\n')):
                lines.append("    # ... (truncated)")
            
            self.add_text(f"**Source:**")
            self.add_code_block('\n'.join(lines))
        except:
            pass
    
    def document_module(self, module_path):
        """Generate documentation for a module"""
        module_name = module_path.stem
        
        print(f"📝 Documenting: {module_name}")
        
        self.add_header(f"{module_name}.py", level=2)
        
        # Import module
        module = self.import_module(module_path)
        if module is None:
            self.add_text("❌ Failed to import module")
            return
        
        # Module docstring
        if module.__doc__:
            self.add_text(f"**Module Description:**")
            self.add_text(module.__doc__.strip())
        
        # List all functions
        functions = []
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) and not name.startswith('_'):
                functions.append((name, obj))
        
        if functions:
            self.add_text(f"\n**Functions:** {len(functions)}")
            
            for func_name, func in sorted(functions):
                self.document_function(func_name, func)
        else:
            self.add_text("\n*No public functions found*")
    
    def generate_docs(self):
        """Generate documentation for all utility modules"""
        self.add_header("StatsmodelsMasterPro - API Documentation", level=1)
        self.add_text(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        
        self.add_text("""
This document provides automatically generated API documentation for all 
utility modules in the StatsmodelsMasterPro project.
        """)
        
        # Table of Contents
        self.add_header("Table of Contents", level=2)
        
        util_files = sorted([f for f in UTILS_DIR.glob("*.py") 
                           if f.name != '__init__.py'])
        
        for i, util_file in enumerate(util_files, 1):
            self.add_text(f"{i}. [{util_file.stem}](#{util_file.stem})")
        
        # Document each module
        for util_file in util_files:
            self.document_module(util_file)
        
        # Save documentation
        output_path = DOCS_DIR / "API_DOCUMENTATION.md"
        with open(output_path, 'w') as f:
            f.write('\n'.join(self.docs))
        
        print(f"\n✅ API documentation generated: {output_path}")
        return output_path


def main():
    """Generate API documentation"""
    print("=" * 70)
    print("StatsmodelsMasterPro - API Documentation Generator")
    print("=" * 70)
    
    generator = APIDocGenerator()
    output_path = generator.generate_docs()
    
    print("\n" + "=" * 70)
    print(f"📄 Documentation saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
