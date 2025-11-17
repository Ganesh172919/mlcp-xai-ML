"""
Complete File Creator
This script creates ALL project files at once
Run this first to set up the entire project structure
"""

import os
from pathlib import Path

# Initialize project structure
print("🚀 Creating Complete Project Structure...")

base_dir = Path(__file__).parent

# Create directories
dirs = ['data', 'notebooks', 'src', 'reports', 'reports/plots', 'docs']
for d in dirs:
    (base_dir / d).mkdir(parents=True, exist_ok=True)

# Create src/__init__.py
(base_dir / 'src' / '__init__.py').write_text('"""Stroke prediction project source code."""\n')

print("✅ Directory structure created")
print("✅ Ready for file creation")
print("\nPlease run the Jupyter notebooks in sequence:")
print("  1. notebooks/1_data_understanding.ipynb")
print("  2. notebooks/2_baseline_models.ipynb")
print("  3. notebooks/3_oversampling_experiments.ipynb")
print("  4. notebooks/4_xai_analysis.ipynb")
print("  5. notebooks/5_final_summary.ipynb")
