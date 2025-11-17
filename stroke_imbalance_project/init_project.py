"""
Project Initialization Script
Creates all necessary directories for the stroke imbalance project
"""
import os
from pathlib import Path

print("🚀 Initializing Stroke Imbalance Project Structure...")
print("="*60)

# Base directory
base_dir = Path(__file__).parent

# Create directory structure
directories = [
    'data',
    'notebooks',
    'src',
    'reports',
    'reports/plots',
    'docs'
]

for dir_name in directories:
    dir_path = base_dir / dir_name
    dir_path.mkdir(parents=True, exist_ok=True)
    print(f"✅ Created: {dir_name}/")

# Create __init__.py for src package
src_init = base_dir / 'src' / '__init__.py'
src_init.write_text('"""Source code package for stroke prediction project."""\n')
print(f"✅ Created: src/__init__.py")

print("="*60)
print("✅ Project structure initialized successfully!")
print("\nNext steps:")
print("1. Run notebooks in order (1 through 5)")
print("2. Check reports/ for generated plots")
print("3. Read docs/ for comprehensive explanations")
