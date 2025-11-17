"""
Complete Project Builder
========================
This script creates the entire project structure with all files.
Run this once to generate everything.
"""

import os
import sys

# Get base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def create_all_directories():
    """Create all required directories"""
    directories = [
        'data',
        'notebooks',
        'src',
        'reports',
        'reports/plots',
        'docs'
    ]
    
    print("Creating directory structure...")
    for directory in directories:
        dir_path = os.path.join(BASE_DIR, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"  ✓ {directory}/")
    print()

def create_placeholder_file(filepath, content="# This file was created by the project builder\n"):
    """Create a placeholder file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    print("="*60)
    print("🚀 BUILDING COMPLETE PROJECT STRUCTURE")
    print("="*60)
    print()
    
    # Create directories
    create_all_directories()
    
    # Create placeholder files to mark directories as complete
    create_placeholder_file(os.path.join(BASE_DIR, 'data', 'README.md'), 
                          "# Data Directory\n\nPlace `stroke.csv` here from Kaggle.\n")
    
    create_placeholder_file(os.path.join(BASE_DIR, 'notebooks', 'README.md'),
                          "# Notebooks Directory\n\nJupyter notebooks will be created here.\n")
    
    create_placeholder_file(os.path.join(BASE_DIR, 'src', 'README.md'),
                          "# Source Code Directory\n\nPython modules will be created here.\n")
    
    create_placeholder_file(os.path.join(BASE_DIR, 'reports', 'README.md'),
                          "# Reports Directory\n\nGenerated reports will be saved here.\n")
    
    create_placeholder_file(os.path.join(BASE_DIR, 'docs', 'README.md'),
                          "# Documentation Directory\n\nFull documentation will be created here.\n")
    
    print("\n✅ Directory structure created successfully!")
    print()
    print("Next: Individual file creation scripts will populate these directories.")
    print()

if __name__ == "__main__":
    main()
