import os
from pathlib import Path

# Create all necessary directories
base_dir = Path(__file__).parent

dirs = [
    base_dir / 'data',
    base_dir / 'notebooks',
    base_dir / 'src',
    base_dir / 'reports' / 'plots',
    base_dir / 'docs'
]

for dir_path in dirs:
    dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Created: {dir_path}")

print("\n✅ All directories created successfully!")
