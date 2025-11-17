# 📦 Installation & Setup Guide

## Complete Installation Instructions for Stroke Prediction Project

---

## Prerequisites

- **Python 3.7+** (preferably 3.8 or 3.9)
- **pip** package manager
- **Git** (optional, for version control)
- **Jupyter** (will be installed with requirements)
- **Internet connection** (for downloading packages and dataset)

---

## Step-by-Step Installation

### Step 1: Verify Python Installation

Open command prompt/terminal and check:

```bash
python --version
```

Should show: `Python 3.7.x` or higher

If not installed, download from: https://www.python.org/downloads/

### Step 2: Create Virtual Environment (Recommended)

```bash
# Navigate to project directory
cd stroke_imbalance_project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

You should see `(venv)` in your command prompt.

### Step 3: Install Dependencies

**Option A: Install from requirements.txt (Recommended)**

```bash
pip install -r requirements.txt
```

**Option B: Install manually**

```bash
# Core libraries
pip install pandas numpy scikit-learn

# Imbalanced learning
pip install imbalanced-learn

# Advanced models
pip install xgboost

# Explainable AI
pip install shap lime

# Visualization
pip install matplotlib seaborn plotly

# Jupyter
pip install jupyter notebook ipykernel

# Optional: GAN-based oversampling
pip install sdv
```

### Step 4: Verify Installation

Create a test file `test_installation.py`:

```python
import pandas as pd
import numpy as np
import sklearn
import imblearn
import xgboost
import shap
import lime
import matplotlib
import seaborn

print("✅ All packages installed successfully!")
print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"imbalanced-learn: {imblearn.__version__}")
print(f"xgboost: {xgboost.__version__}")
print(f"shap: {shap.__version__}")
```

Run it:
```bash
python test_installation.py
```

### Step 5: Generate Project Structure

```bash
# Run the project builder
python BUILD_COMPLETE_PROJECT.py
```

This creates all directories and documentation files.

```bash
# Run the extended generator
python generate_complete_project.py
```

This creates source code files with complete implementations.

### Step 6: Download Dataset

1. Go to Kaggle: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

2. **If you have Kaggle account:**
   - Click "Download" button
   - Extract `healthcare-dataset-stroke-data.csv`

3. **If you don't have Kaggle account:**
   - Create free account (takes 1 minute)
   - Then download

4. **Place the file:**
   ```
   stroke_imbalance_project/
   └── data/
       └── stroke.csv  ← Rename and place here
   ```

### Step 7: Verify Project Structure

Your project should look like this:

```
stroke_imbalance_project/
├── data/
│   ├── stroke.csv              ✅ (you placed this)
│   └── README.md               ✅ (generated)
│
├── src/
│   ├── __init__.py             ✅
│   ├── data_loader.py          ✅
│   ├── preprocessing.py        ✅
│   ├── baseline_model.py       ✅
│   ├── oversampling_methods.py ✅
│   ├── xai_tools.py            ✅
│   └── evaluation.py           ✅
│
├── notebooks/
│   ├── 1_data_understanding.ipynb       (to create)
│   ├── 2_baseline_models.ipynb          (to create)
│   ├── 3_oversampling_experiments.ipynb (to create)
│   ├── 4_xai_analysis.ipynb             (to create)
│   └── 5_final_summary.ipynb            (to create)
│
├── docs/
│   ├── full_project_documentation.md    ✅
│   ├── learning_notes_for_students.md   ✅
│   └── (other documentation files)       ✅
│
├── reports/
│   └── plots/                            ✅
│
├── requirements.txt                      ✅
├── README.md                             ✅
├── PROJECT_SUMMARY.md                    ✅
└── INSTALLATION_GUIDE.md                 ✅ (this file)
```

### Step 8: Test the Code

```bash
# Test data loader
cd src
python data_loader.py

# Expected output:
# ✓ Loaded XXXX records
# ✓ All required columns present
# ✅ Dataset validation passed!
```

```bash
# Test preprocessing
python preprocessing.py

# Expected output:
# ✓ Filled XX missing BMI values
# ✅ Preprocessing complete!
```

```bash
# Test baseline models
python baseline_model.py

# Expected output:
# Training accuracy: X.XXXX
# ✅ All tests passed!
```

### Step 9: Start Jupyter

```bash
# From project root directory
cd ..
jupyter notebook
```

Browser should open automatically at `http://localhost:8888`

---

## Troubleshooting

### Issue 1: "Python is not recognized"

**Solution:** Add Python to PATH

Windows:
1. Search "Environment Variables"
2. Edit PATH
3. Add `C:\PythonXX\` and `C:\PythonXX\Scripts\`
4. Restart command prompt

### Issue 2: "pip install fails"

**Solution:** Update pip

```bash
python -m pip install --upgrade pip
```

### Issue 3: "No module named 'X'"

**Solution:** Install specific package

```bash
pip install X
```

### Issue 4: "SHAP/LIME installation fails"

**Solution:** Install dependencies first

```bash
# Windows
pip install numpy scipy scikit-learn
pip install shap lime

# Mac/Linux (might need C++ compiler)
sudo apt-get install python3-dev  # Linux
brew install gcc  # Mac
pip install shap lime
```

### Issue 5: "Dataset not found"

**Solution:** Check file path and name

```python
# In Python:
import os
os.path.exists('data/stroke.csv')  # Should return True
```

If False:
- Make sure file is named exactly `stroke.csv`
- Check it's in `data/` folder
- Check you're running from project root

### Issue 6: "Jupyter notebook not found"

**Solution:** Install jupyter

```bash
pip install jupyter notebook
```

### Issue 7: "Memory error with SHAP"

**Solution:** Reduce sample size

```python
# In notebooks:
X_sample = X_test.sample(n=100)  # Use smaller sample
shap_values = explainer.explain(X_sample)
```

---

## System Requirements

### Minimum:
- **RAM:** 4 GB
- **Storage:** 500 MB free
- **CPU:** Dual-core processor
- **OS:** Windows 7+, Mac OS X 10.10+, Linux

### Recommended:
- **RAM:** 8 GB+
- **Storage:** 2 GB free
- **CPU:** Quad-core processor
- **OS:** Windows 10, Mac OS X 10.15+, Ubuntu 18.04+

---

## Package Versions

Tested with:

```
pandas==1.3.5
numpy==1.21.6
scikit-learn==1.0.2
imbalanced-learn==0.9.1
xgboost==1.6.1
shap==0.41.0
lime==0.2.0.1
matplotlib==3.5.2
seaborn==0.11.2
jupyter==1.0.0
```

If you have version issues, install these exact versions:

```bash
pip install pandas==1.3.5 numpy==1.21.6 scikit-learn==1.0.2
```

---

## IDE Setup (Optional)

### VS Code

1. Install VS Code: https://code.visualstudio.com/
2. Install Python extension
3. Select interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter" → Choose venv
4. Install Jupyter extension for notebook support

### PyCharm

1. Install PyCharm: https://www.jetbrains.com/pycharm/
2. Open project folder
3. Configure interpreter: Settings → Project → Python Interpreter → Add → Existing Environment → Select venv

### Jupyter Lab (Alternative to Jupyter Notebook)

```bash
pip install jupyterlab
jupyter lab
```

More modern interface, same functionality.

---

## GPU Support (Optional, for faster SHAP)

If you have NVIDIA GPU:

```bash
# Install CUDA toolkit first
# Then:
pip install shap[gpu]
```

Not required, but speeds up SHAP calculations.

---

## Cloud Setup (Alternative to Local)

### Google Colab

1. Go to: https://colab.research.google.com/
2. Upload notebooks
3. Install packages in first cell:
```python
!pip install imbalanced-learn shap lime
```
4. Upload dataset manually or mount Google Drive

### Kaggle Notebooks

1. Go to: https://www.kaggle.com/
2. Create new notebook
3. Add stroke dataset (already on Kaggle!)
4. All packages pre-installed

### Azure Notebooks

1. Go to: https://notebooks.azure.com/
2. Create project
3. Upload files
4. Install requirements

---

## Updating the Project

### Get Latest Code

```bash
# If you made changes and want to reset:
git checkout .

# If you want to update to latest version:
git pull origin main
```

### Update Packages

```bash
pip install --upgrade pandas numpy scikit-learn
pip install --upgrade imbalanced-learn xgboost
pip install --upgrade shap lime matplotlib
```

---

## Uninstallation

### Remove Virtual Environment

```bash
# Deactivate first
deactivate

# Delete venv folder
rm -rf venv  # Mac/Linux
rmdir /s venv  # Windows
```

### Remove All Packages

```bash
pip uninstall -r requirements.txt -y
```

---

## Next Steps After Installation

1. ✅ Installation complete? Great!
2. 📖 Read `PROJECT_SUMMARY.md` for overview
3. 📚 Read `docs/learning_notes_for_students.md` to learn concepts
4. 🧪 Run `src/data_loader.py` to test
5. 📓 Start with `notebooks/1_data_understanding.ipynb`

---

## Getting Help

### If installation fails:

1. **Check Python version:** `python --version` (need 3.7+)
2. **Update pip:** `python -m pip install --upgrade pip`
3. **Try one package at a time:** Identify which fails
4. **Search error message:** Usually someone had same issue
5. **Check disk space:** Need at least 500MB free

### Resources:

- **Python official:** https://www.python.org/
- **pip documentation:** https://pip.pypa.io/
- **Stack Overflow:** Search your error message
- **GitHub Issues:** Check if known issue

---

## Success Checklist

Installation successful if you can:

- [x] Run `python --version` (shows 3.7+)
- [x] Run `pip list` (shows all packages)
- [x] Run `python src/data_loader.py` (loads data)
- [x] Run `jupyter notebook` (opens browser)
- [x] Import all packages without errors

If all checked, you're ready to start! 🎉

---

## Common Commands Reference

```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Deactivate
deactivate

# Install package
pip install package_name

# List installed packages
pip list

# Save current environment
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt

# Start Jupyter
jupyter notebook

# Run Python script
python script.py

# Run tests
python -m pytest  # If using pytest
```

---

**Installation support:** If stuck, re-read this guide carefully. 95% of issues are covered here!

**Ready to learn?** Head to `PROJECT_SUMMARY.md` next!

🎓 **Happy Learning!**
