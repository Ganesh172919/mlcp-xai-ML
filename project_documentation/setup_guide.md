# Setup Guide (Windows, PowerShell)

Follow these steps to get the project running in a clean Python environment.

## 1) Prerequisites
- Python 3.9+ (3.10 recommended)
- Git (optional but recommended)

Verify versions:

```powershell
python --version
git --version
```

## 2) Create and activate a virtual environment
From the repo root (`mlcp-xai-ML`):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

To deactivate later:

```powershell
deactivate
```

## 3) Install required packages
Install core dependencies used by the samples and notebook:

```powershell
pip install --upgrade pip
pip install pandas scikit-learn imbalanced-learn shap xgboost matplotlib seaborn jupyter
```

Notes:
- `shap` and `xgboost` may take a minute to build/install.
- If you use VS Code, select the `.venv` interpreter for the workspace.

## 4) Optional: Save dependencies
```powershell
pip freeze > requirements.txt
```

## 5) Data placement
- If using your own CSV, place it anywhere and pass the path to scripts.
- A raw text dataset was added under `data-all-annotations/` (see `dataset_card.md`). You’ll likely parse it into a tabular format before modeling.

You’re ready to run the samples and notebook. See `running_the_project.md` for commands.
