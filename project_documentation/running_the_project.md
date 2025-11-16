# Running the Project

Below are copy‑paste commands for Windows PowerShell. Run from the repo root (`mlcp-xai-ML`). Ensure your virtual environment is activated (see `setup_guide.md`).

## 1) Run sample scripts

- Data loading and inspection (uses a dummy CSV if none provided):
```powershell
python project_documentation/samples/data_loading.py
```

- Baseline model on imbalanced data:
```powershell
python project_documentation/samples/baseline_model.py
```

- SMOTE oversampling + model training:
```powershell
python project_documentation/samples/oversampling_smote.py
```

- XAI with SHAP on an XGBoost model:
```powershell
python project_documentation/samples/xai_analysis.py
```

Tip: To use your own CSV and target column, import the functions in a short driver script or notebook and pass `(df, target_column, features)` accordingly.

## 2) Run the notebook

- Open the notebook in VS Code and select the `.venv` interpreter.
- File: `imbalanced_data_analysis.ipynb`
- Execute cells top‑to‑bottom.

## 3) Common issues
- Missing packages: re‑run `pip install ...` from `setup_guide.md`.
- Kernel not found in VS Code: select `Python: Select Interpreter` and choose `.venv`.
- SHAP plots not visible in headless runs: save figures instead of showing, or run inside the notebook.
