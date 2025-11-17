# Stroke Imbalance Project: Baseline, SMOTE, and CTGAN + XAI

This project builds a baseline classifier for the imbalanced Kaggle Stroke Prediction Dataset, benchmarks SMOTE and CTGAN-based oversampling, and analyzes effects via SHAP and UMAP.

- Dataset: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
- Main notebooks:
  - `notebooks/1_experiment_pipeline.ipynb`: Full experiment pipeline, metrics, curves, calibration, and artifacts.
  - `notebooks/2_xai_analysis.ipynb`: SHAP comparisons, UMAP projections, statistical distances, decision-boundary checks.
- Artifacts and models are saved under `artifacts/` and `models/`.

## Setup (Windows, PowerShell)

1) Install Python 3.8+ and create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

2) Configure Kaggle API credentials (required for programmatic download):

- Download `kaggle.json` from your Kaggle account (Account > API > Create New Token).
- Place it in `%USERPROFILE%\.kaggle\kaggle.json`.
- Ensure permissions if needed.

3) Run the notebooks in order:

- `notebooks/1_experiment_pipeline.ipynb`
- `notebooks/2_xai_analysis.ipynb`

The pipeline will create:
- `artifacts/data_summary.json`, `artifacts/metrics.json`
- `artifacts/synthetic_ctgan_minority.csv`
- Models under `models/` (CTGAN model saved under `models/ctgan_model/`).

## Notes and Practices

- Stratified train/test split and 5-fold CV for model selection.
- Avoid leakage: SMOTE and CTGAN applied on training data only.
- Reproducibility via a single config with random seeds.
- Evaluation includes ROC AUC, PR AUC, F1 (macro and per-class), precision, recall, and confusion matrix; plus calibration and threshold analysis.

## Citations

- N. V. Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique", 2002. Journal of Artificial Intelligence Research.
- Lei Xu et al., "Modeling Tabular Data using Conditional GAN (CTGAN)", NeurIPS 2019.

## Short Report

See `report.md` for a concise summary template. After running the notebooks, fill in numeric improvements (e.g., minority recall +X, PR AUC +Y) and recommendations.
