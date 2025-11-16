# Learning Path

A practical, step‑by‑step route to learn this project.

## 0) Setup
- Do: `project_documentation/setup_guide.md`

## 1) Core Concepts (skim → return as needed)
- Python refresher: `project_documentation/key_concepts/python_basics.md`
- Data wrangling: `project_documentation/key_concepts/numpy_pandas.md`
- Modeling basics: `project_documentation/key_concepts/scikit_learn_guide.md`
- Imbalance & oversampling: `project_documentation/key_concepts/oversampling_techniques.md`
- XAI fundamentals: `project_documentation/key_concepts/xai_methods.md`

## 2) Big Picture
- Read: `project_documentation/imbalance_oversampling_xai_project_guide.md` (overview, workflow, metrics, XAI)

## 3) Hands‑on Mini‑flow (scripts)
1. Load and inspect a CSV
   - Run: `project_documentation/samples/data_loading.py`
   - Goal: Verify schema and class imbalance.
2. Baseline on imbalanced data
   - Run: `project_documentation/samples/baseline_model.py`
   - Goal: Observe low minority recall/F1.
3. Balance with SMOTE
   - Run: `project_documentation/samples/oversampling_smote.py`
   - Goal: See recall/F1 improve on test set.
4. Explain with SHAP
   - Run: `project_documentation/samples/xai_analysis.py`
   - Goal: Understand feature effects; sanity‑check patterns.

## 4) Notebook Walkthrough
- Open: `imbalanced_data_analysis.ipynb`
- Objective: Recreate the pipeline end‑to‑end with richer visuals and notes.

## 5) Move to Your Data
- Read: `project_documentation/dataset_card.md` if using `data-all-annotations/`.
- Adapt the samples (feature selection, target definition, preprocessing).

## 6) Extend
- Try ADASYN, class weighting, or different models (Tree‑based, Linear, Ensemble).
- Add evaluation: PR curves, ROC AUC, calibration, threshold tuning.
- Deeper XAI: per‑class SHAP, cohort analysis, drift checks.

Keep brief notes as you go—copy results into a personal `notes.md` or a new notebook.
