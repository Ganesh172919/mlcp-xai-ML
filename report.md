# Short Report: CTGAN vs SMOTE vs Baseline (Stroke)

Fill in after running the notebooks:

- Minority prevalence (train): ...%
- Baseline metrics (test): PR AUC=..., Recall1=..., F1_macro=...
- SMOTE metrics (test): PR AUC=..., Recall1=..., delta vs baseline
- CTGAN metrics (test): PR AUC=..., Recall1=..., delta vs baseline and vs SMOTE

Key Findings:
- CTGAN impact on minority recall: +X (absolute), PR AUC: +Y; precision trade-off: ΔP=...
- Are synthetic minority samples similar to real minority? (KS/Wasserstein, UMAP): summarize.
- Decision boundary inclusion (P>0.5): real minority=..., synthetic=...

Recommendations:
- Consider threshold tuning to meet clinical recall targets; monitor precision.
- Try alternative synthesizers (TVAE, Copulas) and ensemble models.
- Perform stability checks across multiple seeds and folds.
