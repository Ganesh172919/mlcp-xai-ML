# Handling Class Imbalance with Oversampling and Explainable AI (XAI)

---
## 1. Project Overview

### Theme
You are building a **binary classification** project where one class (the minority) is severely under‑represented (≤ 20% of samples). The project demonstrates:
- Training baseline models on imbalanced data
- Applying multiple **oversampling** techniques (classical + advanced)
- Retraining and comparing improvements
- Applying **Explainable AI (XAI)** to understand:  
  1. How synthetic (generated) minority samples influence decision boundaries  
  2. Which features drive predictions before & after oversampling  
  3. Whether oversampling introduces artifacts or shifts feature importance in unintended ways

### Why This Matters
In many real-world AI scenarios, the class you care about most (fraud, disease, defect, churn, intrusion) is rare. Unbalanced datasets can trick models into learning shortcuts and achieving deceptively high accuracy by mostly predicting the majority class. Oversampling plus XAI helps to:
- Recover **recall** (catch more true minority cases)
- Avoid **accuracy illusions**
- Provide **transparency** into how synthetic data affects learning
- Increase **trust** for stakeholders (auditors, clinicians, managers)

### Real-World Examples
| Domain | Majority vs Minority | Why Oversampling Helps |
|--------|---------------------|------------------------|
| Fraud Detection | 98% legitimate vs 2% fraud | Without oversampling recall is low; fraud missed |
| Disease Prediction (e.g., rare cancer) | 95% healthy vs 5% positive | Medical false negatives costly |
| Customer Churn | 85% stay vs 15% churn | Business wants to act before churn |
| Manufacturing Defect | 97% good parts vs 3% defective | Need early detection for quality control |
| Cyber Intrusion | 99% normal vs 1% attack | Security teams need high sensitivity |

### Simple English Explanation
Imagine you have a basket of 100 apples and only 5 oranges. If you train someone to distinguish apples vs oranges by showing mostly apples, they will learn to always say "apple" and appear 95% correct—but they fail the real task (identifying oranges). Oversampling creates more believable oranges (synthetic samples) so the learner pays attention to what makes oranges unique.

---
## 2. Dataset Selection Guide

### How to Choose an Imbalanced Dataset
Select a dataset where the minority class proportion is small (≤ 20%). Look for:
- Naturally imbalanced domains (fraud, anomaly, medical, reliability)
- Clear target variable with minority interest
- Enough minority samples to permit interpolation (for SMOTE you generally want ≥ several dozen)

### Steps to Inspect Class Distribution
1. Load dataset into a DataFrame.  
2. Identify target column (`y`).  
3. Count class frequencies: `df[y].value_counts()`  
4. Compute percentage: `value_counts(normalize=True) * 100`  
5. Visualize with bar plot or pie chart.

### Reading Features & Target
- Features: numeric, categorical, ordinal, binary, time-based, text (consider preprocessing)  
- Target: binary label (0 = majority, 1 = minority)  
- Check missing values, outliers, skewness, cardinality of categoricals

### Suitability Checklist
| Criterion | Good Sign |
|-----------|-----------|
| Minority ratio | < 20% |
| Sample size | Total > 1000 preferred (not mandatory) |
| Feature richness | ≥ 5 meaningful features |
| Noise level | Not extreme (excess noise hurts synthetic generation) |
| Domain relevance | Real business or scientific use-case |

### Example Distribution
"90% non-fraud vs 10% fraud" – baseline model may predict "non-fraud" always achieving 90% accuracy yet 0% fraud recall.

### Sources for Datasets
- **Kaggle**: Fraud detection, credit card default, churn, healthcare claims  
- **UCI Repository**: Thyroid disease, credit approval, bank marketing  
- **OpenML**: Imbalanced tasks tagged with minority proportion  
- **Government portals**: Health surveillance, transport safety

Provide dataset citation in README: include source URL, license, description, retrieval date.

---
## 3. Concepts I Must Learn Before Starting

### Core Imbalance Concepts
- **Class Imbalance**: When one class has far fewer samples than the other. Leads to biased learning and misleading accuracy.
- **Oversampling**: Techniques to increase number of minority samples (real or synthetic) to balance classes.
- **SMOTE (Synthetic Minority Over-sampling Technique)**: Generates new minority samples by interpolating between a point and its nearest neighbors in feature space.
- **ADASYN (Adaptive Synthetic Sampling)**: Focuses generation on minority samples that are harder to learn (located near majority class boundary). Emphasizes local density complexity.
- **GAN-based Oversampling**: Uses Generative Adversarial Networks to synthesize realistic minority samples by learning the distribution (e.g., CTGAN for tabular data). More powerful but risk of mode collapse or unrealistic artifacts.

### Data Splitting
- **Train-Test Split**: Always split BEFORE applying oversampling to avoid data leakage. Oversample only the training portion.
- **Stratification**: Ensure both train and test retain original class proportions (use `stratify=y`).

### Model Evaluation Metrics (Beyond Accuracy)
- **Recall (Sensitivity)**: Fraction of actual minority instances correctly identified. Critical when missing a minority case is costly.
- **Precision**: Among predicted minorities, how many are correct. Important when false alarms are costly.
- **F1 Score**: Harmonic mean of precision & recall. Balances both.
- **Balanced Accuracy**: Average of recall for each class (`(TPR + TNR)/2`). Mitigates majority bias.
- **Confusion Matrix**: Table of TP, FP, FN, TN counts; essential to see tradeoffs.
- **ROC AUC**: Measures ranking quality; can be optimistic with imbalance.  
- **PR AUC**: More informative for highly skewed datasets.

### Explainable AI (XAI) Basics
- **Feature Importance**: Global ranking of features impacting predictions (e.g., tree impurity, permutation importance).  
- **SHAP (SHapley Additive exPlanations)**: Game-theoretic approach assigning each feature a contribution value per prediction; consistent across model types (with appropriate explainer).  
- **LIME (Local Interpretable Model-Agnostic Explanations)**: Perturbs data locally around an instance, trains a simple surrogate to approximate model behavior nearby.  
- **Decision Boundaries**: Visual depiction (often in 2D or 3D projections) of how the classifier separates classes; can shift after oversampling.
- **PDP (Partial Dependence Plot)**: Shows average effect of a feature on predicted outcome.
- **ICE (Individual Conditional Expectation)**: Shows per-instance trajectories for feature effect.

### Mini Learning Syllabus
| Topic | Goals | Key Functions |
|-------|-------|---------------|
| Python Basics | Syntax, loops, functions, OOP | `def`, `class`, list/dict comprehensions |
| NumPy | Efficient numeric arrays | `np.array`, `np.mean`, broadcasting |
| Pandas | DataFrame manipulation | `read_csv`, `groupby`, `merge`, `value_counts` |
| Matplotlib & Seaborn | Visualization | `plt.bar`, `sns.countplot`, `sns.pairplot` |
| Scikit-Learn Basics | Modeling pipeline | `train_test_split`, `Pipeline`, `StandardScaler` |
| Imbalanced-Learn | Oversampling algorithms | `SMOTE`, `ADASYN`, `RandomOverSampler` |
| XAI Libraries | Interpretability | `shap.TreeExplainer`, `lime.lime_tabular.LimeTabularExplainer` |

---
## 4. Folder Structure for the Project
A clear modular structure improves reproducibility:
```
project_root/
  data/
    raw/
      dataset.csv
    processed/
      train.csv
      test.csv
  notebooks/
    01_exploration.ipynb
    02_baseline_model.ipynb
    03_oversampling_experiments.ipynb
    04_xai_analysis.ipynb
  src/
    __init__.py
    config.py
    data_loading.py
    preprocessing.py
    oversampling.py
    modeling.py
    evaluation.py
    xai_explainers.py
    visualization.py
  reports/
    figures/
    metrics/
  models/
    baseline_model.pkl
    smote_model.pkl
  project_documentation/
    imbalance_oversampling_xai_project_guide.md
    README.md
    samples/
      data_loading.py
      oversampling_examples.py
      baseline_model.py
      xai_demo.py
  requirements.txt
  pyproject.toml (optional)
  README.md (main)
```

---
## 5. Step-by-Step Guide: How to Start Working

1. **Import Dataset**: Place raw CSV in `data/raw`; load with `pd.read_csv`.  
2. **Explore Data**: Use `df.head()`, `df.info()`, `df.describe()`. Check missing values.  
3. **Visualize Class Imbalance**: `sns.countplot(x=target)` or bar chart of `value_counts()`.  
4. **Create Baseline Model**: Split data (`train_test_split(stratify=y)`), train a simple model (e.g., `LogisticRegression`).  
5. **Choose Oversampling Algorithm**: Start with `RandomOverSampler`, then `SMOTE`, then `ADASYN`. Optionally advanced (BorderlineSMOTE, KMeansSMOTE, GAN).  
6. **Apply Oversampling**: Fit oversampler on `X_train, y_train`. Do NOT oversample test set.  
7. **Train Improved Model**: Retrain same model architecture on balanced training data.  
8. **Compare Metrics**: Build a table: baseline vs oversampled recall, precision, F1, balanced accuracy.  
9. **Apply XAI**: Use SHAP or LIME to inspect changes in feature contributions. Compare baseline vs oversampled SHAP summary plots.  
10. **Write Conclusion**: Summarize improvements, tradeoffs (precision vs recall), potential overfitting or synthetic data artifacts.

---
## 6. Baseline Model Explanation

### Purpose
A baseline anchor clarifies how serious imbalance effects are. It sets the reference for improvement when adding oversampling.

### Training on Imbalanced Data
- Fit simple classifier (Logistic Regression / RandomForest) directly on original training set.  
- Evaluate on untouched test set.  
- Observe often: High accuracy (dominated by majority class), low minority recall.

### Why Accuracy Fails
Accuracy counts all correct predictions equally; with 90% majority class, predicting majority always yields ≥90% accuracy. This hides failure to identify minority cases.

### Real-Life Analogy
Imagine hospital triage identifies 95% of healthy people correctly but misses most actual emergencies. Accuracy looks good, but the mission fails.

### Pseudocode
```python
# Baseline pseudocode
X, y = load_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
model = LogisticRegression(class_weight=None)  # start neutral
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```
Consider running also with `class_weight='balanced'` (reweights classes) *without* oversampling to compare strategies.

---
## 7. Oversampling Technique Implementation

### Random Oversampling
Simple duplication of minority samples until classes balanced. Risk: Overfitting repeated minority points.
```python
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X_train, y_train)
```

### SMOTE – Intuition
Pick a minority sample A. Find its k nearest minority neighbors (say B). Create synthetic point along the line segment between A and B: `A + λ*(B - A)` where λ ∈ [0,1]. This smooths minority manifold and adds diversity.
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)
X_smote, y_smote = smote.fit_resample(X_train, y_train)
```
Variants: `BorderlineSMOTE`, `SVMSMOTE`, `KMeansSMOTE`, `SMOTENC` (categorical), `SMOTEN` (nominal). Use variants when decision boundary complexity or categorical features matter.

### ADASYN – Intuition
Focuses on samples near majority decision boundary (harder ones). Generates more synthetic points where learning difficulty is higher, adaptive density.
```python
from imblearn.over_sampling import ADASYN
ada = ADASYN(random_state=42, n_neighbors=5)
X_ada, y_ada = ada.fit_resample(X_train, y_train)
```

### Combining With Cleaning (Hybrid)
- **SMOTE + TomekLinks**: Generate then remove ambiguous samples.  
- **SMOTEENN**: Synthetic generation + Edited Nearest Neighbours cleaning.
```python
from imblearn.combine import SMOTETomek, SMOTEENN
smote_tomek = SMOTETomek(random_state=42)
X_st, y_st = smote_tomek.fit_resample(X_train, y_train)
```

### GAN-Based Oversampling
Use a generator network (learn minority distribution) and discriminator. For tabular:
- **CTGAN** (Conditional Tabular GAN)  
- **TGAN**, **WGAN**, **TableGAN**  
Pros: Captures complex feature interactions. Cons: More compute, risk unrealistic samples. Validate synthetic quality (t-SNE cluster coherence, correlation preservation).

### Citing Research Papers
In documentation include references (APA/IEEE):
- Chawla et al. (2002) – SMOTE  
- He et al. (2008) – ADASYN  
- Goodfellow et al. (2014) – GAN  
Add DOI links where possible.

### Implementation Workflow
1. Fit baseline model.  
2. Choose oversampler(s).  
3. Apply oversampling ONLY to train.  
4. Train same model architecture.  
5. Evaluate metrics.  
6. Record results in DataFrame (rows: method, cols: metrics).  
7. Visualize improvements (bar chart of recall, F1).  
8. Perform XAI on best model.

### Code Snippet (Unified Pipeline)
```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

methods = {
    'baseline': None,
    'random': RandomOverSampler(random_state=42),
    'smote': SMOTE(random_state=42),
    'adasyn': ADASYN(random_state=42)
}

results = []
for name, oversampler in methods.items():
    if oversampler is None:
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000))
        ])
        pipe.fit(X_train, y_train)
    else:
        pipe = ImbPipeline([
            ('oversample', oversampler),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000))
        ])
        pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    results.append({
        'method': name,
        'f1': f1_score(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred)
    })

import pandas as pd
print(pd.DataFrame(results))
```

### Compare Results
Construct table: increases in recall & F1 should be evident. Watch for precision drop (typical tradeoff).

---
## 8. Explainable AI (XAI) Analysis

### What XAI Means
Tools that clarify model reasoning: which features matter, how changes in inputs shift outputs, and whether synthetic data changed predictive logic.

### Applying SHAP
SHAP assigns each feature a contribution value to each prediction. After oversampling, inspect if minority-relevant features gain higher absolute SHAP values (desired) or if noise features inflate influence (problem).
```python
import shap
# Assume best_pipe is trained oversampled model
explainer = shap.Explainer(best_pipe.named_steps['clf'], best_pipe.named_steps['scaler'].transform(X_train))
shap_values = explainer(best_pipe.named_steps['scaler'].transform(X_test))
shap.summary_plot(shap_values, features=X_test, feature_names=X_test.columns)
```
(Note: For tree models use `shap.TreeExplainer(model)` directly.)

### Using LIME
LIME explains individual predictions locally; compare interpretations for original vs synthetic-like samples (pick minority instances).  
Metaphor: LIME is like a coach replaying one specific move slowly to see which body actions led to success.

### Real vs Synthetic Boundary Visualization
- Reduce dimensionality with **PCA** or **t-SNE**.  
- Plot original minority vs synthetic points to ensure they blend naturally rather than forming disconnected clusters.  
Cricket analogy: Synthetic points should behave like trained junior players integrated with senior players, not like random spectators.

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
pca = PCA(n_components=2)
X_emb = pca.fit_transform(X_smote)
plt.scatter(X_emb[:,0], X_emb[:,1], c=y_smote, cmap='coolwarm', alpha=0.6)
plt.title('PCA Projection After SMOTE')
plt.show()
```

### t-SNE for Local Structure
```python
from sklearn.manifold import TSNE
X_vis = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_smote)
plt.scatter(X_vis[:,0], X_vis[:,1], c=y_smote, cmap='coolwarm', alpha=0.6)
plt.title('t-SNE After Oversampling')
plt.show()
```

### SHAP for Synthetic Minority Instances
Select a subset of synthetic points (if tracked) and compare mean absolute SHAP values vs original minority subset. Look for alignment (similar ranking) — large divergence may indicate unrealistic synthetic generation.

### Expected Insights
- Feature ranking stabilizes or gives more weight to minority-discriminative signals.  
- Balanced model surfaces previously overshadowed features.  
- Decision boundary becomes smoother and less biased toward majority region.

---
## 9. What Sir Assisted Us With (Project Context)

**Instructions Overview**: Guidance emphasized educational understanding of imbalance handling, responsible use of synthetic data, and transparent explanation of model behavior.

**Deadlines**: Interim checkpoint for baseline model, second for oversampling comparison, final evaluation day includes XAI presentation.

**Evaluation Day**: Teams present dataset choice rationale, metric improvements, and interpretability visuals.

**Group Requirement**: Collaborative division—data curation, modeling, oversampling experimentation, XAI analyses. Each member rotates tasks for fairness.

**Fairness Between Teams**: Consistent evaluation rubric:  
1. Data quality & justification  
2. Methodological rigor (preventing leakage)  
3. Metric transparency (not just accuracy)  
4. Interpretability clarity  
5. Ethical considerations (not misrepresenting synthetic data)

**Three Tasks Defined by Sir**:
1. **Dataset Selection**: Pick an imbalanced, meaningful dataset; document class ratios.  
2. **Oversampling**: Implement at least three methods; justify choice; show metrics.  
3. **XAI Analysis**: Provide SHAP/LIME visualizations; interpret changes due to oversampling.

---
## 10. Full Learning (Beginner → Advanced)

### Timeline Suggestion
| Day | Focus | Outcomes |
|-----|-------|----------|
| Day 1 | Python refresher + Pandas | Load & inspect dataset |
| Day 2 | EDA + visualize imbalance | Baseline metrics recorded |
| Day 3 | Implement Random + SMOTE | First improvement table |
| Day 4 | Add ADASYN + hybrids | Compare precision-recall tradeoffs |
| Day 5 | Tune model hyperparameters | Optimized oversampled model |
| Day 6 | Apply SHAP + LIME | Interpretation plots generated |
| Day 7 | Validate synthetic quality (t-SNE/PCA) | Confirm realism |
| Day 8 | Draft report sections | Documentation skeleton complete |
| Day 9 | Polish metrics & visuals | Final consolidated results |
| Day 10 | Rehearse presentation | Team ready for evaluation |

### Coding Skills Inventory (Scikit-Learn & More)
- Data Loading: `pd.read_csv`, `pd.to_datetime`  
- Cleaning: `dropna`, `fillna`, `astype`, categorical encoding (`OneHotEncoder`, `OrdinalEncoder`)  
- Feature Scaling: `StandardScaler`, `MinMaxScaler`, `RobustScaler`  
- Transformers: `ColumnTransformer`, `Pipeline`  
- Model Selection: `train_test_split`, `StratifiedKFold`, `GridSearchCV`, `RandomizedSearchCV`, `cross_val_score`  
- Models: `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`, `XGBClassifier` (external), `SVC`, `KNeighborsClassifier`  
- Metrics: `classification_report`, `confusion_matrix`, `precision_recall_curve`, `roc_curve`, `roc_auc_score`, `f1_score`, `balanced_accuracy_score`, `average_precision_score`  
- Probability Calibration: `CalibratedClassifierCV`  
- Imbalanced-Learn: `RandomOverSampler`, `SMOTE`, `BorderlineSMOTE`, `SMOTENC`, `SMOTEN`, `KMeansSMOTE`, `SVMSMOTE`, `ADASYN`, `SMOTEENN`, `SMOTETomek`, `ClusterCentroids`, `TomekLinks`, `EditedNearestNeighbours`, `BalancedRandomForestClassifier`, `EasyEnsembleClassifier`  
- XAI Libraries: `shap`, `lime`, `eli5`, `sklearn.inspection.partial_dependence`, `permutation_importance`  
- Dimensionality Reduction: `PCA`, `TSNE`, `UMAP` (external)  
- Visualization: `matplotlib.pyplot`, `seaborn`, `plotly.express`

### Advanced Oversampling Considerations
- Parameter tuning for k neighbors in SMOTE  
- Checking feature distributions (KS test) for synthetic vs real  
- Avoiding overlap artifacts (synthetics invading majority clusters)  
- Chain with noise removal (TomekLinks) to clean boundary

---
## 11. How to “Figure Out Things” Step-by-Step

### Research Papers
1. Search keywords: "SMOTE DOI", "ADASYN class imbalance", "GAN tabular synthetic".  
2. Skim abstract → identify objective, method, results.  
3. Read methodology section for algorithm steps; summarize in own words.

### Understand New Algorithms
- Break into components: input, transformation, output.  
- Try small numeric example manually (e.g., 2D points for SMOTE).  
- Reproduce baseline result then apply algorithm incrementally.

### Debug Code
- Print shapes before & after oversampling.  
- Confirm no test leakage: ensure oversampler not fit on test.  
- Verify class distribution after resampling.  
- Use assertions: `assert set(np.unique(y_res)) == {0,1}`.

### Compare Datasets
- Summaries `df.describe()` before & after scaling.  
- Distribution plots (histogram) for key features original vs synthetic.  
- Correlation matrices: difference heatmaps.

### Reason About Results
- If recall rises but precision drops drastically: consider threshold tuning, cost-sensitive methods.  
- If SHAP feature importance shifts dramatically: verify synthetic realism.  
- If F1 stagnates: consider different model architecture or hybrid rebalancing.

### Evaluate Model Output
- Use confusion matrix to isolate minority prediction errors.  
- Plot Precision-Recall curve; analyze area under curve.  
- Calibration curve to check predicted probabilities reliability.

### Practical Advice
- Start simple (RandomOverSampler) before complex (GAN).  
- Keep reproducibility (set random seeds).  
- Document each experiment run with parameters & metrics.  
- Automate metric comparison in a DataFrame.  
- Save top models with `joblib.dump`.

---
## 12. Final Summary Section

### Project Contains
- Dataset selection rationale, imbalance quantification  
- Baseline modeling & evaluation  
- Multiple oversampling techniques, comparative metrics  
- XAI interpretations (SHAP, LIME, PD)  
- Synthetic sample quality assessment (PCA/t-SNE)  
- Structured documentation & reproducible code skeletons

### Why Oversampling Matters
It rescues minority signal from being drowned by majority, enabling models to learn discriminative patterns and reducing false negatives in critical domains.

### Why XAI Matters
It verifies that improvements come from meaningful feature learning, not artifacts. Builds stakeholder trust, surfaces feature reasoning shifts, and highlights potential overfitting to synthetics.

### Project Utility in AI & Data Science
Demonstrates end-to-end responsible handling of skewed data: data integrity, methodological rigor, evaluation sophistication, interpretability—all core professional competencies.

### Skills Gained
- Data profiling & EDA  
- Advanced resampling strategies  
- Metric selection & interpretation with imbalance  
- Pipeline engineering & reproducibility  
- Model interpretability & result communication  
- Research literacy and experiment organization

---
## Extended Reference: Quick Function Catalog

### Data & Preprocessing
`pd.read_csv`, `df.info`, `df.describe`, `df.isna().sum`, `df.value_counts`, `OneHotEncoder`, `StandardScaler`, `ColumnTransformer`, `Pipeline`.

### Modeling
`LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`, `XGBClassifier` (external), `SVC(probability=True)`, `KNeighborsClassifier`, `LightGBM` (external), `BalancedRandomForestClassifier`.

### Evaluation
`classification_report`, `confusion_matrix`, `roc_auc_score`, `precision_recall_curve`, `average_precision_score`, `f1_score`, `balanced_accuracy_score`, `accuracy_score`, `recall_score`, `precision_score`, `cohen_kappa_score`.

### Resampling (Imbalanced-Learn)
`RandomOverSampler`, `SMOTE`, `ADASYN`, `BorderlineSMOTE`, `SVMSMOTE`, `KMeansSMOTE`, `SMOTENC`, `SMOTEN`, `SMOTETomek`, `SMOTEENN`, `ClusterCentroids`, `TomekLinks`, `EditedNearestNeighbours`.

### XAI
`shap.Explainer`, `shap.TreeExplainer`, `lime.lime_tabular.LimeTabularExplainer`, `permutation_importance`, `PartialDependenceDisplay.from_estimator`, `eli5.explain_weights`.

### Dimensionality Reduction & Visualization
`PCA`, `TSNE`, `UMAP`, `sns.countplot`, `sns.heatmap`, `sns.barplot`, `plt.scatter`, `plotly.express.scatter`, `plotly.express.bar`.

---
## Ethical & Practical Considerations
- Avoid misrepresenting synthetic samples as real data.  
- Disclose oversampling approach in reports.  
- Monitor for performance inflation due to leakage.  
- Interpretability should highlight limitations (uncertainty in minority predictions).  
- Ensure fairness: oversampling must not distort demographic subgroup distributions (check subgroup metrics).

---
## Next Steps
- Implement notebooks incrementally following timeline.  
- Extend with threshold tuning (adjust decision threshold to optimize F1 or recall).  
- Explore cost-sensitive learning (`class_weight` or custom loss).  
- Evaluate ensemble of models (e.g., EasyEnsembleClassifier).  
- Package code into reusable module if needed.

---
### End of Guide
This document is intentionally expansive—use it as a playbook and adapt per dataset specifics.
