from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from xgboost import XGBClassifier

from . import config


def get_classifier(scale_pos_weight: Optional[float] = None) -> XGBClassifier:
    params = dict(config.XGB_PARAMS)
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = float(scale_pos_weight)
    return XGBClassifier(**params)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
    y_pred = (y_prob >= 0.5).astype(int)
    roc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1]
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "roc_auc": float(roc),
        "pr_auc": float(pr_auc),
        "precision_class0": float(precision[0]),
        "recall_class0": float(recall[0]),
        "f1_class0": float(f1[0]),
        "precision_class1": float(precision[1]),
        "recall_class1": float(recall[1]),
        "f1_class1": float(f1[1]),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "confusion_matrix": cm,
    }


def cross_validate_model(X: np.ndarray, y: np.ndarray, scale_pos_weight: Optional[float] = None) -> Dict:
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    metrics_list = []
    for train_idx, val_idx in skf.split(X, y):
        Xtr, Xval = X[train_idx], X[val_idx]
        ytr, yval = y[train_idx], y[val_idx]
        clf = get_classifier(scale_pos_weight=scale_pos_weight)
        clf.fit(Xtr, ytr)
        yval_prob = clf.predict_proba(Xval)[:, 1]
        metrics_list.append(compute_metrics(yval, yval_prob))

    # Aggregate means
    agg = {}
    for key in metrics_list[0].keys():
        if key == "confusion_matrix":
            continue
        agg[key] = float(np.mean([m[key] for m in metrics_list]))
    return agg


def train_and_eval(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                   scale_pos_weight: Optional[float] = None) -> Tuple[Dict, Dict, XGBClassifier]:
    # CV metrics
    cv_metrics = cross_validate_model(X_train, y_train, scale_pos_weight=scale_pos_weight)
    # Fit final
    clf = get_classifier(scale_pos_weight=scale_pos_weight)
    clf.fit(X_train, y_train)
    y_test_prob = clf.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, y_test_prob)
    return cv_metrics, test_metrics, clf
