from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from . import config


CATEGORICAL_CANDIDATES = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]
BINARY_AS_NUMERIC = ["hypertension", "heart_disease"]


def get_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    target = config.TARGET_COL
    cols = [c for c in df.columns if c != target]
    cat = [c for c in cols if c in CATEGORICAL_CANDIDATES]
    num = [c for c in cols if c not in cat]
    return num, cat


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor


def fit_transform(preprocessor: ColumnTransformer, X_train: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    Xtr = preprocessor.fit_transform(X_train)
    feature_names = []
    # Numeric names
    num_names = preprocessor.transformers_[0][2]
    feature_names.extend(num_names)

    # Cat names from OneHot
    cat_pipeline = preprocessor.transformers_[1][1]
    ohe = cat_pipeline.named_steps["onehot"]
    cat_feature_names = ohe.get_feature_names_out(preprocessor.transformers_[1][2]).tolist()
    feature_names.extend(cat_feature_names)

    return Xtr, feature_names


def transform(preprocessor: ColumnTransformer, X: pd.DataFrame) -> np.ndarray:
    return preprocessor.transform(X)


def ctgan_prepare_training_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare a DataFrame for CTGAN: impute missing (median/mode), keep original dtypes,
    and return list of discrete columns as CTGAN expects.
    """
    df_ctgan = df.copy()
    # Simple imputations consistent with build_preprocessor
    for col in df_ctgan.columns:
        if col == config.TARGET_COL:
            continue
        if df_ctgan[col].dtype == "O" or col in CATEGORICAL_CANDIDATES:
            mode = df_ctgan[col].mode(dropna=True)
            fill = mode.iloc[0] if not mode.empty else "unknown"
            df_ctgan[col] = df_ctgan[col].fillna(fill)
        else:
            med = df_ctgan[col].median()
            df_ctgan[col] = df_ctgan[col].fillna(med)

    discrete_columns = list(CATEGORICAL_CANDIDATES) + [config.TARGET_COL]
    # Include known binary numeric columns as discrete for CTGAN stability
    for b in BINARY_AS_NUMERIC:
        if b in df_ctgan.columns:
            discrete_columns.append(b)

    # De-duplicate
    discrete_columns = sorted(list(set([c for c in discrete_columns if c in df_ctgan.columns])))
    return df_ctgan, discrete_columns
