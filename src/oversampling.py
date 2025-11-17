from typing import Tuple, List, Optional
import os
import math
import json
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

from . import config


def apply_smote(X: np.ndarray, y: np.ndarray, target_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to reach a desired minority fraction of total.
    imblearn expects sampling_strategy as minority/majority ratio.
    Convert fraction f to ratio r = f/(1-f).
    """
    ratio = target_ratio / (1.0 - target_ratio)
    sm = SMOTE(sampling_strategy=ratio, random_state=config.SEED)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res


def _save_ctgan_model(synth, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    try:
        synth.save(os.path.join(save_dir, "ctgan.pkl"))
    except Exception:
        # Fallback: pickle full object
        import pickle
        with open(os.path.join(save_dir, "ctgan.pkl"), "wb") as f:
            pickle.dump(synth, f)


def _load_ctgan():
    # Prefer ctgan library
    try:
        from ctgan import CTGANSynthesizer
        return CTGANSynthesizer
    except Exception:  # pragma: no cover
        return None


def train_ctgan_and_generate(train_df: pd.DataFrame,
                             discrete_columns: List[str],
                             target_col: str,
                             target_ratio: float,
                             save_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Train CTGAN on training data and generate synthetic minority rows until training set
    reaches desired minority fraction (target_ratio). Tries conditional sampling; if not
    available, trains on minority subset only.
    """
    CTGANSynthesizer = _load_ctgan()
    if CTGANSynthesizer is None:
        raise ImportError("ctgan package not available. Please install 'ctgan'.")

    df = train_df.copy()
    assert target_col in df.columns

    # Count current minority in training
    n_total = len(df)
    n_min = int((df[target_col] == 1).sum())
    n_maj = n_total - n_min

    # desired f = n_min' / (n_min' + n_maj) => n_min' = f/(1-f) * n_maj
    desired_min_count = int(math.ceil((target_ratio / (1.0 - target_ratio)) * n_maj))
    n_to_generate = max(0, desired_min_count - n_min)
    if n_to_generate == 0:
        return pd.DataFrame(columns=df.columns)

    # Train CTGAN on full train with target as discrete for conditioning
    synth = CTGANSynthesizer(epochs=config.CTGAN_EPOCHS, batch_size=config.CTGAN_BATCH_SIZE)
    synth.fit(df, discrete_columns=discrete_columns)

    # Try conditional sampling (stroke==1)
    synth_df = None
    try:
        # Newer ctgan versions allow conditions as dict or DataFrame
        sampled = synth.sample(n_to_generate, conditions={target_col: 1})
        synth_df = sampled
    except Exception:
        # Fallback: train on minority-only data for unconditional sampling
        df_min = df[df[target_col] == 1].copy()
        synth_min = CTGANSynthesizer(epochs=config.CTGAN_EPOCHS, batch_size=config.CTGAN_BATCH_SIZE)
        synth_min.fit(df_min, discrete_columns=[c for c in discrete_columns if c in df_min.columns])
        synth_df = synth_min.sample(n_to_generate)
        if save_dir:
            _save_ctgan_model(synth_min, save_dir)
        else:
            _save_ctgan_model(synth_min, config.CTGAN_DIR)
        return synth_df

    # Save primary synthesizer if we got here
    if save_dir:
        _save_ctgan_model(synth, save_dir)
    else:
        _save_ctgan_model(synth, config.CTGAN_DIR)

    # Ensure target label is set to 1
    if target_col in synth_df.columns:
        synth_df[target_col] = 1
    else:
        synth_df[target_col] = 1

    return synth_df
