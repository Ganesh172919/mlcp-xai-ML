SEED = 42
TARGET_COL = "stroke"
TEST_SIZE = 0.2
CV_FOLDS = 5

# Oversampling targets
SMOTE_TARGET_RATIO = 0.30  # minority fraction of total after resampling (e.g., 0.30 => 30%)
CTGAN_TARGET_RATIO = 0.30

# CTGAN settings
CTGAN_EPOCHS = 300
CTGAN_BATCH_SIZE = 500

# Paths
DATA_DIR = "data"
RAW_DIR = f"{DATA_DIR}/raw"
PROCESSED_DIR = f"{DATA_DIR}/processed"
MODELS_DIR = "models"
CTGAN_DIR = f"{MODELS_DIR}/ctgan_model"
ARTIFACTS_DIR = "artifacts"

# Classifier defaults (XGBoost)
XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": -1,
    "eval_metric": "logloss",
}
