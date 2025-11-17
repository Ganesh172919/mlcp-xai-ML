import json
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np

from . import config


def ensure_dirs() -> None:
    for p in [config.DATA_DIR, config.RAW_DIR, config.PROCESSED_DIR, config.MODELS_DIR, config.CTGAN_DIR, config.ARTIFACTS_DIR]:
        Path(p).mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = None) -> None:
    s = seed if seed is not None else config.SEED
    random.seed(s)
    np.random.seed(s)


def save_json(obj: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
