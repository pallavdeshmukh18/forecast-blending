"""
Global reproducibility and experiment provenance management.
Provides deterministic seed management across numpy, random, torch, and scikit-learn.
"""

import os
import random
import platform
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def set_global_seed(seed: int = 42) -> None:
    """
    Set random seeds for python random, numpy, and torch to guarantee reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_experiment_metadata(
    seed: int = 42,
    config_version: str = "0.6.0",
    model_version: str = "0.6.0",
    dataset_version: str = "synthetic-v1",
    training_period: Optional[str] = None,
    validation_period: Optional[str] = None,
    test_period: Optional[str] = None,
    extra_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Construct standardized reproducibility metadata dictionary for an experiment run.
    """
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "config_version": config_version,
        "model_version": model_version,
        "dataset_version": dataset_version,
        "training_period": training_period,
        "validation_period": validation_period,
        "test_period": test_period,
        "system_info": {
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__ if TORCH_AVAILABLE else "N/A",
        },
        "parameters": extra_params or {},
    }
    return meta


def save_experiment_manifest(metadata: Dict[str, Any], filepath: str | Path) -> None:
    """
    Persist experiment metadata manifest as JSON.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
