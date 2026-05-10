from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def set_global_seed(random_cfg: Any) -> int:
    """Apply a reproducible seed policy from cfg.random."""

    seed = int(_get(random_cfg, "seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(_get(random_cfg, "cudnn_deterministic", True))
    torch.backends.cudnn.benchmark = bool(_get(random_cfg, "cudnn_benchmark", False))

    use_deterministic = bool(_get(random_cfg, "use_deterministic_algorithms", False))
    warn_only = bool(_get(random_cfg, "deterministic_warn_only", True))
    torch.use_deterministic_algorithms(use_deterministic, warn_only=warn_only)

    python_hash_seed = _get(random_cfg, "python_hash_seed", None)
    if python_hash_seed is not None:
        os.environ["PYTHONHASHSEED"] = str(python_hash_seed)

    return seed
