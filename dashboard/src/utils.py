import hashlib
import json
import os
import platform
import random
from pathlib import Path

import numpy as np


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")


def environment():
    from importlib.metadata import version
    names = ["numpy", "pandas", "scipy", "scikit-learn", "torch", "streamlit"]
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {n: version(n) for n in names},
            "cpu_count": os.cpu_count(), "torch_threads": 1}
