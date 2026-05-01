from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional


def load_flash_kmeans() -> Optional[ModuleType]:
    try:
        return importlib.import_module("flash_kmeans")
    except ImportError:
        pass

    repo_root = Path(__file__).resolve().parents[1]
    local_ref = repo_root / "third_party" / "flash-kmeans"
    if not local_ref.is_dir():
        return None

    local_ref_str = str(local_ref)
    if local_ref_str not in sys.path:
        sys.path.insert(0, local_ref_str)

    try:
        return importlib.import_module("flash_kmeans")
    except ImportError:
        return None
