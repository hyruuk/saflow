"""Pytest bootstrap for the repository's historical ``code`` package name."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

loaded_code = sys.modules.get("code")
if loaded_code is not None and not hasattr(loaded_code, "__path__"):
    del sys.modules["code"]
