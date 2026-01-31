"""Feature extraction module for saflow.

This module provides scripts for:
- Welch PSD computation
- FOOOF fitting
- Complexity metrics (LZC, entropy, fractal)
- Trial classification utilities
"""

from code.utils.behavioral import classify_trials_from_vtc

__all__ = ["classify_trials_from_vtc"]
