"""Stage 2: FOOOF fitting with on-demand IN/OUT classification (UNIFIED: sensor/source/atlas).

This script works across all analysis spaces using the same code.
The `--space` parameter determines which Welch PSDs to load.

Usage:
    python -m code.features.compute_fooof --subject 04 --run 02 --space sensor
    python -m code.features.compute_fooof --subject 04 --run 02 --space source --inout-bounds 50 50

Author: Claude (Anthropic)
Date: 2026-01-30
"""

import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from fooof import FOOOFGroup

from code.features.utils import classify_trials_from_vtc
from code.utils.config import load_config
from code.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)
