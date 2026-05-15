"""Resolve which trials/windows to drop, given a configurable rejection rule.

The stats and classification loaders all need the same decision: for one run's
``trial_metadata``, which rows are "bad" and should be excluded? Historically
that was just the ``bad_ar2`` column. This module generalises it to a
configurable rule (``analysis.bad_trial_rule`` in config.yaml) so AR1-only,
AR2-only, their union, and an optional interpolation-count criterion can all
be selected without touching loader code.

Feature ``trial_metadata`` may carry:
  - ``bad_ar2``  : autoreject second pass (post-ICA) bad flag. Always present
                   on current feature files.
  - ``bad_ar1``  : autoreject first pass (pre-ICA) bad flag. Present only after
                   running ``code.utils.backfill_ar1`` (or re-extracting
                   features from raws that carry BAD_AR1 annotations).
  - ``n_interp`` : number of channels autoreject's second pass interpolated for
                   this trial/window (windowed mode: max over the window).
                   Cannot be backfilled — present only on features extracted
                   from a fresh preprocessing pass that emits it.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np

logger = logging.getLogger(__name__)

VALID_RULES = ("ar1", "ar2", "union")

# Modules warn at most once per missing column so a 32-subject load does not
# print the same caveat 192 times.
_warned: set = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        logger.warning(msg)
        _warned.add(key)


def validate_rule(rule: str) -> str:
    """Return ``rule`` lower-cased, or raise ValueError if unknown."""
    r = str(rule).lower()
    if r not in VALID_RULES:
        raise ValueError(
            f"Unknown bad_trial_rule '{rule}'. Choose one of: {', '.join(VALID_RULES)}"
        )
    return r


def _get_flag(meta: Mapping[str, Any], key: str, run_len: int) -> np.ndarray:
    """Return ``meta[key]`` as a bool array of length ``run_len``, or all-False."""
    if key in meta and meta[key] is not None:
        arr = np.asarray(meta[key], dtype=bool)
        if len(arr) == run_len:
            return arr
        _warn_once(
            f"len_{key}",
            f"'{key}' length {len(arr)} != run trial count {run_len}; "
            f"treating this run as if '{key}' were absent.",
        )
    return np.zeros(run_len, dtype=bool)


def compute_run_bad_mask(
    meta: Mapping[str, Any],
    run_len: int,
    rule: str = "ar2",
    interp_threshold: int = 0,
) -> np.ndarray:
    """Boolean bad-trial mask for one run's ``trial_metadata``.

    Args:
        meta: one run's trial_metadata dict (from the feature npz).
        run_len: number of trials/windows in this run (mask length).
        rule: 'ar2' (default), 'ar1', or 'union' (AR1 OR AR2).
        interp_threshold: if > 0, additionally drop trials whose ``n_interp``
            is >= this value. 0 disables the criterion.

    Returns:
        Boolean array (run_len,), True = drop this trial.
    """
    rule = validate_rule(rule)

    if rule in ("ar2", "union"):
        ar2 = _get_flag(meta, "bad_ar2", run_len)
        if "bad_ar2" not in meta:
            _warn_once(
                "miss_ar2",
                "bad_trial_rule needs 'bad_ar2' but it is missing from "
                "trial_metadata — no AR2 trials will be dropped.",
            )
    if rule in ("ar1", "union"):
        ar1 = _get_flag(meta, "bad_ar1", run_len)
        if "bad_ar1" not in meta:
            _warn_once(
                "miss_ar1",
                "bad_trial_rule needs 'bad_ar1' but it is missing from "
                "trial_metadata — run code.utils.backfill_ar1 first. "
                "No AR1 trials will be dropped.",
            )

    if rule == "ar2":
        mask = ar2.copy()
    elif rule == "ar1":
        mask = ar1.copy()
    else:  # union
        mask = ar1 | ar2

    if interp_threshold and interp_threshold > 0:
        if "n_interp" in meta and meta["n_interp"] is not None:
            n_interp = np.asarray(meta["n_interp"], dtype=float)
            if len(n_interp) == run_len:
                mask = mask | (n_interp >= interp_threshold)
            else:
                _warn_once(
                    "len_n_interp",
                    f"'n_interp' length {len(n_interp)} != run trial count "
                    f"{run_len}; interpolation criterion skipped for this run.",
                )
        else:
            _warn_once(
                "miss_interp",
                "interp_reject_threshold > 0 but 'n_interp' is missing from "
                "trial_metadata — it cannot be backfilled; re-extract features "
                "from a fresh preprocessing pass. Criterion has no effect.",
            )

    return mask
