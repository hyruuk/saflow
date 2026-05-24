"""Central IN/OUT zone selection for sliding-window or single-trial epochs.

Four strategies are supported, all anchored on the per-subject ``inout_bounds``
percentiles (default ``[25, 75]``):

* ``"strict"``  (default) — a window is IN only when **every** constituent trial
  has its per-trial ``VTC_filtered`` below the low percentile threshold; OUT only
  when every trial is at or above the high percentile threshold. Thresholds are
  pooled across the subject (or per-run, per ``zoning``). MID / mixed windows
  are excluded from both contrasts. Conceptually clean (every trial in an IN
  window really is IN) at the cost of ~halving the per-condition window count
  vs. the boxcar-mean strategies.

* ``"lenient"`` — same per-trial classification, but a window passes as IN when
  it contains at least one IN trial and **no OUT trial** (the rest may be MID);
  symmetric for OUT. Roughly triples the sample vs. strict.

* ``"vtcfilt"`` — saflow's pre-refactor behaviour. A window is IN/OUT according
  to whether its ``window_vtc_mean`` (boxcar mean of the constituent trials'
  ``VTC_filtered`` values) sits below/above the percentile thresholds computed
  on those window means. In single-trial mode it reduces to per-trial
  ``VTC_filtered`` thresholding.

* ``"vtcraw"`` — cc_saflow's behaviour. Same window-mean thresholding but on
  ``window_vtc_raw_mean`` (boxcar mean of unfiltered ``VTC_raw``). Requires
  welch outputs to include ``window_vtc_raw_mean`` (and ``included_VTC_raw``
  in window mode); regenerate the welch step if you switch to this strategy on
  an older dataset.

The default ``"strict"`` rewrites the historical filenames unchanged (no
selection token). Any non-default strategy adds a ``_sel-{name}`` token to
output filenames so cached outputs from different strategies can coexist (see
:func:`inout_selection_token`).

The module produces *window-level* boolean zone masks. Callers AND-combine
those zones with the task-type mask and (if applicable) the bad-trial mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

DEFAULT_STRATEGY = "strict"
STRATEGIES = ("strict", "lenient", "vtcfilt", "vtcraw")

_WINDOW_MEAN_STRATEGIES = ("vtcfilt", "vtcraw")
_PURITY_STRATEGIES = ("strict", "lenient")


@dataclass
class RunMeta:
    """Per-run metadata required for IN/OUT selection.

    Attributes:
        n_windows: Number of epochs in this run.
        window_vtc_mean: ``(n_windows,)`` boxcar mean of ``VTC_filtered`` for the
            constituent trials of each window. Required for ``vtcfilt``. In
            single-trial mode this is just the per-trial ``VTC_filtered``.
        window_vtc_raw_mean: ``(n_windows,)`` boxcar mean of ``VTC_raw``.
            Required for ``vtcraw``.
        included_VTC: Length-``n_windows`` list of ``(n_events_window,)``
            arrays carrying the per-trial ``VTC_filtered`` values inside each
            window. Required for ``strict``/``lenient``. In single-trial mode
            wrap each scalar in a 1-element array.
    """

    n_windows: int
    window_vtc_mean: Optional[np.ndarray] = None
    window_vtc_raw_mean: Optional[np.ndarray] = None
    included_VTC: Optional[Sequence[np.ndarray]] = None


def validate_strategy(strategy: str) -> str:
    """Normalize and validate a strategy name; raises ValueError on unknown."""
    s = (strategy or "").strip().lower()
    if s not in STRATEGIES:
        raise ValueError(
            f"Unknown inout_selection strategy {strategy!r}; "
            f"expected one of {STRATEGIES}"
        )
    return s


def inout_selection_token(strategy: str) -> str:
    """Filename token for the strategy.

    Returns ``""`` for the default ``"strict"`` so existing filename conventions
    are preserved; returns ``"_sel-{strategy}"`` otherwise. Consumers that parse
    filenames should treat a missing token as the default strategy.
    """
    s = validate_strategy(strategy)
    if s == DEFAULT_STRATEGY:
        return ""
    return f"_sel-{s}"


def build_run_meta_from_welch(meta: Mapping[str, object]) -> RunMeta:
    """Construct a ``RunMeta`` from a welch-PSD ``trial_metadata`` dict.

    Falls back to per-trial fields when window-level fields are absent (i.e.,
    single-trial PSDs computed with ``n_events_window=1``). In window mode the
    presence of ``included_VTC`` signals the absence of a single-trial anchor;
    in that case the per-trial ``VTC_raw`` cannot stand in for
    ``window_vtc_raw_mean`` (would mix anchor-trial-only with window-mean
    semantics) and the field is left ``None`` so the vtcraw strategy errors
    with a regeneration hint instead of silently producing wrong results.
    """
    win_filt = meta.get("window_vtc_mean")
    if win_filt is None and "VTC_filtered" in meta:
        win_filt = np.asarray(meta["VTC_filtered"], dtype=float)
    elif win_filt is not None:
        win_filt = np.asarray(win_filt, dtype=float)

    is_windowed = "included_VTC" in meta
    win_raw = meta.get("window_vtc_raw_mean")
    if win_raw is not None:
        win_raw = np.asarray(win_raw, dtype=float)
    elif not is_windowed and "VTC_raw" in meta:
        # Single-trial fallback: anchor VTC_raw is the per-window value.
        win_raw = np.asarray(meta["VTC_raw"], dtype=float)
    # In windowed mode without ``window_vtc_raw_mean``: leave win_raw=None so
    # vtcraw is the only strategy that fails (with a clear regeneration hint).

    inc_vtc = meta.get("included_VTC")
    if inc_vtc is not None:
        inc_vtc = [np.asarray(v, dtype=float) for v in inc_vtc]
    elif win_filt is not None:
        # Single-trial fallback: each "window" is one trial.
        inc_vtc = [np.asarray([v], dtype=float) for v in np.asarray(win_filt, dtype=float)]

    n_windows = int(len(win_filt) if win_filt is not None
                    else (len(win_raw) if win_raw is not None
                          else (len(inc_vtc) if inc_vtc is not None else 0)))

    return RunMeta(
        n_windows=n_windows,
        window_vtc_mean=win_filt,
        window_vtc_raw_mean=win_raw,
        included_VTC=inc_vtc,
    )


def _bounds_from(values: np.ndarray, lo_pct: float, hi_pct: float) -> Tuple[float, float]:
    if values.size == 0 or np.all(np.isnan(values)):
        return (np.nan, np.nan)
    return (float(np.nanpercentile(values, lo_pct)),
            float(np.nanpercentile(values, hi_pct)))


def _check_field(per_run: Sequence[RunMeta], attr: str, strategy: str) -> None:
    for i, rm in enumerate(per_run):
        if getattr(rm, attr) is None:
            raise ValueError(
                f"inout_selection strategy {strategy!r} requires per-run "
                f"{attr!r} but it is missing on run index {i}. "
                f"If you selected 'vtcraw' on an older dataset, re-run the "
                f"welch step so the metadata gains 'window_vtc_raw_mean' / "
                f"'included_VTC_raw'."
            )


def _per_trial_class(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Per-trial class: -1 (IN), 0 (MID), +1 (OUT)."""
    out = np.zeros_like(values, dtype=int)
    out[values <= lo] = -1
    out[values >= hi] = 1
    return out


def _window_mean_zones(
    per_run: Sequence[RunMeta],
    inout_bounds: Tuple[float, float],
    zoning: str,
    attr: str,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """IN/OUT zones based on a per-window scalar (vtcfilt / vtcraw)."""
    lo_pct, hi_pct = float(inout_bounds[0]), float(inout_bounds[1])
    if zoning == "per-subject":
        pooled = np.concatenate(
            [np.asarray(getattr(rm, attr), dtype=float).ravel() for rm in per_run]
        )
        lo, hi = _bounds_from(pooled, lo_pct, hi_pct)
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for rm in per_run:
        v = np.asarray(getattr(rm, attr), dtype=float)
        if zoning == "per-run":
            lo, hi = _bounds_from(v, lo_pct, hi_pct)
        if np.isnan(lo) or np.isnan(hi):
            out.append((np.zeros(rm.n_windows, dtype=bool),
                        np.zeros(rm.n_windows, dtype=bool)))
            continue
        in_zone = v <= lo
        out_zone = v >= hi
        out.append((in_zone, out_zone))
    return out


def _purity_zones(
    per_run: Sequence[RunMeta],
    inout_bounds: Tuple[float, float],
    zoning: str,
    strategy: str,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """IN/OUT zones via per-trial VTC_filtered classification + window purity."""
    lo_pct, hi_pct = float(inout_bounds[0]), float(inout_bounds[1])
    if zoning == "per-subject":
        pooled = np.concatenate(
            [np.asarray(np.concatenate([np.asarray(arr, dtype=float)
                                        for arr in rm.included_VTC]), dtype=float)
             for rm in per_run]
        )
        lo, hi = _bounds_from(pooled, lo_pct, hi_pct)

    zones: List[Tuple[np.ndarray, np.ndarray]] = []
    for rm in per_run:
        if zoning == "per-run":
            flat = np.concatenate([np.asarray(arr, dtype=float)
                                   for arr in rm.included_VTC])
            lo, hi = _bounds_from(flat, lo_pct, hi_pct)
        if np.isnan(lo) or np.isnan(hi):
            zones.append((np.zeros(rm.n_windows, dtype=bool),
                          np.zeros(rm.n_windows, dtype=bool)))
            continue
        in_zone = np.zeros(rm.n_windows, dtype=bool)
        out_zone = np.zeros(rm.n_windows, dtype=bool)
        for i, arr in enumerate(rm.included_VTC):
            cls = _per_trial_class(np.asarray(arr, dtype=float), lo, hi)
            if strategy == "strict":
                if cls.size and (cls == -1).all():
                    in_zone[i] = True
                if cls.size and (cls == 1).all():
                    out_zone[i] = True
            else:  # lenient
                has_in = (cls == -1).any()
                has_out = (cls == 1).any()
                no_in = not has_in
                no_out = not has_out
                if has_in and no_out:
                    in_zone[i] = True
                if has_out and no_in:
                    out_zone[i] = True
        zones.append((in_zone, out_zone))
    return zones


def compute_inout_zones(
    per_run: Sequence[RunMeta],
    *,
    strategy: str = DEFAULT_STRATEGY,
    inout_bounds: Tuple[float, float] = (25.0, 75.0),
    zoning: str = "per-subject",
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Compute per-run window-level (in_zone, out_zone) bool masks.

    Args:
        per_run: Per-run :class:`RunMeta` records, in the order matching the
            caller's per-run feature/PSD arrays.
        strategy: One of ``"strict"`` / ``"lenient"`` / ``"vtcfilt"`` / ``"vtcraw"``.
        inout_bounds: ``(low_pct, high_pct)`` percentile thresholds (default
            ``(25, 75)``).
        zoning: ``"per-subject"`` pools thresholds across all runs (default);
            ``"per-run"`` computes thresholds within each run.

    Returns:
        ``[(in_zone, out_zone), ...]``, one tuple per run, each array of length
        ``rm.n_windows``. Combine with task-type and bad-trial masks downstream.
    """
    strategy = validate_strategy(strategy)
    if zoning not in ("per-subject", "per-run"):
        raise ValueError(f"zoning must be 'per-subject' or 'per-run', got {zoning!r}")

    if strategy in _WINDOW_MEAN_STRATEGIES:
        attr = "window_vtc_mean" if strategy == "vtcfilt" else "window_vtc_raw_mean"
        _check_field(per_run, attr, strategy)
        return _window_mean_zones(per_run, inout_bounds, zoning, attr)

    if strategy in _PURITY_STRATEGIES:
        _check_field(per_run, "included_VTC", strategy)
        return _purity_zones(per_run, inout_bounds, zoning, strategy)

    raise AssertionError(f"unreachable: strategy={strategy!r}")  # pragma: no cover


def concat_zones(
    per_run_zones: Iterable[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate per-run (in, out) zones into flat masks in run order."""
    per_run_zones = list(per_run_zones)
    in_all = np.concatenate([z[0] for z in per_run_zones]) if per_run_zones else np.array([], dtype=bool)
    out_all = np.concatenate([z[1] for z in per_run_zones]) if per_run_zones else np.array([], dtype=bool)
    return in_all, out_all
