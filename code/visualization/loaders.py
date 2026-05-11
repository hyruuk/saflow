"""Loaders that turn a result file into a uniform `MapResult`.

Each loader handles one file family (statistics or classification). Both produce
the same dataclass so the renderer code is metric-agnostic.

The loaders also know how to produce a helpful error message when expected
files don't exist — see `MissingResultsError` and `Metric.missing_msg`.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from code.visualization.metrics import METRICS, Metric, resolve_pval_key

logger = logging.getLogger(__name__)


@dataclass
class MapResult:
    """One panel's worth of data: a 1-D values array + optional sig mask."""

    feature: str            # full feature name, e.g. 'psd_alpha'
    family: str             # 'psd', 'psd_corrected', 'fooof', 'complexity', 'other'
    label: str              # short label for the panel title
    values: np.ndarray      # (n_spatial,)
    mask: Optional[np.ndarray]  # (n_spatial,) bool — significant entries
    n_significant: int
    metric: Metric
    metadata: Dict          # full JSON metadata, for figure suptitle / provenance
    source_path: Path


class MissingResultsError(Exception):
    """Raised when expected result files cannot be found.

    The string representation is the user-facing instruction (what to run).
    """


# ---------------------------------------------------------------------------
# Family helpers (shared with classification CLI ordering)
# ---------------------------------------------------------------------------

BAND_ORDER = ["delta", "theta", "alpha", "lobeta", "hibeta",
              "gamma1", "gamma2", "gamma3"]
COMPLEXITY_ORDER = [
    "lzc_median",
    "entropy_permutation", "entropy_spectral", "entropy_sample",
    "entropy_approximate", "entropy_svd",
    "fractal_higuchi", "fractal_petrosian", "fractal_katz", "fractal_dfa",
]
FOOOF_ORDER = ["exponent", "offset", "knee", "r_squared"]


def feature_family(feature: str) -> str:
    if feature.startswith("psd_corrected_"):
        return "psd_corrected"
    if feature.startswith("psd_"):
        return "psd"
    if feature.startswith("fooof_"):
        return "fooof"
    if feature.startswith("complexity_"):
        return "complexity"
    return "other"


def short_label(feature: str) -> str:
    for prefix in ("psd_corrected_", "psd_", "fooof_", "complexity_"):
        if feature.startswith(prefix):
            return feature[len(prefix):]
    return feature


def family_sort_key(feature: str) -> Tuple[int, int, str]:
    fam = feature_family(feature)
    sub = short_label(feature)
    fam_idx = {"psd": 0, "psd_corrected": 1, "fooof": 2, "complexity": 3}.get(fam, 4)
    if fam == "psd" or fam == "psd_corrected":
        order = BAND_ORDER
    elif fam == "fooof":
        order = FOOOF_ORDER
    elif fam == "complexity":
        order = COMPLEXITY_ORDER
    else:
        order = []
    pos = order.index(sub) if sub in order else 999
    return (fam_idx, pos, sub)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def _resolve_glob(pattern: str, **fields) -> str:
    """Replace {placeholders} in a glob pattern, defaulting unspecified ones to '*'."""
    out = pattern
    while True:
        m = re.search(r"\{(\w+)\}", out)
        if not m:
            break
        key = m.group(1)
        repl = fields.get(key, "*")
        out = out.replace("{" + key + "}", str(repl))
    return out


def _parse_filename(path: Path, glob_pattern: str) -> Optional[Dict[str, str]]:
    """Extract placeholder values from a filename matching a glob pattern."""
    # Build a regex from the pattern by escaping literals and turning {x} into named groups.
    rex = re.escape(glob_pattern).replace(r"\{", "{").replace(r"\}", "}")
    rex = re.sub(r"\{(\w+)\}", lambda m: f"(?P<{m.group(1)}>[^_]+?)", rex)
    rex = "^" + rex + "$"
    m = re.match(rex, path.name)
    return m.groupdict() if m else None


def find_results_for_metric(
    metric: Metric,
    results_dir: Path,
    feature: Optional[str] = None,
    inout: str = "*",
    extra_filters: Optional[Dict[str, str]] = None,
) -> List[Path]:
    """List files under `results_dir` matching the metric's glob pattern.

    `feature` may be a single name, None (any), or '*'. Other placeholders
    (clf, cv, mode, test, …) come from `extra_filters` and default to '*'.
    """
    if not results_dir.exists():
        return []
    extra_filters = dict(extra_filters or {})
    extra_filters.setdefault("inout", inout)
    glob_pattern = _resolve_glob(
        metric.results_glob,
        feature=feature if feature else "*",
        **extra_filters,
    )
    # Skip chunk partials.
    return sorted(p for p in results_dir.glob(glob_pattern) if "_chunk-" not in p.name)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_metadata(npz_path: Path) -> Dict:
    meta_path = npz_path.with_name(npz_path.name.replace(".npz", ".json").replace("_scores", "_metadata").replace("_results", "_metadata"))
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except Exception as exc:
            logger.warning(f"Could not parse {meta_path}: {exc}")
    return {}


def _extract_feature_from_filename(npz_path: Path, glob_pattern: str) -> str:
    parsed = _parse_filename(npz_path, glob_pattern)
    if parsed and "feature" in parsed:
        return parsed["feature"]
    # Last-ditch: chop "feature-" prefix
    name = npz_path.stem
    m = re.match(r"^feature-([^_]+(?:_[^_]+)*?)(?:_space-|_inout-)", name)
    return m.group(1) if m else name


def load_one(
    npz_path: Path,
    metric: Metric,
    correction: str = "auto",
    alpha: float = 0.05,
) -> MapResult:
    """Load a single .npz into a MapResult."""
    with np.load(npz_path, allow_pickle=True) as npz:
        if metric.value_key not in npz.files:
            raise KeyError(
                f"Metric '{metric.name}' expects key '{metric.value_key}' in "
                f"{npz_path.name}, but file contains: {list(npz.files)}"
            )
        values = np.asarray(npz[metric.value_key]).flatten()
        pkey = resolve_pval_key(metric, correction)
        # Match on full pval key list — pkey is suggested first, but if the file
        # only has a different one, fall back through the metric's list.
        chosen = None
        for candidate in [pkey] + [k for k in metric.pval_keys if k != pkey]:
            if candidate in npz.files:
                chosen = candidate
                break
        mask = None
        if chosen is not None:
            pvals = np.asarray(npz[chosen]).flatten()
            mask = pvals < alpha

    feature = _extract_feature_from_filename(npz_path, metric.results_glob)
    metadata = _load_metadata(npz_path)
    return MapResult(
        feature=feature,
        family=feature_family(feature),
        label=short_label(feature),
        values=values,
        mask=mask,
        n_significant=int(mask.sum()) if mask is not None else 0,
        metric=metric,
        metadata=metadata,
        source_path=npz_path,
    )


def load_for_metric(
    metric_name: str,
    space: str,
    data_root: Path,
    results_subpath: str,
    feature: Optional[str] = None,
    inout: str = "*",
    correction: str = "auto",
    alpha: float = 0.05,
    extra_filters: Optional[Dict[str, str]] = None,
) -> List[MapResult]:
    """Find and load all results for a metric, raising MissingResultsError if none.

    `extra_filters` can pin metric-specific placeholders (e.g. clf='lda', cv='logo').
    """
    if metric_name not in METRICS:
        raise ValueError(
            f"Unknown metric '{metric_name}'. Known: {list(METRICS)}"
        )
    metric = METRICS[metric_name]
    results_dir = (
        data_root
        / results_subpath
        / metric.results_subdir.format(space=space)
    )
    paths = find_results_for_metric(
        metric, results_dir, feature=feature, inout=inout,
        extra_filters=extra_filters,
    )
    if not paths:
        # Build a useful "what to run" message
        feat_display = feature or "<any>"
        clf = (extra_filters or {}).get("clf", "lda")
        msg_lines = [
            f"No '{metric.name}' results found.",
            f"Looked in: {results_dir}",
            f"Pattern:   {_resolve_glob(metric.results_glob, feature=feature or '*', **(extra_filters or {}))}",
            "",
            metric.missing_msg(space=space, feature=feat_display, clf=clf),
        ]
        raise MissingResultsError("\n".join(msg_lines))

    out = []
    for p in paths:
        try:
            out.append(load_one(p, metric, correction=correction, alpha=alpha))
        except KeyError as exc:
            logger.warning(f"Skipping {p.name}: {exc}")
    return out
