"""Metric registry for the unified viz pipeline.

A `Metric` describes one quantity that can be plotted as a row of topomaps or
brain maps. It captures everything the pipeline needs to know:

- which result files to look for (`results_subdir`, `results_glob`)
- which key in the .npz holds the values (`value_key`)
- which key holds the p-values used as a significance mask (`pval_key`)
- how the colour scale should be built (`color_mode` + parameters)
- how to ask the user to produce missing results (`missing_msg`)

Adding a new metric is one entry in `METRICS` — no new code path required.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass(frozen=True)
class Metric:
    name: str
    description: str

    # Where to look for the result files.
    # results_subdir is relative to <data_root>/<paths.results>; the {space}
    # placeholder is filled in by the caller.
    results_subdir: str
    # Glob pattern for the score/result files. Placeholders {feature}, {space},
    # {inout}, {clf}, {cv}, {mode} are filled in by the loader; missing
    # placeholders are replaced with '*'.
    results_glob: str

    # NPZ keys.
    value_key: str
    pval_keys: List[str]  # ordered fallback list

    # Colour scaling.
    color_mode: str  # 'diverging' or 'sequential_above'
    cmap: str
    cbar_label: str
    chance_level: float = 0.0  # used by sequential_above
    # When color_mode == 'sequential_above', clip vmax to [min_vmax, max_vmax]
    sequential_min_span: float = 0.05
    sequential_max_vmax: float = 1.0

    # User-facing instruction for "missing results" errors.
    # Receives **kwargs from the search context (space, feature, missing_str).
    missing_msg: Callable[..., str] = field(
        default=lambda **k: "Run the appropriate analysis task first."
    )


def _stats_missing(space: str, feature: str, **kw) -> str:
    return (
        f"I'd like to plot {feature} ({space}), but I can't find the t-test "
        f"results. Run:\n"
        f"  invoke analysis.stats --features={feature} --space={space}"
    )


def _classification_missing(space: str, feature: str, clf: str = "lda", **kw) -> str:
    return (
        f"I'd like to plot {feature} ({space}), but I can't find the "
        f"classification scores. Run:\n"
        f"  invoke analysis.classify --features={feature} --space={space} "
        f"--clf={clf} --slurm"
    )


METRICS: Dict[str, Metric] = {
    # ----------- Statistics (run_group_statistics) -----------
    "tval": Metric(
        name="tval",
        description="Paired t-statistic (OUT - IN), symmetric diverging.",
        results_subdir="statistics_{space}",
        results_glob="feature-{feature}_inout-{inout}_test-{test}_path-{path}_results.npz",
        value_key="tvals",
        pval_keys=["pvals_corrected_tmax", "pvals_corrected_fdr_bh",
                   "pvals_corrected_bonferroni", "pvals"],
        color_mode="diverging",
        cmap="RdBu_r",
        cbar_label="t-value",
        missing_msg=_stats_missing,
    ),
    "contrast": Metric(
        name="contrast",
        description="Normalised contrast (OUT - IN) / |IN|, diverging.",
        results_subdir="statistics_{space}",
        results_glob="feature-{feature}_inout-{inout}_test-{test}_path-{path}_results.npz",
        value_key="contrast",
        pval_keys=["pvals_corrected_tmax", "pvals_corrected_fdr_bh",
                   "pvals_corrected_bonferroni", "pvals"],
        color_mode="diverging",
        cmap="RdBu_r",
        cbar_label="(OUT − IN) / |IN|",
        missing_msg=_stats_missing,
    ),

    # ----------- Classification (run_classification) -----------
    "roc_auc": Metric(
        name="roc_auc",
        description="Per-spatial cross-validated ROC AUC (chance = 0.5).",
        results_subdir="classification_{space}/group",
        results_glob=(
            "feature-{feature}_space-{space}_inout-{inout}"
            "_clf-{clf}_cv-{cv}_mode-{mode}_scores.npz"
        ),
        value_key="observed",
        pval_keys=["pvals_tmax", "pvals_fdr_bh", "pvals_bonferroni", "pvals_uncorrected"],
        color_mode="sequential_above",
        chance_level=0.5,
        cmap="magma",
        cbar_label="ROC AUC",
        sequential_min_span=0.05,
        sequential_max_vmax=0.80,
        missing_msg=_classification_missing,
    ),
}


# Map a chosen correction name to a pval key inside a metric's NPZ.
def resolve_pval_key(metric: Metric, correction: str) -> Optional[str]:
    """Pick the pval key matching a correction name, falling back to fallbacks.

    `correction` is one of: 'tmax', 'fdr_bh', 'bonferroni', 'uncorrected', 'auto'.
    """
    if correction == "auto":
        return metric.pval_keys[0]
    matches = [k for k in metric.pval_keys if correction in k]
    if matches:
        return matches[0]
    return metric.pval_keys[0]
