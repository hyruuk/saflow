"""Run t-tests on complexity measures (IN vs OUT) and generate figures.

This script:
1. Loads complexity data efficiently (vectorized)
2. Computes subject-level means for IN/OUT
3. Runs paired t-tests with optional corrections
4. Generates topographic figures

Correction methods:
- none: No correction (not recommended)
- fdr: Benjamini-Hochberg FDR (controls false discovery rate)
- bonferroni: Bonferroni correction (controls FWER, very conservative)
- tmax: Maximum statistic permutation (controls FWER, recommended)

The tmax method compares each observed t-statistic to the null distribution
of the maximum absolute t-statistic across all channels. This inherently
controls for multiple comparisons without additional correction.

Usage:
    python -m code.statistics.run_complexity_stats
    python -m code.statistics.run_complexity_stats --correction fdr --alpha 0.05
    python -m code.statistics.run_complexity_stats --correction tmax --n-permutations 10000
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy import stats
from scipy.stats import false_discovery_control

from code.utils.bad_trials import compute_run_bad_mask
from code.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

METRICS = [
    "lzc_median", "entropy_permutation", "entropy_spectral",
    "entropy_sample", "entropy_approximate", "entropy_svd",
    "fractal_higuchi", "fractal_petrosian", "fractal_katz", "fractal_dfa"
]


def load_subject_data(
    npz_path: Path,
    metrics: List[str],
    bad_trial_rule: str = "ar2",
    interp_reject_threshold: int = 0,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Load data from a single NPZ file efficiently.

    Returns:
        Tuple of (data_dict, vtc, task, bad). ``bad`` follows the configured
        bad-trial rule (ar2 / ar1 / union, plus optional interpolation
        threshold); trials are all-good when the relevant flags are absent.
    """
    data = np.load(npz_path, allow_pickle=True)
    meta = data["trial_metadata"].item()

    data_dict = {m: data[m] for m in metrics}
    vtc = np.array(meta["VTC_filtered"])
    task = np.array(meta["task"])
    bad = compute_run_bad_mask(
        meta, len(vtc), bad_trial_rule, interp_reject_threshold
    )

    return data_dict, vtc, task, bad


def compute_subject_aggregates(
    data_root: Path,
    subject: str,
    runs: List[str],
    space: str,
    inout_bounds: Tuple[int, int],
    metrics: List[str],
    drop_bad_trials: bool = True,
    aggregate: str = "median",
    bad_trial_rule: str = "ar2",
    interp_reject_threshold: int = 0,
) -> Tuple[
    Optional[Dict[str, np.ndarray]],
    Optional[Dict[str, np.ndarray]],
    Dict[str, int],
]:
    """Per-subject IN/OUT aggregates for each complexity metric.

    IN/OUT thresholds are computed on **all** trials (including bads).
    Trials flagged ``bad_ar2`` are dropped after masking when
    ``drop_bad_trials`` is True.

    Returns:
        (agg_in, agg_out, counts) where counts has n_in, n_out, n_mid,
        n_bad_in, n_bad_out, n_total. agg_in/out are None if a subject
        ends up empty in either condition after filtering.
    """
    if aggregate not in ("median", "mean"):
        raise ValueError(f"Unknown aggregate '{aggregate}'")
    reducer = np.nanmedian if aggregate == "median" else np.nanmean

    complexity_dir = data_root / "features" / f"complexity_{space}"
    subj_dir = complexity_dir / f"sub-{subject}"

    empty_counts = {
        "n_total": 0, "n_in": 0, "n_out": 0, "n_mid": 0,
        "n_bad_in": 0, "n_bad_out": 0,
    }
    if not subj_dir.exists():
        return None, None, empty_counts

    all_vtc = []
    all_task = []
    all_bad = []
    all_data = {m: [] for m in metrics}

    for run in runs:
        pattern = f"sub-{subject}_*_run-{run}_*_desc-complexity.npz"
        files = list(subj_dir.glob(pattern))
        if not files:
            continue

        data_dict, vtc, task, bad = load_subject_data(
            files[0], metrics, bad_trial_rule, interp_reject_threshold
        )

        all_vtc.append(vtc)
        all_task.append(task)
        all_bad.append(bad)
        for m in metrics:
            all_data[m].append(data_dict[m])

    if not all_vtc:
        return None, None, empty_counts

    all_vtc = np.concatenate(all_vtc)
    all_task = np.concatenate(all_task)
    all_bad_arr = np.concatenate(all_bad)
    for m in metrics:
        all_data[m] = np.concatenate(all_data[m], axis=0)

    inbound = np.nanpercentile(all_vtc, inout_bounds[0])
    outbound = np.nanpercentile(all_vtc, inout_bounds[1])

    task_mask = all_task == "correct_commission"
    in_mask_full = task_mask & (all_vtc <= inbound)
    out_mask_full = task_mask & (all_vtc >= outbound)
    mid_mask_full = task_mask & ~in_mask_full & ~out_mask_full

    n_bad_in = int((in_mask_full & all_bad_arr).sum()) if drop_bad_trials else 0
    n_bad_out = int((out_mask_full & all_bad_arr).sum()) if drop_bad_trials else 0

    if drop_bad_trials:
        in_mask = in_mask_full & ~all_bad_arr
        out_mask = out_mask_full & ~all_bad_arr
    else:
        in_mask = in_mask_full
        out_mask = out_mask_full

    counts = {
        "n_total": int(len(all_vtc)),
        "n_in": int(in_mask.sum()),
        "n_out": int(out_mask.sum()),
        "n_mid": int(mid_mask_full.sum()),
        "n_bad_in": n_bad_in,
        "n_bad_out": n_bad_out,
    }

    if counts["n_in"] == 0 or counts["n_out"] == 0:
        return None, None, counts

    agg_in = {m: reducer(all_data[m][in_mask], axis=0) for m in metrics}
    agg_out = {m: reducer(all_data[m][out_mask], axis=0) for m in metrics}

    return agg_in, agg_out, counts


# Backwards-compatible alias (older callers used compute_subject_means)
def compute_subject_means(*args, **kwargs):
    """Deprecated alias for :func:`compute_subject_aggregates`."""
    agg_in, agg_out, counts = compute_subject_aggregates(*args, **kwargs)
    return agg_in, agg_out, counts["n_in"], counts["n_out"]


def run_paired_ttest(
    subj_means_in: List[Dict[str, np.ndarray]],
    subj_means_out: List[Dict[str, np.ndarray]],
    metrics: List[str],
) -> Dict[str, Dict]:
    """Run paired t-test comparing IN vs OUT across subjects (vectorized)."""
    results = {}

    for metric in metrics:
        in_arr = np.array([s[metric] for s in subj_means_in])
        out_arr = np.array([s[metric] for s in subj_means_out])

        # Vectorized t-test across all channels
        tvals, pvals = stats.ttest_rel(out_arr, in_arr, axis=0)

        # Compute contrast (handle division by zero)
        mean_in = np.mean(in_arr, axis=0)
        mean_out = np.mean(out_arr, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            contrast = (mean_out - mean_in) / mean_in
            contrast[~np.isfinite(contrast)] = 0.0

        results[metric] = {
            "contrast": contrast,
            "tvals": tvals,
            "pvals": pvals,
        }

    return results


def apply_correction(
    results: Dict[str, Dict],
    correction: str,
    alpha: float,
    n_permutations: int = 1000,
    subj_means_in: Optional[List] = None,
    subj_means_out: Optional[List] = None,
) -> Dict[str, Dict]:
    """Apply multiple comparison correction to p-values.

    Args:
        results: Dict with tvals, pvals per metric
        correction: 'none', 'fdr', 'bonferroni', 'tmax'
        alpha: Significance threshold
        n_permutations: Number of permutations (for tmax correction)

    Note on tmax:
        The tmax (maximum statistic) correction controls FWER by comparing each
        observed t-statistic to the null distribution of the MAXIMUM absolute
        t-statistic across all channels. This accounts for multiple comparisons
        without requiring a separate correction step.
    """
    for metric in results:
        pvals = results[metric]["pvals"].copy()
        n_tests = len(pvals)

        # Handle NaN p-values (set to 1.0 for correction purposes)
        nan_mask = np.isnan(pvals)
        if np.any(nan_mask):
            logger.warning(f"{metric}: {np.sum(nan_mask)} channels with NaN p-values")
            pvals[nan_mask] = 1.0

        if correction == "none":
            results[metric]["pvals_corrected"] = pvals
            results[metric]["sig_mask"] = pvals < alpha

        elif correction == "fdr":
            # Benjamini-Hochberg FDR correction
            pvals_corrected = false_discovery_control(pvals, method='bh')
            results[metric]["pvals_corrected"] = pvals_corrected
            results[metric]["sig_mask"] = pvals_corrected < alpha

        elif correction == "bonferroni":
            pvals_corrected = np.minimum(pvals * n_tests, 1.0)
            results[metric]["pvals_corrected"] = pvals_corrected
            results[metric]["sig_mask"] = pvals_corrected < alpha

        elif correction == "tmax":
            # Tmax (maximum statistic) permutation test - controls FWER
            if subj_means_in is None or subj_means_out is None:
                logger.warning("Tmax correction requires subject data, falling back to FDR")
                pvals_corrected = false_discovery_control(pvals, method='bh')
                results[metric]["pvals_corrected"] = pvals_corrected
                results[metric]["sig_mask"] = pvals_corrected < alpha
            else:
                in_arr = np.array([s[metric] for s in subj_means_in])
                out_arr = np.array([s[metric] for s in subj_means_out])
                observed_t = results[metric]["tvals"]

                n_subjects = in_arr.shape[0]
                max_t_perm = np.zeros(n_permutations)

                for perm_i in range(n_permutations):
                    flip = np.random.choice([-1, 1], size=n_subjects)
                    diff = out_arr - in_arr
                    perm_diff = diff * flip[:, np.newaxis]
                    perm_t, _ = stats.ttest_1samp(perm_diff, 0, axis=0)
                    max_t_perm[perm_i] = np.nanmax(np.abs(perm_t))

                pvals_corrected = np.array([
                    np.mean(max_t_perm >= np.abs(t)) if not np.isnan(t) else 1.0
                    for t in observed_t
                ])
                results[metric]["pvals_corrected"] = pvals_corrected
                results[metric]["sig_mask"] = pvals_corrected < alpha

        # Ensure NaN channels are not significant
        results[metric]["sig_mask"][nan_mask] = False

    return results


def get_meg_info(data_root: Path, subject: str = "04") -> mne.Info:
    """Load MEG channel info for topographic plotting."""
    deriv_dir = data_root / "derivatives" / "preprocessed" / f"sub-{subject}" / "meg"
    fif_files = list(deriv_dir.glob("*_meg.fif"))

    if fif_files:
        raw = mne.io.read_raw_fif(fif_files[0], preload=False, verbose=False)
        raw.pick_types(meg=True, ref_meg=False)
        return raw.info

    raise FileNotFoundError("Could not find MEG data for channel info")


def plot_complexity_stats(
    results: Dict[str, Dict],
    info: mne.Info,
    output_path: Path,
    correction: str,
    alpha: float,
):
    """Plot topographic maps of t-values for all complexity metrics."""
    metrics = list(results.keys())
    n_metrics = len(metrics)

    fig, axes = plt.subplots(2, n_metrics, figsize=(2.5 * n_metrics, 5), dpi=150)

    for idx, metric in enumerate(metrics):
        tvals = results[metric]["tvals"]
        sig_mask = results[metric]["sig_mask"]
        vlim = np.max(np.abs(tvals))

        # Plot t-values
        mne.viz.plot_topomap(
            tvals, info, axes=axes[0, idx], show=False,
            cmap="RdBu_r", vlim=(-vlim, vlim), contours=0,
        )
        short_name = metric.replace("entropy_", "").replace("fractal_", "")
        axes[0, idx].set_title(short_name, fontsize=8)

        # Plot with significance mask
        mne.viz.plot_topomap(
            tvals, info, axes=axes[1, idx], show=False,
            cmap="RdBu_r", vlim=(-vlim, vlim),
            mask=sig_mask,
            mask_params=dict(marker="o", markerfacecolor="k", markeredgecolor="k", markersize=2),
            contours=0,
        )
        n_sig = np.sum(sig_mask)
        axes[1, idx].set_title(f"n={n_sig}", fontsize=7)

    axes[0, 0].set_ylabel("T-values", fontsize=9)
    axes[1, 0].set_ylabel(f"{correction}\np<{alpha}", fontsize=8)

    fig.suptitle("Complexity: OUT vs IN", fontsize=11, y=0.98)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved figure to {output_path}")

    return fig


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run complexity statistics")
    parser.add_argument("--space", default="sensor", help="Analysis space (default: sensor)")
    parser.add_argument("--correction", default="fdr",
                        choices=["none", "fdr", "bonferroni", "tmax"],
                        help="Multiple comparison correction: none, fdr, bonferroni, tmax (default: fdr)")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold")
    parser.add_argument("--n-permutations", type=int, default=1000,
                        help="Number of permutations (for permutation correction)")
    parser.add_argument("--metrics", nargs="+", default=None,
                        help="Specific metrics to analyze (default: all)")
    parser.add_argument(
        "--aggregate",
        default="median",
        choices=["median", "mean"],
        help="Per-subject aggregation statistic (default: median, matches cc_saflow)",
    )
    parser.add_argument(
        "--keep-bad-trials",
        action="store_true",
        default=False,
        help="Skip the bad_ar2 filter (keeps trials inside autoreject-rejected "
             "BAD_AR2 windows). Default is to drop them.",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("COMPLEXITY STATISTICS: IN vs OUT")
    logger.info("=" * 60)

    config = load_config()
    data_root = Path(config["paths"]["data_root"])
    subjects = config["bids"]["subjects"]
    runs = config["bids"]["task_runs"]
    inout_bounds = tuple(config["analysis"]["inout_bounds"])
    metrics = args.metrics if args.metrics else METRICS
    analysis_cfg = config.get("analysis", {})
    bad_trial_rule = str(analysis_cfg.get("bad_trial_rule", "ar2"))
    interp_reject_threshold = int(analysis_cfg.get("interp_reject_threshold", 0) or 0)

    logger.info(f"Space: {args.space}")
    logger.info(f"Correction: {args.correction}")
    logger.info(f"Alpha: {args.alpha}")
    logger.info(f"Aggregate: {args.aggregate}")
    logger.info(f"Drop bad trials: {not args.keep_bad_trials} (rule={bad_trial_rule})")
    logger.info(f"Subjects: {len(subjects)}")

    # Process subjects
    logger.info(f"Computing subject-level {args.aggregate}s...")
    subj_means_in = []
    subj_means_out = []
    total_in, total_out, total_bad_excluded = 0, 0, 0
    per_subject: Dict[str, Dict[str, int]] = {}

    for subject in subjects:
        agg_in, agg_out, counts = compute_subject_aggregates(
            data_root, subject, runs, args.space, inout_bounds, metrics,
            drop_bad_trials=not args.keep_bad_trials,
            aggregate=args.aggregate,
            bad_trial_rule=bad_trial_rule,
            interp_reject_threshold=interp_reject_threshold,
        )
        per_subject[subject] = counts
        if agg_in is not None:
            subj_means_in.append(agg_in)
            subj_means_out.append(agg_out)
            total_in += counts["n_in"]
            total_out += counts["n_out"]
            total_bad_excluded += counts["n_bad_in"] + counts["n_bad_out"]
            logger.info(
                f"  sub-{subject}: {counts['n_in']} IN, {counts['n_out']} OUT "
                f"(bad excluded: in={counts['n_bad_in']} out={counts['n_bad_out']})"
            )
        else:
            logger.warning(
                f"  sub-{subject}: dropped (n_in={counts['n_in']}, n_out={counts['n_out']})"
            )

    logger.info(
        f"Total: {len(subj_means_in)} subjects, {total_in} IN, {total_out} OUT trials "
        f"(bad excluded: {total_bad_excluded})"
    )

    # Run t-tests
    logger.info("Running paired t-tests...")
    results = run_paired_ttest(subj_means_in, subj_means_out, metrics)

    # Apply correction
    logger.info(f"Applying {args.correction} correction...")
    results = apply_correction(
        results, args.correction, args.alpha, args.n_permutations,
        subj_means_in, subj_means_out
    )

    for metric in metrics:
        n_sig = np.sum(results[metric]["sig_mask"])
        logger.info(f"  {metric}: {n_sig} significant channels")

    # Get MEG info and plot
    logger.info("Generating figure...")
    info = get_meg_info(data_root)
    output_path = Path("reports") / "figures" / f"complexity_ttest_{args.correction}.png"
    plot_complexity_stats(results, info, output_path, args.correction, args.alpha)

    # Save results
    results_path = data_root / config["paths"]["results"] / f"statistics_{args.space}" / "complexity_ttest_results.npz"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "correction": args.correction,
        "alpha": args.alpha,
        "n_subjects": len(subj_means_in),
        "n_trials_in": total_in,
        "n_trials_out": total_out,
        "n_bad_excluded": total_bad_excluded,
        "aggregate": args.aggregate,
        "drop_bad_trials": not args.keep_bad_trials,
        "bad_trial_rule": bad_trial_rule,
        "interp_reject_threshold": interp_reject_threshold,
    }
    for metric in results:
        save_dict[f"{metric}_tvals"] = results[metric]["tvals"]
        save_dict[f"{metric}_pvals"] = results[metric]["pvals"]
        save_dict[f"{metric}_pvals_corrected"] = results[metric]["pvals_corrected"]
        save_dict[f"{metric}_sig_mask"] = results[metric]["sig_mask"]

    np.savez_compressed(results_path, **save_dict)
    # Companion JSON sidecar with the per-subject breakdown
    import json as _json
    sidecar = results_path.with_suffix("").with_suffix(".json")
    sidecar = results_path.with_name(results_path.stem + "_metadata.json")
    with open(sidecar, "w") as f:
        _json.dump(
            {
                "correction": args.correction,
                "alpha": args.alpha,
                "aggregate": args.aggregate,
                "drop_bad_trials": not args.keep_bad_trials,
                "bad_trial_rule": bad_trial_rule,
                "interp_reject_threshold": interp_reject_threshold,
                "inout_bounds": list(inout_bounds),
                "n_subjects": len(subj_means_in),
                "n_trials_in": int(total_in),
                "n_trials_out": int(total_out),
                "n_bad_excluded": int(total_bad_excluded),
                "per_subject": per_subject,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved results to {results_path}")
    logger.info(f"Saved metadata to {sidecar}")

    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
