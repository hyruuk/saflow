"""Run t-tests on complexity measures (IN vs OUT) and generate figures.

This script:
1. Loads complexity data efficiently (vectorized)
2. Computes subject-level means for IN/OUT
3. Runs paired t-tests with optional corrections
4. Generates topographic figures

Usage:
    python -m code.statistics.run_complexity_stats
    python -m code.statistics.run_complexity_stats --correction fdr --alpha 0.05
    python -m code.statistics.run_complexity_stats --correction permutation --n-permutations 1000
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
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Load data from a single NPZ file efficiently.

    Returns:
        Tuple of (data_dict, vtc, task)
    """
    data = np.load(npz_path, allow_pickle=True)
    meta = data["trial_metadata"].item()

    data_dict = {m: data[m] for m in metrics}
    vtc = np.array(meta["VTC_filtered"])
    task = np.array(meta["task"])

    return data_dict, vtc, task


def compute_subject_means(
    data_root: Path,
    subject: str,
    runs: List[str],
    space: str,
    inout_bounds: Tuple[int, int],
    metrics: List[str],
) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, np.ndarray]], int, int]:
    """Compute subject-level means for IN and OUT conditions (vectorized).

    Returns:
        Tuple of (means_in, means_out, n_in, n_out)
    """
    complexity_dir = data_root / "features" / f"complexity_{space}"
    subj_dir = complexity_dir / f"sub-{subject}"

    if not subj_dir.exists():
        return None, None, 0, 0

    # Collect all data for this subject across runs
    all_vtc = []
    all_task = []
    all_data = {m: [] for m in metrics}

    for run in runs:
        pattern = f"sub-{subject}_*_run-{run}_*_desc-complexity.npz"
        files = list(subj_dir.glob(pattern))
        if not files:
            continue

        data_dict, vtc, task = load_subject_data(files[0], metrics)

        all_vtc.append(vtc)
        all_task.append(task)
        for m in metrics:
            all_data[m].append(data_dict[m])

    if not all_vtc:
        return None, None, 0, 0

    # Concatenate across runs
    all_vtc = np.concatenate(all_vtc)
    all_task = np.concatenate(all_task)
    for m in metrics:
        all_data[m] = np.concatenate(all_data[m], axis=0)

    # Compute IN/OUT bounds for this subject
    inbound = np.nanpercentile(all_vtc, inout_bounds[0])
    outbound = np.nanpercentile(all_vtc, inout_bounds[1])

    # Create masks (vectorized)
    task_mask = all_task == "correct_commission"
    in_mask = task_mask & (all_vtc <= inbound)
    out_mask = task_mask & (all_vtc >= outbound)

    n_in = np.sum(in_mask)
    n_out = np.sum(out_mask)

    if n_in == 0 or n_out == 0:
        return None, None, n_in, n_out

    # Compute means (vectorized)
    means_in = {m: np.nanmean(all_data[m][in_mask], axis=0) for m in metrics}
    means_out = {m: np.nanmean(all_data[m][out_mask], axis=0) for m in metrics}

    return means_in, means_out, n_in, n_out


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
        correction: 'none', 'fdr', 'bonferroni', 'permutation'
        alpha: Significance threshold
        n_permutations: Number of permutations (for permutation correction)
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

        elif correction == "permutation":
            # Max-stat permutation test
            if subj_means_in is None or subj_means_out is None:
                logger.warning("Permutation test requires subject data, falling back to FDR")
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
                        choices=["none", "fdr", "bonferroni", "permutation"],
                        help="Multiple comparison correction (default: fdr)")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold")
    parser.add_argument("--n-permutations", type=int, default=1000,
                        help="Number of permutations (for permutation correction)")
    parser.add_argument("--metrics", nargs="+", default=None,
                        help="Specific metrics to analyze (default: all)")
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

    logger.info(f"Space: {args.space}")
    logger.info(f"Correction: {args.correction}")
    logger.info(f"Alpha: {args.alpha}")
    logger.info(f"Subjects: {len(subjects)}")

    # Process subjects
    logger.info("Computing subject-level means...")
    subj_means_in = []
    subj_means_out = []
    total_in, total_out = 0, 0

    for subject in subjects:
        means_in, means_out, n_in, n_out = compute_subject_means(
            data_root, subject, runs, args.space, inout_bounds, metrics
        )
        if means_in is not None:
            subj_means_in.append(means_in)
            subj_means_out.append(means_out)
            total_in += n_in
            total_out += n_out
            logger.info(f"  sub-{subject}: {n_in} IN, {n_out} OUT")

    logger.info(f"Total: {len(subj_means_in)} subjects, {total_in} IN, {total_out} OUT trials")

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
    results_path = data_root / "features" / f"statistics_{args.space}" / "complexity_ttest_results.npz"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "correction": args.correction,
        "alpha": args.alpha,
        "n_subjects": len(subj_means_in),
        "n_trials_in": total_in,
        "n_trials_out": total_out,
    }
    for metric in results:
        save_dict[f"{metric}_tvals"] = results[metric]["tvals"]
        save_dict[f"{metric}_pvals"] = results[metric]["pvals"]
        save_dict[f"{metric}_pvals_corrected"] = results[metric]["pvals_corrected"]
        save_dict[f"{metric}_sig_mask"] = results[metric]["sig_mask"]

    np.savez_compressed(results_path, **save_dict)
    logger.info(f"Saved results to {results_path}")

    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
