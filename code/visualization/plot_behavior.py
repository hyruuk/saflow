"""Behavioral analysis visualization for saflow.

This script generates publication-quality behavioral figures including:
- VTC (Variability Time Course) plot for individual subjects
- Group-level lapse rate and omission error rate comparisons
- Pre-event reaction time analysis (IN vs OUT conditions)

Usage:
    python -m code.visualization.plot_behavior
    python -m code.visualization.plot_behavior --subject 07 --run 4
    python -m code.visualization.plot_behavior --inout-bounds 50 50

Author: Claude (Anthropic)
Date: 2026-01-31
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from code.utils.behavioral import classify_trials_from_vtc, get_VTC_from_file
from code.utils.colors import EVENT_COLORS, VTC_COLORS, ZONE_COLORS, ZONE_PALETTE
from code.utils.config import load_config
from code.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def plot_VTC(
    VTC_filtered: np.ndarray,
    VTC_raw: np.ndarray,
    IN_mask: np.ndarray,
    OUT_mask: np.ndarray,
    performance_dict: Dict[str, List[int]],
    subject_name: str,
    run: str,
    ax: Optional[plt.Axes] = None,
    fig: Optional[plt.Figure] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot VTC time course with IN/OUT zones and performance markers.

    Args:
        VTC_filtered: Filtered VTC array.
        VTC_raw: Raw VTC array (before filtering).
        IN_mask: Boolean mask for IN zone trials.
        OUT_mask: Boolean mask for OUT zone trials.
        performance_dict: Dictionary with trial indices for each performance type.
        subject_name: Subject identifier.
        run: Run number.
        ax: Existing axes to plot on. If None, creates new figure.
        fig: Existing figure. If None, creates new figure.

    Returns:
        Tuple of (figure, axes).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 5))

    x = np.arange(len(VTC_filtered))

    # Create masked arrays for IN and OUT zones
    VTC_in = np.ma.masked_array(VTC_filtered, mask=~IN_mask)
    VTC_out = np.ma.masked_array(VTC_filtered, mask=~OUT_mask)

    # Plot VTC traces
    ax.plot(x, VTC_raw, color=VTC_COLORS["raw"], alpha=0.3, linewidth=1, label="Raw VTC")
    ax.plot(x, VTC_in, color=ZONE_COLORS["IN"], linewidth=2.5, label="IN zone")
    ax.plot(x, VTC_out, color=ZONE_COLORS["OUT"], linewidth=2.5, label="OUT zone")

    # Mark performance events (positioned around middle of graph)
    # Commission errors (lapses) - red X
    ce_idx = performance_dict.get("commission_error", [])
    if ce_idx:
        ax.scatter(
            ce_idx,
            [2.6] * len(ce_idx),
            marker="x",
            s=80,
            c=EVENT_COLORS["lapse"],
            linewidths=2,
            label=f"Lapse (n={len(ce_idx)})",
            zorder=5,
        )

    # Correct omissions - green circle
    co_idx = performance_dict.get("correct_omission", [])
    if co_idx:
        ax.scatter(
            co_idx,
            [2.5] * len(co_idx),
            marker="o",
            s=40,
            c=EVENT_COLORS["correct_omission"],
            alpha=0.7,
            label=f"Correct omission (n={len(co_idx)})",
            zorder=5,
        )

    # Omission errors - orange triangle
    oe_idx = performance_dict.get("omission_error", [])
    if oe_idx:
        ax.scatter(
            oe_idx,
            [2.4] * len(oe_idx),
            marker="^",
            s=50,
            c=EVENT_COLORS["omission_error"],
            label=f"Omission error (n={len(oe_idx)})",
            zorder=5,
        )

    ax.set_xlabel("Trial", fontsize=12)
    ax.set_ylabel("VTC (|z-score|)", fontsize=12)
    ax.set_title(f"Subject {subject_name}, Run {run}", fontsize=14)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(0, len(VTC_filtered))
    ax.set_ylim(0, 4)

    return fig, ax


def get_behavior_dict(
    files_list: List[str],
    subjects: List[str],
    runs: List[str],
    logs_dir: Path,
    inout_bounds: Tuple[int, int],
    filt_config: Dict,
) -> pd.DataFrame:
    """Aggregate behavioral data across subjects and runs.

    Args:
        files_list: List of behavioral log file names.
        subjects: List of subject IDs.
        runs: List of run numbers.
        logs_dir: Directory containing log files.
        inout_bounds: Tuple of (lower_percentile, upper_percentile) for zone classification.
        filt_config: Filter configuration dictionary.

    Returns:
        DataFrame with behavioral metrics per subject, run, and condition.
    """
    records = []

    for subject in subjects:
        for run in runs:
            try:
                (
                    _,
                    _,
                    VTC_raw,
                    VTC_filtered,
                    _,
                    _,
                    performance_dict,
                    df_response,
                    RT_array,
                ) = get_VTC_from_file(
                    subject=subject,
                    run=run,
                    files_list=files_list,
                    logs_dir=logs_dir,
                    filt_type=filt_config.get("type", "gaussian"),
                    filt_config=filt_config,
                )

                # Classify trials
                zones = classify_trials_from_vtc(VTC_filtered, inout_bounds)

                # Process each condition (IN, OUT)
                for cond, cond_idx in [("IN", zones["IN_idx"]), ("OUT", zones["OUT_idx"])]:
                    cond_set = set(cond_idx)

                    # Split performance by zone
                    ce = [x for x in performance_dict["commission_error"] if x in cond_set]
                    co = [x for x in performance_dict["correct_omission"] if x in cond_set]
                    oe = [x for x in performance_dict["omission_error"] if x in cond_set]
                    cc = [x for x in performance_dict["correct_commission"] if x in cond_set]

                    # Compute rates
                    n_rares = len(co) + len(ce)
                    lapse_rate = len(ce) / n_rares if n_rares > 0 else np.nan
                    n_freq = len(cc) + len(oe)
                    omission_error_rate = len(oe) / n_freq if n_freq > 0 else np.nan

                    # Compute pre-event RTs
                    def get_pre_rt(event_list: List[int], required_pre: List[int]) -> float:
                        """Get mean RT of trials preceding events."""
                        pre_trials = [
                            x - 1
                            for x in event_list
                            if x > 0 and (x - 1) in required_pre
                        ]
                        if pre_trials and len(pre_trials) > 0:
                            pre_rts = RT_array[pre_trials]
                            return float(np.nanmean(pre_rts[pre_rts > 0]))
                        return np.nan

                    cc_set = set(performance_dict["correct_commission"])
                    rt_pre_ce = get_pre_rt(ce, cc_set)
                    rt_pre_co = get_pre_rt(co, cc_set)
                    rt_pre_cc = get_pre_rt(cc, cc_set)

                    records.append(
                        {
                            "subject": subject,
                            "run": run,
                            "cond": cond,
                            "lapse_rate": lapse_rate,
                            "omission_error_rate": omission_error_rate,
                            "commission_error": len(ce),
                            "correct_omission": len(co),
                            "omission_error": len(oe),
                            "correct_commission": len(cc),
                            "n_rare": n_rares,
                            "RT_preCE": rt_pre_ce,
                            "RT_preOC": rt_pre_co,
                            "RT_preCC": rt_pre_cc,
                        }
                    )

            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"Could not process subject {subject}, run {run}: {e}")
                continue

    return pd.DataFrame(records)


def replace_missing_values(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Replace NaN values with subject mean across runs in the same condition.

    Args:
        df: DataFrame with behavioral data.
        col: Column name to fill NaN values.

    Returns:
        DataFrame with NaN values replaced.
    """
    df = df.copy()
    df[col] = df[col].fillna(df.groupby(["subject", "cond"])[col].transform("mean"))

    # Remove subjects that still have NaN (no valid values in any run)
    subjects_with_nan = df[df[col].isna()]["subject"].unique()
    if len(subjects_with_nan) > 0:
        logger.warning(f"Removing subjects with no valid {col} values: {subjects_with_nan}")
        df = df[~df["subject"].isin(subjects_with_nan)]

    return df


def full_behavior_plot(
    files_list: List[str],
    subjects: List[str],
    runs: List[str],
    logs_dir: Path,
    inout_bounds: Tuple[int, int],
    filt_config: Dict,
    example_subject: str = "07",
    example_run: str = "4",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Generate full behavioral analysis figure.

    Creates a publication-quality figure with:
    - Panel A: VTC time course for example subject
    - Panel B: Lapse rate (IN vs OUT)
    - Panel C: Omission error rate (IN vs OUT)
    - Panel D: Pre-event RTs by condition

    Args:
        files_list: List of behavioral log file names.
        subjects: List of subject IDs.
        runs: List of run numbers.
        logs_dir: Directory containing log files.
        inout_bounds: Zone classification bounds.
        filt_config: Filter configuration.
        example_subject: Subject to show in VTC plot.
        example_run: Run to show in VTC plot.
        output_path: Path to save figure. If None, doesn't save.

    Returns:
        Matplotlib figure object.
    """
    # Delay seaborn/statannotations import for optional dependency
    try:
        import seaborn as sns
        from statannotations.Annotator import Annotator
        from statannotations.stats.StatTest import StatTest
    except ImportError as e:
        logger.error(f"Missing visualization dependency: {e}")
        logger.error("Install with: pip install seaborn statannotations")
        raise

    # Get VTC for example subject
    (
        _,
        _,
        VTC_raw,
        VTC_filtered,
        _,
        _,
        performance_dict,
        _,
        _,
    ) = get_VTC_from_file(
        subject=example_subject,
        run=example_run,
        files_list=files_list,
        logs_dir=logs_dir,
        filt_type=filt_config.get("type", "gaussian"),
        filt_config=filt_config,
    )

    # Classify trials for example subject
    zones = classify_trials_from_vtc(VTC_filtered, inout_bounds)

    # Get group behavioral data
    plot_df = get_behavior_dict(
        files_list, subjects, runs, logs_dir, inout_bounds, filt_config
    )

    # Aggregate by subject (mean across runs) - select only numeric columns
    numeric_cols = [
        "lapse_rate", "omission_error_rate", "commission_error",
        "correct_omission", "omission_error", "correct_commission",
        "n_rare", "RT_preCE", "RT_preOC", "RT_preCC"
    ]
    subj_avg_df = plot_df.groupby(["subject", "cond"])[numeric_cols].mean().reset_index()

    # Set up statistics
    custom_test = StatTest(ttest_rel, "Paired t-test", "t-test")
    pairs = [("IN", "OUT")]

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs_outer = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35, height_ratios=[0.9, 0.9, 1.2])

    # Panel A: VTC plot (spans top 2 rows, first 3 columns)
    ax_vtc = fig.add_subplot(gs_outer[:2, :3])
    plot_VTC(
        VTC_filtered,
        VTC_raw,
        zones["IN_mask"],
        zones["OUT_mask"],
        performance_dict,
        subject_name=example_subject,
        run=example_run,
        ax=ax_vtc,
        fig=fig,
    )

    # Panel B: Lapse rate (IN=blue, OUT=orange)
    ax_lapse = fig.add_subplot(gs_outer[2, 0])
    sns.barplot(x="cond", y="lapse_rate", data=subj_avg_df, ax=ax_lapse, errorbar="se", palette=ZONE_PALETTE)
    annot = Annotator(ax_lapse, pairs, data=subj_avg_df, x="cond", y="lapse_rate")
    annot.configure(test=custom_test, comparisons_correction="fdr_bh", text_format="star")
    annot.apply_test().annotate()
    ax_lapse.set_ylabel("Lapse rate")
    ax_lapse.set_xlabel("")

    # Panel C: Omission error rate (IN=blue, OUT=orange)
    ax_oe = fig.add_subplot(gs_outer[2, 1])
    sns.barplot(x="cond", y="omission_error_rate", data=subj_avg_df, ax=ax_oe, errorbar="se", palette=ZONE_PALETTE)
    annot = Annotator(ax_oe, pairs, data=subj_avg_df, x="cond", y="omission_error_rate")
    annot.configure(test=custom_test, comparisons_correction="fdr_bh", text_format="star")
    annot.apply_test().annotate()
    ax_oe.set_ylabel("Omission error rate")
    ax_oe.set_xlabel("")

    # Panel D: Pre-event RTs
    ax_pre = fig.add_subplot(gs_outer[2, 2])

    # Prepare RT data for plotting
    df_corrected = replace_missing_values(plot_df.copy(), "RT_preCE")
    rt_cols = ["RT_preCE", "RT_preOC", "RT_preCC"]
    subj_avg_rt = df_corrected.groupby(["subject", "cond"])[rt_cols].mean().reset_index()
    rt_melt = subj_avg_rt[["RT_preCE", "RT_preOC", "RT_preCC", "cond"]].melt(id_vars="cond")
    rt_melt = rt_melt.replace(
        {"RT_preCE": "Pre-lapse", "RT_preOC": "Pre-CO", "RT_preCC": "Pre-baseline"}
    )

    sns.barplot(x="variable", y="value", hue="cond", data=rt_melt, ax=ax_pre, errorbar="se", palette=ZONE_PALETTE)
    rt_pairs = [
        (("Pre-lapse", "IN"), ("Pre-lapse", "OUT")),
        (("Pre-CO", "IN"), ("Pre-CO", "OUT")),
        (("Pre-baseline", "IN"), ("Pre-baseline", "OUT")),
    ]
    annot = Annotator(ax_pre, rt_pairs, data=rt_melt, x="variable", y="value", hue="cond")
    annot.configure(test=custom_test, comparisons_correction="fdr_bh", text_format="star")
    annot.apply_test().annotate()
    ax_pre.set_ylabel("RT (s)")
    ax_pre.set_xlabel("")
    ax_pre.legend(title="", loc="lower left")

    # Add panel labels
    for ax, letter in [(ax_vtc, "A"), (ax_lapse, "B"), (ax_oe, "C"), (ax_pre, "D")]:
        ax.text(
            -0.02 if ax == ax_vtc else -0.05,
            1.03,
            letter,
            transform=ax.transAxes,
            size=14,
            weight="bold",
        )

    # Save figure
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved figure to {output_path}")

    return fig


def main():
    """Main entry point for behavioral visualization."""
    parser = argparse.ArgumentParser(
        description="Generate behavioral analysis figures for saflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="07",
        help="Example subject for VTC plot (default: 07)",
    )
    parser.add_argument(
        "--run",
        type=str,
        default="4",
        help="Example run for VTC plot (default: 4)",
    )
    parser.add_argument(
        "--inout-bounds",
        type=int,
        nargs=2,
        default=[25, 75],
        metavar=("LOWER", "UPPER"),
        help="Percentile bounds for IN/OUT classification (default: 25 75)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for figure (default: reports/figures/behavior/)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(__name__, level=log_level)

    # Load config
    config = load_config(args.config)

    # Get paths
    data_root = Path(config["paths"]["data_root"])
    logs_dir = data_root / "sourcedata" / "behav"
    reports_dir = Path(config["paths"]["reports"])

    # Get subject and run lists
    subjects = config["bids"]["subjects"]
    runs = config["bids"]["task_runs"]

    # Get filter config
    filt_config = config["behavioral"]["vtc"]["filter"]

    # Get list of log files
    files_list = os.listdir(logs_dir)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        bounds_str = f"{args.inout_bounds[0]}_{args.inout_bounds[1]}"
        output_path = reports_dir / "figures" / "behavior" / f"behavior_panel_{bounds_str}.png"

    logger.info(f"Generating behavioral figure with bounds {args.inout_bounds}")
    logger.info(f"Example subject: {args.subject}, run: {args.run}")

    # Generate figure
    fig = full_behavior_plot(
        files_list=files_list,
        subjects=subjects,
        runs=runs,
        logs_dir=logs_dir,
        inout_bounds=tuple(args.inout_bounds),
        filt_config=filt_config,
        example_subject=args.subject,
        example_run=args.run,
        output_path=output_path,
    )

    logger.info("Done!")
    plt.show()


if __name__ == "__main__":
    main()
