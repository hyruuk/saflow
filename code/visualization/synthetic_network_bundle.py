"""Generate a full synthetic results tree for the network story panel.

Writes the set of npz files that ``network_story_panel.py`` reads, under
a sandbox directory whose layout mirrors the real ``data_root/results/``
tree. Combined with the panel's ``--data-root`` override, this lets us
iterate on the figure layout before the real HPC results land.

Layout written under ``<out-root>/results/``:

  statistics_<space>/
      feature-<feat>_inout-<X>_test-paired_ttest_level-<lvl>_type-<trial>_results.npz
      group/networks/
          stats-networks_yeo{N}_type-<trial>_correction-<corr>.npz
          coherence_feature-<feat>_yeo{N}_type-<trial>.npz
  classification_<space>/
      group/
          feature-<feat>_space-<space>_inout-<X>_clf-<clf>_cv-<cv>_mode-univariate_level-<lvl>_type-<trial>_scores.npz
      group_mf/networks/
          classif-networks_yeo{N}_scope-per-family_type-<trial>_clf-<clf>_cv-<cv>.npz
          classif-networks_yeo{N}_scope-per-feature_type-<trial>_clf-<clf>_cv-<cv>.npz
          importance-networks_yeo{N}_label-<label>_clf-<clf>_cv-<cv>_type-<trial>.npz

All tiers are derived from one shared (n_parcels, n_features) "cell-effect"
map so the figure tells a coherent story: parcels hot in Tier 1A are
significant in Tier 2, classified well in Tier 1B / Tier 3, important in
Tier 4, and coherence-positive in the relevant feature panels.

Usage:
    python -m code.visualization.synthetic_network_bundle \\
        --out-root /tmp/synth_network --space schaefer_400 --yeo 7
    python -m code.visualization.network_story_panel \\
        --data-root /tmp/synth_network --space schaefer_400 \\
        --trial-type correct --yeo 7 --no-yeo-overlay
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# Same five "panel columns" as network_story_panel.FEATURE_COLUMNS.
PANEL_FEATURES: Tuple[str, ...] = (
    "psd_corrected_alpha",
    "psd_corrected_theta",
    "fooof_exponent",
    "fooof_offset",
    "complexity_lzc_median",
)

# Wider set fed to Tier-4 joint importance (matches expand_feature_set("all")).
ALL_FEATURES: Tuple[str, ...] = (
    "fooof_exponent", "fooof_offset", "fooof_r_squared",
    "psd_delta", "psd_theta", "psd_alpha", "psd_lobeta", "psd_hibeta",
    "psd_gamma1", "psd_gamma2", "psd_gamma3",
    "psd_corrected_delta", "psd_corrected_theta", "psd_corrected_alpha",
    "psd_corrected_lobeta", "psd_corrected_hibeta",
    "psd_corrected_gamma1", "psd_corrected_gamma2", "psd_corrected_gamma3",
    "complexity_lzc_median", "complexity_entropy_permutation",
    "complexity_entropy_spectral", "complexity_entropy_svd",
    "complexity_fractal_petrosian",
)

FAMILIES: Tuple[str, ...] = ("psds_corrected", "fooof", "complexity")
TRIAL_TYPES: Tuple[str, ...] = ("alltrials", "correct", "lapse")

# Features that get "real signal" in the synthetic effect map.
HOT_FEATURES = {
    "psd_alpha", "psd_corrected_alpha",
    "psd_theta", "psd_corrected_theta",
    "fooof_exponent", "fooof_offset",
    "complexity_lzc_median",
}

# Trial-type modulation: lapse has a larger effect than correct (a common
# pattern in our analyses), alltrials sits between the two.
TRIAL_EFFECT_GAIN = {"correct": 0.85, "alltrials": 1.0, "lapse": 1.25}

SCHAEFER_REAL_BUNDLE = Path(
    "/home/hyruuk/DATA/cocolab/saflow/results/classification_schaefer_400/"
    "group_mf/feature-combined-29_space-schaefer_400_inout-2575_clf-logistic_"
    "cv-logo_axis-bundle_imp-permutation_level-epoch_type-alltrials_mf_scores.npz"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_real_schaefer_names() -> List[str]:
    """Load canonical Schaefer-400 ROI names.

    Prefers reading from an existing real bundle (preserves the actual
    ordering used by the rest of the pipeline). Falls back to MNE's
    ``read_labels_from_annot`` so the helper works on a fresh checkout.
    """
    if SCHAEFER_REAL_BUNDLE.exists():
        with np.load(SCHAEFER_REAL_BUNDLE, allow_pickle=True) as npz:
            return [str(n) for n in npz["spatial_names"]]

    logger.info("Real Schaefer bundle not found; loading names from annot.")
    from code.visualization.plot_surface import _load_roi_names_from_atlas
    names = _load_roi_names_from_atlas("schaefer_400")
    if not names:
        raise RuntimeError(
            "Cannot resolve Schaefer-400 ROI names. Run `invoke get.atlases` "
            "or generate at least one real feature file first."
        )
    return names


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Vectorized Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, dtype=float).ravel()
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty_like(p)
    out[order] = adj
    return out.reshape(pvals.shape)


def _build_hotspot_mask(
    parcel_names: Sequence[str],
    feature_names: Sequence[str],
    yeo: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Synthesize a sparse parcel × feature effect map.

    Returns
    -------
    cell_effect : (n_parcels, n_features) in [0, 1]ish — effect strength
        biased toward Default/Cont parcels × HOT_FEATURES.
    parcel_score : (n_parcels,) per-parcel max-effect (used for the
        per-spatial AUC story so Tier 1B aligns with Tier 1A).
    """
    from code.utils.yeo_networks import get_network_assignments

    n_parcels = len(parcel_names)
    n_features = len(feature_names)

    feat_weights = np.array(
        [1.0 if f in HOT_FEATURES else 0.0 for f in feature_names]
    )

    assignments = get_network_assignments(parcel_names, n_networks=7)
    hot_idx = np.zeros(n_parcels, dtype=float)
    for i, net in enumerate(assignments):
        if net in ("Default", "Cont"):
            hot_idx[i] = rng.uniform(0.5, 1.0)
        elif net in ("DorsAttn", "SalVentAttn"):
            hot_idx[i] = rng.uniform(0.15, 0.55)
        elif net == "Unknown":
            hot_idx[i] = 0.0
        elif rng.random() < 0.05:
            hot_idx[i] = rng.uniform(0.1, 0.4)

    sign = np.where(np.array(
        [a in ("Default",) for a in assignments]
    ), -1.0, 1.0)

    cell_effect = (
        sign[:, None] * hot_idx[:, None] * feat_weights[None, :]
        * rng.uniform(0.6, 1.0, size=(n_parcels, n_features))
    )
    parcel_score = np.abs(cell_effect).max(axis=1)
    return cell_effect, parcel_score


def _per_parcel_stats(
    cell_effect_col: np.ndarray,
    rng: np.random.Generator,
    *,
    t_scale: float = 4.0,
    noise: float = 0.6,
) -> Dict[str, np.ndarray]:
    """Synthesize one feature's per-parcel paired-t stats payload."""
    n = cell_effect_col.size
    tvals = cell_effect_col * t_scale + rng.normal(0, noise, n)
    # Two-sided p from a heuristic normal mapping (rough but OK for layout).
    from scipy.stats import norm
    pvals_unc = 2 * norm.sf(np.abs(tvals))
    pvals_fdr = _bh_fdr(pvals_unc)
    pvals_bonf = np.minimum(pvals_unc * n, 1.0)
    # Synthetic tmax (max-statistic) — slightly stricter than FDR
    pvals_tmax = np.minimum(pvals_unc * (n * 0.4), 1.0)
    # Cohen's d ≈ t / sqrt(N) with N ~32 subjects
    d_vals = tvals / np.sqrt(32)
    return {
        "tvals": tvals,
        "pvals_uncorrected": pvals_unc,
        "pvals_fdr_bh": pvals_fdr,
        "pvals_bonferroni": pvals_bonf,
        "pvals_tmax": pvals_tmax,
        "cohens_d": d_vals,
    }


def _per_parcel_classif(
    cell_effect_col: np.ndarray,
    rng: np.random.Generator,
    *,
    auc_scale: float = 0.18,
    noise: float = 0.025,
) -> Dict[str, np.ndarray]:
    """Synthesize one feature's per-parcel univariate-classif AUC payload."""
    n = cell_effect_col.size
    observed = 0.5 + np.abs(cell_effect_col) * auc_scale + rng.normal(0, noise, n)
    observed = np.clip(observed, 0.45, 0.95)
    # Mock perm: scores around chance with noise
    n_perms = 200
    perm = 0.5 + rng.normal(0, noise, (n_perms, n))
    pvals_unc = (np.sum(perm >= observed[None, :], axis=0) + 1) / (n_perms + 1)
    tmax_dist = np.nanmax(perm, axis=1)
    pvals_tmax = (np.sum(tmax_dist[:, None] >= observed[None, :], axis=0) + 1) / (n_perms + 1)
    pvals_fdr = _bh_fdr(pvals_unc)
    pvals_bonf = np.minimum(pvals_unc * n, 1.0)
    return {
        "observed": observed,
        "perm_scores": perm,
        "pvals_uncorrected": pvals_unc,
        "pvals_tmax": pvals_tmax,
        "pvals_fdr_bh": pvals_fdr,
        "pvals_bonferroni": pvals_bonf,
    }


def _aggregate_to_network(
    values: np.ndarray,
    parcel_names: Sequence[str],
    yeo: int,
    agg: str = "mean",
) -> Tuple[np.ndarray, List[str]]:
    """Mirror code.utils.yeo_networks.aggregate_to_networks (last-axis variant)."""
    from code.utils.yeo_networks import (
        UNKNOWN_NETWORK,
        get_network_assignments,
        network_order,
    )

    assignments = get_network_assignments(parcel_names, n_networks=yeo)
    nets = list(network_order(yeo))
    out = np.full(len(nets), np.nan)
    for j, net in enumerate(nets):
        idx = np.where(assignments == net)[0]
        if idx.size == 0:
            continue
        vals = values[..., idx]
        if agg == "mean":
            out[j] = float(np.nanmean(vals))
        elif agg == "signed_count":
            out[j] = float((vals > 0).sum() - (vals < 0).sum())
        elif agg == "sum":
            out[j] = float(np.nansum(vals))
    return out, nets


def _write_npz(path: Path, payload: Dict[str, np.ndarray],
               provenance: Optional[Dict] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if provenance is not None:
        np.savez(path, **payload, meta=np.asarray(json.dumps(provenance)))
    else:
        np.savez(path, **payload)


# ---------------------------------------------------------------------------
# Per-trial-type writers
# ---------------------------------------------------------------------------

def _write_per_parcel_stats(
    stats_dir: Path,
    feature: str,
    trial: str,
    inout: str,
    level: str,
    cell_col: np.ndarray,
    rng: np.random.Generator,
) -> None:
    payload = _per_parcel_stats(cell_col, rng)
    fname = (f"feature-{feature}_inout-{inout}_test-paired_ttest"
             f"_level-{level}_type-{trial}_results.npz")
    _write_npz(stats_dir / fname, payload,
               provenance={"synthetic": True, "feature": feature,
                           "trial": trial, "level": level})


def _write_per_parcel_classif(
    clf_group_dir: Path,
    feature: str,
    trial: str,
    inout: str,
    clf: str,
    cv: str,
    space: str,
    level: str,
    cell_col: np.ndarray,
    rng: np.random.Generator,
) -> None:
    payload = _per_parcel_classif(cell_col, rng)
    fname = (f"feature-{feature}_space-{space}_inout-{inout}"
             f"_clf-{clf}_cv-{cv}_mode-univariate_level-{level}"
             f"_type-{trial}_scores.npz")
    _write_npz(clf_group_dir / fname, payload,
               provenance={"synthetic": True, "feature": feature,
                           "trial": trial, "level": level})


def _write_tier2_stats_agg(
    stats_net_dir: Path,
    trial: str,
    correction: str,
    yeo: int,
    cell_effect: np.ndarray,
    parcel_names: List[str],
    feature_names: Sequence[str],
    rng: np.random.Generator,
) -> None:
    from code.utils.yeo_networks import network_order, get_network_assignments

    nets = list(network_order(yeo))
    n_feat = len(feature_names)
    n_nets = len(nets)

    t_mean = np.full((n_feat, n_nets), np.nan)
    d_mean = np.full((n_feat, n_nets), np.nan)
    sc_sig = np.full((n_feat, n_nets), np.nan)
    p_pool = np.full((n_feat, n_nets), np.nan)

    assignments = get_network_assignments(parcel_names, n_networks=yeo)
    keep = assignments != "Unknown"

    for i, feat in enumerate(feature_names):
        stats = _per_parcel_stats(cell_effect[:, i], rng)
        tvals = stats["tvals"][keep]
        pvals_corr = stats[f"pvals_{ 'fdr_bh' if correction=='fdr' else correction}"][keep]
        d_arr = stats["cohens_d"][keep]
        ch_keep = [c for c, k in zip(parcel_names, keep) if k]

        t_block, _ = _aggregate_to_network(tvals, ch_keep, yeo, agg="mean")
        d_block, _ = _aggregate_to_network(d_arr, ch_keep, yeo, agg="mean")
        sig_t = np.where(pvals_corr < 0.05, tvals, 0.0)
        sc_block, _ = _aggregate_to_network(sig_t, ch_keep, yeo, agg="signed_count")
        t_mean[i, :] = t_block
        d_mean[i, :] = d_block
        sc_sig[i, :] = sc_block

        # Stouffer-pooled p per network (signed-z combination)
        from scipy.stats import norm
        net_assign_keep = assignments[keep]
        for j, net in enumerate(nets):
            idx = np.where(net_assign_keep == net)[0]
            if idx.size == 0:
                continue
            p = np.clip(pvals_corr[idx], 1e-300, 1.0)
            z = norm.isf(p / 2.0) * np.sign(tvals[idx])
            z_comb = z.sum() / np.sqrt(z.size) if z.size else np.nan
            p_pool[i, j] = 2.0 * norm.sf(abs(z_comb))

    n_per_net = np.array([
        int(np.sum((assignments[keep] == net))) for net in nets
    ], dtype=int)

    payload = {
        "features": np.array(feature_names),
        "network_names": np.array(nets),
        "tvals_mean": t_mean,
        "d_mean": d_mean,
        "signed_count_sig": sc_sig,
        "pooled_p_stouffer": p_pool,
        "n_parcels_per_network": n_per_net,
    }
    out = stats_net_dir / (
        f"stats-networks_yeo{yeo}_type-{trial}_correction-{correction}.npz"
    )
    _write_npz(out, payload, provenance={
        "synthetic": True, "trial": trial, "correction": correction, "yeo": yeo,
    })


def _write_tier3_per_family(
    classif_net_dir: Path,
    trial: str,
    clf: str,
    cv: str,
    yeo: int,
    cell_effect: np.ndarray,
    parcel_names: List[str],
    feature_names: Sequence[str],
    rng: np.random.Generator,
) -> None:
    from code.utils.yeo_networks import network_order, get_network_assignments

    nets = list(network_order(yeo))
    n_nets = len(nets)
    families = list(FAMILIES)

    # Per-family signal = mean |effect| over the family's hot features, by network
    family_to_features = {
        "psds_corrected": [f for f in feature_names if f.startswith("psd_corrected_")],
        "fooof":          [f for f in feature_names if f.startswith("fooof_")],
        "complexity":     [f for f in feature_names if f.startswith("complexity_")],
    }
    feat_to_idx = {f: i for i, f in enumerate(feature_names)}
    assignments = get_network_assignments(parcel_names, n_networks=yeo)

    scores = np.full((n_nets, len(families)), np.nan)
    pvals = np.full((n_nets, len(families)), np.nan)

    for j, fam in enumerate(families):
        cols = [feat_to_idx[f] for f in family_to_features[fam] if f in feat_to_idx]
        if not cols:
            continue
        fam_effect = np.abs(cell_effect[:, cols]).mean(axis=1)
        for i, net in enumerate(nets):
            idx = np.where(assignments == net)[0]
            if idx.size == 0:
                continue
            mean_eff = float(fam_effect[idx].mean())
            scores[i, j] = np.clip(
                0.5 + mean_eff * 0.25 + rng.normal(0, 0.012), 0.45, 0.92
            )
            # Heuristic p: stronger signal → smaller p
            pvals[i, j] = max(0.0005, np.exp(-5 * mean_eff) + rng.normal(0, 0.01))

    payload = {
        "network_names": np.array(nets),
        "families": np.array(families),
        "scores": scores,
        "pvals": pvals,
    }
    out = classif_net_dir / (
        f"classif-networks_yeo{yeo}_scope-per-family_type-{trial}"
        f"_clf-{clf}_cv-{cv}.npz"
    )
    _write_npz(out, payload, provenance={
        "synthetic": True, "scope": "per-family", "trial": trial, "yeo": yeo,
    })


def _write_tier3_per_feature(
    classif_net_dir: Path,
    trial: str,
    clf: str,
    cv: str,
    yeo: int,
    cell_effect: np.ndarray,
    parcel_names: List[str],
    feature_names: Sequence[str],
    rng: np.random.Generator,
) -> None:
    from code.utils.yeo_networks import network_order, get_network_assignments

    nets = list(network_order(yeo))
    n_nets = len(nets)
    n_feat = len(feature_names)
    assignments = get_network_assignments(parcel_names, n_networks=yeo)

    scores = np.full((n_nets, n_feat), np.nan)
    pvals = np.full((n_nets, n_feat), np.nan)

    for j, feat in enumerate(feature_names):
        col_effect = np.abs(cell_effect[:, j])
        for i, net in enumerate(nets):
            idx = np.where(assignments == net)[0]
            if idx.size == 0:
                continue
            mean_eff = float(col_effect[idx].mean())
            scores[i, j] = np.clip(
                0.5 + mean_eff * 0.22 + rng.normal(0, 0.012), 0.45, 0.92
            )
            pvals[i, j] = max(0.0005, np.exp(-5 * mean_eff) + rng.normal(0, 0.01))

    payload = {
        "network_names": np.array(nets),
        "features": np.array(feature_names),
        "scores": scores,
        "pvals": pvals,
    }
    out = classif_net_dir / (
        f"classif-networks_yeo{yeo}_scope-per-feature_type-{trial}"
        f"_clf-{clf}_cv-{cv}.npz"
    )
    _write_npz(out, payload, provenance={
        "synthetic": True, "scope": "per-feature", "trial": trial, "yeo": yeo,
    })


def _write_tier4_importance(
    classif_net_dir: Path,
    trial: str,
    clf: str,
    cv: str,
    yeo: int,
    label: str,
    cell_effect: np.ndarray,
    parcel_names: List[str],
    feature_names: Sequence[str],
    rng: np.random.Generator,
) -> None:
    from code.utils.yeo_networks import network_order, get_network_assignments

    nets = list(network_order(yeo))
    n_nets = len(nets)
    n_feat = len(feature_names)
    assignments = get_network_assignments(parcel_names, n_networks=yeo)

    # Per-parcel importance ≈ |effect| with noise (mimics permutation-importance)
    parcel_imp = np.abs(cell_effect) + rng.normal(0, 0.02,
                                                  cell_effect.shape).clip(min=0)
    importance_sum = np.full((n_nets, n_feat), np.nan)
    importance_mean = np.full((n_nets, n_feat), np.nan)
    for i, net in enumerate(nets):
        idx = np.where(assignments == net)[0]
        if idx.size == 0:
            continue
        importance_sum[i, :] = parcel_imp[idx].sum(axis=0)
        importance_mean[i, :] = parcel_imp[idx].mean(axis=0)

    payload = {
        "network_names": np.array(nets),
        "features": np.array(feature_names),
        "importance_sum": importance_sum,
        "importance_mean": importance_mean,
    }
    out = classif_net_dir / (
        f"importance-networks_yeo{yeo}_label-{label}_clf-{clf}_cv-{cv}"
        f"_type-{trial}.npz"
    )
    _write_npz(out, payload, provenance={
        "synthetic": True, "trial": trial, "yeo": yeo, "label": label,
    })


def _write_tier4_coherence(
    stats_net_dir: Path,
    feature: str,
    trial: str,
    yeo: int,
    cell_col: np.ndarray,
    parcel_names: List[str],
    rng: np.random.Generator,
) -> None:
    """Synth within > between r for hot features, ~0 for cold."""
    from code.utils.yeo_networks import network_order, get_network_assignments

    nets = list(network_order(yeo))
    assignments = get_network_assignments(parcel_names, n_networks=yeo)

    is_hot = feature in HOT_FEATURES
    within = np.full(len(nets), np.nan)
    between = np.full(len(nets), np.nan)
    for j, net in enumerate(nets):
        idx_in = np.where(assignments == net)[0]
        idx_out = np.where((assignments != net)
                           & (assignments != "Unknown"))[0]
        if idx_in.size == 0:
            continue
        if is_hot and net in ("Default", "Cont", "DorsAttn"):
            within[j] = 0.35 + rng.normal(0, 0.06)
            between[j] = 0.05 + rng.normal(0, 0.04)
        elif is_hot:
            within[j] = 0.12 + rng.normal(0, 0.05)
            between[j] = 0.04 + rng.normal(0, 0.04)
        else:
            within[j] = rng.normal(0, 0.06)
            between[j] = rng.normal(0, 0.05)

    payload = {
        "network_names": np.array(nets),
        "within_r": within,
        "between_r": between,
    }
    out = stats_net_dir / (
        f"coherence_feature-{feature}_yeo{yeo}_type-{trial}.npz"
    )
    _write_npz(out, payload, provenance={
        "synthetic": True, "feature": feature, "trial": trial, "yeo": yeo,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", required=True,
                   help="Sandbox directory to write the synthetic results "
                        "tree under (will create out-root/results/...).")
    p.add_argument("--space", default="schaefer_400",
                   choices=["schaefer_400"],
                   help="Atlas space. Only schaefer_400 is wired (need Schaefer "
                        "labels to derive Yeo assignments).")
    p.add_argument("--yeo", type=int, default=7, choices=[7, 17])
    p.add_argument("--trial-types", nargs="+", default=list(TRIAL_TYPES),
                   help="Trial-type variants to generate (default: all three).")
    p.add_argument("--features", nargs="+", default=list(PANEL_FEATURES),
                   help="Panel-column features (Tier 1 & 2 & coherence). "
                        "Default: 5 panel columns.")
    p.add_argument("--all-features", nargs="+", default=list(ALL_FEATURES),
                   help="Wider feature set for Tier-3 per-feature heatmap "
                        "and Tier-4 importance. Default: combined-24 list.")
    p.add_argument("--inout", default="2575")
    p.add_argument("--clf", default="logistic")
    p.add_argument("--classif-cv", default="logo")
    p.add_argument("--stats-level", default="average",
                   choices=["average", "epoch"])
    p.add_argument("--classif-level", default="epoch",
                   choices=["epoch", "average"])
    p.add_argument("--correction", default="fdr",
                   choices=["fdr", "tmax", "bonferroni", "uncorrected"],
                   help="Stats correction baked into the Tier-2 filename.")
    p.add_argument("--mf-label", default="all",
                   help="Tier-4 importance bundle label (must match the "
                        "panel's --mf-label).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_root = Path(args.out_root).resolve()
    results_root = out_root / "results"
    stats_dir = results_root / f"statistics_{args.space}"
    stats_net_dir = stats_dir / "group" / "networks"
    clf_dir = results_root / f"classification_{args.space}"
    clf_group_dir = clf_dir / "group"
    classif_net_dir = clf_dir / "group_mf" / "networks"

    rng = np.random.default_rng(args.seed)
    parcel_names = _load_real_schaefer_names()
    n_parcels = len(parcel_names)
    logger.info(f"Loaded {n_parcels} Schaefer parcels")

    # The wider feature set drives Tier-3 per-feature heatmap and Tier-4
    # importance; the 5 panel columns are a subset of it.
    feature_set = list(dict.fromkeys(list(args.all_features) + list(args.features)))
    panel_features = list(args.features)

    base_effect, _ = _build_hotspot_mask(
        parcel_names, feature_set, yeo=args.yeo, rng=rng,
    )

    for trial in args.trial_types:
        gain = TRIAL_EFFECT_GAIN.get(trial, 1.0)
        cell_effect = base_effect * gain
        feat_to_idx = {f: i for i, f in enumerate(feature_set)}

        logger.info(f"[{trial}] gain={gain:.2f}")

        # --- Tier 1A: per-parcel stats (panel features only) ---
        for feat in panel_features:
            col = cell_effect[:, feat_to_idx[feat]]
            _write_per_parcel_stats(
                stats_dir, feat, trial, args.inout, args.stats_level,
                col, rng,
            )

        # --- Tier 1B: per-parcel univariate classif (panel features only) ---
        for feat in panel_features:
            col = cell_effect[:, feat_to_idx[feat]]
            _write_per_parcel_classif(
                clf_group_dir, feat, trial, args.inout, args.clf,
                args.classif_cv, args.space, args.classif_level,
                col, rng,
            )

        # --- Tier 2: per-network stats aggregate ---
        _write_tier2_stats_agg(
            stats_net_dir, trial, args.correction, args.yeo,
            cell_effect, parcel_names, feature_set, rng,
        )

        # --- Tier 3a: per-family classif ---
        _write_tier3_per_family(
            classif_net_dir, trial, args.clf, args.classif_cv, args.yeo,
            cell_effect, parcel_names, feature_set, rng,
        )

        # --- Tier 3b: per-feature classif ---
        _write_tier3_per_feature(
            classif_net_dir, trial, args.clf, args.classif_cv, args.yeo,
            cell_effect, parcel_names, feature_set, rng,
        )

        # --- Tier 4: joint importance ---
        _write_tier4_importance(
            classif_net_dir, trial, args.clf, args.classif_cv, args.yeo,
            args.mf_label, cell_effect, parcel_names, feature_set, rng,
        )

        # --- Tier 4: coherence (per panel feature) ---
        for feat in panel_features:
            col = cell_effect[:, feat_to_idx[feat]]
            _write_tier4_coherence(
                stats_net_dir, feat, trial, args.yeo,
                col, parcel_names, rng,
            )

    manifest = {
        "synthetic": True,
        "timestamp": datetime.now().isoformat(),
        "seed": args.seed,
        "space": args.space,
        "yeo": args.yeo,
        "trial_types": args.trial_types,
        "panel_features": panel_features,
        "wider_features": feature_set,
        "n_parcels": n_parcels,
    }
    (out_root / "synthetic_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    logger.info(f"Wrote synthetic tree to {out_root}")
    logger.info(f"Render with:")
    logger.info(
        f"  python -m code.visualization.network_story_panel "
        f"--data-root {out_root} --space {args.space} "
        f"--yeo {args.yeo} --trial-type correct"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
