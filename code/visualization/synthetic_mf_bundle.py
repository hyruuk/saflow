"""Generate a synthetic multifeature bundle npz for panel prototyping.

Writes a file in the same shape as ``aggregate_multifeature.aggregate`` so the
panel can be exercised end-to-end before the real combined-24 reruns finish.

Structure (per axis):
  per-cell/observed         (n_spatial, n_features)
  per-cell/perm_scores      (n_perms, n_spatial, n_features)
  per-cell/pvals_uncorrected/tmax/fdr_bh/bonferroni
  per-spatial/observed      (n_spatial,)
  per-spatial/perm_scores   (n_perms, n_spatial)
  per-spatial/importances   (n_spatial, n_features)
  per-spatial/pvals_*
  per-feature/observed      (n_features,)
  per-feature/perm_scores   (n_perms, n_features)
  per-feature/importances   (n_features, n_spatial)
  per-feature/pvals_*
  joint/observed            scalar
  joint/perm_scores         (n_perms,)
  joint/importances         (n_spatial, n_features)
  spatial_names, feature_names

Usage:
    python -m code.visualization.synthetic_mf_bundle --space schaefer_400 \
        --trial-type alltrials --out synthetic_bundle.npz
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# Combined feature list (matches expand_feature_set("all")).
DEFAULT_FEATURES = [
    "fooof_exponent", "fooof_offset", "fooof_r_squared",
    "psd_delta", "psd_theta", "psd_alpha", "psd_lobeta", "psd_hibeta",
    "psd_gamma1", "psd_gamma2", "psd_gamma3",
    "psd_corrected_delta", "psd_corrected_theta", "psd_corrected_alpha",
    "psd_corrected_lobeta", "psd_corrected_hibeta",
    "psd_corrected_gamma1", "psd_corrected_gamma2", "psd_corrected_gamma3",
    "complexity_lzc_median", "complexity_entropy_permutation",
    "complexity_entropy_spectral", "complexity_entropy_svd",
    "complexity_fractal_petrosian",
]

# Spatial-name templates for synthetic generation. For schaefer_400 we copy
# names from a real bundle if available so brain-surface rendering works;
# otherwise we fabricate them.
SCHAEFER_REAL_BUNDLE = (
    "/home/hyruuk/DATA/cocolab/saflow/results/classification_schaefer_400/"
    "group_mf/feature-combined-29_space-schaefer_400_inout-2575_clf-logistic_"
    "cv-logo_axis-bundle_imp-permutation_level-epoch_type-alltrials_mf_scores.npz"
)


def _real_schaefer_names() -> List[str]:
    """Read 402 real Schaefer ROI names from the existing bundle (if present)."""
    p = Path(SCHAEFER_REAL_BUNDLE)
    if not p.exists():
        return [f"schaefer_400_parcel_{i:03d}" for i in range(402)]
    with np.load(p, allow_pickle=True) as npz:
        return list(npz["spatial_names"])


def _spatial_names(space: str, n_spatial: int) -> List[str]:
    if space == "schaefer_400":
        names = _real_schaefer_names()
        return names[:n_spatial] if len(names) >= n_spatial else names
    return [f"{space}_unit_{i:03d}" for i in range(n_spatial)]


def _hotspot_mask(
    spatial_names: List[str],
    feature_names: List[str],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize a sparse 'real effect' map.

    Returns:
        cell_effect:   (n_spatial, n_features), 0..1 effect strength per cell
        feature_score: (n_features,), 0..1 per-feature effect (max over space)
        spatial_score: (n_spatial,), 0..1 per-spatial effect (max over feat)
    """
    n_spatial = len(spatial_names)
    n_features = len(feature_names)

    # Pick "hot" features (alpha, theta, lzc, fooof_exponent) by name pattern.
    hot_features = {
        "psd_alpha", "psd_corrected_alpha", "psd_theta", "psd_corrected_theta",
        "fooof_exponent", "complexity_lzc_median",
    }
    feat_weights = np.array([
        1.0 if f in hot_features else 0.0 for f in feature_names
    ])

    # Random spatial hotspots — ~10% of parcels get boosted, with structure
    # biased toward parcels whose name contains 'Default' or 'Cont' (Yeo
    # default-mode / control networks, common attention-related findings).
    hot_idx = np.zeros(n_spatial, dtype=float)
    for i, name in enumerate(spatial_names):
        if "Default" in name or "Cont_" in name:
            hot_idx[i] = rng.uniform(0.4, 1.0)
        elif rng.random() < 0.05:
            hot_idx[i] = rng.uniform(0.2, 0.7)

    # Cell effect = outer product with noise
    cell_effect = (
        hot_idx[:, None] * feat_weights[None, :]
        * rng.uniform(0.5, 1.0, size=(n_spatial, n_features))
    )
    return cell_effect, feat_weights, hot_idx


def _build_axis_payloads(
    spatial_names: List[str],
    feature_names: List[str],
    n_perms: int = 1000,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Build all per-axis arrays for the synthetic bundle."""
    rng = np.random.default_rng(seed)
    n_spatial = len(spatial_names)
    n_features = len(feature_names)

    cell_effect, feat_score, spatial_score = _hotspot_mask(
        spatial_names, feature_names, rng,
    )

    payload: Dict[str, np.ndarray] = {}

    # ---- per-cell ------------------------------------------------------
    # Observed AUC = 0.5 + effect * scale + noise
    pc_obs = 0.5 + cell_effect * 0.12 + rng.normal(0, 0.012, (n_spatial, n_features))
    pc_perm = 0.5 + rng.normal(0, 0.012, (n_perms, n_spatial, n_features))
    pc_obs_flat = pc_obs.ravel()
    pc_perm_flat = pc_perm.reshape(n_perms, -1)
    pc_unc = (np.sum(pc_perm_flat >= pc_obs_flat[None, :], axis=0) + 1) / (n_perms + 1)
    pc_tmax_dist = np.nanmax(pc_perm_flat, axis=1)
    pc_tmax = (np.sum(pc_tmax_dist[:, None] >= pc_obs_flat[None, :], axis=0) + 1) / (n_perms + 1)
    # Simple BH-FDR
    pc_fdr_flat = _bh_fdr(pc_unc)
    pc_bonf_flat = np.minimum(pc_unc * pc_unc.size, 1.0)

    payload["per-cell/observed"] = pc_obs
    payload["per-cell/perm_scores"] = pc_perm
    payload["per-cell/pvals_uncorrected"] = pc_unc.reshape(pc_obs.shape)
    payload["per-cell/pvals_tmax"] = pc_tmax.reshape(pc_obs.shape)
    payload["per-cell/pvals_fdr_bh"] = pc_fdr_flat.reshape(pc_obs.shape)
    payload["per-cell/pvals_bonferroni"] = pc_bonf_flat.reshape(pc_obs.shape)

    # ---- per-spatial (multivariate over all features per parcel) -------
    ps_signal = spatial_score * 0.20
    ps_obs = 0.5 + ps_signal + rng.normal(0, 0.015, n_spatial)
    ps_perm = 0.5 + rng.normal(0, 0.015, (n_perms, n_spatial))
    ps_unc = (np.sum(ps_perm >= ps_obs[None, :], axis=0) + 1) / (n_perms + 1)
    ps_tmax_dist = np.nanmax(ps_perm, axis=1)
    ps_tmax = (np.sum(ps_tmax_dist[:, None] >= ps_obs[None, :], axis=0) + 1) / (n_perms + 1)
    ps_fdr = _bh_fdr(ps_unc)
    ps_bonf = np.minimum(ps_unc * n_spatial, 1.0)
    # Importances: same shape as cell_effect with extra noise (the per-parcel
    # classifier "found" the hot features).
    ps_imp = np.abs(cell_effect) + rng.normal(0, 0.02, (n_spatial, n_features)).clip(0)

    payload["per-spatial/observed"] = ps_obs
    payload["per-spatial/perm_scores"] = ps_perm
    payload["per-spatial/pvals_uncorrected"] = ps_unc
    payload["per-spatial/pvals_tmax"] = ps_tmax
    payload["per-spatial/pvals_fdr_bh"] = ps_fdr
    payload["per-spatial/pvals_bonferroni"] = ps_bonf
    payload["per-spatial/importances"] = ps_imp

    # ---- per-feature (multivariate over all parcels per feature) -------
    # Hot features have a clear signal; cold features barely above 0.5.
    pf_signal = feat_score * 0.15 + rng.uniform(0, 0.04, n_features)
    pf_obs = 0.5 + pf_signal + rng.normal(0, 0.015, n_features)
    pf_perm = 0.5 + rng.normal(0, 0.015, (n_perms, n_features))
    pf_unc = (np.sum(pf_perm >= pf_obs[None, :], axis=0) + 1) / (n_perms + 1)
    pf_tmax_dist = np.nanmax(pf_perm, axis=1)
    pf_tmax = (np.sum(pf_tmax_dist[:, None] >= pf_obs[None, :], axis=0) + 1) / (n_perms + 1)
    pf_fdr = _bh_fdr(pf_unc)
    pf_bonf = np.minimum(pf_unc * n_features, 1.0)
    pf_imp = np.abs(cell_effect.T) + rng.normal(0, 0.02, (n_features, n_spatial)).clip(0)

    payload["per-feature/observed"] = pf_obs
    payload["per-feature/perm_scores"] = pf_perm
    payload["per-feature/pvals_uncorrected"] = pf_unc
    payload["per-feature/pvals_tmax"] = pf_tmax
    payload["per-feature/pvals_fdr_bh"] = pf_fdr
    payload["per-feature/pvals_bonferroni"] = pf_bonf
    payload["per-feature/importances"] = pf_imp

    # ---- joint -------------------------------------------------------
    joint_signal = float(np.mean(cell_effect)) * 0.5
    j_obs = 0.5 + joint_signal + rng.normal(0, 0.005)
    j_perm = 0.5 + rng.normal(0, 0.005, n_perms)
    payload["joint/observed"] = np.asarray(j_obs)
    payload["joint/perm_scores"] = j_perm
    payload["joint/importances"] = cell_effect + rng.normal(0, 0.02, (n_spatial, n_features))

    # ---- shared ------------------------------------------------------
    payload["spatial_names"] = np.asarray(spatial_names)
    payload["feature_names"] = np.asarray(feature_names)

    return payload


def _bh_fdr(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Vectorized Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, dtype=float).ravel()
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    # Enforce monotonicity from the largest p down.
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty_like(p)
    out[order] = adj
    return out.reshape(pvals.shape)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Generate a synthetic multifeature bundle for panel prototyping."
    )
    p.add_argument("--space", default="schaefer_400",
                   choices=["sensor", "schaefer_400"])
    p.add_argument("--trial-type", default="alltrials")
    p.add_argument("--features", nargs="+", default=DEFAULT_FEATURES,
                   help="Feature names to synthesize (default: combined-24).")
    p.add_argument("--n-spatial", type=int, default=None,
                   help="Number of spatial units (default: 402 for schaefer, 270 for sensor).")
    p.add_argument("--n-perms", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="synthetic_mf_bundle.npz",
                   help="Output npz path (a metadata sibling .json is also written).")
    args = p.parse_args()

    n_spatial = args.n_spatial or (402 if args.space == "schaefer_400" else 270)
    feature_names = list(args.features)
    spatial_names = _spatial_names(args.space, n_spatial)

    payload = _build_axis_payloads(
        spatial_names, feature_names,
        n_perms=args.n_perms, seed=args.seed,
    )

    out_npz = Path(args.out)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **payload)

    meta = {
        "synthetic": True,
        "space": args.space,
        "trial_type": args.trial_type,
        "n_spatial": n_spatial,
        "n_features": len(feature_names),
        "n_perms": args.n_perms,
        "seed": args.seed,
        "features": feature_names,
        "spatial_names_source": (
            "real_bundle" if args.space == "schaefer_400" and Path(SCHAEFER_REAL_BUNDLE).exists()
            else "synthetic"
        ),
        "timestamp": datetime.now().isoformat(),
    }
    out_meta = out_npz.with_name(out_npz.name.replace("_scores.npz", "_metadata.json"))
    if out_meta == out_npz:
        out_meta = out_npz.with_suffix(".meta.json")
    out_meta.write_text(json.dumps(meta, indent=2))

    print(f"Wrote synthetic bundle -> {out_npz}")
    print(f"Wrote metadata          -> {out_meta}")
    print(f"n_spatial={n_spatial}  n_features={len(feature_names)}  n_perms={args.n_perms}")
    print(f"joint observed AUC = {float(payload['joint/observed']):.3f}")
    print(f"per-spatial max AUC = {payload['per-spatial/observed'].max():.3f}, "
          f"min = {payload['per-spatial/observed'].min():.3f}")
    print(f"per-cell tmax sig cells: "
          f"{int(np.sum(payload['per-cell/pvals_tmax'] < 0.05))} / "
          f"{payload['per-cell/pvals_tmax'].size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
