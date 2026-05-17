"""Subject-spectrum (Path 1) feature loader for group-level statistics.

This module implements the "fit FOOOF on aggregated spectra" path for
PSD-derived features (raw band power, FOOOF-corrected band power, FOOOF
aperiodic parameters). Aggregation pools **all good trials** of a
``(subject, condition)`` across the subject's runs into a single mean PSD
(arithmetic mean of the per-epoch Welch PSDs), then fits FOOOF **once per
condition** — exactly two fits per subject (IN, OUT) per spatial unit.

Trade-offs vs the per-trial path:

- FOOOF is fit once per ``(subject, condition, channel)`` on a clean,
  trial-averaged spectrum — much more robust than fitting per 0.852 s
  trial. Aperiodic params and corrected PSDs are correspondingly less
  noisy.
- Each trial only contributes to its own condition's aperiodic. Within
  a subject the IN and OUT corrected spectra are therefore corrected by
  *different* aperiodic estimates — by design, this isolates the
  periodic IN/OUT difference from any IN/OUT difference in the 1/f
  baseline. Don't feed these features to a classifier (the corrected
  values encode their own condition label).

Complexity features are not handled here — they are per-trial scalars and
go through the existing trial-level loader + subject-median aggregation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from code.features.utils import select_window_mask
from code.utils.bad_trials import compute_run_bad_mask

logger = logging.getLogger(__name__)


# Feature-type → (feat-family, sub-key)
# - psd_<band>           → ("psd",        band_name)
# - psd_corrected_<band> → ("psd_corr",   band_name)
# - fooof_exponent       → ("fooof",      "exponent")
# - fooof_offset         → ("fooof",      "offset")
# - fooof_r_squared      → ("fooof",      "r_squared")
def _classify_feature(feature_type: str) -> Tuple[str, str]:
    if feature_type.startswith("psd_corrected_"):
        return "psd_corr", feature_type.replace("psd_corrected_", "")
    if feature_type.startswith("psd_"):
        return "psd", feature_type.replace("psd_", "")
    if feature_type.startswith("fooof_"):
        return "fooof", feature_type.replace("fooof_", "")
    raise ValueError(
        f"Feature '{feature_type}' is not PSD-derived; subject-spectrum path "
        f"only supports psd_*, psd_corrected_*, fooof_*."
    )


def _fit_fooof_group_on_psd(
    mean_psd: np.ndarray,
    freq_bins: np.ndarray,
    freq_range: Tuple[float, float],
    fooof_params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit FOOOF on a per-channel mean PSD and return aperiodic + corrected.

    Args:
        mean_psd: (n_channels, n_freqs) mean PSD (linear power).
        freq_bins: (n_freqs,) frequency vector in Hz.
        freq_range: (fmin, fmax) FOOOF fitting range.
        fooof_params: dict of FOOOF init kwargs.

    Returns:
        (exponent, offset, r_squared, corrected_psd) where the first three
        are shape ``(n_channels,)`` and ``corrected_psd`` is shape
        ``(n_channels, n_freqs)`` in log10 units (the residual after
        subtracting the extrapolated aperiodic component over the full
        frequency range).
    """
    from fooof import FOOOFGroup

    fg = FOOOFGroup(
        peak_width_limits=fooof_params.get("peak_width_limits", [1, 8]),
        max_n_peaks=fooof_params.get("max_n_peaks", 4),
        min_peak_height=fooof_params.get("min_peak_height", 0.10),
        peak_threshold=fooof_params.get("peak_threshold", 2.0),
        aperiodic_mode=fooof_params.get("aperiodic_mode", "fixed"),
        verbose=False,
    )
    fg.fit(freq_bins, mean_psd, freq_range=freq_range, n_jobs=1)

    n_chans = mean_psd.shape[0]
    exponent = np.full(n_chans, np.nan)
    offset = np.full(n_chans, np.nan)
    r_squared = np.full(n_chans, np.nan)
    corrected = np.full_like(mean_psd, np.nan, dtype=float)

    safe_freqs = np.where(freq_bins > 0, freq_bins, np.nan)
    log_psd = np.log10(np.where(mean_psd > 0, mean_psd, np.nan))

    for ch in range(n_chans):
        fm = fg.get_fooof(ch)
        if not fm.has_model:
            continue
        offs = fm.aperiodic_params_[0]
        expn = fm.aperiodic_params_[-1]  # last param: exponent (knee mode adds a knee in between)
        exponent[ch] = expn
        offset[ch] = offs
        r_squared[ch] = fm.r_squared_
        aperiodic_full = offs - np.log10(safe_freqs ** expn)
        corrected[ch] = log_psd[ch] - aperiodic_full

    return exponent, offset, r_squared, corrected


def load_subject_spectrum_features(
    feature_types: List[str],
    space: str,
    inout_bounds: Tuple[int, int],
    config: Dict[str, Any],
    subjects: Optional[List[str]] = None,
    drop_bad_trials: bool = True,
    n_jobs: int = -1,
    trial_type: str = "alltrials",
    zoning: str = "per-subject",
    n_events_window: int = 8,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]]:
    """Load PSD-derived features through Path 1 (subject-spectrum FOOOF).

    Per ``(subject, condition)``: pool every good epoch's Welch PSD across
    the subject's runs, take the arithmetic mean to get one spectrum, fit
    FOOOF once on it, then derive each requested feature. Exactly two FOOOF
    fits per subject (IN, OUT).

    Returns a dict mapping ``feature_type -> (X, y, groups, metadata)``,
    matching the shape contract of :func:`load_all_features_batched`. ``X``
    has shape ``(1, 2 * n_subjects_kept, n_spatial)`` where rows alternate
    IN, OUT for each kept subject; ``y`` is ``[0, 1, 0, 1, ...]``;
    ``groups`` is ``[s0, s0, s1, s1, ...]``. The existing paired-t-test
    code path consumes this directly.

    Only feature_types from the PSD/FOOOF families are supported. Mix of
    ``psd_*``, ``psd_corrected_*`` and ``fooof_*`` is fine — the FOOOF fit
    is computed once per ``(subject, condition)`` and reused across every
    requested feature.
    """
    if not feature_types:
        raise ValueError("feature_types must be non-empty")
    parsed = [(ft, *_classify_feature(ft)) for ft in feature_types]

    analysis_cfg = config.get("analysis", {})
    bad_trial_rule = str(analysis_cfg.get("bad_trial_rule", "ar2"))
    interp_reject_threshold = int(analysis_cfg.get("interp_reject_threshold", 0) or 0)

    # Where the raw welch PSDs live
    data_root = Path(config["paths"]["data_root"])
    welch_root = data_root / config["paths"]["features"] / f"welch_psds_{space}"
    if not welch_root.exists():
        raise FileNotFoundError(f"Welch PSD folder not found: {welch_root}")

    if subjects is None:
        subjects = list(config["bids"]["subjects"])
    runs = list(config["bids"]["task_runs"])
    freq_bands: Dict[str, List[float]] = (
        config.get("features", {}).get("frequency_bands", {})
    )
    fooof_cfg_list = config.get("features", {}).get("fooof", [{}])
    fooof_params = fooof_cfg_list[0] if isinstance(fooof_cfg_list, list) else fooof_cfg_list
    freq_range = tuple(fooof_params.get("freq_range", [2.0, 40.0]))

    # Container for each subject's per-condition aggregate (mean across runs)
    # subj_cond_agg[subj] = {"IN": {feat_key: np.ndarray (n_chans,)},
    #                       "OUT": {feat_key: np.ndarray (n_chans,)}}
    # Where feat_key is the *full* feature_type (e.g. 'psd_alpha').
    subj_cond_agg: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    per_subject_counts: Dict[str, Dict[str, Any]] = {}
    input_git_hashes: set = set()
    bad_metadata_present = False

    # File desc suffix depends on the window size used by compute_welch_psd
    desc_suffix = "welch" if n_events_window <= 1 else f"welchw{n_events_window}"

    for subj_idx, subject in enumerate(
        tqdm(subjects, desc="Loading subjects (subject-spectrum)", unit="subj")
    ):
        subj_dir = welch_root / f"sub-{subject}"
        if not subj_dir.exists():
            continue

        # First pass: collect VTC + tasks + included_task + bads from every run
        # so per-subject (or per-run) IN/OUT thresholds can be computed on ALL
        # epochs (matches Path-2 default).
        all_vtc: List[np.ndarray] = []
        all_task: List[np.ndarray] = []
        all_inc_task: List[List[np.ndarray]] = []
        all_bad: List[np.ndarray] = []
        run_psd_files: List[Path] = []

        for run in runs:
            files = list(
                subj_dir.glob(f"sub-{subject}_*_run-{run}_*_desc-{desc_suffix}_psds.npz")
            )
            if not files:
                continue
            file_path = files[0]
            run_psd_files.append(file_path)

            params_file = file_path.with_name(file_path.stem + "_params.json")
            if params_file.exists():
                try:
                    params = json.loads(params_file.read_text())
                    if "git_hash" in params:
                        input_git_hashes.add(params["git_hash"])
                except Exception:
                    pass

            with np.load(file_path, allow_pickle=True) as npz:
                meta = npz["trial_metadata"].item()
                # Use window_vtc_mean as the anchor in window mode, else
                # per-trial VTC_filtered.
                if "window_vtc_mean" in meta:
                    run_vtc = np.asarray(meta["window_vtc_mean"], dtype=float)
                else:
                    run_vtc = np.asarray(meta["VTC_filtered"], dtype=float)
                all_vtc.append(run_vtc)
                all_task.append(np.asarray(meta["task"]))
                if "included_task" in meta:
                    all_inc_task.append([np.asarray(t) for t in meta["included_task"]])
                else:
                    all_inc_task.append([np.array([t]) for t in meta["task"]])
                all_bad.append(
                    compute_run_bad_mask(
                        meta, len(run_vtc), bad_trial_rule, interp_reject_threshold
                    )
                )

        if not run_psd_files:
            continue

        subj_vtc = np.concatenate(all_vtc)
        subj_task = np.concatenate(all_task)
        subj_inc_flat = [arr for run_list in all_inc_task for arr in run_list]
        subj_bad = np.concatenate(all_bad) if all_bad else np.zeros_like(subj_vtc, dtype=bool)
        if subj_bad.any():
            bad_metadata_present = True

        # Pooled (subject-level) thresholds; per-run bounds applied below.
        pooled_inbound = np.nanpercentile(subj_vtc, inout_bounds[0])
        pooled_outbound = np.nanpercentile(subj_vtc, inout_bounds[1])

        # Walk runs again: collect every good IN / OUT epoch PSD, then pool
        # them across the subject's runs into a single mean spectrum per
        # condition (pooled grand mean — one FOOOF fit per condition).
        in_epoch_psds: List[np.ndarray] = []
        out_epoch_psds: List[np.ndarray] = []
        n_in_total, n_out_total, n_bad_in, n_bad_out = 0, 0, 0, 0
        freqs: Optional[np.ndarray] = None

        for file_path, vtc, task, inc_task_run, bad in zip(
            run_psd_files, all_vtc, all_task, all_inc_task, all_bad
        ):
            with np.load(file_path, allow_pickle=True) as npz:
                psd_block = np.asarray(npz["psds"])  # (n_epochs, n_chans, n_freqs)
                freqs = np.asarray(npz["freqs"])

            if zoning == "per-run":
                if np.all(np.isnan(vtc)):
                    continue
                inbound = np.nanpercentile(vtc, inout_bounds[0])
                outbound = np.nanpercentile(vtc, inout_bounds[1])
            else:
                inbound = pooled_inbound
                outbound = pooled_outbound

            task_mask = select_window_mask(
                included_task_per_epoch=inc_task_run,
                task_per_epoch=task,
                type_how=trial_type,
            )
            in_zone = task_mask & (vtc <= inbound)
            out_zone = task_mask & (vtc >= outbound)
            n_bad_in += int((in_zone & bad).sum()) if drop_bad_trials else 0
            n_bad_out += int((out_zone & bad).sum()) if drop_bad_trials else 0

            good_mask = ~bad if drop_bad_trials else np.ones_like(bad, dtype=bool)
            in_mask = in_zone & good_mask
            out_mask = out_zone & good_mask
            if in_mask.any():
                in_epoch_psds.append(psd_block[in_mask])
                n_in_total += int(in_mask.sum())
            if out_mask.any():
                out_epoch_psds.append(psd_block[out_mask])
                n_out_total += int(out_mask.sum())

        # Need both conditions populated somewhere across the runs.
        if not in_epoch_psds or not out_epoch_psds or freqs is None:
            continue

        # Pooled grand-mean PSD per condition → one FOOOF fit each.
        cond_agg: Dict[str, Dict[str, np.ndarray]] = {}
        for cond, epoch_psds in (("IN", in_epoch_psds), ("OUT", out_epoch_psds)):
            mean_psd = np.nanmean(np.concatenate(epoch_psds, axis=0), axis=0)
            exp, offs, r2, corr = _fit_fooof_group_on_psd(
                mean_psd, freqs, freq_range, fooof_params
            )
            cond_agg[cond] = dict(
                mean_psd=mean_psd,
                corrected_psd=corr,
                exponent=exp,
                offset=offs,
                r_squared=r2,
            )
        # Now derive the requested features from the run-mean aggregates.
        cond_features: Dict[str, Dict[str, np.ndarray]] = {"IN": {}, "OUT": {}}
        # Cache band masks once per call
        for ft, family, sub_key in parsed:
            if family in ("psd", "psd_corr"):
                if sub_key not in freq_bands:
                    raise ValueError(
                        f"Band '{sub_key}' not in config.features.frequency_bands"
                    )
                fmin, fmax = freq_bands[sub_key]
                # Band mask aligned to the run-level frequency vector
                # (identical across runs; captured in the pooling loop above).
                fmask = (freqs >= fmin) & (freqs <= fmax)
                for cond in ("IN", "OUT"):
                    if family == "psd":
                        # Average linear-power band across band freqs
                        cond_features[cond][ft] = np.nanmean(
                            cond_agg[cond]["mean_psd"][:, fmask], axis=1
                        )
                    else:  # psd_corr
                        # corrected is in log10 units; band-average is fine
                        cond_features[cond][ft] = np.nanmean(
                            cond_agg[cond]["corrected_psd"][:, fmask], axis=1
                        )
            elif family == "fooof":
                if sub_key not in ("exponent", "offset", "r_squared"):
                    raise ValueError(
                        f"Unsupported fooof sub-key '{sub_key}'. "
                        f"Use exponent, offset, or r_squared."
                    )
                for cond in ("IN", "OUT"):
                    cond_features[cond][ft] = cond_agg[cond][sub_key]
            else:
                raise AssertionError(f"Unhandled family {family}")

        subj_cond_agg[subject] = cond_features
        per_subject_counts[subject] = {
            "n_in": n_in_total,
            "n_out": n_out_total,
            "n_bad_in": n_bad_in,
            "n_bad_out": n_bad_out,
        }

    if not subj_cond_agg:
        raise ValueError("No data loaded for any subject in subject-spectrum mode")

    # Pack each requested feature into the (X, y, groups, metadata) contract.
    kept_subjects = list(subj_cond_agg.keys())
    n_subj = len(kept_subjects)
    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]] = {}
    total_in = sum(per_subject_counts[s]["n_in"] for s in kept_subjects)
    total_out = sum(per_subject_counts[s]["n_out"] for s in kept_subjects)
    total_bad = sum(
        per_subject_counts[s]["n_bad_in"] + per_subject_counts[s]["n_bad_out"]
        for s in kept_subjects
    )

    for ft, family, sub_key in parsed:
        in_rows = np.stack([subj_cond_agg[s]["IN"][ft] for s in kept_subjects], axis=0)
        out_rows = np.stack([subj_cond_agg[s]["OUT"][ft] for s in kept_subjects], axis=0)
        # interleave IN, OUT per subject so groups[2*i] == groups[2*i+1]
        n_spatial = in_rows.shape[1]
        X = np.empty((1, 2 * n_subj, n_spatial), dtype=float)
        y = np.empty(2 * n_subj, dtype=int)
        groups = np.empty(2 * n_subj, dtype=int)
        for i in range(n_subj):
            X[0, 2 * i, :] = in_rows[i]
            X[0, 2 * i + 1, :] = out_rows[i]
            y[2 * i] = 0
            y[2 * i + 1] = 1
            groups[2 * i] = i
            groups[2 * i + 1] = i

        metadata = {
            "feature_type": ft,
            "space": space,
            "inout_bounds": list(inout_bounds),
            "n_subjects": n_subj,
            "n_trials": int(2 * n_subj),  # subject-condition rows
            "n_in": total_in,
            "n_out": total_out,
            "n_bad_excluded": total_bad,
            "drop_bad_trials": bool(drop_bad_trials),
            "bad_trial_rule": bad_trial_rule,
            "interp_reject_threshold": interp_reject_threshold,
            "bad_ar2_metadata_present": bool(bad_metadata_present),
            "analysis_mode": "subject-spectrum",
            "fooof_freq_range": list(freq_range),
            "fooof_params": fooof_params,
            "per_subject": per_subject_counts,
            "input_git_hashes": sorted(input_git_hashes),
        }
        out[ft] = (X, y, groups, metadata)

    logger.info(
        f"Subject-spectrum load done: {len(feature_types)} feature(s), "
        f"{n_subj} subjects, IN={total_in} OUT={total_out} "
        f"(bad excluded {total_bad})"
    )
    return out


# ---------------------------------------------------------------------------
# Single-channel spectra (for the FOOOF C-D-E-F panel figure)
# ---------------------------------------------------------------------------

# Full-range log10 curves vs. fit-range log10 curves — kept apart because the
# aperiodic/corrected curves are extrapolated over every PSD bin while the
# FOOOF periodic model only exists inside the fit range.
_FULL_CURVE_KEYS = ("raw", "aperiodic", "corrected")
_FIT_CURVE_KEYS = ("periodic", "full_model")
_SCALAR_KEYS = ("exponent", "offset")


def _fit_fooof_single_curve(
    psd_1d: np.ndarray,
    freqs: np.ndarray,
    freq_range: Tuple[float, float],
    fooof_params: Dict[str, Any],
) -> Optional[Dict[str, np.ndarray]]:
    """FOOOF-decompose one channel's spectrum into the panel-C..F curves.

    Returns a dict with full-range log10 curves (``raw``, ``aperiodic``,
    ``corrected``) plus the fit-range FOOOF model (``periodic`` = summed
    Gaussian peaks, ``full_model`` = aperiodic + peaks), and the aperiodic
    scalars. ``None`` if the model failed to fit.
    """
    from fooof import FOOOF

    fm = FOOOF(
        peak_width_limits=fooof_params.get("peak_width_limits", [1, 8]),
        max_n_peaks=fooof_params.get("max_n_peaks", 4),
        min_peak_height=fooof_params.get("min_peak_height", 0.10),
        peak_threshold=fooof_params.get("peak_threshold", 2.0),
        aperiodic_mode=fooof_params.get("aperiodic_mode", "fixed"),
        verbose=False,
    )
    try:
        fm.fit(freqs, psd_1d, freq_range=freq_range)
    except Exception:
        return None
    if not getattr(fm, "has_model", False):
        return None

    offset = float(fm.aperiodic_params_[0])
    exponent = float(fm.aperiodic_params_[-1])  # knee mode adds a knee before it

    safe_freqs = np.where(freqs > 0, freqs, np.nan)
    raw_log = np.log10(np.where(psd_1d > 0, psd_1d, np.nan))
    aperiodic_full = offset - np.log10(safe_freqs ** exponent)

    return dict(
        raw=raw_log,                                       # (n_freqs,)
        aperiodic=aperiodic_full,                          # (n_freqs,)
        corrected=raw_log - aperiodic_full,                # (n_freqs,)
        fit_freqs=np.asarray(fm.freqs, dtype=float),       # (n_freqs_fit,)
        periodic=np.asarray(fm._peak_fit, dtype=float),    # (n_freqs_fit,)
        full_model=np.asarray(fm.fooofed_spectrum_, float),# (n_freqs_fit,)
        exponent=exponent,
        offset=offset,
    )


def load_channel_spectra(
    channel_index: int,
    space: str,
    inout_bounds: Tuple[int, int],
    config: Dict[str, Any],
    *,
    subjects: Optional[List[str]] = None,
    drop_bad_trials: bool = True,
    bad_trial_rule: str = "ar2",
    interp_reject_threshold: int = 0,
    trial_type: str = "alltrials",
    zoning: str = "per-subject",
    n_events_window: int = 8,
) -> Dict[str, Any]:
    """Per-subject IN/OUT spectra + FOOOF decomposition for one spatial unit.

    Mirrors the aggregation of :func:`load_subject_spectrum_features` (pooled
    mean PSD across all good trials of a ``(subject, condition)``, one FOOOF
    fit per condition) but keeps the *full curves* — raw spectrum, aperiodic
    fit, corrected spectrum and FOOOF periodic model — for a single
    channel/region so the Figure-3 C..F panel can be reproduced.

    Args:
        channel_index: index into the spatial axis of the welch PSD arrays
            (same ordering as the saved statistics maps).
        space: 'sensor' or atlas name (selects the welch_psds_<space> folder).
        inout_bounds: (low_pct, high_pct) VTC percentiles for IN/OUT.
        config: loaded config dict.
        subjects: subject list (default: config.bids.subjects).
        drop_bad_trials: exclude bad trials before the median PSD.
        bad_trial_rule: 'ar2' | 'ar1' | 'union' (pass the value the matching
            statistics run used so the spectra line up with the stats map).
        interp_reject_threshold: extra n_interp-based rejection (0 disables).
        trial_type: window/trial selection passed to ``select_window_mask``.
        zoning: 'per-subject' (pooled VTC thresholds) or 'per-run'.
        n_events_window: trials per welch window (selects the desc suffix).

    Returns:
        dict with ``freqs`` (full vector), ``freqs_fit`` (FOOOF fit range),
        ``subjects`` (kept ids), ``channel_index``, and ``IN``/``OUT`` blocks.
        Each block maps ``raw``/``aperiodic``/``corrected`` -> arrays of shape
        ``(n_subjects, n_freqs)``, ``periodic``/``full_model`` -> arrays of
        shape ``(n_subjects, n_freqs_fit)``, and ``exponent``/``offset`` ->
        arrays of shape ``(n_subjects,)``.
    """
    data_root = Path(config["paths"]["data_root"])
    welch_root = data_root / config["paths"]["features"] / f"welch_psds_{space}"
    if not welch_root.exists():
        raise FileNotFoundError(f"Welch PSD folder not found: {welch_root}")

    if subjects is None:
        subjects = list(config["bids"]["subjects"])
    runs = list(config["bids"]["task_runs"])

    fooof_cfg_list = config.get("features", {}).get("fooof", [{}])
    fooof_params = fooof_cfg_list[0] if isinstance(fooof_cfg_list, list) else fooof_cfg_list
    freq_range = tuple(fooof_params.get("freq_range", [2.0, 40.0]))

    desc_suffix = "welch" if n_events_window <= 1 else f"welchw{n_events_window}"

    # Per-subject per-condition mean curves, keyed by curve name.
    subj_blocks: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    freqs_full: Optional[np.ndarray] = None
    freqs_fit: Optional[np.ndarray] = None

    for subject in tqdm(subjects, desc="Loading subjects (channel spectra)", unit="subj"):
        subj_dir = welch_root / f"sub-{subject}"
        if not subj_dir.exists():
            continue

        # First pass: collect VTC + task metadata so IN/OUT thresholds can be
        # computed on every epoch (matches load_subject_spectrum_features).
        all_vtc: List[np.ndarray] = []
        all_task: List[np.ndarray] = []
        all_inc_task: List[List[np.ndarray]] = []
        all_bad: List[np.ndarray] = []
        run_psd_files: List[Path] = []

        for run in runs:
            files = list(
                subj_dir.glob(f"sub-{subject}_*_run-{run}_*_desc-{desc_suffix}_psds.npz")
            )
            if not files:
                continue
            file_path = files[0]
            run_psd_files.append(file_path)
            with np.load(file_path, allow_pickle=True) as npz:
                meta = npz["trial_metadata"].item()
                if "window_vtc_mean" in meta:
                    run_vtc = np.asarray(meta["window_vtc_mean"], dtype=float)
                else:
                    run_vtc = np.asarray(meta["VTC_filtered"], dtype=float)
                all_vtc.append(run_vtc)
                all_task.append(np.asarray(meta["task"]))
                if "included_task" in meta:
                    all_inc_task.append([np.asarray(t) for t in meta["included_task"]])
                else:
                    all_inc_task.append([np.array([t]) for t in meta["task"]])
                all_bad.append(
                    compute_run_bad_mask(
                        meta, len(run_vtc), bad_trial_rule, interp_reject_threshold
                    )
                )

        if not run_psd_files:
            continue

        subj_vtc = np.concatenate(all_vtc)
        pooled_inbound = np.nanpercentile(subj_vtc, inout_bounds[0])
        pooled_outbound = np.nanpercentile(subj_vtc, inout_bounds[1])

        # Second pass: pool every good IN / OUT epoch at the selected channel
        # across runs, mean into one spectrum per condition, FOOOF fit once.
        in_chan_psds: List[np.ndarray] = []
        out_chan_psds: List[np.ndarray] = []

        for file_path, vtc, task, inc_task_run, bad in zip(
            run_psd_files, all_vtc, all_task, all_inc_task, all_bad
        ):
            with np.load(file_path, allow_pickle=True) as npz:
                psd_block = np.asarray(npz["psds"])  # (n_epochs, n_chans, n_freqs)
                freqs = np.asarray(npz["freqs"], dtype=float)
            if freqs_full is None:
                freqs_full = freqs
            if channel_index >= psd_block.shape[1]:
                raise IndexError(
                    f"channel_index {channel_index} out of range "
                    f"(welch PSD has {psd_block.shape[1]} channels)"
                )

            if zoning == "per-run":
                if np.all(np.isnan(vtc)):
                    continue
                inbound = np.nanpercentile(vtc, inout_bounds[0])
                outbound = np.nanpercentile(vtc, inout_bounds[1])
            else:
                inbound, outbound = pooled_inbound, pooled_outbound

            task_mask = select_window_mask(
                included_task_per_epoch=inc_task_run,
                task_per_epoch=task,
                type_how=trial_type,
            )
            good_mask = ~bad if drop_bad_trials else np.ones_like(bad, dtype=bool)
            in_mask = task_mask & (vtc <= inbound) & good_mask
            out_mask = task_mask & (vtc >= outbound) & good_mask
            if in_mask.any():
                in_chan_psds.append(psd_block[in_mask, channel_index, :])
            if out_mask.any():
                out_chan_psds.append(psd_block[out_mask, channel_index, :])

        if not in_chan_psds or not out_chan_psds:
            continue

        in_psd = np.nanmean(np.concatenate(in_chan_psds, axis=0), axis=0)
        out_psd = np.nanmean(np.concatenate(out_chan_psds, axis=0), axis=0)
        in_fit = _fit_fooof_single_curve(in_psd, freqs_full, freq_range, fooof_params)
        out_fit = _fit_fooof_single_curve(out_psd, freqs_full, freq_range, fooof_params)
        if in_fit is None or out_fit is None:
            continue  # FOOOF failed: skip the subject to keep IN/OUT paired
        if freqs_fit is None:
            freqs_fit = in_fit["fit_freqs"]

        cond_block: Dict[str, Dict[str, np.ndarray]] = {}
        for label, fit in (("IN", in_fit), ("OUT", out_fit)):
            block: Dict[str, np.ndarray] = {}
            for key in _FULL_CURVE_KEYS + _FIT_CURVE_KEYS:
                block[key] = np.asarray(fit[key], dtype=float)
            for key in _SCALAR_KEYS:
                block[key] = float(fit[key])
            cond_block[label] = block
        subj_blocks[subject] = cond_block

    if not subj_blocks:
        raise ValueError(
            f"No data loaded for channel {channel_index} in space '{space}'. "
            f"Are the welch PSDs present under {welch_root}?"
        )

    kept = list(subj_blocks.keys())
    out: Dict[str, Any] = {
        "freqs": freqs_full,
        "freqs_fit": freqs_fit,
        "subjects": kept,
        "channel_index": int(channel_index),
        "n_subjects": len(kept),
    }
    for label in ("IN", "OUT"):
        block: Dict[str, np.ndarray] = {}
        for key in _FULL_CURVE_KEYS + _FIT_CURVE_KEYS:
            block[key] = np.stack([subj_blocks[s][label][key] for s in kept], axis=0)
        for key in _SCALAR_KEYS:
            block[key] = np.array([subj_blocks[s][label][key] for s in kept], dtype=float)
        out[label] = block

    logger.info(
        f"Channel-spectra load done: channel {channel_index}, space '{space}', "
        f"{len(kept)} subjects"
    )
    return out
