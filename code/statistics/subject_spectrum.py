"""Subject-spectrum (Path 1) feature loader for group-level statistics.

This module implements the "fit FOOOF on aggregated spectra" path for
PSD-derived features (raw band power, FOOOF-corrected band power, FOOOF
aperiodic parameters). Aggregation is done per ``(subject, run, condition)``
on **good trials only** using the median PSD across trials, then averaged
across the 6 runs of a subject (arithmetic mean) to give one value per
``(subject, condition)`` per spatial unit.

Trade-offs vs the per-trial path:

- FOOOF is fit once per ``(subject, run, condition, channel)`` on a clean,
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


def _compute_run_condition_block(
    psd_block: np.ndarray,
    bad_mask: np.ndarray,
    in_mask: np.ndarray,
    out_mask: np.ndarray,
    freqs: np.ndarray,
    freq_range: Tuple[float, float],
    fooof_params: Dict[str, Any],
) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
    """Run-level: median PSD per condition (good trials), then FOOOF.

    Returns a dict ``{"IN": {...}, "OUT": {...}}`` where each inner dict
    has keys ``mean_psd``, ``corrected_psd``, ``exponent``, ``offset``,
    ``r_squared``. Returns ``None`` if either condition has zero good
    trials in this run (run is skipped).
    """
    good_mask = ~bad_mask
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for label, cond_mask in (("IN", in_mask), ("OUT", out_mask)):
        cond_good = cond_mask & good_mask
        if not cond_good.any():
            return None
        # Median across good trials of this condition → (n_chans, n_freqs)
        cond_psd = np.median(psd_block[cond_good], axis=0)
        exp, offs, r2, corr = _fit_fooof_group_on_psd(
            cond_psd, freqs, freq_range, fooof_params
        )
        out[label] = dict(
            mean_psd=cond_psd,
            corrected_psd=corr,
            exponent=exp,
            offset=offs,
            r_squared=r2,
        )
    return out


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

    Per ``(subject, run, condition)``: median PSD across good trials, FOOOF
    fit on that median, then derive each requested feature. Per
    ``(subject, condition)``: arithmetic mean across runs.

    Returns a dict mapping ``feature_type -> (X, y, groups, metadata)``,
    matching the shape contract of :func:`load_all_features_batched`. ``X``
    has shape ``(1, 2 * n_subjects_kept, n_spatial)`` where rows alternate
    IN, OUT for each kept subject; ``y`` is ``[0, 1, 0, 1, ...]``;
    ``groups`` is ``[s0, s0, s1, s1, ...]``. The existing paired-t-test
    code path consumes this directly.

    Only feature_types from the PSD/FOOOF families are supported. Mix of
    ``psd_*``, ``psd_corrected_*`` and ``fooof_*`` is fine — the FOOOF fit
    is computed once per ``(subject, run, condition)`` and reused across
    every requested feature.
    """
    if not feature_types:
        raise ValueError("feature_types must be non-empty")
    parsed = [(ft, *_classify_feature(ft)) for ft in feature_types]

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
                if "bad_ar2" in meta:
                    all_bad.append(np.asarray(meta["bad_ar2"], dtype=bool))
                else:
                    all_bad.append(np.zeros(len(run_vtc), dtype=bool))

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

        # Now walk runs again, splitting each run's epochs by VTC + trial-type,
        # applying the bad filter, computing the median PSD + FOOOF fit.
        run_blocks: List[Dict[str, Dict[str, np.ndarray]]] = []
        n_in_total, n_out_total, n_bad_in, n_bad_out = 0, 0, 0, 0

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
            in_mask_full = task_mask & (vtc <= inbound)
            out_mask_full = task_mask & (vtc >= outbound)
            n_bad_in += int((in_mask_full & bad).sum()) if drop_bad_trials else 0
            n_bad_out += int((out_mask_full & bad).sum()) if drop_bad_trials else 0

            block = _compute_run_condition_block(
                psd_block=psd_block,
                bad_mask=bad if drop_bad_trials else np.zeros_like(bad, dtype=bool),
                in_mask=in_mask_full,
                out_mask=out_mask_full,
                freqs=freqs,
                freq_range=freq_range,
                fooof_params=fooof_params,
            )
            if block is None:
                continue

            # Bookkeeping (counts of good IN/OUT trials retained in this run)
            good_mask = ~bad if drop_bad_trials else np.ones_like(bad, dtype=bool)
            n_in_total += int((in_mask_full & good_mask).sum())
            n_out_total += int((out_mask_full & good_mask).sum())
            run_blocks.append(block)

        if not run_blocks:
            continue

        # Average the per-run aggregates → one value per (subj, cond) per channel.
        per_cond_runs: Dict[str, Dict[str, List[np.ndarray]]] = {
            "IN": {k: [] for k in ("mean_psd", "corrected_psd", "exponent", "offset", "r_squared")},
            "OUT": {k: [] for k in ("mean_psd", "corrected_psd", "exponent", "offset", "r_squared")},
        }
        for block in run_blocks:
            for cond in ("IN", "OUT"):
                for k in per_cond_runs[cond]:
                    per_cond_runs[cond][k].append(block[cond][k])

        cond_agg: Dict[str, Dict[str, np.ndarray]] = {}
        for cond in ("IN", "OUT"):
            cond_agg[cond] = {
                k: np.mean(np.stack(per_cond_runs[cond][k], axis=0), axis=0)
                for k in per_cond_runs[cond]
            }
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
                # We need band mask aligned to the frequency vector. Use the
                # run-level freqs by reading one of the run's PSD files.
                with np.load(run_psd_files[0], allow_pickle=True) as npz:
                    freqs = np.asarray(npz["freqs"])
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
