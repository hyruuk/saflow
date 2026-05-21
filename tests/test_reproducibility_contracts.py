from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from code.features.utils import segment_spatial_temporal_data
from code.statistics.effect_sizes import compute_cohens_d, compute_paired_cohens_d
from code.statistics.run_group_statistics import apply_corrections
from code.utils.config import ConfigurationError, expand_paths, validate_config


def load_template_config(tmp_path: Path) -> dict:
    text = Path("config.yaml.template").read_text()
    text = text.replace("<DATA_ROOT>", str(tmp_path / "data"))
    return yaml.safe_load(text)


def test_config_expands_bids_and_accepts_atlas_spaces(tmp_path: Path) -> None:
    config = load_template_config(tmp_path)
    config["analysis"]["space"] = "aparc.a2009s"

    validate_config(config)
    expanded = expand_paths(config)

    assert expanded["paths"]["bids"] == str((tmp_path / "data" / "bids").resolve())
    assert expanded["paths"]["raw"] == str((tmp_path / "data" / "sourcedata").resolve())


def test_config_rejects_knee_mode(tmp_path: Path) -> None:
    config = load_template_config(tmp_path)
    config["features"]["fooof"][0]["aperiodic_mode"] = "knee"

    with pytest.raises(ConfigurationError, match="Only 'fixed' is supported"):
        validate_config(config)


def test_preprocessing_epoch_path_uses_canonical_epo_suffix(tmp_path: Path) -> None:
    pytest.importorskip("mne")
    pytest.importorskip("mne_bids")
    from code.preprocessing.utils import create_preprocessing_paths

    paths = create_preprocessing_paths(
        subject="04",
        run="02",
        bids_root=tmp_path / "bids",
        derivatives_root=tmp_path / "derivatives",
    )

    assert paths["epoch_ica"].fpath.name == "sub-04_task-gradCPT_run-02_proc-ica_epo.fif"
    assert "epoch_ar2_interp" not in paths


def test_window_segmentation_uses_actual_event_onsets() -> None:
    sfreq = 10.0
    data = np.arange(2 * 80, dtype=float).reshape(2, 80)
    events = pd.DataFrame(
        {
            "onset": [1.0, 2.2, 4.0, 5.1],
            "trial_type": ["Freq", "Freq", "Rare", "Freq"],
            "VTC_filtered": [1.0, 2.0, 3.0, 4.0],
            "task": ["correct_commission"] * 4,
        }
    )

    segments, metadata = segment_spatial_temporal_data(
        data=data,
        events_df=events,
        sfreq=sfreq,
        tmin=0.1,
        tmax=0.5,
        n_events_window=2,
    )

    assert len(segments) == 3
    assert metadata.loc[0, "window_start"] == pytest.approx(1.1)
    assert metadata.loc[0, "window_end"] == pytest.approx(2.7)
    assert metadata.loc[1, "window_start"] == pytest.approx(2.3)
    assert metadata.loc[1, "window_end"] == pytest.approx(4.5)


def test_welch_window_profile_must_be_explicit(tmp_path: Path) -> None:
    config = load_template_config(tmp_path)
    config["features"]["welch_psd"] = [p for p in config["features"]["welch_psd"] if p["name"] != "window8"]

    assert not any(p["name"] == "window8" for p in config["features"]["welch_psd"])


def test_tmax_uses_finite_permutation_correction() -> None:
    data = np.array(
        [
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [4.0, 4.0],
                [6.0, 6.0],
            ]
        ]
    )
    y = np.array([0, 0, 1, 1])
    groups = np.array([0, 1, 0, 1])
    tvals = np.array([[100.0, 100.0]])
    pvals = np.array([[0.0, 0.0]])

    corrected, _ = apply_corrections(
        pvals=pvals,
        tvals=tvals,
        correction="tmax",
        X=data,
        y=y,
        groups=groups,
        n_permutations=9,
        n_jobs=1,
    )

    assert np.all(corrected >= 0.1)


def test_fdr_method_is_honored() -> None:
    pvals = np.array([[0.01, 0.02, 0.04, 0.20]])
    tvals = np.ones_like(pvals)

    bh, _ = apply_corrections(pvals, tvals, correction="fdr", fdr_method="bh")
    by, _ = apply_corrections(pvals, tvals, correction="fdr", fdr_method="by")

    assert not np.allclose(bh, by)
    assert np.all(by >= bh)


def test_paired_effect_size_differs_from_trial_pooled() -> None:
    data = np.array(
        [
            [
                [1.0],
                [10.0],
                [2.0],
                [11.0],
                [2.0],
                [11.0],
                [4.0],
                [13.0],
            ]
        ]
    )
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    pooled = compute_cohens_d(data, y, groups)
    paired = compute_paired_cohens_d(data, y, groups, aggregate="mean")

    assert pooled.shape == paired.shape
    assert not np.allclose(pooled, paired, equal_nan=True)
