"""Protected paper and slide rendering tests."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from code.paper_panels.render import (
    _assert_overwrite_allowed,
    render_paper_panels,
)


def test_synthetic_panel_writes_watermarked_paper_and_exact_slide_exports(
    tmp_path: Path,
):
    outputs = render_paper_panels(
        panel="panel2",
        analysis_id=None,
        analysis_root=None,
        reports_root=tmp_path,
    )
    paper = tmp_path / "figures" / "paper" / "panel2_multifeature_decoding.png"
    slide = (
        tmp_path
        / "figures"
        / "slides"
        / "panel2_multifeature_decoding"
        / "01_A_model_performance.png"
    )
    svg = slide.with_suffix(".svg")
    assert paper in outputs
    assert slide in outputs
    assert svg in outputs
    assert Image.open(slide).size == (2560, 1440)
    metadata = json.loads((slide.parent / f"{slide.name}.json").read_text())
    assert metadata["data_mode"] == "synthetic"
    assert metadata["render_parameters"]["synthetic_watermark"]
    assert "SYNTHETIC DATA" in svg.read_text()


def test_synthetic_cannot_overwrite_real_but_real_can_replace_synthetic(
    tmp_path: Path,
):
    figure = tmp_path / "panel.png"
    sidecar = tmp_path / "panel.png.json"
    figure.write_bytes(b"figure")
    sidecar.write_text('{"data_mode": "synthetic"}')
    _assert_overwrite_allowed(figure, sidecar, "real")
    sidecar.write_text('{"data_mode": "real"}')
    with pytest.raises(PermissionError, match="cannot overwrite real"):
        _assert_overwrite_allowed(figure, sidecar, "synthetic")
    with pytest.raises(FileExistsError, match="immutable"):
        _assert_overwrite_allowed(figure, sidecar, "real")


def test_real_mode_requires_complete_real_bundle(tmp_path: Path):
    analysis_root = tmp_path / "analyses"
    bundle = analysis_root / "analysis-1" / "panel1"
    bundle.mkdir(parents=True)
    frequency = np.linspace(2, 120, 24)
    np.savez_compressed(
        bundle / "observed.npz",
        raw_psd_modulation=np.ones((7, 40)),
        raw_psd_auc=np.full((7, 40), 0.6),
        frequency=frequency,
        spectrum_in=-np.log10(frequency),
        spectrum_out=-np.log10(frequency) - 0.02,
        aperiodic_spectrum_in=-np.log10(frequency),
        aperiodic_spectrum_out=-np.log10(frequency) - 0.02,
        corrected_spectrum_in=np.zeros_like(frequency),
        corrected_spectrum_out=np.zeros_like(frequency),
        periodic_spectrum_in=np.zeros_like(frequency),
        periodic_spectrum_out=np.zeros_like(frequency),
        fooof_modulation=np.ones((3, 40)),
        fooof_auc=np.full((3, 40), 0.6),
        corrected_psd_modulation=np.ones((7, 40)),
        corrected_psd_auc=np.full((7, 40), 0.6),
    )
    (bundle / "observed.json").write_text(
        '{"provenance": {"data_mode": "real"}}'
    )
    outputs = render_paper_panels(
        panel="panel1",
        analysis_id="analysis-1",
        analysis_root=analysis_root,
        reports_root=tmp_path / "reports",
    )
    sidecar = outputs[0].with_name(f"{outputs[0].name}.json")
    assert json.loads(sidecar.read_text())["data_mode"] == "real"
