"""Protected composite and slide rendering tests."""

import json
from pathlib import Path

import numpy as np
import pytest
import matplotlib.pyplot as plt
from PIL import Image

from code.analysis.render import (
    _assert_overwrite_allowed,
    _plot_matrix,
    render_panels,
)


def test_synthetic_panel_writes_watermarked_composite_and_exact_slide_exports(
    tmp_path: Path,
):
    outputs = render_panels(
        panel="panel2",
        analysis_id=None,
        analysis_root=None,
        reports_root=tmp_path,
    )
    composite = (
        tmp_path / "figures" / "manuscript" / "panel2_multifeature_decoding.png"
    )
    slide = (
        tmp_path
        / "figures"
        / "slides"
        / "panel2_multifeature_decoding"
        / "01_A_model_performance.png"
    )
    svg = slide.with_suffix(".svg")
    assert composite in outputs
    assert slide in outputs
    assert svg in outputs
    assert Image.open(slide).size == (2560, 1440)
    metadata = json.loads((slide.parent / f"{slide.name}.json").read_text())
    assert metadata["data_mode"] == "synthetic"
    assert metadata["render_parameters"]["synthetic_watermark"]
    assert "SYNTHETIC DATA" in svg.read_text()
    assert composite.with_suffix(".json").exists()
    assert composite.with_suffix(".txt").exists()
    assert (
        slide.parent / "panel2_multifeature_decoding.txt"
    ).exists()


def test_synthetic_cannot_overwrite_real_but_real_can_replace_synthetic(
    tmp_path: Path,
):
    figure = tmp_path / "panel.png"
    sidecar = tmp_path / "panel.json"
    figure.write_bytes(b"figure")
    sidecar.write_text('{"data_mode": "synthetic"}')
    _assert_overwrite_allowed(figure, sidecar, "real")
    sidecar.write_text('{"data_mode": "real"}')
    with pytest.raises(PermissionError, match="cannot overwrite real"):
        _assert_overwrite_allowed(figure, sidecar, "synthetic")
    with pytest.raises(FileExistsError, match="immutable"):
        _assert_overwrite_allowed(figure, sidecar, "real")


def test_panel3_writes_manuscript_composite_and_captions(tmp_path: Path):
    outputs = render_panels(
        panel="panel3",
        analysis_id=None,
        analysis_root=None,
        reports_root=tmp_path,
    )
    composite = (
        tmp_path / "figures" / "manuscript" / "panel3_network_dynamics.png"
    )
    slide_caption = (
        tmp_path
        / "figures"
        / "slides"
        / "panel3_network_dynamics"
        / "panel3_network_dynamics.txt"
    )
    assert composite in outputs
    assert composite.with_suffix(".json").exists()
    assert composite.with_suffix(".txt").exists()
    assert slide_caption.exists()


def test_panel3_matrix_uses_yeo_and_feature_display_names():
    figure, axis = plt.subplots()
    _plot_matrix(axis, {"matrix": np.zeros((7, 9))}, 0)

    assert [label.get_text() for label in axis.get_yticklabels()] == [
        "Visual",
        "Somatomotor",
        "Dorsal Attention",
        "Salience/Ventral Attention",
        "Limbic",
        "Frontoparietal Control",
        "Default Mode",
    ]
    assert [label.get_text() for label in axis.get_xticklabels()] == [
        "FOOOF exponent",
        "FOOOF offset",
        "Corrected Theta",
        "Corrected Alpha",
        "Corrected Low Beta",
        "Corrected High Beta",
        "Corrected Gamma 1",
        "Corrected Gamma 2",
        "Corrected Gamma 3",
    ]
    plt.close(figure)


def test_generic_renderer_rejects_panel1_alternate():
    with pytest.raises(ValueError, match="invoke viz.panel1"):
        render_panels(
            panel="panel1", analysis_id=None, analysis_root=None,
            reports_root=Path("reports"),
        )
