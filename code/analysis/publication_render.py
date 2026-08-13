"""Render all publication panels with their established panel-specific layouts."""

from __future__ import annotations

import argparse
from pathlib import Path

from code.analysis.provenance import active_analysis_id, resolve_analysis_directory
from code.analysis.render import render_panels
from code.visualization.panel1_bundle import render_bundle


def main() -> None:
    """Render legacy-layout Panel 1 and the current Panel 2/3 layouts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--analysis-id")
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    arguments = parser.parse_args()
    analysis_id = arguments.analysis_id or active_analysis_id(arguments.analysis_root)
    analysis_dir = resolve_analysis_directory(arguments.analysis_root, analysis_id)
    render_bundle(analysis_dir / "feature_modulation", arguments.reports_root)
    for panel in ("panel2", "panel3"):
        render_panels(
            panel=panel,
            analysis_id=analysis_id,
            analysis_root=arguments.analysis_root,
            reports_root=arguments.reports_root,
        )


if __name__ == "__main__":
    main()
