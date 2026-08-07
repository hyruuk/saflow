"""Command-line orchestration for immutable corrected Figure 3 analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import csv
from pathlib import Path

from code.figure3.contracts import (
    PANEL1_FEATURES,
    PANEL23_FEATURES,
    PANEL_COMPONENTS,
    PANEL_SPECS,
    frequency_band_manifest,
    schema_catalog,
)
from code.figure3.dag import (
    DEFAULT_FIGURE3_RESOURCES,
    bound_paper_dag,
    build_paper_dag,
    build_submission_plan,
)
from code.figure3.preflight import inspect_inputs, write_reports
from code.figure3.provenance import (
    config_hash,
    create_analysis_id,
    initialize,
    validate_analysis_id,
)
from code.figure3.slurm_execution import submit_dag
from code.figure3.panel_validator import validate_panel
from code.utils.logging_config import setup_logging
from code.utils.config import load_config

LOGGER = logging.getLogger(__name__)


def _load_config(path: Path) -> dict:
    """Load, validate, and expand the shared project configuration."""
    return load_config(str(path))


def _analysis_root(config: dict, override: str | None) -> Path:
    if override:
        return Path(override)
    directory = config.get("figure3", {}).get("processed_directory", "figure3")
    return Path(config["paths"]["data_root"]) / "processed" / directory


def run_preflight(args: argparse.Namespace) -> Path:
    """Initialize an immutable ID and record required input checks."""
    config = _load_config(Path(args.config))
    analysis_id = args.analysis_id or create_analysis_id(config, Path.cwd())
    analysis_dir = initialize(
        _analysis_root(config, args.analysis_root), analysis_id, config, vars(args), Path.cwd()
    )
    requirements = {
        "spaces": ["sensor", "schaefer_400"],
        "trial_sets": ["alltrials", "correct", "lapse"],
        "primary_trial_set": "alltrials",
        "feature_families": {
            "panel1": list(PANEL1_FEATURES),
            "panel2": list(PANEL23_FEATURES),
            "panel3": list(PANEL23_FEATURES),
        },
        "frequency_bands": frequency_band_manifest(),
        "panels": PANEL_SPECS,
        "label_definition": "reflected-gaussian-filtered-vtc_strict-eight-trial",
        "primary_rejection": "any-bad-constituent",
    }
    subjects = args.subjects.split() if args.subjects else config["bids"]["subjects"]
    runs = args.runs.split() if args.runs else config["bids"]["task_runs"]
    input_report = inspect_inputs(config, subjects, runs)
    report = {"analysis_id": analysis_id, **requirements, **input_report}
    write_reports(report, analysis_dir / "qc")
    (analysis_dir / "preflight_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (analysis_dir / "manifests" / "feature_families.json").write_text(
        json.dumps(requirements["feature_families"], indent=2) + "\n"
    )
    (analysis_dir / "manifests" / "schemas.json").write_text(
        json.dumps(schema_catalog(), indent=2) + "\n"
    )
    LOGGER.info("Figure 3 preflight initialized %s", analysis_dir)
    return analysis_dir


def run_analysis(args: argparse.Namespace) -> Path:
    """Validate preflight and write an authoritative execution manifest.

    HPC fitting is resumable: workers write permutation chunks beneath
    ``classification/chunks`` and ``statistics/chunks``. This command never
    replaces completed bundles and refuses an uninitialized analysis ID.
    """
    validate_analysis_id(args.analysis_id)
    config = _load_config(Path(args.config))
    analysis_dir = _analysis_root(config, args.analysis_root) / args.analysis_id
    preflight = analysis_dir / "preflight_report.json"
    if not preflight.exists() or json.loads(preflight.read_text()).get("status") != "passed":
        raise RuntimeError("a passing figure3-preflight report is required")
    manifest = {
        "analysis_id": args.analysis_id,
        "n_permutations": args.n_permutations,
        "seed": args.seed,
        "minimum_circular_offset": args.minimum_circular_offset,
        "spaces": ["sensor", "schaefer_400"],
        "status": "configured",
        "hpc_authoritative": True,
    }
    path = analysis_dir / "permutation_manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise FileExistsError("immutable permutation manifest already differs")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def plan_dag(args: argparse.Namespace) -> Path:
    """Write a complete inspectable DAG without submitting any jobs."""
    validate_analysis_id(args.analysis_id)
    config = _load_config(Path(args.config))
    analysis_dir = _analysis_root(config, args.analysis_root) / args.analysis_id
    if not analysis_dir.exists():
        raise FileNotFoundError(f"analysis does not exist: {analysis_dir}")
    subjects = args.subjects.split() if args.subjects else config["bids"]["subjects"]
    runs = args.runs.split() if args.runs else config["bids"]["task_runs"]
    spaces = args.spaces.split()
    manifest = build_paper_dag(
        args.analysis_id,
        subjects,
        runs,
        spaces,
        include_exploratory=args.include_exploratory,
    )
    path = analysis_dir / "manifests" / "dag.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def run_full_pipeline(args: argparse.Namespace) -> Path:
    """Create and submit the bounded DAG, or render it completely in dry-run."""
    config = _load_config(Path(args.config))
    if not args.dry_run and shutil.which("sbatch") is None:
        raise RuntimeError(
            "SLURM submission requires sbatch; use --dry-run on non-SLURM hosts"
        )
    analysis_id = args.analysis_id or create_analysis_id(config, Path.cwd())
    validate_analysis_id(analysis_id)
    root = _analysis_root(config, args.analysis_root)
    analysis_dir = root / analysis_id
    if analysis_dir.exists():
        raise FileExistsError(f"immutable analysis already exists: {analysis_dir}")
    initialize(root, analysis_id, config, vars(args), Path.cwd())
    subjects = args.subjects.split() if args.subjects else config["bids"]["subjects"]
    runs = args.runs.split() if args.runs else config["bids"]["task_runs"]
    figure3_config = config.get("figure3", {})
    graph = build_paper_dag(
        analysis_id,
        subjects,
        runs,
        args.spaces.split(),
        include_exploratory=args.include_exploratory,
        map_chunk_count=_chunk_count(
            figure3_config, "map_permutations", "map_chunk_size", 10_000, 250
        ),
        decoding_chunk_count=_chunk_count(
            figure3_config,
            "decoding_permutations",
            "decoding_chunk_size",
            1_000,
            25,
        ),
    )
    graph = bound_paper_dag(
        graph,
        start_at=args.start_at,
        stop_after=args.stop_after,
        skip=_split_stages(args.skip),
    )
    provenance = json.loads((analysis_dir / "provenance.json").read_text())
    graph["provenance"].update(
        {
            "analysis_id": analysis_id,
            "config_hash": config_hash(config),
            "git_commit": provenance["git"]["commit"],
            "git_dirty": provenance["git"]["dirty"],
            "dry_run": bool(args.dry_run),
            "submission_status": "dry_run" if args.dry_run else "ready",
        }
    )
    graph["submission_plan"] = build_submission_plan(
        graph,
        figure3_config.get("resources", DEFAULT_FIGURE3_RESOURCES),
        config.get("computing", {}).get("slurm", {}),
    )
    path = analysis_dir / "manifests" / "dag.json"
    path.write_text(json.dumps(graph, indent=2, sort_keys=True) + "\n")
    submission = submit_dag(
        graph,
        analysis_dir,
        config,
        dry_run=bool(args.dry_run),
    )
    graph["scheduler"] = submission
    graph["provenance"]["submitted"] = not args.dry_run
    graph["provenance"]["submission_status"] = (
        "dry_run" if args.dry_run else "submitted"
    )
    path.write_text(json.dumps(graph, indent=2, sort_keys=True) + "\n")
    LOGGER.info("Full pipeline manifest ready: %s", path)
    return path


def _chunk_count(
    config: dict, total_key: str, size_key: str, total_default: int, size_default: int
) -> int:
    """Resolve a validated exact chunk count from configuration defaults."""
    total = int(config.get(total_key, total_default))
    size = int(config.get(size_key, size_default))
    if total < 1 or size < 1 or total % size:
        raise ValueError(f"{size_key} must divide {total_key}")
    return total // size


def resume_pipeline(args: argparse.Namespace) -> Path:
    """Audit an immutable DAG and submit a dependency-safe invalid-cell wave."""
    validate_analysis_id(args.analysis_id)
    config = _load_config(Path(args.config))
    analysis_dir = _analysis_root(config, args.analysis_root) / args.analysis_id
    dag_path = analysis_dir / "manifests" / "dag.json"
    if not dag_path.exists():
        raise FileNotFoundError(f"immutable DAG manifest not found: {dag_path}")
    dag = json.loads(dag_path.read_text())
    selected = []
    complete = []
    for expected in dag.get("expected_outputs", []):
        status_path = analysis_dir / expected["status_path"]
        reason = _invalid_cell_reason(status_path, expected, dag["provenance"])
        record = {**expected, "reason": reason}
        (selected if reason else complete).append(record)
    resume = {
        "analysis_id": args.analysis_id,
        "dry_run": bool(args.dry_run),
        "completed_cell_count": len(complete),
        "resubmit_cell_count": len(selected),
        "cells_to_resubmit": selected,
        "completed_cells": complete,
        "deletes_completed_chunks": False,
    }
    ready, deferred = _resume_submission_wave(dag, selected)
    resume["cells_submitted_this_wave"] = ready
    resume["cells_deferred_to_next_wave"] = deferred
    submission = submit_dag(
        dag,
        analysis_dir,
        config,
        dry_run=bool(args.dry_run),
        selected_cells=ready,
    )
    resume["scheduler"] = submission
    path = analysis_dir / "manifests" / "resume.json"
    path.write_text(json.dumps(resume, indent=2, sort_keys=True) + "\n")
    return path


def _resume_submission_wave(
    dag: dict, invalid_cells: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Select a dependency-safe recovery wave without recomputing valid cells."""
    by_node: dict[str, list[dict]] = {}
    for cell in invalid_cells:
        by_node.setdefault(cell["node"], []).append(cell)
    deferred_nodes: set[str] = set()
    changed = True
    while changed:
        changed = False
        for edge in dag["edges"]:
            upstream = edge["upstream"]
            downstream = edge["downstream"]
            upstream_cells = by_node.get(upstream, [])
            downstream_cells = by_node.get(downstream, [])
            if not downstream_cells or downstream in deferred_nodes:
                continue
            if upstream in deferred_nodes:
                deferred_nodes.add(downstream)
                changed = True
                continue
            if edge["dependency"] == "aftercorr" and upstream_cells:
                upstream_indices = {
                    (cell.get("subject"), cell.get("run"))
                    for cell in upstream_cells
                }
                downstream_indices = {
                    (cell.get("subject"), cell.get("run"))
                    for cell in downstream_cells
                }
                if upstream_indices != downstream_indices:
                    deferred_nodes.add(downstream)
                    changed = True
    ready = [
        cell for cell in invalid_cells if cell["node"] not in deferred_nodes
    ]
    deferred = [
        cell for cell in invalid_cells if cell["node"] in deferred_nodes
    ]
    return ready, deferred


def _split_stages(value: str | None) -> list[str]:
    """Parse comma- or whitespace-separated public stage names."""
    return value.replace(",", " ").split() if value else []


def _invalid_cell_reason(
    path: Path, expected: dict, provenance: dict
) -> str | None:
    """Return why a cell must be resubmitted, or None when compatible."""
    if not path.exists():
        return "missing"
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return "corrupt"
    if payload.get("status") != "complete":
        return "failed_or_incomplete"
    compatibility = {
        "analysis_id": expected.get("analysis_id", provenance.get("analysis_id")),
        "node": expected["node"],
        "cell_index": expected["cell_index"],
        "config_hash": provenance.get("config_hash"),
        "git_commit": provenance.get("git_commit"),
    }
    for field, value in compatibility.items():
        if value is not None and payload.get(field) != value:
            return f"incompatible_{field}"
    return None


def export_analysis(args: argparse.Namespace) -> Path:
    """Copy only compact bundles, metadata, tables, and figures."""
    validate_analysis_id(args.analysis_id)
    source = Path(args.analysis_root) / args.analysis_id
    destination = Path(args.destination)
    if destination.exists():
        raise FileExistsError(f"export destination exists: {destination}")
    if not (source / "preflight_report.json").exists():
        raise ValueError("source is not a Figure 3 analysis")
    destination.mkdir(parents=True)
    excluded = {"chunks", "partials", "subject_features", "inputs", "slurm"}
    copied = []
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if not path.is_file() or excluded.intersection(relative.parts):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(str(relative))
    table = _write_compact_summary_table(destination)
    copied.append(str(table.relative_to(destination)))
    files = []
    for relative in sorted(set(copied)):
        path = destination / relative
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (destination / "export_manifest.json").write_text(
        json.dumps(
            {
                "analysis_id": args.analysis_id,
                "files": files,
                "omitted": [
                    "chunks",
                    "partials",
                    "subject_features",
                    "inputs",
                    "slurm",
                ],
                "contains_subject_feature_matrices": False,
            },
            indent=2,
        )
        + "\n"
    )
    return destination


def _write_compact_summary_table(destination: Path) -> Path:
    """Export scalar observed summaries without subject feature matrices."""
    rows = []
    for panel in PANEL_SPECS:
        path = destination / panel / "observed.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        for key, value in _scalar_items(payload.get("summary", payload)):
            rows.append({"panel": panel, "metric": key, "value": value})
    table = destination / "tables" / "observed_summary.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("panel", "metric", "value"))
        writer.writeheader()
        writer.writerows(rows)
    return table


def _scalar_items(value: object, prefix: str = "") -> list[tuple[str, object]]:
    """Flatten JSON scalar fields for automated compact tables."""
    if isinstance(value, dict):
        return [
            item
            for key, nested in value.items()
            for item in _scalar_items(nested, f"{prefix}.{key}".strip("."))
        ]
    if isinstance(value, list):
        return []
    if value is None or isinstance(value, (str, int, float, bool)):
        return [(prefix, value)]
    return []


def audit_analysis(args: argparse.Namespace) -> Path:
    """Audit complete bundles and rendered paper/slide provenance."""
    validate_analysis_id(args.analysis_id)
    config = _load_config(Path(args.config))
    analysis_dir = _analysis_root(config, args.analysis_root) / args.analysis_id
    reports_root = Path(args.reports_root)
    findings = []
    modes = set()
    for panel, spec in PANEL_SPECS.items():
        try:
            validate_panel(
                args.config,
                args.analysis_id,
                str(_analysis_root(config, args.analysis_root)),
                panel,
            )
        except (FileNotFoundError, ValueError) as error:
            findings.append(
                {
                    "panel": panel,
                    "kind": "bundle_validation",
                    "status": "failed",
                    "reason": str(error),
                }
            )
        bundle = analysis_dir / panel / "observed.json"
        paper = reports_root / "figures" / "paper" / spec["paper_filename"]
        sidecar = paper.with_name(f"{paper.name}.json")
        for kind, path in (("bundle", bundle), ("paper", paper), ("sidecar", sidecar)):
            if not path.exists():
                findings.append({"panel": panel, "kind": kind, "status": "missing"})
        if sidecar.exists():
            metadata = json.loads(sidecar.read_text())
            modes.add(metadata.get("data_mode"))
            if metadata.get("analysis_id") != args.analysis_id:
                findings.append(
                    {"panel": panel, "kind": "sidecar", "status": "wrong_analysis_id"}
                )
        slide_directory = (
            reports_root
            / "figures"
            / "slides"
            / spec["slide_directory"]
        )
        for index, component in enumerate(PANEL_COMPONENTS[panel], start=1):
            stem = f"{index:02d}_{component}"
            for suffix in (".png", ".svg"):
                slide = slide_directory / f"{stem}{suffix}"
                slide_sidecar = slide.with_name(f"{slide.name}.json")
                if not slide.exists() or not slide_sidecar.exists():
                    findings.append(
                        {
                            "panel": panel,
                            "kind": f"slide_{suffix[1:]}",
                            "status": "missing",
                            "component": component,
                        }
                    )
                    continue
                slide_metadata = json.loads(slide_sidecar.read_text())
                if (
                    slide_metadata.get("analysis_id") != args.analysis_id
                    or slide_metadata.get("data_mode") != "real"
                ):
                    findings.append(
                        {
                            "panel": panel,
                            "kind": f"slide_{suffix[1:]}",
                            "status": "incompatible_provenance",
                            "component": component,
                        }
                    )
    dag_path = analysis_dir / "manifests" / "dag.json"
    if dag_path.exists() and json.loads(dag_path.read_text()).get(
        "include_exploratory"
    ):
        exploratory = analysis_dir / "exploratory" / "sidekick_manifest.json"
        if not exploratory.exists():
            findings.append(
                {"panel": "all", "kind": "exploratory", "status": "missing"}
            )
    status = "passed" if not findings and modes == {"real"} else "failed"
    report = {
        "analysis_id": args.analysis_id,
        "status": status,
        "required_data_mode": "real",
        "observed_data_modes": sorted(str(mode) for mode in modes),
        "findings": findings,
    }
    path = analysis_dir / "final_analysis_audit.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if status != "passed":
        raise RuntimeError(f"final analysis audit failed; see {path}")
    return path


def inventory_legacy(args: argparse.Namespace) -> Path:
    """Hash legacy Figure 3 outputs without moving, deleting, or modifying them."""
    entries = []
    source = Path(args.source)
    for path in sorted(source.rglob("*")) if source.exists() else []:
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            entries.append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest})
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"read_only": True, "moved": False, "deleted": False,
                                    "source": str(source), "files": entries}, indent=2) + "\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """Build the Figure 3 workflow parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--config", default="config.yaml")
    preflight.add_argument("--analysis-id")
    preflight.add_argument("--analysis-root")
    preflight.add_argument("--subjects")
    preflight.add_argument("--runs")
    run = commands.add_parser("run")
    run.add_argument("--config", default="config.yaml")
    run.add_argument("--analysis-id", required=True)
    run.add_argument("--analysis-root")
    run.add_argument("--n-permutations", type=int, default=1000)
    run.add_argument("--minimum-circular-offset", type=int, default=24)
    run.add_argument("--seed", type=int, default=42)
    dag = commands.add_parser("dag")
    dag.add_argument("--config", default="config.yaml")
    dag.add_argument("--analysis-id", required=True)
    dag.add_argument("--analysis-root")
    dag.add_argument("--subjects")
    dag.add_argument("--runs")
    dag.add_argument("--spaces", default="sensor schaefer_400")
    dag.add_argument(
        "--include-exploratory", action=argparse.BooleanOptionalAction, default=True
    )
    full = commands.add_parser("full")
    full.add_argument("--config", default="config.yaml")
    full.add_argument("--analysis-id")
    full.add_argument("--analysis-root")
    full.add_argument("--start-at")
    full.add_argument("--stop-after")
    full.add_argument("--skip")
    full.add_argument("--subjects")
    full.add_argument("--runs")
    full.add_argument("--spaces", default="sensor schaefer_400")
    full.add_argument(
        "--include-exploratory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    full.add_argument("--dry-run", action="store_true")
    resume = commands.add_parser("resume")
    resume.add_argument("--config", default="config.yaml")
    resume.add_argument("--analysis-id", required=True)
    resume.add_argument("--analysis-root")
    resume.add_argument("--dry-run", action="store_true")
    audit = commands.add_parser("audit")
    audit.add_argument("--config", default="config.yaml")
    audit.add_argument("--analysis-id", required=True)
    audit.add_argument("--analysis-root")
    audit.add_argument("--reports-root", default="reports")
    export = commands.add_parser("export")
    export.add_argument("--analysis-id", required=True)
    export.add_argument("--analysis-root", required=True)
    export.add_argument("--destination", required=True)
    legacy = commands.add_parser("legacy-inventory")
    legacy.add_argument("--source", required=True)
    legacy.add_argument("--manifest", required=True)
    return parser


def main() -> None:
    """Run the selected Figure 3 workflow command."""
    args = build_parser().parse_args()
    setup_logging("figure3", log_file="figure3.log",
                  config={"paths": {"logs": "logs"}, "logging": {"level": "INFO"}})
    functions = {
        "preflight": run_preflight,
        "run": run_analysis,
        "dag": plan_dag,
        "full": run_full_pipeline,
        "resume": resume_pipeline,
        "audit": audit_analysis,
        "export": export_analysis,
        "legacy-inventory": inventory_legacy,
    }
    functions[args.command](args)


if __name__ == "__main__":
    main()
