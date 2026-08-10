"""Command-line orchestration for immutable corrected Saflow analysis analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import csv
from pathlib import Path

from code.analysis.contracts import (
    FEATURE_MODULATION_FEATURES,
    CORRECTED_FEATURES,
    PANEL_COMPONENTS,
    PANEL_SPECS,
    PANEL_ANALYSES,
    frequency_band_manifest,
    schema_catalog,
)
from code.analysis.execution_plan import (
    DEFAULT_ANALYSIS_RESOURCES,
    DEFAULT_NODE_RESOURCES,
    bound_execution_plan,
    build_execution_plan,
    build_submission_plan,
)
from code.analysis.preflight import inspect_inputs, write_reports
from code.analysis.provenance import (
    config_hash,
    create_analysis_id,
    initialize,
    validate_analysis_id,
)
from code.analysis.slurm_execution import (
    execute_plan_locally,
    submit_execution_plan,
)
from code.analysis.result_validator import validate_analysis_result
from code.utils.logging_config import setup_logging
from code.utils.config import load_config

LOGGER = logging.getLogger(__name__)
SLURM_JOB_CEILING = 900


def _load_config(path: Path) -> dict:
    """Load, validate, and expand the shared project configuration."""
    return load_config(str(path))


def _analysis_root(config: dict, override: str | None) -> Path:
    if override:
        return Path(override)
    directory = config.get("analysis_workflow", {}).get(
        "processed_directory", "analysis_workflow"
    )
    return Path(config["paths"]["data_root"]) / "processed" / directory


def run_preflight(args: argparse.Namespace) -> Path:
    """Initialize an immutable ID and record required input checks."""
    config = _load_config(Path(args.config))
    analysis_id = args.analysis_id or create_analysis_id(config, Path.cwd())
    analysis_dir = initialize(
        _analysis_root(config, args.analysis_root),
        analysis_id,
        config,
        vars(args),
        Path.cwd(),
    )
    requirements = {
        "spaces": ["sensor", "schaefer_400"],
        "trial_sets": ["alltrials", "correct", "lapse"],
        "primary_trial_set": "alltrials",
        "feature_families": {
            "feature_modulation": list(FEATURE_MODULATION_FEATURES),
            "multifeature_decoding": list(CORRECTED_FEATURES),
            "network_dynamics": list(CORRECTED_FEATURES),
        },
        "frequency_bands": frequency_band_manifest(),
        "panels": PANEL_SPECS,
        "label_definition": "reflected-gaussian-filtered-vtc_strict-eight-trial",
        "coupling_label_definition": "opposite-state-free-with-mid",
        "primary_rejection": "any-bad-constituent",
    }
    subjects = args.subjects.split() if args.subjects else config["bids"]["subjects"]
    runs = args.runs.split() if args.runs else config["bids"]["task_runs"]
    input_report = inspect_inputs(config, subjects, runs)
    report = {"analysis_id": analysis_id, **requirements, **input_report}
    write_reports(report, analysis_dir / "qc")
    (analysis_dir / "preflight_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    (analysis_dir / "manifests" / "feature_families.json").write_text(
        json.dumps(requirements["feature_families"], indent=2) + "\n"
    )
    (analysis_dir / "manifests" / "schemas.json").write_text(
        json.dumps(schema_catalog(), indent=2) + "\n"
    )
    LOGGER.info("Saflow analysis preflight initialized %s", analysis_dir)
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
    if (
        not preflight.exists()
        or json.loads(preflight.read_text()).get("status") != "passed"
    ):
        raise RuntimeError("a passing analysis_workflow-preflight report is required")
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


def plan_execution(args: argparse.Namespace) -> Path:
    """Write a complete inspectable execution plan without submitting any jobs."""
    validate_analysis_id(args.analysis_id)
    config = _load_config(Path(args.config))
    analysis_dir = _analysis_root(config, args.analysis_root) / args.analysis_id
    if not analysis_dir.exists():
        raise FileNotFoundError(f"analysis does not exist: {analysis_dir}")
    subjects = args.subjects.split() if args.subjects else config["bids"]["subjects"]
    runs = args.runs.split() if args.runs else config["bids"]["task_runs"]
    spaces = args.spaces.split()
    analysis_workflow_config = config.get("analysis_workflow", {})
    manifest = build_execution_plan(
        args.analysis_id,
        subjects,
        runs,
        spaces,
        include_exploratory=args.include_exploratory,
        map_chunk_count=_chunk_count(
            analysis_workflow_config,
            "map_permutations",
            "map_chunk_size",
            10_000,
            250,
        ),
        decoding_chunk_count=_chunk_count(
            analysis_workflow_config,
            "decoding_permutations",
            "decoding_chunk_size",
            1_000,
            25,
        ),
        map_chunks_per_job=int(analysis_workflow_config.get("map_chunks_per_job", 5)),
        decoding_chunks_per_job=int(
            analysis_workflow_config.get("decoding_chunks_per_job", 5)
        ),
    )
    path = analysis_dir / "manifests" / "execution_plan.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def run_all_pipeline(args: argparse.Namespace) -> Path:
    """Create and execute the bounded plan locally or through explicit SLURM."""
    config = _load_config(Path(args.config))
    if args.slurm and not args.dry_run and shutil.which("sbatch") is None:
        raise RuntimeError(
            "SLURM submission requires sbatch; use --dry-run on non-SLURM hosts"
        )
    analysis_id = args.analysis_id or create_analysis_id(config, Path.cwd())
    validate_analysis_id(analysis_id)
    root = _analysis_root(config, args.analysis_root)
    analysis_dir = root / analysis_id
    if analysis_dir.exists():
        raise FileExistsError(f"immutable analysis already exists: {analysis_dir}")
    subjects = args.subjects.split() if args.subjects else config["bids"]["subjects"]
    runs = args.runs.split() if args.runs else config["bids"]["task_runs"]
    analysis_workflow_config = config.get("analysis_workflow", {})
    graph = build_execution_plan(
        analysis_id,
        subjects,
        runs,
        args.spaces.split(),
        include_exploratory=args.include_exploratory,
        map_chunk_count=_chunk_count(
            analysis_workflow_config, "map_permutations", "map_chunk_size", 10_000, 250
        ),
        decoding_chunk_count=_chunk_count(
            analysis_workflow_config,
            "decoding_permutations",
            "decoding_chunk_size",
            1_000,
            25,
        ),
        map_chunks_per_job=int(analysis_workflow_config.get("map_chunks_per_job", 5)),
        decoding_chunks_per_job=int(
            analysis_workflow_config.get("decoding_chunks_per_job", 5)
        ),
    )
    graph = bound_execution_plan(
        graph,
        start_at=args.start_at,
        stop_after=args.stop_after,
        skip=_split_stages(args.skip),
    )
    capacity = None
    if args.slurm:
        capacity = _available_submission_capacity(config, dry_run=args.dry_run)
        required = len(graph["expected_outputs"])
        if required > capacity:
            raise RuntimeError(
                "complete pipeline requires "
                f"{required} available SLURM job slots, but only "
                f"{capacity} remain below the 900-job ceiling; wait for at "
                f"least {required - capacity} queued/running jobs to finish, "
                "then rerun pipeline.all"
            )
    initialize(root, analysis_id, config, vars(args), Path.cwd())
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
    analysis_resources = {
        **DEFAULT_ANALYSIS_RESOURCES,
        **analysis_workflow_config.get("resources", {}),
    }
    graph["submission_plan"] = build_submission_plan(
        graph,
        analysis_resources,
        config.get("computing", {}).get("slurm", {}),
        _node_resources(analysis_workflow_config),
    )
    path = analysis_dir / "manifests" / "execution_plan.json"
    path.write_text(json.dumps(graph, indent=2, sort_keys=True) + "\n")
    if args.slurm:
        invalid = list(graph["expected_outputs"])
        ready = invalid
        deferred: list[dict] = []
        submission = submit_execution_plan(
            graph,
            analysis_dir,
            config,
            dry_run=bool(args.dry_run),
            selected_cells=ready,
        )
        submission["capacity"] = capacity
        submission["submitted_cell_count"] = len(ready)
        submission["deferred_cell_count"] = len(deferred)
        submission["expected_cell_count"] = len(invalid)
    else:
        submission = execute_plan_locally(
            graph,
            analysis_dir,
            config,
            dry_run=bool(args.dry_run),
        )
        deferred = []
    graph["scheduler"] = submission
    graph["provenance"]["execution_mode"] = "slurm" if args.slurm else "local"
    graph["provenance"]["submitted"] = bool(args.slurm and not args.dry_run)
    graph["provenance"]["submission_status"] = (
        "dry_run" if args.dry_run else ("wave_submitted" if deferred else "submitted")
    )
    path.write_text(json.dumps(graph, indent=2, sort_keys=True) + "\n")
    LOGGER.info("Full pipeline manifest ready: %s", path)
    if deferred:
        LOGGER.info(
            "Submitted %d cells; deferred %d to a later pipeline.resume wave",
            len(ready),
            len(deferred),
        )
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
    """Audit an immutable execution plan and submit a dependency-safe invalid-cell wave."""
    validate_analysis_id(args.analysis_id)
    config = _load_config(Path(args.config))
    analysis_dir = _analysis_root(config, args.analysis_root) / args.analysis_id
    if args.slurm and not args.dry_run:
        active = _active_submission_jobs(analysis_dir)
        if active:
            raise RuntimeError(
                "previous submission wave is still active; wait before resume: "
                + ", ".join(active)
            )
    plan_path = analysis_dir / "manifests" / "execution_plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(
            f"immutable execution plan manifest not found: {plan_path}"
        )
    plan = json.loads(plan_path.read_text())
    plan["submission_plan"] = build_submission_plan(
        plan,
        {
            **DEFAULT_ANALYSIS_RESOURCES,
            **config.get("analysis_workflow", {}).get("resources", {}),
        },
        config.get("computing", {}).get("slurm", {}),
        _node_resources(config.get("analysis_workflow", {})),
    )
    selected = []
    complete = []
    for expected in plan.get("expected_outputs", []):
        status_path = analysis_dir / expected["status_path"]
        reason = _invalid_cell_reason(status_path, expected, plan["provenance"])
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
        "scheduler_resources_refreshed": True,
    }
    ready, deferred = _resume_submission_wave(plan, selected)
    capacity = None
    if args.slurm:
        capacity = _available_submission_capacity(config, dry_run=args.dry_run)
        ready, capacity_deferred = _capacity_limited_wave(plan, ready, capacity)
        deferred.extend(capacity_deferred)
    resume["cells_submitted_this_wave"] = ready
    resume["cells_deferred_to_next_wave"] = deferred
    if args.slurm:
        submission = submit_execution_plan(
            plan,
            analysis_dir,
            config,
            dry_run=bool(args.dry_run),
            selected_cells=ready,
        )
        submission["capacity"] = capacity
    else:
        submission = execute_plan_locally(
            plan,
            analysis_dir,
            config,
            dry_run=bool(args.dry_run),
            selected_cells=ready,
        )
    resume["scheduler"] = submission
    resume["scheduler"]["submitted_cell_count"] = len(ready)
    resume["scheduler"]["deferred_cell_count"] = len(deferred)
    path = analysis_dir / "manifests" / "resume.json"
    path.write_text(json.dumps(resume, indent=2, sort_keys=True) + "\n")
    return path


def _node_resources(analysis_workflow_config: dict) -> dict:
    """Merge safe per-node scheduler defaults with user overrides."""
    return {
        **DEFAULT_NODE_RESOURCES,
        **analysis_workflow_config.get("node_resources", {}),
    }


def _resume_submission_wave(
    plan: dict, invalid_cells: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Select a dependency-safe recovery wave without recomputing valid cells."""
    by_node: dict[str, list[dict]] = {}
    for cell in invalid_cells:
        by_node.setdefault(cell["node"], []).append(cell)
    deferred_nodes: set[str] = set()
    changed = True
    while changed:
        changed = False
        for edge in plan["edges"]:
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
                    (cell.get("subject"), cell.get("run")) for cell in upstream_cells
                }
                downstream_indices = {
                    (cell.get("subject"), cell.get("run")) for cell in downstream_cells
                }
                if upstream_indices != downstream_indices:
                    deferred_nodes.add(downstream)
                    changed = True
    ready = [cell for cell in invalid_cells if cell["node"] not in deferred_nodes]
    deferred = [cell for cell in invalid_cells if cell["node"] in deferred_nodes]
    return ready, deferred


def _capacity_limited_wave(
    plan: dict, invalid_cells: list[dict], capacity: int
) -> tuple[list[dict], list[dict]]:
    """Select a dependency-safe prefix that fits the scheduler job ceiling."""
    if capacity < 1:
        raise RuntimeError("no SLURM submission capacity is currently available")
    by_node: dict[str, list[dict]] = {}
    for cell in invalid_cells:
        by_node.setdefault(cell["node"], []).append(cell)
    incoming = {
        node["name"]: [
            edge for edge in plan["edges"] if edge["downstream"] == node["name"]
        ]
        for node in plan["nodes"]
    }
    selected: list[dict] = []
    selected_by_node: dict[str, list[dict]] = {}
    for node in _ordered_execution_nodes(plan):
        name = node["name"]
        candidates = by_node.get(name, [])
        if not candidates:
            continue
        upstream_incomplete = [
            edge
            for edge in incoming[name]
            if by_node.get(edge["upstream"])
            and len(selected_by_node.get(edge["upstream"], []))
            != len(by_node[edge["upstream"]])
        ]
        if upstream_incomplete:
            continue
        remaining = capacity - len(selected)
        if remaining < 1:
            break
        has_recovering_aftercorr = any(
            edge["dependency"] == "aftercorr" and by_node.get(edge["upstream"])
            for edge in incoming[name]
        )
        chosen = candidates
        if len(chosen) > remaining:
            if has_recovering_aftercorr:
                continue
            chosen = candidates[:remaining]
        selected.extend(chosen)
        selected_by_node[name] = chosen
    selected_keys = {(cell["node"], int(cell["cell_index"])) for cell in selected}
    deferred = [
        cell
        for cell in invalid_cells
        if (cell["node"], int(cell["cell_index"])) not in selected_keys
    ]
    return selected, deferred


def _ordered_execution_nodes(plan: dict) -> list[dict]:
    """Return plan nodes only after all their upstream nodes."""
    by_name = {node["name"]: node for node in plan["nodes"]}
    pending = list(by_name)
    ordered = []
    while pending:
        ready = [
            name
            for name in pending
            if all(
                edge["upstream"] not in pending
                for edge in plan["edges"]
                if edge["downstream"] == name
            )
        ]
        if not ready:
            raise ValueError("execution plan contains a dependency cycle")
        for name in ready:
            ordered.append(by_name[name])
            pending.remove(name)
    return ordered


def _available_submission_capacity(config: dict, *, dry_run: bool) -> int:
    """Return new array-element capacity below the strict 900-job ceiling."""
    slurm = config.get("computing", {}).get("slurm", {})
    maximum = min(
        int(slurm.get("max_submitted_jobs", SLURM_JOB_CEILING)),
        SLURM_JOB_CEILING,
    )
    reserve = int(slurm.get("submission_job_reserve", 0))
    current = 0 if dry_run else _current_slurm_job_count()
    return maximum - reserve - current


def _current_slurm_job_count() -> int:
    """Count the user's currently queued/running SLURM array elements."""
    result = subprocess.run(
        ["squeue", "-h", "-r", "-u", os.environ.get("USER", ""), "-o", "%i"],
        check=True,
        capture_output=True,
        text=True,
    )
    return len([line for line in result.stdout.splitlines() if line.strip()])


def _active_submission_jobs(analysis_dir: Path) -> list[str]:
    """Return still-active job IDs from the most recent submission wave."""
    journal = analysis_dir / "manifests" / "submission_journal.json"
    if not journal.exists() or shutil.which("squeue") is None:
        return []
    job_ids = list(json.loads(journal.read_text()).get("job_ids", {}).values())
    real_ids = [job_id for job_id in job_ids if not str(job_id).startswith("dry-")]
    if not real_ids:
        return []
    result = subprocess.run(
        ["squeue", "-h", "-j", ",".join(real_ids), "-o", "%A"],
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(set(result.stdout.split()))


def _split_stages(value: str | None) -> list[str]:
    """Parse comma- or whitespace-separated public stage names."""
    return value.replace(",", " ").split() if value else []


def _invalid_cell_reason(path: Path, expected: dict, provenance: dict) -> str | None:
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
        raise ValueError("source is not a Saflow analysis analysis")
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
    for analysis in PANEL_ANALYSES.values():
        path = destination / analysis / "observed.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        for key, value in _scalar_items(payload.get("summary", payload)):
            rows.append({"analysis": analysis, "metric": key, "value": value})
    table = destination / "tables" / "observed_summary.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("analysis", "metric", "value"))
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
    """Audit complete bundles and rendered composite/slide provenance."""
    validate_analysis_id(args.analysis_id)
    config = _load_config(Path(args.config))
    analysis_dir = _analysis_root(config, args.analysis_root) / args.analysis_id
    reports_root = Path(args.reports_root)
    findings = []
    modes = set()
    for panel, spec in PANEL_SPECS.items():
        analysis = PANEL_ANALYSES[panel]
        try:
            validate_analysis_result(
                args.config,
                args.analysis_id,
                str(_analysis_root(config, args.analysis_root)),
                analysis,
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
        bundle = analysis_dir / analysis / "observed.json"
        composite = reports_root / "figures" / "paper" / spec["composite_filename"]
        sidecar = composite.with_name(f"{composite.name}.json")
        for kind, path in (
            ("bundle", bundle),
            ("composite", composite),
            ("sidecar", sidecar),
        ):
            if not path.exists():
                findings.append({"panel": panel, "kind": kind, "status": "missing"})
        if sidecar.exists():
            metadata = json.loads(sidecar.read_text())
            modes.add(metadata.get("data_mode"))
            if metadata.get("analysis_id") != args.analysis_id:
                findings.append(
                    {"panel": panel, "kind": "sidecar", "status": "wrong_analysis_id"}
                )
        slide_directory = reports_root / "figures" / "slides" / spec["slide_directory"]
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
    plan_path = analysis_dir / "manifests" / "execution_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()).get(
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
    path = analysis_dir / "analysis_audit.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if status != "passed":
        raise RuntimeError(f"analysis audit failed; see {path}")
    return path


def inventory_legacy(args: argparse.Namespace) -> Path:
    """Hash legacy Saflow analysis outputs without moving, deleting, or modifying them."""
    entries = []
    source = Path(args.source)
    for path in sorted(source.rglob("*")) if source.exists() else []:
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            entries.append(
                {"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest}
            )
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "read_only": True,
                "moved": False,
                "deleted": False,
                "source": str(source),
                "files": entries,
            },
            indent=2,
        )
        + "\n"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """Build the Saflow analysis workflow parser."""
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
    plan = commands.add_parser("plan")
    plan.add_argument("--config", default="config.yaml")
    plan.add_argument("--analysis-id", required=True)
    plan.add_argument("--analysis-root")
    plan.add_argument("--subjects")
    plan.add_argument("--runs")
    plan.add_argument("--spaces", default="sensor schaefer_400")
    plan.add_argument(
        "--include-exploratory", action=argparse.BooleanOptionalAction, default=True
    )
    all_pipeline = commands.add_parser("all")
    all_pipeline.add_argument("--config", default="config.yaml")
    all_pipeline.add_argument("--analysis-id")
    all_pipeline.add_argument("--analysis-root")
    all_pipeline.add_argument("--start-at")
    all_pipeline.add_argument("--stop-after")
    all_pipeline.add_argument("--skip")
    all_pipeline.add_argument("--subjects")
    all_pipeline.add_argument("--runs")
    all_pipeline.add_argument("--spaces", default="sensor schaefer_400")
    all_pipeline.add_argument(
        "--include-exploratory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    all_pipeline.add_argument("--slurm", action="store_true")
    all_pipeline.add_argument("--dry-run", action="store_true")
    resume = commands.add_parser("resume")
    resume.add_argument("--config", default="config.yaml")
    resume.add_argument("--analysis-id", required=True)
    resume.add_argument("--analysis-root")
    resume.add_argument("--slurm", action="store_true")
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
    """Run the selected Saflow analysis workflow command."""
    args = build_parser().parse_args()
    setup_logging(
        "analysis_pipeline",
        log_file="analysis_pipeline.log",
        config={"paths": {"logs": "logs"}, "logging": {"level": "INFO"}},
    )
    functions = {
        "preflight": run_preflight,
        "run": run_analysis,
        "plan": plan_execution,
        "all": run_all_pipeline,
        "resume": resume_pipeline,
        "audit": audit_analysis,
        "export": export_analysis,
        "legacy-inventory": inventory_legacy,
    }
    functions[args.command](args)


if __name__ == "__main__":
    main()
