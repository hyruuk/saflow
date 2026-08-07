"""Render and submit immutable Panel analysis execution plan cells to SLURM."""

from __future__ import annotations

import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from code.analysis.cell_status import execute_cell
from code.analysis.execution_plan import stage_for_node
from code.utils.slurm import (
    render_slurm_script,
    submit_job_array,
    submit_slurm_job,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FEATURE_NODES = {
    "input_validation",
    "run_preprocessing",
    "run_source",
    "run_features",
    "sensor_feature_validator",
    "schaefer_400_feature_validator",
}
OBSERVED_NODES = {
    "panel1_statistics",
    "panel2_observed_models",
    "panel3_factorial_maps",
    "panel3_coupling",
}
PERMUTATION_NODES = {
    "panel1_decoding_permutations",
    "panel2_permutation_chunks",
}
AGGREGATION_NODES = {
    "panel1_aggregator",
    "panel2_aggregator",
    "panel3_aggregator",
}
VALIDATOR_NODES = {
    "panel1_validator",
    "panel2_validator",
    "panel3_validator",
}
FINALIZATION_NODES = {
    "exploratory_analyses",
    "compact_export_tables_slides",
    "figure_composites",
    "analysis_audit",
}
IMPLEMENTED_NODES = (
    RAW_FEATURE_NODES
    | OBSERVED_NODES
    | PERMUTATION_NODES
    | AGGREGATION_NODES
    | VALIDATOR_NODES
    | FINALIZATION_NODES
)


def submit_execution_plan(
    manifest: dict[str, Any],
    analysis_dir: Path,
    config: dict[str, Any],
    *,
    dry_run: bool,
    selected_cells: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Render all selected cells and submit topologically valid job arrays."""
    selected = _selected_indices(manifest, selected_cells)
    retained = _topological_nodes(manifest, [
        node for node in manifest["nodes"] if selected.get(node["name"])
    ])
    unsupported = sorted(
        node["name"] for node in retained if node["name"] not in IMPLEMENTED_NODES
    )
    if unsupported and not dry_run:
        raise RuntimeError(
            "production submission blocked; node adapters are incomplete: "
            + ", ".join(unsupported)
        )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    script_root = analysis_dir / "slurm" / "scripts"
    log_root = Path(config["paths"]["logs"]) / "analysis_pipeline" / analysis_dir.name
    job_ids: dict[str, str] = {}
    submissions = []
    plan = {record["name"]: record for record in manifest["submission_plan"]}
    for node in retained:
        name = node["name"]
        records = [
            cell
            for cell in manifest["node_cells"][name]
            if cell["index"] in selected[name]
        ]
        scripts = [
            _render_cell(
                manifest,
                analysis_dir,
                config,
                plan[name],
                cell,
                script_root,
                log_root,
                timestamp,
            )
            for cell in records
        ]
        dependencies, dependency_type = _dependencies(
            manifest, name, job_ids, selected
        )
        job_id = _submit_node(
            node,
            scripts,
            plan[name],
            script_root,
            timestamp,
            dependencies,
            dependency_type,
            dry_run,
            config,
        )
        if not dry_run and job_id is None:
            raise RuntimeError(f"SLURM submission failed for node {name}")
        resolved_id = job_id or f"dry-{name}"
        job_ids[name] = resolved_id
        submissions.append(
            {
                "node": name,
                "job_id": resolved_id,
                "cell_indices": sorted(selected[name]),
                "dependencies": dependencies,
                "dependency_type": dependency_type,
                "supported": name in IMPLEMENTED_NODES,
                "dry_run": dry_run,
            }
        )
        _write_submission_journal(
            analysis_dir,
            manifest["analysis_id"],
            job_ids,
            submissions,
        )
    return {"job_ids": job_ids, "submissions": submissions}


def execute_plan_locally(
    manifest: dict[str, Any],
    analysis_dir: Path,
    config: dict[str, Any],
    *,
    dry_run: bool,
    selected_cells: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Render and sequentially execute selected cells without SLURM."""
    selected = _selected_indices(manifest, selected_cells)
    retained = _topological_nodes(
        manifest,
        [node for node in manifest["nodes"] if selected.get(node["name"])],
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    script_root = analysis_dir / "local" / "scripts"
    log_root = Path(config["paths"]["logs"]) / "analysis_pipeline" / analysis_dir.name
    plan = {record["name"]: record for record in manifest["submission_plan"]}
    executions = []
    for node in retained:
        name = node["name"]
        records = [
            cell
            for cell in manifest["node_cells"][name]
            if cell["index"] in selected[name]
        ]
        for cell in records:
            script = _render_cell(
                manifest,
                analysis_dir,
                config,
                plan[name],
                cell,
                script_root,
                log_root,
                timestamp,
            )
            spec = script.parent / f"cell-{cell['index']:04d}.json"
            if not dry_run:
                execute_cell(spec)
            executions.append(
                {
                    "node": name,
                    "cell_index": cell["index"],
                    "spec_path": str(spec),
                    "dry_run": dry_run,
                }
            )
    return {"mode": "local", "executions": executions}


def _write_submission_journal(
    analysis_dir: Path,
    analysis_id: str,
    job_ids: dict[str, str],
    submissions: list[dict[str, Any]],
) -> None:
    """Persist every successful submission before attempting its downstream."""
    path = analysis_dir / "manifests" / "submission_journal.json"
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(
            {
                "analysis_id": analysis_id,
                "job_ids": job_ids,
                "submissions": submissions,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary.replace(path)


def _topological_nodes(
    manifest: dict[str, Any], nodes: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Order retained nodes so every submitted dependency already has an ID."""
    by_name = {node["name"]: node for node in nodes}
    pending = list(by_name)
    ordered = []
    while pending:
        ready = [
            name
            for name in pending
            if all(
                edge["upstream"] not in by_name
                or edge["upstream"] in {node["name"] for node in ordered}
                for edge in manifest["edges"]
                if edge["downstream"] == name
            )
        ]
        if not ready:
            raise ValueError("execution plan contains a cycle or unresolved dependency")
        for name in ready:
            ordered.append(by_name[name])
            pending.remove(name)
    return ordered


def _selected_indices(
    manifest: dict[str, Any], selected: Sequence[dict[str, Any]] | None
) -> dict[str, set[int]]:
    """Resolve full-run or resume-selected cell indices by node."""
    if selected is None:
        return {
            name: {cell["index"] for cell in cells}
            for name, cells in manifest["node_cells"].items()
        }
    resolved = {name: set() for name in manifest["node_cells"]}
    for cell in selected:
        resolved[cell["node"]].add(int(cell["cell_index"]))
    return resolved


def _render_cell(
    manifest: dict[str, Any],
    analysis_dir: Path,
    config: dict[str, Any],
    plan: dict[str, Any],
    cell: dict[str, Any],
    script_root: Path,
    log_root: Path,
    timestamp: str,
) -> Path:
    """Render one status-wrapped cell script and immutable command spec."""
    name = plan["name"]
    log_root.mkdir(parents=True, exist_ok=True)
    cell_dir = script_root / name
    cell_dir.mkdir(parents=True, exist_ok=True)
    commands = commands_for_cell(
        name,
        cell,
        manifest["analysis_id"],
        analysis_dir.parent,
        config,
        subjects=sorted({item["subject"] for item in manifest["array_cells"]}),
        runs=sorted({item["run"] for item in manifest["array_cells"]}),
        spaces=manifest["spaces"],
        include_exploratory=manifest["include_exploratory"],
    )
    expected = next(
        item
        for item in manifest["expected_outputs"]
        if item["node"] == name and item["cell_index"] == cell["index"]
    )
    spec = {
        "analysis_id": manifest["analysis_id"],
        "node": name,
        "cell_index": cell["index"],
        "config_hash": manifest["provenance"]["config_hash"],
        "git_commit": manifest["provenance"]["git_commit"],
        "status_path": str(analysis_dir / expected["status_path"]),
        "commands": commands,
    }
    spec_path = cell_dir / f"cell-{cell['index']:04d}.json"
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    resources = _base_resources(config, plan, log_root)
    script_path = cell_dir / f"cell-{cell['index']:04d}.sh"
    render_slurm_script(
        "analysis_cell.sh.j2",
        {
            **resources,
            "job_name": f"saflow_{name[:30]}",
            "timestamp": timestamp,
            "spec_path": shlex.quote(str(spec_path)),
            "config_path": shlex.quote(
                str(analysis_dir / "resolved_config.yaml")
            ),
        },
        output_path=script_path,
    )
    return script_path


def commands_for_cell(
    node: str,
    cell: dict[str, Any],
    analysis_id: str,
    analysis_root: Path,
    config: dict[str, Any],
    *,
    subjects: Sequence[str] = (),
    runs: Sequence[str] = (),
    spaces: Sequence[str] = (),
    include_exploratory: bool = True,
) -> list[list[str]]:
    """Return ordered commands executed inside one scheduler allocation."""
    python = str(Path(config["paths"]["venv"]) / "bin" / "python")
    invoke = str(Path(config["paths"]["venv"]) / "bin" / "invoke")
    subject = cell.get("subject")
    run = cell.get("run")
    if node == "run_preprocessing":
        return [
            [
                invoke,
                "pipeline.bids",
                f"--subjects={subject}",
                f"--runs={run}",
            ],
            [
                invoke,
                "pipeline.preprocess",
                f"--subject={subject}",
                f"--runs={run}",
            ],
        ]
    if node == "run_source":
        return [
            [
                invoke,
                "pipeline.source-recon",
                f"--subject={subject}",
                f"--runs={run}",
            ],
            [
                invoke,
                "pipeline.atlas",
                f"--subject={subject}",
                f"--runs={run}",
                "--atlases=schaefer_400",
            ],
        ]
    if node == "run_features":
        return _feature_commands(
            invoke,
            str(subject),
            str(run),
            spaces,
            include_exploratory,
        )
    if node == "panel1_decoding_permutations":
        return _panel1_permutation_commands(
            python,
            cell,
            analysis_id,
            analysis_root,
            config,
            subjects,
            runs,
        )
    if node == "panel2_permutation_chunks":
        return _panel2_permutation_commands(
            python,
            cell,
            analysis_id,
            analysis_root,
            subjects,
            runs,
        )
    return [
        command_for_cell(
            node,
            cell,
            analysis_id,
            analysis_root,
            config,
            subjects=subjects,
            runs=runs,
        )
    ]


def command_for_cell(
    node: str,
    cell: dict[str, Any],
    analysis_id: str,
    analysis_root: Path,
    config: dict[str, Any],
    *,
    subjects: Sequence[str] = (),
    runs: Sequence[str] = (),
) -> list[str]:
    """Return one concrete command for a non-bundled execution-plan cell."""
    python = str(Path(config["paths"]["venv"]) / "bin" / "python")
    if node == "input_validation":
        return [python, "-m", "code.utils.validation", "--check-inputs"]
    if node in {"sensor_feature_validator", "schaefer_400_feature_validator"}:
        command = [
            python,
            "-m",
            "code.analysis.validation_runner",
            "--space",
            node.removesuffix("_feature_validator"),
        ]
        if subjects:
            command.extend(["--subjects", " ".join(subjects)])
        if runs:
            command.extend(["--runs", " ".join(runs)])
        return command
    if node in OBSERVED_NODES:
        command = [
            python,
            "-m",
            "code.analysis.observed_runner",
            "--analysis-id",
            analysis_id,
            "--analysis-root",
            str(analysis_root),
            "--node",
            node,
            "--cell-index",
            str(cell["index"]),
        ]
        for field in ("feature", "model"):
            if cell.get(field):
                command.extend([f"--{field}", cell[field]])
        if subjects:
            command.extend(["--subjects", " ".join(subjects)])
        if runs:
            command.extend(["--runs", " ".join(runs)])
        return command
    if node in AGGREGATION_NODES:
        panel = node.removesuffix("_aggregator")
        command = [
            python,
            "-m",
            "code.analysis.aggregate_runner",
            "--analysis-id",
            analysis_id,
            "--analysis-root",
            str(analysis_root),
            "--panel",
            panel,
        ]
        if subjects:
            command.extend(["--subjects", " ".join(subjects)])
        if runs:
            command.extend(["--runs", " ".join(runs)])
        return command
    if node in VALIDATOR_NODES:
        return [
            python,
            "-m",
            "code.analysis.panel_validator",
            "--analysis-id",
            analysis_id,
            "--analysis-root",
            str(analysis_root),
            "--panel",
            node.removesuffix("_validator"),
        ]
    if node == "exploratory_analyses":
        resources = config.get("panel_analysis", {}).get("resources", {}).get(
            "maps", {}
        )
        return [
            python,
            "-m",
            "code.analysis.exploratory_runner",
            "--analysis-id",
            analysis_id,
            "--analysis-root",
            str(analysis_root),
            "--n-permutations",
            str(config.get("panel_analysis", {}).get("decoding_permutations", 1_000)),
            "--jobs",
            str(resources.get("cpus", 4)),
        ]
    if node == "compact_export_tables_slides":
        return [
            python,
            "-m",
            "code.analysis.workflow",
            "export",
            "--analysis-id",
            analysis_id,
            "--analysis-root",
            str(analysis_root),
            "--destination",
            str(Path(config["paths"]["reports"]) / "exports" / analysis_id),
        ]
    if node == "figure_composites":
        return [
            python,
            "-m",
            "code.analysis.render",
            "--panel",
            "all",
            "--analysis-id",
            analysis_id,
            "--analysis-root",
            str(analysis_root),
            "--reports-root",
            str(config["paths"]["reports"]),
        ]
    if node == "analysis_audit":
        return [
            python,
            "-m",
            "code.analysis.workflow",
            "audit",
            "--analysis-id",
            analysis_id,
            "--analysis-root",
            str(analysis_root),
            "--reports-root",
            str(config["paths"]["reports"]),
        ]
    return [
        python,
        "-c",
        (
            "raise RuntimeError('Panel analysis node adapter is not implemented: "
            f"{node}')"
        ),
    ]


def _feature_commands(
    invoke: str,
    subject: str,
    run: str,
    spaces: Sequence[str],
    include_exploratory: bool,
) -> list[list[str]]:
    """Build sequential PSD, FOOOF, and optional complexity commands."""
    commands = []
    for space in spaces:
        common = [f"--subject={subject}", f"--runs={run}", f"--space={space}"]
        commands.append([invoke, "pipeline.features.psd", *common])
        commands.append([invoke, "pipeline.features.fooof", *common])
        if include_exploratory:
            commands.append([invoke, "pipeline.features.complexity", *common])
    return commands


def _panel1_permutation_commands(
    python: str,
    cell: dict[str, Any],
    analysis_id: str,
    analysis_root: Path,
    config: dict[str, Any],
    subjects: Sequence[str],
    runs: Sequence[str],
) -> list[list[str]]:
    """Build one command per immutable Panel 1 chunk in a scheduler batch."""
    jobs = str(
        config.get("panel_analysis", {})
        .get("resources", {})
        .get("maps", {})
        .get("cpus", 4)
    )
    commands = []
    for chunk_index, cell_index in zip(
        cell["chunk_indices"],
        cell["chunk_cell_indices"],
        strict=True,
    ):
        command = [
            python,
            "-m",
            "code.analysis.panel1_decoding_runner",
            "--analysis-id",
            analysis_id,
            "--analysis-root",
            str(analysis_root),
            "--feature",
            cell["feature"],
            "--chunk-index",
            str(chunk_index),
            "--cell-index",
            str(cell_index),
            "--jobs",
            jobs,
            "--skip-valid",
        ]
        if subjects:
            command.extend(["--subjects", " ".join(subjects)])
        if runs:
            command.extend(["--runs", " ".join(runs)])
        commands.append(command)
    return commands


def _panel2_permutation_commands(
    python: str,
    cell: dict[str, Any],
    analysis_id: str,
    analysis_root: Path,
    subjects: Sequence[str],
    runs: Sequence[str],
) -> list[list[str]]:
    """Build one command per immutable Panel 2 chunk in a scheduler batch."""
    commands = []
    for chunk_index in cell["chunk_indices"]:
        command = [
            python,
            "-m",
            "code.analysis.panel2_permutation_runner",
            "--analysis-id",
            analysis_id,
            "--analysis-root",
            str(analysis_root),
            "--chunk-index",
            str(chunk_index),
            "--cell-index",
            str(chunk_index),
            "--skip-valid",
        ]
        if subjects:
            command.extend(["--subjects", " ".join(subjects)])
        if runs:
            command.extend(["--runs", " ".join(runs)])
        commands.append(command)
    return commands


def _dependencies(
    manifest: dict[str, Any],
    node: str,
    job_ids: dict[str, str],
    selected: dict[str, set[int]],
) -> tuple[list[str], str]:
    """Resolve one node's homogeneous SLURM dependency clause."""
    edges = [
        edge for edge in manifest["edges"] if edge["downstream"] == node
    ]
    active = [edge for edge in edges if selected.get(edge["upstream"])]
    types = {edge["dependency"] for edge in active}
    if len(types) > 1:
        raise ValueError(f"node {node} has mixed dependency types: {types}")
    if types == {"aftercorr"}:
        downstream_indices = selected[node]
        for edge in active:
            if selected[edge["upstream"]] != downstream_indices:
                raise ValueError(
                    f"resume aftercorr subsets differ: {edge['upstream']} -> {node}"
                )
    return (
        [job_ids[edge["upstream"]] for edge in active],
        next(iter(types), "afterok"),
    )


def _submit_node(
    node: dict[str, Any],
    scripts: list[Path],
    plan: dict[str, Any],
    script_root: Path,
    timestamp: str,
    dependencies: list[str],
    dependency_type: str,
    dry_run: bool,
    config: dict[str, Any],
) -> str | None:
    """Submit one array or singleton with its resolved dependencies."""
    resources = _base_resources(
        config,
        plan,
        Path(config["paths"]["logs"]) / "panel_analysis",
    )
    if node["array"]:
        return submit_job_array(
            scripts,
            f"saflow_{node['name']}",
            resources,
            script_root / node["name"],
            timestamp,
            max_concurrent=int(
                config.get("computing", {}).get("slurm", {}).get(
                    "array_throttle", 0
                )
            ),
            dependencies=dependencies or None,
            dep_type=dependency_type,
            dry_run=dry_run,
        )
    return submit_slurm_job(
        scripts[0],
        dependencies=dependencies or None,
        dep_type=dependency_type,
        dry_run=dry_run,
    )


def _base_resources(
    config: dict[str, Any], plan: dict[str, Any], log_root: Path
) -> dict[str, Any]:
    """Translate the Panel analysis resource contract to the shared template."""
    slurm = config["computing"]["slurm"]
    return {
        "account": slurm["account"],
        "partition": slurm.get("partition", ""),
        "cpus": plan["resources"]["cpus"],
        "mem": plan["resources"]["mem"],
        "time": plan["resources"]["time"],
        "log_dir": str(log_root),
        "venv_path": str(config["paths"]["venv"]),
        "project_root": str(PROJECT_ROOT),
    }
