"""Declarative, inspectable SLURM graph for the analysis pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Sequence

from code.analysis.contracts import FEATURE_MODULATION_FEATURES, CORRECTED_FEATURES

PIPELINE_STAGES = (
    "validation",
    "bids",
    "preprocess",
    "source",
    "atlas",
    "features",
    "analyses",
    "export",
    "render",
    "audit",
)
DEFAULT_ANALYSIS_RESOURCES = {
    "maps": {"time": "08:00:00", "memory_gb": 16, "cpus": 4},
    "decoding": {"time": "12:00:00", "memory_gb": 24, "cpus": 4},
    "permutation_batches": {
        "time": "3-00:00:00",
        "memory_gb": 24,
        "cpus": 4,
    },
    "rendering": {"time": "01:00:00", "memory_gb": 8, "cpus": 2},
}
DEFAULT_NODE_RESOURCES = {
    "feature_modulation_statistics": {
        "time": "12:00:00",
        "memory_gb": 64,
        "cpus": 4,
    },
    "network_factorial_modulation": {
        "time": "12:00:00",
        "memory_gb": 64,
        "cpus": 4,
    },
    "network_coupling": {
        "time": "12:00:00",
        "memory_gb": 64,
        "cpus": 4,
    },
    "multifeature_decoding_models": {
        "time": "3-00:00:00",
        "memory_gb": 32,
        "cpus": 4,
    },
    "exploratory_analyses": {
        "time": "3-00:00:00",
        "memory_gb": 32,
        "cpus": 4,
    },
}
BUNDLE_MINIMUM_RESOURCES = {
    "run_preprocessing": {"time": "18:00:00", "mem": "64G", "cpus": 12},
    "run_source": {"time": "05:00:00", "mem": "256G", "cpus": 1},
    "run_features": {"time": "1-12:00:00", "mem": "96G", "cpus": 12},
}


@dataclass(frozen=True)
class ExecutionNode:
    """Describe one pipeline node without submitting it."""

    name: str
    kind: str
    array: bool = False
    exploratory: bool = False


@dataclass(frozen=True)
class ExecutionDependency:
    """Describe one dependency and its SLURM dependency type."""

    upstream: str
    downstream: str
    dependency: str


def build_execution_plan(
    analysis_id: str,
    subjects: Sequence[str],
    runs: Sequence[str],
    spaces: Sequence[str] = ("sensor", "schaefer_400"),
    *,
    include_exploratory: bool = True,
    map_chunk_count: int = 40,
    decoding_chunk_count: int = 40,
    map_chunks_per_job: int = 5,
    decoding_chunks_per_job: int = 5,
) -> dict[str, Any]:
    """Build the immutable raw-to-panels execution plan manifest."""
    nodes = [
        ExecutionNode("input_validation", "validator"),
        ExecutionNode("run_preprocessing", "worker", array=True),
    ]
    edges = [
        ExecutionDependency("input_validation", "run_preprocessing", "afterok"),
    ]
    feature_upstream = "run_preprocessing"
    if "schaefer_400" in spaces:
        nodes.append(ExecutionNode("run_source", "worker", array=True))
        edges.append(
            ExecutionDependency("run_preprocessing", "run_source", "aftercorr")
        )
        feature_upstream = "run_source"
    nodes.append(ExecutionNode("run_features", "worker", array=True))
    edges.append(
        ExecutionDependency(feature_upstream, "run_features", "aftercorr")
    )
    _add_feature_validators(nodes, edges, spaces)
    _add_analysis_branches(nodes, edges, include_exploratory)
    cells = [
        {"index": index, "subject": subject, "run": run}
        for index, (subject, run) in enumerate(product(subjects, runs))
    ]
    manifest = {
        "analysis_id": analysis_id,
        "nodes": [asdict(node) for node in nodes],
        "edges": [asdict(edge) for edge in edges],
        "array_cells": cells,
        "spaces": list(spaces),
        "include_exploratory": include_exploratory,
        "provenance": {"status": "planned", "submitted": False},
    }
    manifest["node_cells"] = _build_node_cells(
        manifest,
        map_chunk_count=map_chunk_count,
        decoding_chunk_count=decoding_chunk_count,
        map_chunks_per_job=map_chunks_per_job,
        decoding_chunks_per_job=decoding_chunks_per_job,
    )
    return manifest


def bound_execution_plan(
    manifest: dict[str, Any],
    *,
    start_at: str | None = None,
    stop_after: str | None = None,
    skip: Sequence[str] = (),
) -> dict[str, Any]:
    """Return a stage-bounded plan while retaining inspectable exclusions."""
    aliases = {
        "bids": "preprocess",
        "atlas": "source",
        "source-recon": "source",
        "source_recon": "source",
        "feature-modulation": "analyses",
        "multifeature-decoding": "analyses",
        "network-dynamics": "analyses",
        "panels": "render",
    }
    start = aliases.get(start_at, start_at) if start_at else PIPELINE_STAGES[0]
    stop = aliases.get(stop_after, stop_after) if stop_after else PIPELINE_STAGES[-1]
    skipped = {aliases.get(name, name) for name in skip}
    if start not in PIPELINE_STAGES or stop not in PIPELINE_STAGES:
        raise ValueError(f"stage bounds must be in {PIPELINE_STAGES}")
    if PIPELINE_STAGES.index(start) > PIPELINE_STAGES.index(stop):
        raise ValueError("start-at occurs after stop-after")
    retained_stages = set(
        PIPELINE_STAGES[
            PIPELINE_STAGES.index(start) : PIPELINE_STAGES.index(stop) + 1
        ]
    ) - skipped
    nodes = manifest["nodes"]
    retained_names = {
        node["name"]
        for node in nodes
        if stage_for_node(node["name"]) in retained_stages
    }
    bounded = {
        **manifest,
        "nodes": [node for node in nodes if node["name"] in retained_names],
        "edges": [
            edge
            for edge in manifest["edges"]
            if edge["upstream"] in retained_names and edge["downstream"] in retained_names
        ],
        "bounds": {
            "start_at": start,
            "stop_after": stop,
            "skipped_stages": sorted(skipped),
        },
        "excluded_nodes": [
            node["name"] for node in nodes if node["name"] not in retained_names
        ],
    }
    bounded["expected_outputs"] = expected_output_cells(bounded)
    return bounded


def stage_for_node(name: str) -> str:
    """Map an execution plan node to its public bounded-execution stage."""
    if name == "input_validation":
        return "validation"
    if name == "run_preprocessing":
        return "preprocess"
    if name == "run_source":
        return "source"
    if name == "run_features":
        return "features"
    if (
        name.endswith("_feature_validator")
        or name.startswith(("sensor_", "schaefer_400_"))
        or name == "run_features"
    ):
        return "features"
    if name == "analysis_export":
        return "export"
    if name == "panel_generation":
        return "render"
    if name == "analysis_audit":
        return "audit"
    return "analyses"


def expected_output_cells(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand every retained node into immutable expected output cells."""
    outputs = []
    for node in manifest["nodes"]:
        cells = manifest["node_cells"].get(node["name"], [{"index": 0}])
        for cell in cells:
            outputs.append(
                {
                    "node": node["name"],
                    "cell_index": cell["index"],
                    "subject": cell.get("subject"),
                    "run": cell.get("run"),
                    "feature": cell.get("feature"),
                    "model": cell.get("model"),
                    "chunk_index": cell.get("chunk_index"),
                    "chunk_indices": cell.get("chunk_indices"),
                    "status_path": (
                        f"manifests/cells/{node['name']}/"
                        f"cell-{cell['index']:04d}.json"
                    ),
                }
            )
    return outputs


def _build_node_cells(
    manifest: dict[str, Any],
    *,
    map_chunk_count: int,
    decoding_chunk_count: int,
    map_chunks_per_job: int,
    decoding_chunks_per_job: int,
) -> dict[str, list[dict[str, Any]]]:
    """Define the scientifically meaningful index mapping for every node."""
    subject_run_nodes = {
        "run_preprocessing",
        "run_source",
        "run_features",
    }
    node_cells: dict[str, list[dict[str, Any]]] = {}
    for node in manifest["nodes"]:
        name = node["name"]
        if not node["array"]:
            node_cells[name] = [{"index": 0}]
        elif name in subject_run_nodes:
            node_cells[name] = manifest["array_cells"]
        elif name == "feature_modulation_statistics":
            node_cells[name] = _named_cells("feature", FEATURE_MODULATION_FEATURES)
        elif name == "feature_modulation_decoding_permutations":
            node_cells[name] = _feature_chunk_batch_cells(
                FEATURE_MODULATION_FEATURES,
                map_chunk_count,
                map_chunks_per_job,
            )
        elif name == "multifeature_decoding_models":
            node_cells[name] = _named_cells(
                "model", ("state", "lapse_within_IN", "lapse_within_OUT")
            )
        elif name == "multifeature_decoding_permutations":
            node_cells[name] = _chunk_batch_cells(
                decoding_chunk_count,
                decoding_chunks_per_job,
            )
        elif name in {"network_factorial_modulation", "network_coupling"}:
            node_cells[name] = _named_cells("feature", CORRECTED_FEATURES)
        else:
            node_cells[name] = [{"index": 0}]
    return node_cells


def _named_cells(field: str, values: Sequence[str]) -> list[dict[str, Any]]:
    """Create ordered cells identified by a named scientific dimension."""
    return [
        {"index": index, field: value} for index, value in enumerate(values)
    ]


def _chunk_batch_cells(
    chunk_count: int,
    chunks_per_job: int,
) -> list[dict[str, Any]]:
    """Group ordered immutable chunks into sequential scheduler cells."""
    if chunk_count < 1 or chunks_per_job < 1:
        raise ValueError("chunk counts and chunks per job must be positive")
    batches = []
    for start in range(0, chunk_count, chunks_per_job):
        indices = list(range(start, min(start + chunks_per_job, chunk_count)))
        batches.append({"index": len(batches), "chunk_indices": indices})
    return batches


def _feature_chunk_batch_cells(
    features: Sequence[str],
    chunk_count: int,
    chunks_per_job: int,
) -> list[dict[str, Any]]:
    """Group each feature's chunks without mixing scientific families."""
    batches = []
    for feature_index, feature in enumerate(features):
        for batch in _chunk_batch_cells(chunk_count, chunks_per_job):
            chunk_indices = batch["chunk_indices"]
            batches.append(
                {
                    "index": len(batches),
                    "feature": feature,
                    "chunk_indices": chunk_indices,
                    "chunk_cell_indices": [
                        feature_index * chunk_count + chunk_index
                        for chunk_index in chunk_indices
                    ],
                }
            )
    return batches


def build_submission_plan(
    manifest: dict[str, Any],
    resources: dict[str, Any],
    stage_resources: dict[str, Any] | None = None,
    node_resources: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Resolve node resources and dependency expressions in topological order.

    The returned records are scheduler-neutral until submission assigns real
    job IDs. Dry runs use stable ``dry-<node>`` identifiers so the complete
    dependency graph remains inspectable without requiring SLURM.
    """
    nodes = {node["name"]: node for node in manifest["nodes"]}
    incoming: dict[str, list[dict[str, str]]] = {name: [] for name in nodes}
    for edge in manifest["edges"]:
        incoming[edge["downstream"]].append(edge)
    plan = []
    for node in manifest["nodes"]:
        name = node["name"]
        dependencies = [
            {
                "node": edge["upstream"],
                "type": edge["dependency"],
                "job_id": f"dry-{edge['upstream']}",
            }
            for edge in incoming[name]
        ]
        plan.append(
            {
                **node,
                "job_id": f"dry-{name}",
                "array_size": len(manifest["node_cells"][name]),
                "resources": _resources_for_node(
                    name,
                    resources,
                    stage_resources or {},
                    node_resources or DEFAULT_NODE_RESOURCES,
                ),
                "dependencies": dependencies,
            }
        )
    _validate_aftercorr_alignment(manifest, plan)
    return plan


def _resources_for_node(
    name: str,
    resources: dict[str, Any],
    stage_resources: dict[str, Any],
    node_resources: dict[str, Any],
) -> dict[str, Any]:
    """Choose a configured resource class for one scientific endpoint."""
    if name in node_resources:
        selected = node_resources[name]
        return {
            "class": f"node:{name}",
            "time": selected["time"],
            "mem": f"{selected['memory_gb']}G",
            "cpus": selected["cpus"],
        }
    raw_key = _raw_resource_key(name)
    if raw_key and raw_key in stage_resources:
        selected = stage_resources[raw_key]
        if name in BUNDLE_MINIMUM_RESOURCES:
            return _bounded_bundle_resources(
                name,
                raw_key,
                selected,
            )
        return {
            "class": raw_key,
            "time": selected["time"],
            "mem": str(selected["mem"]),
            "cpus": selected["cpus"],
        }
    if "permutation" in name:
        key = (
            "permutation_batches"
            if "permutation_batches" in resources
            else "decoding"
        )
    elif (
        "decoding" in name
        or name == "multifeature_decoding_models"
        or name == "exploratory_analyses"
    ):
        key = "decoding"
    elif name in {"analysis_export", "panel_generation"}:
        key = "rendering"
    else:
        key = "maps"
    selected = resources[key]
    return {
        "class": key,
        "time": selected["time"],
        "mem": f"{selected['memory_gb']}G",
        "cpus": selected["cpus"],
    }


def _bounded_bundle_resources(
    name: str,
    resource_class: str,
    configured: dict[str, Any],
) -> dict[str, Any]:
    """Apply safe lower bounds to sequential multi-stage run allocations."""
    minimum = BUNDLE_MINIMUM_RESOURCES[name]
    configured_time = str(configured["time"])
    time = (
        configured_time
        if _slurm_time_seconds(configured_time)
        >= _slurm_time_seconds(minimum["time"])
        else minimum["time"]
    )
    configured_mem = str(configured["mem"])
    memory = (
        configured_mem
        if _memory_gb(configured_mem) >= _memory_gb(minimum["mem"])
        else minimum["mem"]
    )
    return {
        "class": resource_class,
        "time": time,
        "mem": memory,
        "cpus": max(int(configured["cpus"]), int(minimum["cpus"])),
    }


def _slurm_time_seconds(value: str) -> int:
    """Convert SLURM ``[days-]HH:MM:SS`` time to seconds."""
    day_text, clock = value.split("-", 1) if "-" in value else ("0", value)
    hours, minutes, seconds = (int(part) for part in clock.split(":"))
    return int(day_text) * 86_400 + hours * 3_600 + minutes * 60 + seconds


def _memory_gb(value: str) -> float:
    """Convert a SLURM memory request in G or M to GiB."""
    normalized = value.strip().upper()
    if normalized.endswith("G"):
        return float(normalized[:-1])
    if normalized.endswith("M"):
        return float(normalized[:-1]) / 1024
    raise ValueError(f"memory must use G or M units: {value}")


def _raw_resource_key(name: str) -> str | None:
    """Map raw-to-feature nodes to established SLURM resource sections."""
    if name == "input_validation":
        return "bids"
    if name == "run_preprocessing":
        return "preprocessing"
    if name == "run_source":
        return "source_reconstruction"
    if name == "run_features":
        return "features"
    if name.endswith("_feature_validator"):
        return "report"
    return None


def _validate_aftercorr_alignment(
    manifest: dict[str, Any], plan: Sequence[dict[str, Any]]
) -> None:
    """Ensure every aftercorr edge connects identical array index mappings."""
    by_name = {record["name"]: record for record in plan}
    for edge in manifest["edges"]:
        if edge["dependency"] != "aftercorr":
            continue
        upstream = by_name[edge["upstream"]]
        downstream = by_name[edge["downstream"]]
        if not upstream["array"] or not downstream["array"]:
            raise ValueError("aftercorr is valid only between job arrays")
        upstream_cells = manifest["node_cells"][edge["upstream"]]
        downstream_cells = manifest["node_cells"][edge["downstream"]]
        if upstream_cells != downstream_cells:
            raise ValueError(
                f"aftercorr index mismatch: {edge['upstream']} -> {edge['downstream']}"
            )


def _add_feature_validators(
    nodes: list[ExecutionNode],
    edges: list[ExecutionDependency],
    spaces: Sequence[str],
) -> None:
    """Add one completeness barrier per requested spatial representation."""
    for space in spaces:
        validator = f"{space}_feature_validator"
        nodes.append(ExecutionNode(validator, "validator"))
        edges.append(
            ExecutionDependency("run_features", validator, "afterany")
        )


def _add_analysis_branches(
    nodes: list[ExecutionNode], edges: list[ExecutionDependency], include_exploratory: bool
) -> None:
    """Add concurrent scientific analyses, export, render, and audit barriers."""
    source = "schaefer_400_feature_validator"
    analysis_workers = {
        "feature_modulation": (
            "feature_modulation_statistics",
            "feature_modulation_decoding_permutations",
        ),
        "multifeature_decoding": (
            "multifeature_decoding_models",
            "multifeature_decoding_permutations",
        ),
        "network_dynamics": (
            "network_factorial_modulation",
            "network_coupling",
        ),
    }
    for analysis, worker_names in analysis_workers.items():
        validator = f"{analysis}_validator"
        aggregator = f"{analysis}_results"
        nodes.extend(ExecutionNode(name, "worker", array=True) for name in worker_names)
        nodes.extend(
            [ExecutionNode(aggregator, "aggregator"), ExecutionNode(validator, "validator")]
        )
        for worker_name in worker_names:
            edges.append(ExecutionDependency(source, worker_name, "afterok"))
            edges.append(ExecutionDependency(worker_name, aggregator, "afterany"))
        edges.append(ExecutionDependency(aggregator, validator, "afterok"))
    if include_exploratory:
        nodes.append(ExecutionNode("exploratory_analyses", "worker", array=True, exploratory=True))
        edges.append(ExecutionDependency(source, "exploratory_analyses", "afterok"))
    nodes.extend([
        ExecutionNode("analysis_export", "exporter"),
        ExecutionNode("panel_generation", "renderer"),
        ExecutionNode("analysis_audit", "validator"),
    ])
    for analysis in analysis_workers:
        edges.append(
            ExecutionDependency(f"{analysis}_validator", "analysis_export", "afterok")
        )
    if include_exploratory:
        edges.append(
            ExecutionDependency(
                "exploratory_analyses",
                "analysis_export",
                "afterok",
            )
        )
    edges.extend([
        ExecutionDependency("analysis_export", "panel_generation", "afterok"),
        ExecutionDependency("panel_generation", "analysis_audit", "afterok"),
    ])
