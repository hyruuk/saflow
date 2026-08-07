"""Declarative, inspectable SLURM graph for the paper-panel pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Sequence

from code.paper_panels.contracts import PANEL1_FEATURES, PANEL23_FEATURES

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
DEFAULT_PAPER_PANEL_RESOURCES = {
    "maps": {"time": "08:00:00", "memory_gb": 16, "cpus": 4},
    "decoding": {"time": "12:00:00", "memory_gb": 24, "cpus": 4},
    "rendering": {"time": "01:00:00", "memory_gb": 8, "cpus": 2},
}


@dataclass(frozen=True)
class DagNode:
    """Describe one pipeline node without submitting it."""

    name: str
    kind: str
    array: bool = False
    exploratory: bool = False


@dataclass(frozen=True)
class DagEdge:
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
) -> dict[str, Any]:
    """Build the immutable raw-to-panels execution plan manifest."""
    nodes = [
        DagNode("input_validation", "validator"),
        DagNode("bids_reflected_vtc", "worker", array=True),
        DagNode("preprocessing", "worker", array=True),
        DagNode("source_reconstruction", "worker", array=True),
        DagNode("schaefer_400_atlas", "worker", array=True),
    ]
    edges = [
        DagEdge("input_validation", "bids_reflected_vtc", "afterok"),
        DagEdge("bids_reflected_vtc", "preprocessing", "aftercorr"),
        DagEdge("preprocessing", "source_reconstruction", "aftercorr"),
        DagEdge("source_reconstruction", "schaefer_400_atlas", "aftercorr"),
    ]
    _add_feature_branches(nodes, edges, spaces, include_exploratory)
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
    )
    return manifest


def bound_execution_plan(
    manifest: dict[str, Any],
    *,
    start_at: str | None = None,
    stop_after: str | None = None,
    skip: Sequence[str] = (),
) -> dict[str, Any]:
    """Return a stage-bounded graph while retaining inspectable exclusions."""
    aliases = {
        "source-recon": "source",
        "source_recon": "source",
        "panel1": "analyses",
        "panel2": "analyses",
        "panel3": "analyses",
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
    """Map a execution plan node to its public bounded-execution stage."""
    if name == "input_validation":
        return "validation"
    if name == "bids_reflected_vtc":
        return "bids"
    if name == "preprocessing":
        return "preprocess"
    if name == "source_reconstruction":
        return "source"
    if name == "schaefer_400_atlas":
        return "atlas"
    if (
        "feature" in name
        or "complexity" in name
        or name.endswith(("_psd", "_fooof_corrected_psd"))
    ):
        return "features"
    if name == "compact_export_tables_slides":
        return "export"
    if name == "paper_composites":
        return "render"
    if name == "final_analysis_audit":
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
) -> dict[str, list[dict[str, Any]]]:
    """Define the scientifically meaningful index mapping for every node."""
    subject_run_nodes = {
        "bids_reflected_vtc",
        "preprocessing",
        "source_reconstruction",
        "schaefer_400_atlas",
        "sensor_psd",
        "sensor_fooof_corrected_psd",
        "sensor_complexity_exploratory",
        "schaefer_400_psd",
        "schaefer_400_fooof_corrected_psd",
        "schaefer_400_complexity_exploratory",
    }
    node_cells: dict[str, list[dict[str, Any]]] = {}
    for node in manifest["nodes"]:
        name = node["name"]
        if not node["array"]:
            node_cells[name] = [{"index": 0}]
        elif name in subject_run_nodes:
            node_cells[name] = manifest["array_cells"]
        elif name == "panel1_statistics":
            node_cells[name] = _named_cells("feature", PANEL1_FEATURES)
        elif name == "panel1_decoding_permutations":
            node_cells[name] = _feature_chunk_cells(
                PANEL1_FEATURES, map_chunk_count
            )
        elif name == "panel2_observed_models":
            node_cells[name] = _named_cells(
                "model", ("state", "lapse_within_IN", "lapse_within_OUT")
            )
        elif name == "panel2_permutation_chunks":
            node_cells[name] = _chunk_cells(decoding_chunk_count)
        elif name in {"panel3_factorial_maps", "panel3_coupling"}:
            node_cells[name] = _named_cells("feature", PANEL23_FEATURES)
        else:
            node_cells[name] = [{"index": 0}]
    return node_cells


def _named_cells(field: str, values: Sequence[str]) -> list[dict[str, Any]]:
    """Create ordered cells identified by a named scientific dimension."""
    return [
        {"index": index, field: value} for index, value in enumerate(values)
    ]


def _chunk_cells(count: int) -> list[dict[str, Any]]:
    """Create ordered immutable permutation chunk cells."""
    if count < 1:
        raise ValueError("chunk count must be positive")
    return [
        {"index": index, "chunk_index": index} for index in range(count)
    ]


def _feature_chunk_cells(
    features: Sequence[str], chunk_count: int
) -> list[dict[str, Any]]:
    """Create synchronized feature × permutation-chunk cells."""
    return [
        {
            "index": feature_index * chunk_count + chunk_index,
            "feature": feature,
            "chunk_index": chunk_index,
        }
        for feature_index, feature in enumerate(features)
        for chunk_index in range(chunk_count)
    ]


def build_submission_plan(
    manifest: dict[str, Any],
    resources: dict[str, Any],
    stage_resources: dict[str, Any] | None = None,
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
                    name, resources, stage_resources or {}
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
) -> dict[str, Any]:
    """Choose a configured resource class for one scientific endpoint."""
    raw_key = _raw_resource_key(name)
    if raw_key and raw_key in stage_resources:
        selected = stage_resources[raw_key]
        return {
            "class": raw_key,
            "time": selected["time"],
            "mem": str(selected["mem"]),
            "cpus": selected["cpus"],
        }
    if (
        "decoding" in name
        or "permutation" in name
        or name == "panel2_observed_models"
        or name == "exploratory_analyses"
    ):
        key = "decoding"
    elif name in {"compact_export_tables_slides", "paper_composites"}:
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


def _raw_resource_key(name: str) -> str | None:
    """Map raw-to-feature nodes to established SLURM resource sections."""
    if name in {"input_validation", "bids_reflected_vtc"}:
        return "bids"
    if name == "preprocessing":
        return "preprocessing"
    if name == "source_reconstruction":
        return "source_reconstruction"
    if name == "schaefer_400_atlas":
        return "atlas"
    if (
        name.endswith(("_psd", "_fooof_corrected_psd"))
        or "complexity" in name
    ):
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


def _add_feature_branches(
    nodes: list[DagNode],
    edges: list[DagEdge],
    spaces: Sequence[str],
    include_exploratory: bool,
) -> None:
    """Add PSD, FOOOF, corrected-PSD, complexity, and validation nodes."""
    for space in spaces:
        upstream = "preprocessing" if space == "sensor" else "schaefer_400_atlas"
        psd = f"{space}_psd"
        fooof = f"{space}_fooof_corrected_psd"
        validator = f"{space}_feature_validator"
        nodes.extend([
            DagNode(psd, "worker", array=True),
            DagNode(fooof, "worker", array=True),
            DagNode(validator, "validator"),
        ])
        edges.extend([
            DagEdge(upstream, psd, "aftercorr"),
            DagEdge(psd, fooof, "aftercorr"),
            DagEdge(psd, validator, "afterany"),
            DagEdge(fooof, validator, "afterany"),
        ])
        if include_exploratory:
            complexity = f"{space}_complexity_exploratory"
            nodes.append(DagNode(complexity, "worker", array=True, exploratory=True))
            edges.extend([
                DagEdge(upstream, complexity, "aftercorr"),
                DagEdge(complexity, validator, "afterany"),
            ])


def _add_analysis_branches(
    nodes: list[DagNode], edges: list[DagEdge], include_exploratory: bool
) -> None:
    """Add concurrent panel inference, export, render, and audit barriers."""
    source = "schaefer_400_feature_validator"
    panel_workers = {
        "panel1": ("statistics", "decoding_permutations"),
        "panel2": ("observed_models", "permutation_chunks"),
        "panel3": ("factorial_maps", "coupling"),
    }
    for panel, worker_names in panel_workers.items():
        validator = f"{panel}_validator"
        aggregator = f"{panel}_aggregator"
        nodes.extend(DagNode(f"{panel}_{name}", "worker", array=True) for name in worker_names)
        nodes.extend([DagNode(aggregator, "aggregator"), DagNode(validator, "validator")])
        for worker_name in worker_names:
            worker = f"{panel}_{worker_name}"
            edges.append(DagEdge(source, worker, "afterok"))
            edges.append(DagEdge(worker, aggregator, "afterany"))
        edges.append(DagEdge(aggregator, validator, "afterok"))
    if include_exploratory:
        nodes.append(DagNode("exploratory_analyses", "worker", array=True, exploratory=True))
        edges.append(DagEdge(source, "exploratory_analyses", "afterok"))
    nodes.extend([
        DagNode("compact_export_tables_slides", "exporter"),
        DagNode("paper_composites", "renderer"),
        DagNode("final_analysis_audit", "validator"),
    ])
    for panel in panel_workers:
        edges.append(DagEdge(f"{panel}_validator", "compact_export_tables_slides", "afterok"))
    if include_exploratory:
        edges.append(
            DagEdge(
                "exploratory_analyses",
                "compact_export_tables_slides",
                "afterok",
            )
        )
    edges.extend([
        DagEdge("compact_export_tables_slides", "paper_composites", "afterok"),
        DagEdge("paper_composites", "final_analysis_audit", "afterok"),
    ])
