"""Phase D execution plan, resume, and protected export tests."""

import json
import sys
from argparse import Namespace
from pathlib import Path

from code.paper_panels.execution_plan import (
    bound_execution_plan,
    build_execution_plan,
    build_submission_plan,
)
from code.paper_panels.workflow import (
    _capacity_limited_wave,
    _invalid_cell_reason,
    _resume_submission_wave,
    build_parser,
)
from code.paper_panels.workflow import export_analysis
from code.paper_panels.cell_status import execute_cell


def test_full_plan_bounds_features_and_keeps_dependency_barriers():
    raw = build_execution_plan(
        "paper-20260102T030405Z-gabc-c123456789abc",
        ["04", "05"],
        ["02", "03"],
        include_exploratory=True,
    )
    bounded = bound_execution_plan(raw, stop_after="features")
    names = {node["name"] for node in bounded["nodes"]}
    assert "schaefer_400_feature_validator" in names
    assert "panel1_statistics" not in names
    assert {edge["dependency"] for edge in bounded["edges"]} == {
        "afterok",
        "aftercorr",
        "afterany",
    }
    preprocessing = [
        cell
        for cell in bounded["expected_outputs"]
        if cell["node"] == "preprocessing"
    ]
    assert len(preprocessing) == 4


def test_panel_branches_are_concurrent_after_validated_schaefer_inputs():
    plan = bound_execution_plan(
        build_execution_plan("analysis", ["04"], ["02"]),
        start_at="analyses",
        stop_after="render",
    )
    workers = {
        edge["downstream"]
        for edge in plan["edges"]
        if edge["upstream"] == "schaefer_400_feature_validator"
    }
    # The upstream validator is outside the bounded graph, so the three panel
    # branches have no artificial edges between one another.
    assert not workers
    names = {node["name"] for node in plan["nodes"]}
    assert {"panel1_statistics", "panel2_observed_models", "panel3_coupling"} <= names
    assert not any(
        edge["upstream"].startswith("panel1")
        and edge["downstream"].startswith(("panel2", "panel3"))
        for edge in plan["edges"]
    )


def test_submission_plan_records_resources_arrays_and_dependency_types():
    plan = bound_execution_plan(
        build_execution_plan("analysis", ["04", "05"], ["02", "03"]),
        stop_after="features",
    )
    resources = {
        "maps": {"time": "01:00:00", "memory_gb": 8, "cpus": 2},
        "decoding": {"time": "02:00:00", "memory_gb": 16, "cpus": 4},
        "rendering": {"time": "00:30:00", "memory_gb": 4, "cpus": 1},
    }
    plan = build_submission_plan(plan, resources)
    by_name = {record["name"]: record for record in plan}
    assert by_name["preprocessing"]["array_size"] == 4
    dependency = by_name["source_reconstruction"]["dependencies"][0]
    assert dependency == {
        "node": "preprocessing",
        "type": "aftercorr",
        "job_id": "dry-preprocessing",
    }
    assert by_name["sensor_feature_validator"]["array_size"] == 1


def test_scientific_arrays_use_feature_model_and_chunk_cells():
    plan = bound_execution_plan(
        build_execution_plan(
            "analysis",
            ["04", "05"],
            ["02", "03"],
            map_chunk_count=2,
            decoding_chunk_count=3,
        ),
        start_at="analyses",
        stop_after="analyses",
    )
    cells = plan["node_cells"]
    assert len(cells["panel1_statistics"]) == 17
    assert len(cells["panel1_decoding_permutations"]) == 34
    assert [cell["model"] for cell in cells["panel2_observed_models"]] == [
        "state",
        "lapse_within_IN",
        "lapse_within_OUT",
    ]
    assert len(cells["panel2_permutation_chunks"]) == 3
    assert len(cells["panel3_factorial_maps"]) == 10
    assert all("subject" not in cell for cell in cells["panel3_coupling"])


def test_resume_reason_selects_only_invalid_cells(tmp_path: Path):
    expected = {"node": "panel2_permutation_chunks", "cell_index": 3}
    provenance = {
        "analysis_id": "analysis",
        "config_hash": "config",
        "git_commit": "commit",
    }
    path = tmp_path / "cell.json"
    assert _invalid_cell_reason(path, expected, provenance) == "missing"
    path.write_text("{bad json")
    assert _invalid_cell_reason(path, expected, provenance) == "corrupt"
    payload = {
        "status": "complete",
        "analysis_id": "analysis",
        "node": expected["node"],
        "cell_index": 3,
        "config_hash": "config",
        "git_commit": "commit",
    }
    path.write_text(json.dumps(payload))
    assert _invalid_cell_reason(path, expected, provenance) is None
    payload["config_hash"] = "different"
    path.write_text(json.dumps(payload))
    assert (
        _invalid_cell_reason(path, expected, provenance)
        == "incompatible_config_hash"
    )


def test_compact_export_omits_chunks_and_writes_hashed_table(tmp_path: Path):
    analysis_id = "paper-20260102T030405Z-gabc-c123456789abc"
    source = tmp_path / "source" / analysis_id
    (source / "panel1" / "chunks").mkdir(parents=True)
    (source / "panel1" / "observed.json").write_text(
        '{"summary": {"metrics": {"auc": 0.75}}}'
    )
    (source / "panel1" / "observed.npz").write_bytes(b"compact")
    (source / "panel1" / "chunks" / "private.npz").write_bytes(b"private")
    (source / "preflight_report.json").write_text("{}")
    destination = tmp_path / "export"
    export_analysis(
        Namespace(
            analysis_id=analysis_id,
            analysis_root=str(tmp_path / "source"),
            destination=str(destination),
        )
    )
    assert not (destination / "panel1" / "chunks").exists()
    table = destination / "tables" / "observed_summary.csv"
    assert "panel1,metrics.auc,0.75" in table.read_text()
    manifest = json.loads((destination / "export_manifest.json").read_text())
    assert not manifest["contains_subject_feature_matrices"]
    assert all("sha256" in item for item in manifest["files"])


def test_cell_status_wrapper_records_compatible_complete_status(tmp_path: Path):
    status = tmp_path / "status.json"
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "analysis_id": "analysis",
                "node": "input_validation",
                "cell_index": 0,
                "config_hash": "config",
                "git_commit": "commit",
                "status_path": str(status),
                "command": [sys.executable, "-c", "raise SystemExit(0)"],
            }
        )
    )
    execute_cell(spec)
    payload = json.loads(status.read_text())
    assert payload["status"] == "complete"
    assert payload["return_code"] == 0
    assert payload["analysis_id"] == "analysis"


def test_resume_defers_misaligned_aftercorr_subsets_without_valid_recompute():
    plan = bound_execution_plan(
        build_execution_plan("analysis", ["04", "05"], ["02"]),
        stop_after="preprocess",
    )
    invalid = [
        {
            **next(
                cell
                for cell in plan["expected_outputs"]
                if cell["node"] == "bids_reflected_vtc"
                and cell["cell_index"] == 0
            ),
            "reason": "failed",
        },
        {
            **next(
                cell
                for cell in plan["expected_outputs"]
                if cell["node"] == "preprocessing"
                and cell["cell_index"] == 1
            ),
            "reason": "missing",
        },
    ]
    ready, deferred = _resume_submission_wave(plan, invalid)
    assert [cell["node"] for cell in ready] == ["bids_reflected_vtc"]
    assert [cell["node"] for cell in deferred] == ["preprocessing"]


def test_submission_wave_stays_below_rorqual_capacity():
    plan = bound_execution_plan(
        build_execution_plan(
            "analysis",
            [f"{index:02d}" for index in range(1, 33)],
            [f"{index:02d}" for index in range(1, 7)],
        )
    )
    ready, deferred = _capacity_limited_wave(
        plan, list(plan["expected_outputs"]), capacity=900
    )
    assert 0 < len(ready) <= 900
    assert deferred
    selected_nodes = {cell["node"] for cell in ready}
    assert "input_validation" in selected_nodes
    assert "bids_reflected_vtc" in selected_nodes


def test_all_pipeline_requires_explicit_slurm_flag():
    local = build_parser().parse_args(["all", "--dry-run"])
    cluster = build_parser().parse_args(["all", "--slurm", "--dry-run"])
    assert not local.slurm
    assert cluster.slurm
