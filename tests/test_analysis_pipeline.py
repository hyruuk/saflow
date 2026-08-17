"""Phase D execution plan, resume, and protected export tests."""

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from code.analysis.execution_plan import (
    DEFAULT_NODE_RESOURCES,
    bound_execution_plan,
    build_execution_plan,
    build_submission_plan,
)
from code.analysis.workflow import (
    SLURM_JOB_CEILING,
    _active_submission_nodes,
    _available_submission_capacity,
    _capacity_limited_wave,
    _downstream_nodes,
    _invalid_cell_reason,
    _resolve_analysis,
    _resume_submission_wave,
    build_parser,
)
from code.analysis.workflow import export_analysis
from code.analysis.cell_status import execute_cell


def test_full_plan_bounds_features_and_keeps_dependency_barriers():
    raw = build_execution_plan(
        "analysis-20260102T030405Z-gabc-c123456789abc",
        ["04", "05"],
        ["02", "03"],
        include_exploratory=True,
    )
    bounded = bound_execution_plan(raw, stop_after="features")
    names = {node["name"] for node in bounded["nodes"]}
    assert "schaefer_400_feature_validator" in names
    assert "feature_modulation_statistics" not in names
    assert {edge["dependency"] for edge in bounded["edges"]} == {
        "afterok",
        "aftercorr",
        "afterany",
    }
    preprocessing = [
        cell
        for cell in bounded["expected_outputs"]
        if cell["node"] == "run_preprocessing"
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
    assert {"feature_modulation_statistics", "multifeature_decoding_models", "network_coupling"} <= names
    assert not any(
        edge["upstream"].startswith("feature_modulation")
        and edge["downstream"].startswith(("multifeature_decoding", "network_dynamics"))
        for edge in plan["edges"]
    )


def test_execution_plan_can_target_panel1_analysis_only():
    plan = bound_execution_plan(
        build_execution_plan(
            "analysis", ["04"], ["02"],
            analyses=("feature_modulation",), include_exploratory=False,
        ),
        start_at="analyses", stop_after="analyses",
    )
    names = {node["name"] for node in plan["nodes"]}
    assert "feature_modulation_statistics" in names
    assert "feature_modulation_results" in names
    assert not any(name.startswith("multifeature") for name in names)
    assert not any(name.startswith("network_") for name in names)


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
    assert by_name["run_preprocessing"]["array_size"] == 4
    dependency = by_name["run_source"]["dependencies"][0]
    assert dependency == {
        "node": "run_preprocessing",
        "type": "aftercorr",
        "job_id": "dry-run_preprocessing",
    }
    assert by_name["sensor_feature_validator"]["array_size"] == 1


def test_expensive_analysis_nodes_have_recovery_sized_resources():
    plan = bound_execution_plan(
        build_execution_plan("analysis", ["04"], ["02"]),
        start_at="analyses",
        stop_after="analyses",
    )
    resources = {
        "maps": {"time": "01:00:00", "memory_gb": 8, "cpus": 2},
        "decoding": {"time": "02:00:00", "memory_gb": 16, "cpus": 4},
        "permutation_batches": {
            "time": "3-00:00:00",
            "memory_gb": 24,
            "cpus": 4,
        },
        "rendering": {"time": "00:30:00", "memory_gb": 4, "cpus": 1},
    }
    by_name = {
        record["name"]: record
        for record in build_submission_plan(
            plan, resources, node_resources=DEFAULT_NODE_RESOURCES
        )
    }
    assert by_name["feature_modulation_statistics"]["resources"] == {
        "class": "node:feature_modulation_statistics",
        "time": "12:00:00",
        "mem": "64G",
        "cpus": 4,
    }
    assert by_name["network_coupling"]["resources"]["mem"] == "64G"
    assert by_name["feature_modulation_results"]["resources"]["mem"] == "64G"
    assert by_name["network_dynamics_results"]["resources"]["mem"] == "64G"
    assert by_name["multifeature_decoding_models"]["resources"]["time"] == (
        "3-00:00:00"
    )
    assert by_name["multifeature_decoding_permutations"]["resources"]["mem"] == (
        "24G"
    )


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
    assert len(cells["feature_modulation_statistics"]) == 16
    assert len(cells["feature_modulation_decoding_permutations"]) == 16
    assert all(
        cell["feature"] != "fooof_r_squared"
        for cell in cells["feature_modulation_decoding_permutations"]
    )
    assert cells["feature_modulation_decoding_permutations"][0]["chunk_indices"] == [0, 1]
    assert [cell["model"] for cell in cells["multifeature_decoding_models"]] == [
        "state",
        "lapse_within_IN",
        "lapse_within_OUT",
    ]
    assert len(cells["multifeature_decoding_permutations"]) == 1
    assert cells["multifeature_decoding_permutations"][0]["chunk_indices"] == [0, 1, 2]
    assert len(cells["network_factorial_modulation"]) == 9
    assert all("subject" not in cell for cell in cells["network_coupling"])


def test_resume_reason_selects_only_invalid_cells(tmp_path: Path):
    expected = {"node": "multifeature_decoding_permutations", "cell_index": 3}
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
    analysis_id = "analysis-20260102T030405Z-gabc-c123456789abc"
    source = tmp_path / "source" / analysis_id
    (source / "feature_modulation" / "chunks").mkdir(parents=True)
    (source / "feature_modulation" / "observed.json").write_text(
        '{"summary": {"metrics": {"auc": 0.75}}}'
    )
    (source / "feature_modulation" / "observed.npz").write_bytes(b"compact")
    (source / "feature_modulation" / "chunks" / "private.npz").write_bytes(b"private")
    (source / "preflight_report.json").write_text("{}")
    destination = tmp_path / "export"
    export_analysis(
        Namespace(
            analysis_id=analysis_id,
            analysis_root=str(tmp_path / "source"),
            destination=str(destination),
        )
    )
    assert not (destination / "feature_modulation" / "chunks").exists()
    table = destination / "tables" / "observed_summary.csv"
    assert "feature_modulation,metrics.auc,0.75" in table.read_text()
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
                "execution_git_commit": "recovery-commit",
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
    assert payload["execution_git_commit"] == "recovery-commit"


def test_cell_status_stops_bundle_after_first_failed_step(tmp_path: Path):
    status = tmp_path / "status.json"
    marker = tmp_path / "must-not-run"
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "analysis_id": "analysis",
                "node": "run_preprocessing",
                "cell_index": 0,
                "config_hash": "config",
                "git_commit": "commit",
                "status_path": str(status),
                "commands": [
                    [sys.executable, "-c", "raise SystemExit(0)"],
                    [sys.executable, "-c", "raise SystemExit(7)"],
                    [
                        sys.executable,
                        "-c",
                        f"from pathlib import Path; Path({str(marker)!r}).touch()",
                    ],
                ],
            }
        )
    )

    with pytest.raises(subprocess.CalledProcessError):
        execute_cell(spec)
    payload = json.loads(status.read_text())
    assert payload["status"] == "failed"
    assert payload["return_code"] == 7
    assert len(payload["steps"]) == 2
    assert not marker.exists()


def test_resume_defers_misaligned_aftercorr_subsets_without_valid_recompute():
    plan = bound_execution_plan(
        build_execution_plan("analysis", ["04", "05"], ["02"]),
        stop_after="source",
    )
    invalid = [
        {
            **next(
                cell
                for cell in plan["expected_outputs"]
                if cell["node"] == "run_preprocessing"
                and cell["cell_index"] == 0
            ),
            "reason": "failed",
        },
        {
            **next(
                cell
                for cell in plan["expected_outputs"]
                if cell["node"] == "run_source"
                and cell["cell_index"] == 1
            ),
            "reason": "missing",
        },
    ]
    ready, deferred = _resume_submission_wave(plan, invalid)
    assert [cell["node"] for cell in ready] == ["run_preprocessing"]
    assert [cell["node"] for cell in deferred] == ["run_source"]


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
    assert not deferred
    selected_nodes = {cell["node"] for cell in ready}
    assert "input_validation" in selected_nodes
    assert "run_preprocessing" in selected_nodes
    assert len(ready) == 762


def test_submission_capacity_counts_existing_jobs_and_never_exceeds_900(
    monkeypatch,
):
    config = {
        "computing": {
            "slurm": {
                "max_submitted_jobs": 900,
                "submission_job_reserve": 25,
            }
        }
    }
    monkeypatch.setattr(
        "code.analysis.workflow._current_slurm_job_count",
        lambda: 127,
    )
    assert SLURM_JOB_CEILING == 900
    assert _available_submission_capacity(config, dry_run=False) == 748
    assert _available_submission_capacity(config, dry_run=True) == 875


def test_active_submission_nodes_preserve_node_identity(tmp_path, monkeypatch):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "submission_journal.json").write_text(
        json.dumps(
            {
                "job_ids": {
                    "multifeature_decoding_models": "101",
                    "network_coupling": "202",
                    "dry_node": "dry-example",
                }
            }
        )
    )
    monkeypatch.setattr("code.analysis.workflow.shutil.which", lambda _: "/usr/bin/squeue")
    monkeypatch.setattr(
        "code.analysis.workflow.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "101\n", ""),
    )
    assert _active_submission_nodes(tmp_path) == {
        "multifeature_decoding_models": "101"
    }


def test_active_branch_blocks_only_its_downstream_nodes():
    plan = build_execution_plan("analysis", ["04"], ["02"])
    blocked = _downstream_nodes(plan, {"multifeature_decoding_models"})
    assert "multifeature_decoding_results" in blocked
    assert "multifeature_decoding_validator" in blocked
    assert "analysis_export" in blocked
    assert "network_coupling" not in blocked
    assert "network_dynamics_results" not in blocked


def test_four_subject_pipeline_fits_in_one_complete_submission():
    plan = bound_execution_plan(
        build_execution_plan(
            "analysis",
            [f"{index:02d}" for index in range(1, 5)],
            [f"{index:02d}" for index in range(2, 8)],
        )
    )
    ready, deferred = _capacity_limited_wave(
        plan, list(plan["expected_outputs"]), capacity=900
    )
    assert len(ready) == 258
    assert not deferred
    assert {
        "feature_modulation_results",
        "multifeature_decoding_results",
        "network_dynamics_results",
    } <= {cell["node"] for cell in ready}


def test_all_pipeline_requires_explicit_slurm_flag():
    local = build_parser().parse_args(["all", "--dry-run"])
    cluster = build_parser().parse_args(["all", "--slurm", "--dry-run"])
    assert not local.slurm
    assert cluster.slurm


def test_user_facing_commands_default_to_active_analysis(tmp_path: Path):
    analysis_id = "analysis-20260102T030405Z-gabc-c123456789abc"
    active = tmp_path / "processed" / "panel_analysis" / "main"
    active.mkdir(parents=True)
    (active / "provenance.json").write_text(
        json.dumps({"analysis_id": analysis_id}) + "\n"
    )
    config = {
        "paths": {"data_root": str(tmp_path)},
        "analysis_workflow": {"processed_directory": "panel_analysis"},
    }

    resolved_id, resolved_directory = _resolve_analysis(config, None, None)

    assert resolved_id == analysis_id
    assert resolved_directory == active
    assert build_parser().parse_args(["plan"]).analysis_id is None
    assert build_parser().parse_args(["run"]).analysis_id is None
