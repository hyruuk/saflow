"""Concurrency regression tests for shared BIDS empty-room derivatives."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from code.bids import generate_bids


def test_concurrent_noise_writers_create_subject_derivative_once(
    tmp_path: Path, monkeypatch
) -> None:
    """Concurrent run cells must serialize and reuse one noise derivative."""
    bids_root = tmp_path / "bids"
    write_count = 0
    count_lock = Lock()

    def write_noise(
        subject: str,
        recording_date: str,
        noise_files: dict[str, Path],
        output_root: Path,
    ) -> None:
        del recording_date, noise_files
        nonlocal write_count
        with count_lock:
            write_count += 1
        meg_dir = output_root / f"sub-{subject}" / "meg"
        meg_dir.mkdir(parents=True, exist_ok=True)
        (meg_dir / f"sub-{subject}_task-noise_meg.fif").write_bytes(b"fif")
        (meg_dir / f"sub-{subject}_task-noise_meg.json").write_text("{}\n")
        (meg_dir / f"sub-{subject}_task-noise_channels.tsv").write_text(
            "name\ttype\n"
        )

    monkeypatch.setattr(generate_bids, "_write_noise_recording", write_noise)
    noise_files = {"20200101": tmp_path / "emptyroom.ds"}
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [
            executor.submit(
                generate_bids.copy_noise_to_subject,
                "08",
                "20200101",
                noise_files,
                bids_root,
            )
            for _ in range(6)
        ]
        for future in futures:
            future.result()

    assert write_count == 1
    assert generate_bids._noise_derivative_is_complete(bids_root, "08")


def test_slurm_template_isolates_mne_runtime_state() -> None:
    """Every array element must avoid the shared user-level MNE lock."""
    template = Path("slurm/templates/base.sh.j2").read_text()
    assert "export MNE_DONTWRITE_HOME=true" in template
    assert 'export MPLCONFIGDIR="$SAFLOW_RUNTIME_DIR/matplotlib"' in template
    assert "${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-0}" in template
