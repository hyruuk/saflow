"""SLURM job submission utilities for HPC execution.

This module provides helper functions for submitting and managing SLURM jobs
on compute clusters.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)


def get_slurm_template_dir() -> Path:
    """Get path to SLURM templates directory."""
    # Get project root (saflow/)
    project_root = Path(__file__).parent.parent.parent
    template_dir = project_root / "slurm" / "templates"

    if not template_dir.exists():
        raise FileNotFoundError(f"SLURM template directory not found: {template_dir}")

    return template_dir


def render_slurm_script(
    template_name: str,
    context: Dict,
    output_path: Optional[Path] = None,
) -> str:
    """Render a SLURM job script from a Jinja2 template.

    Args:
        template_name: Name of template file (e.g., "preprocessing.sh.j2")
        context: Dictionary of variables to pass to template
        output_path: Optional path to save rendered script

    Returns:
        Rendered script as string

    Examples:
        >>> script = render_slurm_script(
        ...     "preprocessing.sh.j2",
        ...     {"subject": "04", "run": "02", "cpus": 12, "mem": "32G"}
        ... )
    """
    template_dir = get_slurm_template_dir()

    # Create Jinja2 environment
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    # Load template
    template = env.get_template(template_name)

    # Render template
    script = template.render(**context)

    # Save if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(script)
        logger.debug(f"Rendered SLURM script saved to: {output_path}")

    return script


def submit_slurm_job(
    script_path: Path,
    job_name: Optional[str] = None,
    dependencies: Optional[List[str]] = None,
    dry_run: bool = False,
) -> Optional[str]:
    """Submit a SLURM job script using sbatch.

    Args:
        script_path: Path to SLURM script file
        job_name: Optional job name (overrides script's #SBATCH --job-name)
        dependencies: Optional list of job IDs this job depends on
        dry_run: If True, print command without submitting

    Returns:
        Job ID as string, or None if dry_run=True or submission failed

    Examples:
        >>> job_id = submit_slurm_job(Path("preprocessing_04_02.sh"))
        >>> print(f"Submitted job: {job_id}")
    """
    if not script_path.exists():
        raise FileNotFoundError(f"SLURM script not found: {script_path}")

    # Build sbatch command
    cmd = ["sbatch"]

    if job_name:
        cmd.extend(["--job-name", job_name])

    if dependencies:
        # Format: --dependency=afterok:jobid1:jobid2
        dep_str = f"afterok:{':'.join(dependencies)}"
        cmd.extend(["--dependency", dep_str])

    cmd.append(str(script_path))

    # Log command
    cmd_str = " ".join(cmd)
    logger.info(f"Submitting SLURM job: {cmd_str}")

    if dry_run:
        print(f"[DRY RUN] Would submit: {cmd_str}")
        return None

    # Submit job
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        # Parse job ID from output: "Submitted batch job 12345"
        output = result.stdout.strip()
        job_id = output.split()[-1]

        logger.info(f"Job submitted successfully: {job_id}")
        print(f"✓ Submitted job {job_id}: {script_path.name}")

        return job_id

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to submit job: {e.stderr}")
        print(f"✗ Job submission failed: {e.stderr}")
        return None


def check_job_status(job_id: str) -> Optional[Dict]:
    """Check status of a SLURM job.

    Args:
        job_id: Job ID to check

    Returns:
        Dictionary with job info (state, time, etc.), or None if job not found

    Examples:
        >>> status = check_job_status("12345")
        >>> print(status["State"])
    """
    cmd = [
        "sacct",
        "-j", job_id,
        "--format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS",
        "--noheader",
        "--parsable2",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        # Parse output
        lines = result.stdout.strip().split("\n")
        if not lines:
            return None

        # First line is the main job (not job steps)
        fields = lines[0].split("|")

        return {
            "JobID": fields[0],
            "JobName": fields[1],
            "State": fields[2],
            "ExitCode": fields[3],
            "Elapsed": fields[4],
            "MaxRSS": fields[5],
        }

    except subprocess.CalledProcessError:
        logger.warning(f"Could not get status for job {job_id}")
        return None


def cancel_job(job_id: str) -> bool:
    """Cancel a running or pending SLURM job.

    Args:
        job_id: Job ID to cancel

    Returns:
        True if successful, False otherwise

    Examples:
        >>> cancel_job("12345")
    """
    cmd = ["scancel", job_id]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"Cancelled job {job_id}")
        print(f"✓ Cancelled job {job_id}")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to cancel job {job_id}: {e.stderr}")
        print(f"✗ Failed to cancel job {job_id}")
        return False


def get_user_jobs(user: Optional[str] = None) -> List[Dict]:
    """Get list of user's SLURM jobs.

    Args:
        user: Username (default: current user)

    Returns:
        List of job dictionaries with status info

    Examples:
        >>> jobs = get_user_jobs()
        >>> for job in jobs:
        ...     print(f"{job['JobID']}: {job['State']}")
    """
    cmd = ["squeue", "-u"]

    if user:
        cmd.append(user)
    else:
        # Get current user
        import getpass
        cmd.append(getpass.getuser())

    cmd.extend([
        "--format=%i|%j|%t|%M|%D|%C|%m",
        "--noheader",
    ])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        jobs = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue

            fields = line.split("|")
            jobs.append({
                "JobID": fields[0],
                "JobName": fields[1],
                "State": fields[2],
                "Time": fields[3],
                "Nodes": fields[4],
                "CPUs": fields[5],
                "Memory": fields[6],
            })

        return jobs

    except subprocess.CalledProcessError:
        logger.warning("Could not get user jobs")
        return []


def save_job_manifest(
    job_ids: List[str],
    manifest_path: Path,
    metadata: Optional[Dict] = None,
) -> None:
    """Save list of submitted job IDs to a manifest file.

    This is useful for tracking batches of jobs and implementing
    dependencies or post-processing.

    Args:
        job_ids: List of SLURM job IDs
        manifest_path: Path to save manifest JSON
        metadata: Optional additional metadata to include

    Examples:
        >>> save_job_manifest(
        ...     ["12345", "12346", "12347"],
        ...     Path("logs/slurm/preprocess_manifest.json"),
        ...     metadata={"stage": "preprocessing", "timestamp": "2024-01-01"}
        ... )
    """
    manifest = {
        "job_ids": job_ids,
        "num_jobs": len(job_ids),
    }

    if metadata:
        manifest["metadata"] = metadata

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Saved job manifest: {manifest_path}")
    print(f"✓ Saved job manifest: {manifest_path}")


def load_job_manifest(manifest_path: Path) -> Dict:
    """Load a job manifest file.

    Args:
        manifest_path: Path to manifest JSON file

    Returns:
        Dictionary with job_ids and metadata

    Examples:
        >>> manifest = load_job_manifest(Path("logs/slurm/preprocess_manifest.json"))
        >>> print(f"Jobs: {manifest['job_ids']}")
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    return manifest
