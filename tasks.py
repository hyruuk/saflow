"""Invoke tasks for saflow MEG pipeline.

This module provides organized tasks using invoke's Collection feature.
Tasks are grouped into namespaces for clean organization.

Usage:
    invoke --list                          # List all tasks
    invoke dev.check.dataset               # Check dataset completeness
    invoke dev.check.qc --subject=04       # Run data quality checks
    invoke dev.check.code                  # Run linting/formatting checks
    invoke pipeline.preprocess --subject=04
    invoke pipeline.features.fooof --subject=04
    invoke pipeline.features.all --slurm   # All features on HPC
    invoke analysis.stats --features=all
    invoke analysis.classify --features=all

Namespaces:
    dev.check           - Data validation (dataset, qc, code)
    dev                 - Development tasks (test, clean)
    env                 - Environment tasks (setup, info)
    get                 - Data downloads (atlases)
    pipeline            - Data processing pipeline (bids, preprocess, source-recon, atlas)
    pipeline.features   - Feature extraction (psd, fooof, complexity, all)
    analysis            - Statistical analysis (stats, classify)
    viz                 - Visualization (behavior)

SLURM Support:
    Most pipeline tasks support --slurm for HPC execution:
    - invoke pipeline.preprocess --slurm
    - invoke pipeline.source-recon --slurm
    - invoke pipeline.features.psd --slurm
    - invoke pipeline.features.all --slurm --space=aparc.a2009s
"""

import os
from pathlib import Path

from invoke import Collection, task

# ==============================================================================
# Configuration
# ==============================================================================

PROJECT_ROOT = Path(__file__).parent
SAFLOW_PKG = PROJECT_ROOT / "saflow"
CODE_DIR = PROJECT_ROOT / "code"
TESTS_DIR = PROJECT_ROOT / "tests"
DOCS_DIR = PROJECT_ROOT / "docs"


def get_python_executable():
    """Get the Python executable, preferring venv if it exists."""
    venv_python = PROJECT_ROOT / "env" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return "python"


def get_env_with_pythonpath():
    """Get environment dict with PYTHONPATH set to project root."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    return env


# Feature-set shortcuts shared by analysis.stats and analysis.classify so the
# two CLIs accept identical --features values.
_FEATURE_SHORTCUTS = {"psds", "psds_corrected", "fooof", "complexity", "all"}


def resolve_features(features):
    """Resolve a `--features` value into a concrete list of feature names.

    Accepts:
      - A single feature name: "fooof_exponent"
      - A space-separated list: "fooof_exponent psd_alpha"
      - A shortcut: "psds", "psds_corrected", "fooof", "complexity", "all"
      - A mix: "fooof psd_alpha"  (shortcut + extra feature)

    Shortcuts expand via classification.run_classification.expand_feature_set,
    which reads bands from config.yaml so PSD families track config.
    """
    from code.classification.run_classification import expand_feature_set
    from code.utils.config import load_config

    if not features:
        raise ValueError("--features is required (single name, list, or shortcut: "
                         "psds, psds_corrected, fooof, complexity, all)")

    tokens = features.split() if isinstance(features, str) else list(features)
    config = None
    resolved = []
    for tok in tokens:
        if tok in _FEATURE_SHORTCUTS:
            if config is None:
                config = load_config()
            resolved.extend(expand_feature_set(tok, config))
        else:
            resolved.append(tok)

    seen = set()
    return [f for f in resolved if not (f in seen or seen.add(f))]


# Trial-type shortcut shared by analysis.stats and analysis.classify. "all"
# expands to the three variants we report on: every window, baseline-only
# (no errors) and lapse windows (>=1 commission error).
_TRIAL_TYPE_SETS = {"all": ["alltrials", "correct", "lapse"]}


def resolve_trial_types(trial_type):
    """Resolve a `--trial-type` value into a concrete list.

    Accepts a single type ('alltrials', 'correct', 'rare', 'lapse',
    'correct_commission') or the shortcut 'all' → alltrials + correct
    (baseline) + lapse.
    """
    if not trial_type:
        return list(_TRIAL_TYPE_SETS["all"])
    if trial_type in _TRIAL_TYPE_SETS:
        return list(_TRIAL_TYPE_SETS[trial_type])
    return [trial_type]


# ==============================================================================
# dev.check.* Tasks - Validation & Quality Checks
# ==============================================================================

@task
def check_dataset(c, verbose=False):
    """Check dataset completeness - which files exist for each subject.

    Scans all data directories (sourcedata, BIDS, derivatives, processed)
    and reports which files are present/missing for each subject.

    Outputs a summary table showing:
    - MEG raw files (sourcedata)
    - Behavioral logfiles (sourcedata)
    - BIDS converted files
    - Preprocessed files
    - Feature files

    Examples:
        invoke dev.check.dataset
        invoke dev.check.dataset --verbose
    """
    python_exe = get_python_executable()
    cmd = [python_exe, "-m", "code.qc.check_dataset"]

    if verbose:
        cmd.append("--verbose")

    c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath())


@task
def check_qc(c, subject=None, runs=None, output_dir="reports/qc", verbose=False):
    """Run data quality checks on BIDS MEG data.

    By default, checks all subjects. Use --subject to check a single subject.

    Reports on:
    - ISI (Inter-Stimulus Interval) consistency and outliers
    - Response rates and reaction times
    - Channel quality (flat, noisy, saturated channels)
    - Event/trigger counts and timing
    - Data integrity (sampling rate, duration)
    - Head motion (if HPI available)

    Examples:
        invoke dev.check.qc                    # All subjects (default)
        invoke dev.check.qc --subject=04       # Single subject
        invoke dev.check.qc --subject=04 --runs="02 03 04"
        invoke dev.check.qc --verbose
    """
    python_exe = get_python_executable()
    cmd = [python_exe, "-m", "code.qc.check_qc"]

    if subject:
        cmd.extend(["--subject", subject])
    else:
        cmd.append("--all")

    if runs:
        cmd.extend(["--runs"] + runs.split())

    cmd.extend(["--output-dir", output_dir])

    if verbose:
        cmd.append("--verbose")

    c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath())


@task
def check_code(c, fix=False):
    """Run code quality checks (lint, format, typecheck).

    Checks:
    - Ruff linting (style, errors, complexity)
    - Ruff formatting (code style consistency)
    - Mypy type checking (type annotations)

    Examples:
        invoke dev.check.code
        invoke dev.check.code --fix
    """
    print("=" * 80)
    print("Code Quality Checks")
    print("=" * 80)

    all_passed = True

    # Linting
    print("\n[1/3] Linting with ruff...")
    lint_cmd = ["ruff", "check", str(SAFLOW_PKG), str(CODE_DIR)]
    if fix:
        lint_cmd.append("--fix")
    result = c.run(" ".join(lint_cmd), warn=True, pty=True)
    if result.exited != 0:
        all_passed = False
        if not fix:
            print("  → Run with --fix to auto-fix issues")

    # Formatting
    print("\n[2/3] Checking format with ruff...")
    format_cmd = ["ruff", "format", "--check", str(SAFLOW_PKG), str(CODE_DIR)]
    if fix:
        format_cmd.remove("--check")
    result = c.run(" ".join(format_cmd), warn=True, pty=True)
    if result.exited != 0:
        all_passed = False
        if not fix:
            print("  → Run with --fix to auto-format")

    # Type checking
    print("\n[3/3] Type checking with mypy...")
    type_cmd = ["mypy", str(SAFLOW_PKG), str(CODE_DIR)]
    result = c.run(" ".join(type_cmd), warn=True, pty=True)
    if result.exited != 0:
        all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ All code quality checks passed!")
    else:
        print("✗ Some checks failed")
    print("=" * 80)


# ==============================================================================
# dev.* Tasks - Development & Testing
# ==============================================================================

@task
def test(c, verbose=False, coverage=True, markers=None):
    """Run tests with pytest.

    Examples:
        invoke dev.test
        invoke dev.test --verbose
        invoke dev.test --markers="not slow"
    """
    cmd = ["pytest"]
    if verbose:
        cmd.append("-v")
    if coverage:
        cmd.extend(["--cov=saflow", "--cov=code", "--cov-report=term-missing"])
    if markers:
        cmd.append(f"-m '{markers}'")

    print(f"Running: {' '.join(cmd)}")
    c.run(" ".join(cmd), pty=True)


@task
def test_fast(c):
    """Run only fast tests (exclude slow markers)."""
    test(c, markers="not slow")


@task
def clean(c, bytecode=True, cache=True, coverage=True, build=True, logs=False):
    """Clean generated files.

    Examples:
        invoke dev.clean
        invoke dev.clean --logs
    """
    patterns = []
    if bytecode:
        patterns.extend(["**/__pycache__", "**/*.pyc", "**/*.pyo"])
    if cache:
        patterns.extend([".pytest_cache", ".mypy_cache", ".ruff_cache"])
    if coverage:
        patterns.extend(["htmlcov", ".coverage", "coverage.xml"])
    if build:
        patterns.extend(["**/*.egg-info", "build", "dist"])
    if logs:
        patterns.append("logs")

    print("Cleaning generated files...")
    removed_count = 0

    for pattern in patterns:
        if "**" in pattern:
            for path in PROJECT_ROOT.rglob(pattern.replace("**/", "")):
                if path.exists():
                    if path.is_file():
                        path.unlink()
                        removed_count += 1
                    elif path.is_dir():
                        c.run(f"rm -rf {path}", hide=True)
                        removed_count += 1
        else:
            path = PROJECT_ROOT / pattern
            if path.exists():
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    c.run(f"rm -rf {path}", hide=True)
                removed_count += 1

    print(f"✓ Cleaned {removed_count} items")


@task(pre=[check_code, test])
def precommit(c):
    """Run pre-commit checks (code checks + tests)."""
    print("\n" + "=" * 80)
    print("✓ All pre-commit checks passed!")
    print("=" * 80)


# ==============================================================================
# env.* Tasks - Environment Management
# ==============================================================================

@task
def setup(c, mode="basic", python="python3.9", force=False):
    """Run the setup script to create development environment.

    Examples:
        invoke env.setup
        invoke env.setup --mode=dev
        invoke env.setup --mode=all --force
    """
    cmd = ["./setup.sh"]
    if mode != "basic":
        cmd.append(f"--{mode}")
    if python != "python3.9":
        cmd.extend(["--python", python])
    if force:
        cmd.append("--force")

    print(f"Running: {' '.join(cmd)}")
    c.run(" ".join(cmd), pty=True)


@task
def info(c):
    """Display project information."""
    print("=" * 80)
    print("Saflow Project Information")
    print("=" * 80)
    print(f"Project root:     {PROJECT_ROOT}")
    print(f"Saflow package:   {SAFLOW_PKG}")
    print(f"Code directory:   {CODE_DIR}")
    print()

    config_file = PROJECT_ROOT / "config.yaml"
    print("Configuration:")
    print(f"  config.yaml: {'✓ exists' if config_file.exists() else '✗ missing'}")

    venv_dir = PROJECT_ROOT / "env"
    print(f"\nVirtual environment: {'✓ exists' if venv_dir.exists() else '✗ missing'}")

    if venv_dir.exists():
        python_exe = venv_dir / "bin" / "python"
        if python_exe.exists():
            result = c.run(f"{python_exe} --version", hide=True)
            print(f"  Python version: {result.stdout.strip()}")
    print("=" * 80)


@task
def validate_config(c):
    """Validate configuration file."""
    print("Validating configuration...")
    cmd = [
        "python", "-c",
        "'from code.utils.config import load_config; "
        "config = load_config(); "
        "print(\"✓ Configuration is valid\")'"
    ]
    c.run(" ".join(cmd), pty=True)


@task(pre=[clean])
def rebuild(c, mode="dev"):
    """Clean and rebuild environment."""
    setup(c, mode=mode, force=True)


# ==============================================================================
# get.* Tasks - Data Downloads
# ==============================================================================

@task
def get_atlases(c):
    """Download FreeSurfer atlas files (Schaefer, Destrieux).

    Downloads atlas annotation files required for source-level parcellation:
    - Destrieux (aparc.a2009s): 148 ROIs
    - Schaefer 100, 200, 400 parcels (7 Networks)

    Files are downloaded to the fsaverage/label directory specified in config.yaml.

    Examples:
        invoke get.atlases
    """
    print("=" * 80)
    print("Downloading FreeSurfer Atlases")
    print("=" * 80)

    script_path = PROJECT_ROOT / "scripts" / "download_atlases.sh"

    if not script_path.exists():
        print(f"ERROR: Download script not found: {script_path}")
        return

    c.run(str(script_path), pty=True)


# ==============================================================================
# pipeline.* Tasks - Data Processing Pipeline
# ==============================================================================

@task
def validate_inputs(c, data_root=None, verbose=False):
    """Validate that raw input data is present and complete.

    Examples:
        invoke pipeline.validate-inputs
        invoke pipeline.validate-inputs --verbose
    """
    print("=" * 80)
    print("Validating Input Data")
    print("=" * 80)

    python_exe = get_python_executable()
    cmd = [python_exe, "-m", "code.utils.validation", "--check-inputs"]

    if data_root:
        cmd.extend(["--data-root", str(data_root)])
    if verbose:
        cmd.append("--verbose")

    c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath())


@task
def bids(c, input_dir=None, output_dir=None, subjects=None, log_level="INFO", dry_run=False):
    """Run BIDS conversion (Stage 0).

    Examples:
        invoke pipeline.bids
        invoke pipeline.bids --subjects "04 05 06"
        invoke pipeline.bids --dry-run
    """
    print("=" * 80)
    print("BIDS Conversion - Stage 0")
    print("=" * 80)

    python_exe = get_python_executable()
    cmd = [python_exe, "-m", "code.bids.generate_bids"]

    if input_dir:
        cmd.extend(["--input", str(input_dir)])
    if output_dir:
        cmd.extend(["--output", str(output_dir)])
    if subjects:
        cmd.extend(["--subjects"] + subjects.split())
    cmd.extend(["--log-level", log_level])
    if dry_run:
        cmd.append("--dry-run")

    print(f"\nRunning: {' '.join(cmd)}\n")
    c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath())


@task
def preprocess(c, subject=None, runs=None, bids_root=None, log_level="INFO",
               skip_existing=True, crop=None, slurm=False, dry_run=False):
    """Run MEG preprocessing (Stage 1).

    Pipeline: Filter -> Epoch (Freq+Rare only) -> AR1 -> ICA -> AR2 (fit+transform)
    Outputs: continuous Raw + BAD annotations, ICA epochs, AR2-interpolated epochs.

    Examples:
        invoke pipeline.preprocess --subject=04
        invoke pipeline.preprocess --subject=04 --runs="02 03"
        invoke pipeline.preprocess --subject=04 --crop=50  # Quick test with 50s
        invoke pipeline.preprocess --slurm
    """
    print("=" * 80)
    print("MEG Preprocessing - Stage 1")
    if crop:
        print(f"[TEST MODE] Cropping to first {crop} seconds")
    print("=" * 80)

    if slurm:
        _preprocess_slurm(c, subject, runs, bids_root, log_level, skip_existing, dry_run)
    else:
        from code.utils.config import load_config
        config = load_config()
        subjects = [subject] if subject else config["bids"]["subjects"]
        print(f"Processing {len(subjects)} subject(s): {', '.join(subjects)}\n")
        for subj in subjects:
            print(f"\n{'='*40}")
            print(f"Subject: {subj}")
            print(f"{'='*40}")
            _preprocess_local(c, subj, runs, bids_root, log_level, skip_existing, crop)


@task
def preprocess_report(c, subject=None, dataset=False):
    """Generate preprocessing aggregate reports.

    Examples:
        invoke pipeline.preprocess-report --subject=04
        invoke pipeline.preprocess-report --dataset
        invoke pipeline.preprocess-report --subject=04 --dataset
    """
    from code.utils.config import load_config
    config = load_config()

    # Default: all subjects + dataset report
    if not subject and not dataset:
        subjects = config["bids"]["subjects"]
        dataset = True
    else:
        subjects = [subject] if subject else []

    python_exe = get_python_executable()
    for subj in subjects:
        cmd = [python_exe, "-m", "code.preprocessing.aggregate_reports", "-s", subj]
        print(f"\nRunning: {' '.join(cmd)}\n")
        c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath(), warn=True)

    if dataset:
        cmd = [python_exe, "-m", "code.preprocessing.aggregate_reports", "--dataset"]
        print(f"\nRunning: {' '.join(cmd)}\n")
        c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath())


@task
def source_recon(c, subject=None, runs=None, bids_root=None, log_level="INFO",
                 skip_existing=True, slurm=False, dry_run=False):
    """Run source reconstruction (Stage 2).

    By default processes all subjects from config. Use --subject for a single subject.

    Examples:
        invoke pipeline.source-recon                # All subjects (default)
        invoke pipeline.source-recon --subject=04  # Single subject
        invoke pipeline.source-recon --slurm       # All subjects on HPC
    """
    from code.utils.config import load_config

    print("=" * 80)
    print("Source Reconstruction - Stage 2")
    print("=" * 80)

    if slurm:
        _source_recon_slurm(c, subject, runs, bids_root, log_level, skip_existing, dry_run)
    else:
        config = load_config()
        subjects = [subject] if subject else config["bids"]["subjects"]
        print(f"Processing {len(subjects)} subject(s): {', '.join(subjects)}\n")

        for subj in subjects:
            print(f"\n{'='*40}")
            print(f"Subject: {subj}")
            print(f"{'='*40}")
            _source_recon_local(c, subj, runs, bids_root, log_level, skip_existing)


@task
def atlas(c, subject=None, runs=None, atlases=None, skip_existing=True, slurm=False, dry_run=False):
    """Apply atlas parcellation to morphed source estimates (Stage 2b).

    Extracts ROI-level time series from vertex-level source estimates using
    cortical parcellations. By default applies:
    - aparc.a2009s (Destrieux, 148 ROIs)
    - schaefer_100, schaefer_200, schaefer_400 (Schaefer parcellations)

    Output: derivatives/atlas_timeseries_{atlas}/sub-{subject}/*.npz

    By default processes all subjects from config. Use --subject for a single subject.

    Examples:
        invoke pipeline.atlas                      # All subjects (default)
        invoke pipeline.atlas --subject=04         # Single subject
        invoke pipeline.atlas --subject=04 --atlases="aparc.a2009s"
        invoke pipeline.atlas --slurm              # All subjects on cluster
    """
    from code.utils.config import load_config

    print("=" * 80)
    print("Atlas Application - Stage 2b")
    print("=" * 80)

    if slurm:
        _atlas_slurm(c, subject, runs, atlases, skip_existing, dry_run)
        return

    config = load_config()
    subjects = [subject] if subject else config["bids"]["subjects"]
    print(f"Processing {len(subjects)} subject(s): {', '.join(subjects)}\n")

    python_exe = get_python_executable()

    for subj in subjects:
        print(f"\n{'='*40}")
        print(f"Subject: {subj}")
        print(f"{'='*40}")

        cmd = [python_exe, "-m", "code.source_reconstruction.apply_atlas"]
        cmd.extend(["--subject", subj])

        if runs:
            cmd.extend(["--runs", runs])

        if atlases:
            cmd.extend(["--atlases", atlases])

        if skip_existing:
            cmd.append("--skip-existing")
        else:
            cmd.append("--no-skip-existing")

        print(f"Running: {' '.join(cmd)}\n")
        c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath(), warn=True)

    print(f"\n✓ Atlas application complete for {len(subjects)} subject(s)")


# ==============================================================================
# pipeline.features.* Tasks - Feature Extraction
# ==============================================================================

@task
def fooof(c, subject=None, runs=None, space="sensor", skip_existing=True, slurm=False, dry_run=False,
          n_events_window=8):
    """Extract FOOOF aperiodic parameters and corrected PSDs.

    Computes:
    - Aperiodic parameters (exponent, offset, knee)
    - Goodness of fit (r_squared, error)
    - Corrected PSDs (aperiodic component removed)

    FOOOF parameters (freq_range, aperiodic_mode) come from config.yaml.

    Args:
        n_events_window: Window size used by upstream Welch (1 = single-trial,
            8 = cc_saflow's sliding window, default). Determines which Welch
            file to load and which desc suffix to write.

    Examples:
        invoke pipeline.features.fooof --subject=04
        invoke pipeline.features.fooof --subject=04 --space=aparc.a2009s
        invoke pipeline.features.fooof --slurm  # All subjects on cluster
        invoke pipeline.features.fooof --n-events-window=1  # single-trial mode
    """
    print("=" * 80)
    print("Feature Extraction - FOOOF")
    print("=" * 80)

    if slurm:
        _features_slurm(c, "fooof", subject, runs, space, skip_existing=skip_existing,
                        dry_run=dry_run, n_events_window=n_events_window)
        return

    from code.utils.config import load_config
    config = load_config()
    subjects = [subject] if subject else config["bids"]["subjects"]
    print(f"Processing {len(subjects)} subject(s)\n")
    for subj in subjects:
        _features_local(c, "fooof", subj, runs, space, skip_existing=skip_existing,
                        n_events_window=n_events_window)
    print(f"\n✓ FOOOF extraction complete")


@task
def psd(c, subject=None, runs=None, space="sensor", skip_existing=True, slurm=False, dry_run=False,
        n_events_window=8):
    """Extract power spectral density features.

    Computes:
    - Welch PSD estimates per epoch
    - Saves with IN/OUT classification metadata

    Args:
        n_events_window: Number of consecutive stim trials per epoch.
            1 = single-trial, 8 = cc_saflow's sliding window (default).

    Examples:
        invoke pipeline.features.psd --subject=04
        invoke pipeline.features.psd --subject=04 --space=aparc.a2009s
        invoke pipeline.features.psd --slurm  # All subjects on cluster
        invoke pipeline.features.psd --n-events-window=1  # single-trial mode
    """
    print("=" * 80)
    print("Feature Extraction - PSD")
    print("=" * 80)

    if slurm:
        _features_slurm(c, "psd", subject, runs, space, skip_existing=skip_existing,
                        dry_run=dry_run, n_events_window=n_events_window)
        return

    from code.utils.config import load_config
    config = load_config()
    subjects = [subject] if subject else config["bids"]["subjects"]
    print(f"Processing {len(subjects)} subject(s)\n")
    for subj in subjects:
        _features_local(c, "psd", subj, runs, space, skip_existing=skip_existing,
                        n_events_window=n_events_window)
    print(f"\n✓ PSD extraction complete")


@task
def complexity(c, subject=None, runs=None, space="sensor", complexity_type="lzc entropy fractal",
               overwrite=False, slurm=False, dry_run=False, n_events_window=8):
    """Extract complexity and entropy measures.

    Computes:
    - Lempel-Ziv Complexity (LZC)
    - Entropy measures (permutation, spectral, sample, approximate, SVD)
    - Fractal dimensions (Higuchi, Petrosian, Katz, DFA)

    Args:
        n_events_window: Number of consecutive stim trials per epoch.
            1 = single-trial, 8 = cc_saflow's sliding window (default).

    Examples:
        invoke pipeline.features.complexity --subject=04
        invoke pipeline.features.complexity --subject=04 --space=aparc.a2009s
        invoke pipeline.features.complexity --slurm  # All subjects on cluster
    """
    print("=" * 80)
    print("Feature Extraction - Complexity")
    print("=" * 80)

    if slurm:
        _features_slurm(c, "complexity", subject, runs, space, skip_existing=not overwrite,
                        complexity_types=complexity_type, dry_run=dry_run,
                        n_events_window=n_events_window)
        return

    from code.utils.config import load_config
    config = load_config()
    subjects = [subject] if subject else config["bids"]["subjects"]
    print(f"Processing {len(subjects)} subject(s)\n")
    for subj in subjects:
        _features_local(c, "complexity", subj, runs, space, skip_existing=not overwrite,
                        complexity_types=complexity_type, n_events_window=n_events_window)
    print(f"\n✓ Complexity extraction complete")


@task
def extract_all(c, subject=None, runs=None, space="sensor", overwrite=False, slurm=False, dry_run=False,
                n_events_window=8):
    """Extract all feature types (PSD, FOOOF, complexity).

    Order: PSD -> FOOOF -> Complexity (FOOOF depends on PSD)

    Args:
        n_events_window: Number of consecutive stim trials per epoch.
            1 = single-trial, 8 = cc_saflow's sliding window (default).

    Examples:
        invoke pipeline.features.all --subject=04
        invoke pipeline.features.all --subject=04 --space=aparc.a2009s
        invoke pipeline.features.all --slurm  # All subjects on cluster
        invoke pipeline.features.all --n-events-window=1  # single-trial mode
    """
    print("=" * 80)
    print(f"Feature Extraction - All Features (space={space}, window={n_events_window})")
    print("=" * 80)

    if slurm:
        _features_slurm(c, "all", subject, runs, space, skip_existing=not overwrite,
                        dry_run=dry_run, n_events_window=n_events_window)
        return

    from code.utils.config import load_config
    config = load_config()
    subjects = [subject] if subject else config["bids"]["subjects"]
    skip_existing = not overwrite
    print(f"Processing {len(subjects)} subject(s)\n")

    for subj in subjects:
        print(f"\n{'='*40}\nSubject: {subj}\n{'='*40}")
        print("\n[1/3] Extracting PSD features...")
        _features_local(c, "psd", subj, runs, space, skip_existing=skip_existing,
                        n_events_window=n_events_window)
        print("\n[2/3] Extracting FOOOF features...")
        _features_local(c, "fooof", subj, runs, space, skip_existing=skip_existing,
                        n_events_window=n_events_window)
        print("\n[3/3] Extracting Complexity features...")
        _features_local(c, "complexity", subj, runs, space, skip_existing=skip_existing,
                        n_events_window=n_events_window)

    print("\n" + "=" * 80)
    print("✓ All feature extraction complete!")
    print("=" * 80)


# ==============================================================================
# analysis.* Tasks - Statistical Analysis
# ==============================================================================

@task
def stats(c, features="all", space="sensor", test="paired_ttest",
          correction="tmax", alpha=0.05, n_permutations=10000,
          n_jobs=1, average_trials=True, aggregate="median", visualize=False,
          continue_on_error=True,
          trial_type="all", zoning="per-run", n_events_window=8,
          slurm_time=None, slurm_mem=None, slurm_cpus=None,
          slurm=False, dry_run=False):
    """Run group-level statistical analysis (IN vs OUT) on one or more features.

    With no arguments, runs every feature through every trial-type variant
    (`--features=all --trial-type=all`).

    `--features` accepts (default: 'all'):
      - A single feature name: --features=fooof_exponent
      - A space-separated list: --features="fooof_exponent psd_alpha"
      - A shortcut: --features=psds | psds_corrected | fooof | complexity | all

    `--trial-type` accepts (default: 'all'):
      - A single type: alltrials | correct | rare | lapse | correct_commission
      - The shortcut 'all' → alltrials + correct (baseline) + lapse, each run
        separately into its own `_type-<...>` result files.

    Each feature is tested independently (single-feature framework). PSD bands
    are read from config.yaml; "all" expands to FOOOF + PSD + PSD-corrected +
    complexity. Complexity features automatically use the dedicated complexity
    stats script; everything else uses run_group_statistics.

    Defaults match the analysis-workflow path (subject-level paired t-test +
    tmax permutation correction). Pass --no-average-trials for the trial-level
    independent t-test path.

    --aggregate ('median' default | 'mean') controls the per-subject reducer
    for the subject-trial-median path (complexity features). It is ignored in
    subject-spectrum (PSD/FOOOF) and single-trials modes — those force their
    own reducer.

    Examples:
        invoke analysis.stats --features=fooof_exponent
        invoke analysis.stats --features=psds --space=schaefer_400
        invoke analysis.stats --features=all --n-jobs=4
        invoke analysis.stats --features=complexity --correction=fdr
        invoke analysis.stats --features=complexity --aggregate=mean
        invoke analysis.stats --features="psd_alpha psd_theta" --no-average-trials
        invoke analysis.stats --features=psds --slurm
    """
    feature_list = resolve_features(features)
    trial_types = resolve_trial_types(trial_type)
    test_label = "paired t-test (subject-level)" if average_trials else "independent t-test (trial-level)"
    print("=" * 80)
    print(f"analysis.stats | space={space}  correction={correction}  α={alpha}")
    print(f"               | n_perm={n_permutations}  test={test_label}")
    print(f"               | features ({len(feature_list)}): {' '.join(feature_list)}")
    print(f"               | trial-types ({len(trial_types)}): {' '.join(trial_types)}")
    print("=" * 80)

    if slurm:
        for tt in trial_types:
            print(f"\n{'#' * 80}\n# SLURM submit | trial-type: {tt}\n{'#' * 80}")
            _stats_slurm(
                c,
                feature_list=feature_list,
                space=space,
                test=test,
                correction=correction,
                alpha=alpha,
                n_permutations=n_permutations,
                n_jobs=n_jobs,
                average_trials=average_trials,
                aggregate=aggregate,
                visualize=visualize,
                trial_type=tt,
                zoning=zoning,
                n_events_window=n_events_window,
                slurm_time=slurm_time,
                slurm_mem=slurm_mem,
                slurm_cpus=slurm_cpus,
                dry_run=dry_run,
            )
        return

    python_exe = get_python_executable()

    # Split into complexity features (use dedicated script) and the rest.
    complexity_feats = [f for f in feature_list
                        if f == "complexity" or f.startswith("complexity_")]
    other_feats = [f for f in feature_list if f not in complexity_feats]

    failures = []

    other_mode = "subject-spectrum" if average_trials else "single-trials"
    complexity_mode = "subject-trial-median" if average_trials else "single-trials"

    for tt in trial_types:
        print(f"\n{'#' * 80}\n# trial-type: {tt}\n{'#' * 80}")

        if other_feats:
            cmd = [python_exe, "-m", "code.statistics.run_group_statistics",
                   "--feature-type", *other_feats,
                   "--space", space, "--test", test,
                   "--correction", correction, "--alpha", str(alpha),
                   "--n-permutations", str(n_permutations), "--n-jobs", str(n_jobs),
                   "--analysis-mode", other_mode,
                   "--aggregate", aggregate,
                   "--trial-type", tt,
                   "--zoning", zoning,
                   "--n-events-window", str(n_events_window)]
            if visualize:
                cmd.append("--visualize")
            print(f"\n>>> run_group_statistics on {len(other_feats)} feature(s) (mode={other_mode})")
            print(f"Running: {' '.join(cmd)}\n")
            result = c.run(" ".join(cmd), pty=True,
                           env=get_env_with_pythonpath(), warn=continue_on_error)
            if result is not None and getattr(result, "exited", 0) != 0:
                failures.append((f"run_group_statistics[{tt}]", f"exit {result.exited}"))

        if complexity_feats:
            cmd = [python_exe, "-m", "code.statistics.run_group_statistics",
                   "--feature-type", *complexity_feats,
                   "--space", space, "--test", test,
                   "--correction", correction, "--alpha", str(alpha),
                   "--n-permutations", str(n_permutations), "--n-jobs", str(n_jobs),
                   "--analysis-mode", complexity_mode,
                   "--aggregate", aggregate,
                   "--trial-type", tt,
                   "--zoning", zoning,
                   "--n-events-window", str(n_events_window)]
            print(f"\n>>> complexity stats on {len(complexity_feats)} metric(s) (mode={complexity_mode})")
            print(f"Running: {' '.join(cmd)}\n")
            result = c.run(" ".join(cmd), pty=True,
                           env=get_env_with_pythonpath(), warn=continue_on_error)
            if result is not None and getattr(result, "exited", 0) != 0:
                failures.append((f"complexity_stats[{tt}]", f"exit {result.exited}"))

    print("\n" + "=" * 80)
    if failures:
        print(f"Done with {len(failures)} failure(s):")
        for fam, msg in failures:
            print(f"  - {fam}: {msg}")
    else:
        print("All features completed.")
    print("=" * 80)


@task
def classify(c, features="all", clf="logistic", cv="auto",
             space="sensor", mode="univariate", n_permutations=1000,
             no_balance=False, n_jobs=-1, continue_on_error=False,
             combine_features=False, importances=False, label=None,
             n_chunks=1, seed=42, aggregate=True, delete_chunks=False,
             trial_type="all", zoning="per-run", n_events_window=8,
             average_trials=True,
             slurm_time=None, slurm_mem=None, slurm_cpus=None,
             slurm=False, dry_run=False):
    """Run classification analysis (IN vs OUT) on one or more features.

    With no arguments, runs every feature through every trial-type variant
    (`--features=all --trial-type=all`).

    `--features` accepts (default: 'all'):
      - A single feature name: --features=fooof_exponent
      - A space-separated list: --features="fooof_exponent psd_alpha"
      - A shortcut: --features=psds | psds_corrected | fooof | complexity | all

    `--trial-type` accepts (default: 'all'):
      - A single type: alltrials | correct | rare | lapse | correct_commission
      - The shortcut 'all' → alltrials + correct (baseline) + lapse, each run
        separately into its own `_type-<...>` score files.

    Spatial mode:
      - univariate (default): per-channel/ROI classifier + shared-permutation t-max
      - multivariate: pool spatial dim, single permutation_test_score

    Trial averaging & CV:
      - Default: each subject's trials are averaged to one IN + one OUT vector,
        and --cv=auto resolves to GroupKFold(k=6) over subjects.
      - --no-average-trials: classify single trials; --cv=auto resolves to
        leave-one-subject-out instead.
      - --cv (logo|stratified|group) forces a specific splitter regardless.

    Feature handling:
      - Default loop: each feature is classified independently (single-feature framework).
      - --combine-features: stack all selected features into a single classification
        (great for RF feature importance with --importances).

    Examples:
        # Single feature, per-channel LDA + tmax (recommended for single features)
        invoke analysis.classify --features=fooof_exponent --clf=lda

        # All 8 PSD bands, each as its own classification
        invoke analysis.classify --features=psds --space=sensor

        # Every feature on disk, each as its own classification
        invoke analysis.classify --features=all --space=schaefer_400

        # Combine all complexity metrics, RF per ROI, save importances
        invoke analysis.classify --features=complexity --space=schaefer_400 \\
            --clf=rf --combine-features --importances

        # Combine + multivariate: one big RF over (n_features × n_spatial)
        invoke analysis.classify --features=all --combine-features \\
            --mode=multivariate --clf=rf --importances

        # Submit each feature as a separate SLURM job
        invoke analysis.classify --features=psds --slurm

        # Single-trial classification with leave-one-subject-out CV
        invoke analysis.classify --features=fooof_exponent --no-average-trials
    """
    print("=" * 80)
    print("Classification Analysis")
    print("=" * 80)

    feature_list = resolve_features(features)
    trial_types = resolve_trial_types(trial_type)
    print(f"Features ({len(feature_list)}): {' '.join(feature_list)}")
    print(f"Trial-types ({len(trial_types)}): {' '.join(trial_types)}")

    if slurm:
        for tt in trial_types:
            print(f"\n{'#' * 80}\n# SLURM submit | trial-type: {tt}\n{'#' * 80}")
            _classify_slurm(
                c,
                feature_list=feature_list,
                clf=clf,
                cv=cv,
                space=space,
                mode=mode,
                n_permutations=n_permutations,
                no_balance=no_balance,
                n_jobs=n_jobs,
                continue_on_error=continue_on_error,
                combine_features=combine_features,
                importances=importances,
                label=label,
                n_chunks=n_chunks,
                seed=seed,
                aggregate=aggregate,
                delete_chunks=delete_chunks,
                trial_type=tt,
                zoning=zoning,
                n_events_window=n_events_window,
                average_trials=average_trials,
                slurm_time=slurm_time,
                slurm_mem=slurm_mem,
                slurm_cpus=slurm_cpus,
                dry_run=dry_run,
            )
        return

    if n_chunks > 1:
        print(
            f"NOTE: --n-chunks={n_chunks} requires running each chunk separately. "
            f"Use --slurm to fan out automatically, or pass --chunk-idx N to the "
            f"underlying script directly."
        )
        return

    python_exe = get_python_executable()
    for tt in trial_types:
        print(f"\n{'#' * 80}\n# trial-type: {tt}\n{'#' * 80}")
        cmd = [python_exe, "-m", "code.classification.run_classification",
               "--feature", *feature_list]
        cmd.extend(["--clf", clf])
        cmd.extend(["--cv", cv])
        cmd.extend(["--space", space])
        cmd.extend(["--mode", mode])
        cmd.extend(["--n-permutations", str(n_permutations)])
        cmd.extend(["--n-jobs", str(n_jobs)])
        cmd.extend(["--trial-type", tt])
        cmd.extend(["--zoning", zoning])
        cmd.extend(["--n-events-window", str(n_events_window)])
        cmd.extend(["--seed", str(seed)])
        if not average_trials:
            cmd.append("--no-average-trials")
        if no_balance:
            cmd.append("--no-balance")
        if continue_on_error:
            cmd.append("--continue-on-error")
        if combine_features:
            cmd.append("--combine-features")
        if importances:
            cmd.append("--importances")
        if label:
            cmd.extend(["--label", label])

        print(f"\nRunning: {' '.join(cmd)}\n")
        c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath())


# ==============================================================================
# viz.* Tasks - Visualization
# ==============================================================================

@task
def viz_stats(c, feature_type="fooof_exponent", space="sensor", alpha=0.05,
              cmap=None, show=False, save=True):
    """Visualize saved statistical results as topographic or surface maps.

    For sensor space: creates a panel of topomaps (t-values).
    For source/atlas spaces: creates inflated brain surface maps (t-values).

    Colormap is resolved from config.yaml visualization section based on
    feature type. Use --cmap to override.

    Convention (for contrast colormaps):
        Red  = OUT > IN (positive t-values)
        Blue = IN > OUT (negative t-values)

    Args:
        feature_type: Feature to visualize (e.g., fooof_exponent, psd_alpha)
        space: Analysis space ('sensor', 'source', or atlas name like 'schaefer_400')
        alpha: Significance threshold for marking
        cmap: Override colormap (default: from config.yaml)
        show: Display the figure interactively
        save: Save the figure to reports/figures/

    Examples:
        invoke viz.stats --feature-type=fooof_exponent --show
        invoke viz.stats --feature-type=psd_alpha --space=schaefer_400
        invoke viz.stats --feature-type=psd --space=aparc.a2009s --save
        invoke viz.stats --feature-type=psd --cmap=coolwarm
    """
    python_exe = get_python_executable()
    cmd = [python_exe, "-m", "code.visualization.run_viz_stats"]
    cmd.extend(["--feature-type", feature_type])
    cmd.extend(["--space", space])
    cmd.extend(["--alpha", str(alpha)])
    if cmap:
        cmd.extend(["--cmap", cmap])
    if show:
        cmd.append("--show")
    if not save:
        cmd.append("--no-save")

    print(f"Running: {' '.join(cmd)}\n")
    c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath())


@task
def viz_maps(c, metric, space="sensor", feature=None, feature_set=None,
             family=None, clf="lda", cv="logo", mode="univariate",
             test="paired_ttest", correction="auto", alpha=0.05,
             cmap=None, output_subdir=None, config="config.yaml"):
    """Unified rows-of-maps viz (stats + classification, sensor + atlas).

    One row of topomaps (sensor) or inflated-brain panels (source/atlas) per
    feature family. Auto-discovers result files for the chosen metric and
    prints exactly what to run when nothing is found.

    Args:
        metric: 'tval' | 'contrast' | 'roc_auc' (see code.visualization.metrics).
        space: 'sensor', 'source', or atlas name (e.g., 'schaefer_400').
        feature: space-separated feature names (e.g., 'psd_alpha psd_theta').
        feature_set: shortcut family ('psds', 'psds_corrected', 'fooof',
            'complexity', 'all').
        family: filter rendering to one family when features span multiple.
        clf, cv, mode: classification result filters.
        test: statistics test filter.
        correction: which p-value correction for the significance mask
            ('auto', 'tmax', 'fdr_bh', 'bonferroni', 'uncorrected').
        alpha: significance threshold for the mask.
        cmap: override colormap (default: from metric).
        output_subdir: subfolder under reports/figures/ (default: 'classification').

    Examples:
        invoke viz.maps --metric=roc_auc --space=sensor --feature-set=psds
        invoke viz.maps --metric=roc_auc --space=schaefer_400 --feature-set=all
        invoke viz.maps --metric=tval --space=sensor --feature=fooof_exponent
        invoke viz.maps --metric=contrast --space=sensor --feature-set=psds
    """
    python_exe = get_python_executable()
    cmd = [python_exe, "-m", "code.visualization.run_viz",
           "--metric", metric, "--space", space,
           "--clf", clf, "--cv", cv, "--mode", mode,
           "--test", test, "--correction", correction,
           "--alpha", str(alpha), "--config", config]
    if feature:
        cmd.extend(["--feature"] + feature.split())
    if feature_set:
        cmd.extend(["--feature-set", feature_set])
    if family:
        cmd.extend(["--family", family])
    if cmap:
        cmd.extend(["--cmap", cmap])
    if output_subdir:
        cmd.extend(["--output-subdir", output_subdir])

    print(f"Running: {' '.join(cmd)}\n")
    c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath())


@task
def spectra(c, space="sensor", stat_feature="fooof_exponent", select_by="corrected",
            subject=None, n_events_window=8, show=False, save=True,
            config="config.yaml"):
    """Reproduce Figure 3 C-F: the FOOOF spectral decomposition panel.

    Selects the most significant sensor/region from the FOOOF exponent
    group-statistics map, then renders, at that unit, the four-panel FOOOF
    figure for IN vs OUT:

        C  Raw spectrum (PSD)
        D  Aperiodic component
        E  Corrected spectrum (PSDc)
        F  Periodic components

    The selected unit name is written into panel C (plus an exponent t-map
    inset for sensor space). Curves are group means +/- SEM unless --subject
    pins a single subject (the manuscript's example-subject case).

    Args:
        space: 'sensor' or atlas name (e.g. 'schaefer_400').
        stat_feature: statistics map used to pick the unit (default fooof_exponent).
        select_by: rank by 'corrected' or 'uncorrected' p-values.
        subject: restrict spectra to one subject (default: group average).
        n_events_window: trials per welch window (welch desc suffix).
        show: display the figure interactively.
        save: write the figure to reports/figures/statistics/.

    Examples:
        invoke viz.spectra
        invoke viz.spectra --space=sensor --select-by=uncorrected
        invoke viz.spectra --subject=07
    """
    python_exe = get_python_executable()
    cmd = [python_exe, "-m", "code.visualization.run_viz_spectra",
           "--space", space, "--stat-feature", stat_feature,
           "--select-by", select_by, "--n-events-window", str(n_events_window),
           "--config", config]
    if subject:
        cmd.extend(["--subject", str(subject)])
    if show:
        cmd.append("--show")
    if not save:
        cmd.append("--no-save")

    print(f"Running: {' '.join(cmd)}\n")
    c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath())


@task
def behavior(c, subject="07", run="4", inout_bounds="25 75", output=None, verbose=False):
    """Generate behavioral analysis figure.

    Examples:
        invoke viz.behavior
        invoke viz.behavior --subject=04 --run=3
    """
    print("=" * 80)
    print("Behavioral Analysis Visualization")
    print("=" * 80)

    python_exe = get_python_executable()
    cmd = [python_exe, "-m", "code.visualization.plot_behavior"]
    cmd.extend(["--subject", subject])
    cmd.extend(["--run", run])

    bounds = inout_bounds.split()
    if len(bounds) == 2:
        cmd.extend(["--inout-bounds", bounds[0], bounds[1]])

    if output:
        cmd.extend(["--output", output])
    if verbose:
        cmd.append("--verbose")

    print(f"\nRunning: {' '.join(cmd)}\n")
    c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath())


# ==============================================================================
# Helper Functions (Private)
# ==============================================================================

def _preprocess_local(c, subject, runs=None, bids_root=None, log_level="INFO", skip_existing=True, crop=None):
    """Run preprocessing locally."""
    python_exe = get_python_executable()
    cmd = [python_exe, "-m", "code.preprocessing.run_preprocessing"]
    cmd.extend(["--subject", subject])

    if runs:
        cmd.extend(["--runs"] + runs.split())
    if bids_root:
        cmd.extend(["--bids-root", str(bids_root)])
    cmd.extend(["--log-level", log_level])
    if skip_existing:
        cmd.append("--skip-existing")
    if crop:
        cmd.extend(["--crop", str(crop)])

    print(f"\nRunning: {' '.join(cmd)}\n")
    c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath())


def _preprocess_slurm(c, subject=None, runs=None, bids_root=None,
                      log_level="INFO", skip_existing=True, dry_run=False):
    """Submit preprocessing jobs to SLURM."""
    from datetime import datetime
    from code.utils.config import load_config
    from code.utils.slurm import render_slurm_script, submit_slurm_job, save_job_manifest

    print("\n[SLURM Mode] Submitting preprocessing jobs to cluster\n")

    config = load_config()
    subjects = [subject] if subject else config["bids"]["subjects"]
    run_list = runs.split() if runs else config["bids"]["task_runs"]

    if not bids_root:
        data_root = Path(config["paths"]["data_root"])
        bids_root = data_root / "bids"
    else:
        bids_root = Path(bids_root)

    slurm_config = config["computing"]["slurm"]
    preproc_resources = slurm_config["preprocessing"]

    log_dir = PROJECT_ROOT / config["paths"]["logs"] / "slurm" / "preprocessing"
    script_dir = PROJECT_ROOT / "slurm" / "scripts" / "preprocessing"
    log_dir.mkdir(parents=True, exist_ok=True)
    script_dir.mkdir(parents=True, exist_ok=True)

    venv_path = PROJECT_ROOT / config["paths"]["venv"]
    job_ids = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Subjects: {len(subjects)}, Runs: {len(run_list)}")
    print(f"Total jobs: {len(subjects) * len(run_list)}")

    # Track per-subject job IDs for dependent report jobs
    subject_job_ids = {subj: [] for subj in subjects}

    for subj in subjects:
        for run in run_list:
            job_name = f"preproc_sub-{subj}_run-{run}"
            context = {
                "job_name": job_name,
                "account": slurm_config["account"],
                "partition": slurm_config.get("partition", ""),
                "time": preproc_resources["time"],
                "cpus": preproc_resources["cpus"],
                "mem": preproc_resources["mem"],
                "venv_path": str(venv_path),
                "project_root": str(PROJECT_ROOT),
                "log_dir": str(log_dir),
                "subject": subj,
                "run": run,
                "bids_root": str(bids_root),
                "log_level": log_level,
                "skip_existing": skip_existing,
            }

            script_path = script_dir / f"{job_name}_{timestamp}.sh"
            render_slurm_script("preprocessing.sh.j2", context, output_path=script_path)
            job_id = submit_slurm_job(script_path, dry_run=dry_run)
            if job_id:
                job_ids.append(job_id)
                subject_job_ids[subj].append(job_id)

    # Submit dependent report jobs (per-subject, each also regenerates dataset report)
    # Each per-subject report job generates its subject report AND the dataset report.
    # This eliminates the fragile dependency chain where a separate dataset job must
    # wait for all subject report jobs. The last subject to complete produces the
    # final correct dataset report (progressive build).
    report_job_ids = []
    report_resources = slurm_config.get("report", {"time": "00:15:00", "cpus": 2, "mem": "4G"})

    for subj in subjects:
        deps = subject_job_ids[subj]  # empty if preprocessing was skipped
        job_name = f"report_sub-{subj}"
        context = {
            "job_name": job_name,
            "account": slurm_config["account"],
            "partition": slurm_config.get("partition", ""),
            "time": report_resources["time"],
            "cpus": report_resources["cpus"],
            "mem": report_resources["mem"],
            "venv_path": str(venv_path),
            "project_root": str(PROJECT_ROOT),
            "log_dir": str(log_dir),
            "subject": subj,
            "timestamp": timestamp,
        }
        script_path = script_dir / f"{job_name}_{timestamp}.sh"
        render_slurm_script("preprocess_report.sh.j2", context, output_path=script_path)
        rjob_id = submit_slurm_job(
            script_path,
            dependencies=deps if deps else None,
            dep_type="afterany",  # run even if some preprocessing runs failed
            dry_run=dry_run,
        )
        if rjob_id:
            report_job_ids.append(rjob_id)

    all_job_ids = job_ids + report_job_ids
    if all_job_ids:
        manifest_path = log_dir / f"preprocessing_manifest_{timestamp}.json"
        save_job_manifest(all_job_ids, manifest_path, metadata={
            "stage": "preprocessing",
            "timestamp": timestamp,
            "subjects": subjects,
            "runs": run_list,
            "preprocessing_job_ids": job_ids,
            "report_job_ids": report_job_ids,
        })
        print(f"\n✓ Submitted {len(job_ids)} preprocessing jobs")
        print(f"✓ Submitted {len(report_job_ids)} report jobs (with dependencies)")


def _source_recon_local(c, subject, runs=None, bids_root=None, log_level="INFO", skip_existing=True):
    """Run source reconstruction locally."""
    python_exe = get_python_executable()
    cmd = [python_exe, "-m", "code.source_reconstruction.run_inverse_solution"]
    cmd.extend(["--subject", subject])

    if runs:
        cmd.extend(["--runs", runs])
    if bids_root:
        cmd.extend(["--bids-root", str(bids_root)])
    cmd.extend(["--log-level", log_level])
    cmd.append("--skip-existing" if skip_existing else "--no-skip-existing")

    print(f"\nRunning: {' '.join(cmd)}\n")
    c.run(" ".join(cmd), env=get_env_with_pythonpath(), pty=True)


def _source_recon_slurm(c, subject=None, runs=None, bids_root=None,
                        log_level="INFO", skip_existing=True, dry_run=False):
    """Submit source reconstruction jobs to SLURM."""
    from datetime import datetime
    from code.utils.config import load_config
    from code.utils.slurm import render_slurm_script, save_job_manifest, submit_slurm_job

    config = load_config()
    subjects = [subject] if subject else config["bids"]["subjects"]
    run_list = runs.split() if runs else config["bids"]["task_runs"]

    slurm_config = config["computing"]["slurm"]
    if not slurm_config.get("enabled", False):
        print("ERROR: SLURM is not enabled in config.yaml")
        return

    src_recon_resources = slurm_config.get("source_reconstruction", {})
    if not src_recon_resources:
        print("ERROR: No source_reconstruction resources in config.yaml")
        return

    data_root = Path(config["paths"]["data_root"])
    if bids_root:
        bids_root = Path(bids_root)
    else:
        bids_root = data_root / "bids"

    venv_path = Path(config["paths"]["venv"])
    if not venv_path.is_absolute():
        venv_path = PROJECT_ROOT / venv_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = PROJECT_ROOT / "slurm" / "scripts" / "source_reconstruction"
    script_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PROJECT_ROOT / config["paths"]["logs"] / "slurm" / "source_reconstruction"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSubjects: {len(subjects)}, Runs: {len(run_list)}")
    print(f"Total jobs: {len(subjects) * len(run_list)}")

    job_ids = []
    for subj in subjects:
        for run in run_list:
            job_name = f"srcrecon_sub-{subj}_run-{run}"
            context = {
                "job_name": job_name,
                "account": slurm_config["account"],
                "partition": slurm_config.get("partition", ""),
                "cpus": src_recon_resources["cpus"],
                "mem": src_recon_resources["mem"],
                "time": src_recon_resources["time"],
                "log_dir": str(log_dir),
                "venv_path": str(venv_path),
                "project_root": str(PROJECT_ROOT),
                "subject": subj,
                "run": run,
                "bids_root": str(bids_root),
                "log_level": log_level,
                "skip_existing": skip_existing,
            }

            script_path = script_dir / f"{job_name}_{timestamp}.sh"
            render_slurm_script("source_reconstruction.sh.j2", context, output_path=script_path)

            if not dry_run:
                try:
                    job_id = submit_slurm_job(script_path, job_name=job_name, dry_run=False)
                    job_ids.append(job_id)
                except Exception as e:
                    print(f"  ✗ Failed to submit: {e}")

    if job_ids:
        manifest_path = log_dir / f"source_reconstruction_manifest_{timestamp}.json"
        save_job_manifest(job_ids, manifest_path, metadata={
            "stage": "source_reconstruction",
            "timestamp": timestamp,
            "subjects": subjects,
            "runs": run_list,
        })
        print(f"\n✓ Submitted {len(job_ids)} source reconstruction jobs")


def _atlas_slurm(c, subject=None, runs=None, atlases=None,
                 skip_existing=True, dry_run=False):
    """Submit atlas application jobs to SLURM."""
    from datetime import datetime
    from code.utils.config import load_config
    from code.utils.slurm import render_slurm_script, save_job_manifest, submit_slurm_job

    print("\n[SLURM Mode] Submitting atlas application jobs to cluster\n")

    config = load_config()
    subjects = [subject] if subject else config["bids"]["subjects"]
    run_list = runs.split() if runs else config["bids"]["task_runs"]

    slurm_config = config["computing"]["slurm"]
    if not slurm_config.get("enabled", False):
        print("ERROR: SLURM is not enabled in config.yaml")
        return

    # Get atlas-specific resources from config
    atlas_resources = slurm_config.get("atlas", {})
    if not atlas_resources:
        print("ERROR: No atlas resources in config.yaml (computing.slurm.atlas)")
        return

    venv_path = Path(config["paths"]["venv"])
    if not venv_path.is_absolute():
        venv_path = PROJECT_ROOT / venv_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = PROJECT_ROOT / "slurm" / "scripts" / "atlas"
    script_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PROJECT_ROOT / config["paths"]["logs"] / "slurm" / "atlas"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Subjects: {len(subjects)}, Runs: {len(run_list)}")
    print(f"Total jobs: {len(subjects)}")  # One job per subject (all runs)

    job_ids = []
    for subj in subjects:
        job_name = f"atlas_sub-{subj}"
        runs_str = " ".join(run_list)
        context = {
            "job_name": job_name,
            "account": slurm_config["account"],
            "partition": slurm_config.get("partition", ""),
            "cpus": atlas_resources["cpus"],
            "mem": atlas_resources["mem"],
            "time": atlas_resources["time"],
            "log_dir": str(log_dir),
            "venv_path": str(venv_path),
            "project_root": str(PROJECT_ROOT),
            "subject": subj,
            "runs": runs_str,
            "atlases": atlases,
            "skip_existing": skip_existing,
            "timestamp": timestamp,
        }

        script_path = script_dir / f"{job_name}_{timestamp}.sh"
        render_slurm_script("atlas.sh.j2", context, output_path=script_path)

        if not dry_run:
            try:
                job_id = submit_slurm_job(script_path, job_name=job_name, dry_run=False)
                if job_id:
                    job_ids.append(job_id)
            except Exception as e:
                print(f"  ✗ Failed to submit: {e}")
        else:
            print(f"[DRY RUN] Would submit: {script_path.name}")

    if job_ids:
        manifest_path = log_dir / f"atlas_manifest_{timestamp}.json"
        save_job_manifest(job_ids, manifest_path, metadata={
            "stage": "atlas",
            "timestamp": timestamp,
            "subjects": subjects,
            "runs": run_list,
        })
        print(f"\n✓ Submitted {len(job_ids)} atlas application jobs")
    elif dry_run:
        print(f"\n[DRY RUN] Would have submitted {len(subjects)} jobs")


def _features_local(c, feature_type, subject, runs=None, space="sensor",
                    skip_existing=True, complexity_types=None, log_level="INFO",
                    n_events_window=8):
    """Run feature extraction locally."""
    from code.utils.config import load_config

    config = load_config()
    run_list = runs.split() if runs else config["bids"]["task_runs"]
    python_exe = get_python_executable()

    for run in run_list:
        print(f"\n[Processing run {run}]")

        if feature_type == "psd":
            cmd = [python_exe, "-m", "code.features.compute_welch_psd"]
            cmd.extend(["--subject", subject, "--run", run, "--space", space])
            cmd.extend(["--n-events-window", str(n_events_window)])
            cmd.extend(["--log-level", log_level])
            cmd.append("--skip-existing" if skip_existing else "--no-skip-existing")

        elif feature_type == "fooof":
            cmd = [python_exe, "-m", "code.features.compute_fooof"]
            cmd.extend(["--subject", subject, "--run", run, "--space", space])
            cmd.extend(["--n-events-window", str(n_events_window)])
            cmd.extend(["--log-level", log_level])
            cmd.append("--skip-existing" if skip_existing else "--no-skip-existing")

        elif feature_type == "complexity":
            cmd = [python_exe, "-m", "code.features.compute_complexity"]
            cmd.extend(["--subject", subject, "--run", run, "--space", space])
            cmd.extend(["--n-events-window", str(n_events_window)])
            if complexity_types:
                cmd.extend(["--complexity-type"] + complexity_types.split())
            if not skip_existing:
                cmd.append("--overwrite")

        else:
            print(f"ERROR: Unknown feature type '{feature_type}'")
            return

        print(f"Running: {' '.join(cmd)}\n")
        result = c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath(), warn=True)
        if result.exited != 0:
            print(f"WARNING: Run {run} failed")


def _features_slurm(c, feature_type, subject=None, runs=None, space="sensor",
                    skip_existing=True, complexity_types=None, log_level="INFO", dry_run=False,
                    n_events_window=8):
    """Submit feature extraction jobs to SLURM."""
    from datetime import datetime
    from code.utils.config import load_config
    from code.utils.slurm import render_slurm_script, save_job_manifest, submit_slurm_job

    print(f"\n[SLURM Mode] Submitting {feature_type} feature extraction jobs to cluster\n")

    config = load_config()
    subjects = [subject] if subject else config["bids"]["subjects"]
    run_list = runs.split() if runs else config["bids"]["task_runs"]

    slurm_config = config["computing"]["slurm"]
    if not slurm_config.get("enabled", False):
        print("ERROR: SLURM is not enabled in config.yaml")
        return

    features_resources = slurm_config.get("features", {})
    if not features_resources:
        print("ERROR: No features resources in config.yaml")
        return

    venv_path = Path(config["paths"]["venv"])
    if not venv_path.is_absolute():
        venv_path = PROJECT_ROOT / venv_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = PROJECT_ROOT / "slurm" / "scripts" / "features"
    script_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PROJECT_ROOT / config["paths"]["logs"] / "slurm" / "features"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Feature type: {feature_type}")
    print(f"Space: {space}")
    print(f"Subjects: {len(subjects)}, Runs: {len(run_list)}")
    print(f"Total jobs: {len(subjects) * len(run_list)}")

    job_ids = []
    for subj in subjects:
        for run in run_list:
            job_name = f"{feature_type}_sub-{subj}_run-{run}_{space}"
            context = {
                "job_name": job_name,
                "account": slurm_config["account"],
                "partition": slurm_config.get("partition", ""),
                "cpus": features_resources["cpus"],
                "mem": features_resources["mem"],
                "time": features_resources["time"],
                "log_dir": str(log_dir),
                "venv_path": str(venv_path),
                "project_root": str(PROJECT_ROOT),
                "feature_type": feature_type,
                "subject": subj,
                "run": run,
                "space": space,
                "log_level": log_level,
                "skip_existing": skip_existing,
                "complexity_types": complexity_types,
                "n_events_window": n_events_window,
                "timestamp": timestamp,
            }

            script_path = script_dir / f"{job_name}_{timestamp}.sh"
            render_slurm_script("features.sh.j2", context, output_path=script_path)

            if not dry_run:
                try:
                    job_id = submit_slurm_job(script_path, job_name=job_name, dry_run=False)
                    if job_id:
                        job_ids.append(job_id)
                except Exception as e:
                    print(f"  ✗ Failed to submit: {e}")
            else:
                print(f"[DRY RUN] Would submit: {script_path.name}")

    if job_ids:
        manifest_path = log_dir / f"{feature_type}_manifest_{timestamp}.json"
        save_job_manifest(job_ids, manifest_path, metadata={
            "stage": f"features_{feature_type}",
            "feature_type": feature_type,
            "space": space,
            "timestamp": timestamp,
            "subjects": subjects,
            "runs": run_list,
        })
        print(f"\n✓ Submitted {len(job_ids)} {feature_type} feature extraction jobs")
    elif dry_run:
        print(f"\n[DRY RUN] Would have submitted {len(subjects) * len(run_list)} jobs")


def _stats_slurm(c, feature_list, space="sensor", test="paired_ttest",
                 correction="tmax", alpha=0.05, n_permutations=10000,
                 n_jobs=1, average_trials=True, aggregate="median",
                 visualize=False,
                 trial_type="alltrials", zoning="per-run",
                 n_events_window=8,
                 slurm_time=None, slurm_mem=None, slurm_cpus=None,
                 dry_run=False):
    """Submit one statistics job per feature to SLURM."""
    from datetime import datetime
    from code.utils.config import load_config
    from code.utils.slurm import render_slurm_script, save_job_manifest, submit_slurm_job

    print(f"\n[SLURM Mode] Submitting statistics jobs to cluster\n")

    config = load_config()
    slurm_config = config["computing"]["slurm"]
    if not slurm_config.get("enabled", False):
        print("ERROR: SLURM is not enabled in config.yaml")
        return

    stats_resources = slurm_config.get("statistics", {})
    if not stats_resources:
        print("ERROR: No statistics resources in config.yaml (computing.slurm.statistics)")
        return

    features = list(feature_list)
    if not features:
        print("ERROR: no features to submit")
        return

    venv_path = Path(config["paths"]["venv"])
    if not venv_path.is_absolute():
        venv_path = PROJECT_ROOT / venv_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = PROJECT_ROOT / "slurm" / "scripts" / "statistics"
    script_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PROJECT_ROOT / config["paths"]["logs"] / "slurm" / "statistics"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Space: {space}  Test: {test}  Correction: {correction}")
    print(f"n_permutations: {n_permutations}  alpha: {alpha}  average_trials: {average_trials}")
    print(f"Jobs: {len(features)} (one per feature)")

    cpus_resolved = slurm_cpus if slurm_cpus is not None else stats_resources["cpus"]
    mem_resolved = slurm_mem if slurm_mem is not None else stats_resources["mem"]
    time_resolved = slurm_time if slurm_time is not None else stats_resources["time"]
    print(f"Resources: cpus={cpus_resolved}  mem={mem_resolved}  time={time_resolved}")

    base_resources = dict(
        account=slurm_config["account"],
        partition=slurm_config.get("partition", ""),
        cpus=cpus_resolved,
        mem=mem_resolved,
        time=time_resolved,
        log_dir=str(log_dir),
        venv_path=str(venv_path),
        project_root=str(PROJECT_ROOT),
    )

    def _mode_for(feat: str) -> str:
        is_complexity = feat == "complexity" or feat.startswith("complexity_")
        if not average_trials:
            return "single-trials"
        return "subject-trial-median" if is_complexity else "subject-spectrum"

    all_job_ids = []
    for feat in features:
        job_name = f"stats_{feat}_{space}_{test}"
        context = {
            **base_resources,
            "job_name": job_name,
            "feature_label": feat,
            "space": space,
            "test": test,
            "correction": correction,
            "alpha": alpha,
            "n_permutations": n_permutations,
            "n_jobs": n_jobs,
            "analysis_mode": _mode_for(feat),
            "aggregate": aggregate,
            "visualize": visualize,
            "trial_type": trial_type,
            "zoning": zoning,
            "n_events_window": n_events_window,
            "timestamp": timestamp,
        }
        script_path = script_dir / f"{job_name}_{timestamp}.sh"
        render_slurm_script("statistics.sh.j2", context, output_path=script_path)

        if dry_run:
            print(f"[DRY RUN] Would submit: {script_path.name}")
            continue

        try:
            jid = submit_slurm_job(script_path, job_name=job_name, dry_run=False)
            if jid:
                all_job_ids.append(jid)
        except Exception as e:
            print(f"  ✗ Failed to submit {job_name}: {e}")

    if all_job_ids:
        manifest_path = log_dir / f"statistics_manifest_{timestamp}.json"
        save_job_manifest(all_job_ids, manifest_path, metadata={
            "stage": "statistics",
            "space": space,
            "test": test,
            "correction": correction,
            "alpha": alpha,
            "n_permutations": n_permutations,
            "features": features,
            "timestamp": timestamp,
        })
        print(f"\n✓ Submitted {len(all_job_ids)} job(s); manifest: {manifest_path}")
    elif dry_run:
        print(f"\n[DRY RUN] Would have submitted {len(features)} job(s)")


def _classify_slurm(c, feature_list, clf="logistic", cv="auto",
                    space="sensor", mode="univariate", n_permutations=1000,
                    no_balance=False, n_jobs=-1, continue_on_error=False,
                    combine_features=False, importances=False, label=None,
                    n_chunks=1, seed=42, aggregate=True, delete_chunks=False,
                    trial_type="alltrials", zoning="per-run",
                    n_events_window=8, average_trials=True,
                    slurm_time=None, slurm_mem=None, slurm_cpus=None,
                    dry_run=False):
    """Submit classification jobs to SLURM.

    Job fan-out:
      - One classification per feature (or one per combined-feature run).
      - When n_chunks > 1 (univariate only), each classification is split into
        n_chunks parallel jobs over the spatial dimension. All chunks share the
        same seed so the permutation y-shuffle sequence is identical, and an
        aggregation job (afterok dependency) merges them into the final output.
    """
    from datetime import datetime
    from code.utils.config import load_config
    from code.utils.slurm import render_slurm_script, save_job_manifest, submit_slurm_job

    print(f"\n[SLURM Mode] Submitting classification jobs to cluster\n")

    # run_classification resolves --cv=auto from trial averaging; mirror that
    # here so the chunk-aggregation job builds the matching output filename.
    cv_resolved = cv
    if cv == "auto":
        cv_resolved = "group" if average_trials else "logo"

    config = load_config()
    slurm_config = config["computing"]["slurm"]
    if not slurm_config.get("enabled", False):
        print("ERROR: SLURM is not enabled in config.yaml")
        return

    classification_resources = slurm_config.get("classification", {})
    if not classification_resources:
        print("ERROR: No classification resources in config.yaml (computing.slurm.classification)")
        return

    if n_chunks > 1 and mode != "univariate":
        print("ERROR: --n-chunks requires --mode=univariate (multivariate has no spatial dim to split)")
        return

    features = list(feature_list)
    if not features:
        print("ERROR: no features to submit")
        return

    venv_path = Path(config["paths"]["venv"])
    if not venv_path.is_absolute():
        venv_path = PROJECT_ROOT / venv_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = PROJECT_ROOT / "slurm" / "scripts" / "classification"
    script_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PROJECT_ROOT / config["paths"]["logs"] / "slurm" / "classification"
    log_dir.mkdir(parents=True, exist_ok=True)

    # One classification per feature, OR one combined classification.
    # Each entry: (label_for_filename, feature_args_for_cli, feature_list)
    if combine_features:
        classifications = [(
            label or f"combined-{len(features)}",
            "--feature " + " ".join(features),
            features,
        )]
    else:
        classifications = [(feat, f"--feature {feat}", [feat]) for feat in features]

    print(f"Space: {space}  Mode: {mode}  Classifier: {clf}  CV: {cv_resolved}")
    print(f"n_permutations: {n_permutations}  combine_features: {combine_features}  "
          f"average_trials: {average_trials}")
    print(f"Classifications: {len(classifications)}  chunks per classification: {n_chunks}")

    # Resolve resources: per-invocation overrides win over config defaults
    cpus_resolved = slurm_cpus if slurm_cpus is not None else classification_resources["cpus"]
    mem_resolved = slurm_mem if slurm_mem is not None else classification_resources["mem"]
    time_resolved = slurm_time if slurm_time is not None else classification_resources["time"]
    print(f"Resources: cpus={cpus_resolved}  mem={mem_resolved}  time={time_resolved}")

    base_resources = dict(
        account=slurm_config["account"],
        partition=slurm_config.get("partition", ""),
        cpus=cpus_resolved,
        mem=mem_resolved,
        time=time_resolved,
        log_dir=str(log_dir),
        venv_path=str(venv_path),
        project_root=str(PROJECT_ROOT),
    )

    all_job_ids = []
    aggregator_job_ids = []

    for feat_label, feature_args, feat_list in classifications:
        chunk_indices = list(range(n_chunks))
        chunk_job_ids = []

        for chunk_idx in chunk_indices:
            chunk_suffix = f"_chunk-{chunk_idx}of{n_chunks}" if n_chunks > 1 else ""
            job_name = f"classify_{feat_label}_{space}_{mode}_{clf}{chunk_suffix}"
            context = {
                **base_resources,
                "job_name": job_name,
                "feature_label": feat_label,
                "feature_args": feature_args,
                "space": space,
                "mode": mode,
                "clf": clf,
                "cv": cv,
                "average_trials": average_trials,
                "n_permutations": n_permutations,
                "n_jobs": n_jobs,
                "no_balance": no_balance,
                "combine_features": combine_features,
                "importances": importances,
                "continue_on_error": continue_on_error,
                "label": label,
                "n_chunks": n_chunks,
                "chunk_idx": chunk_idx,
                "seed": seed,
                "trial_type": trial_type,
                "zoning": zoning,
                "n_events_window": n_events_window,
                "timestamp": timestamp,
            }
            script_path = script_dir / f"{job_name}_{timestamp}.sh"
            render_slurm_script("classification.sh.j2", context, output_path=script_path)

            if dry_run:
                print(f"[DRY RUN] Would submit: {script_path.name}")
                continue

            try:
                jid = submit_slurm_job(script_path, job_name=job_name, dry_run=False)
                if jid:
                    chunk_job_ids.append(jid)
                    all_job_ids.append(jid)
            except Exception as e:
                print(f"  ✗ Failed to submit {job_name}: {e}")

        # Submit aggregation job with afterok dependency on this classification's chunks
        if n_chunks > 1 and aggregate:
            agg_job_name = f"aggregate_{feat_label}_{space}_{mode}_{clf}"
            agg_context = {
                **base_resources,
                # Aggregation is light — one CPU, modest RAM, short walltime.
                "cpus": 1,
                "mem": "8G",
                "time": "0:30:00",
                "job_name": agg_job_name,
                "feature_label": feat_label,
                "space": space,
                "mode": mode,
                "clf": clf,
                "cv": cv_resolved,
                "combined": combine_features,
                "delete_chunks": delete_chunks,
                "trial_type": trial_type,
                "timestamp": timestamp,
            }
            agg_script = script_dir / f"{agg_job_name}_{timestamp}.sh"
            render_slurm_script("classification_aggregate.sh.j2", agg_context, output_path=agg_script)

            if dry_run:
                print(f"[DRY RUN] Would submit aggregator: {agg_script.name} "
                      f"(afterok:{','.join(chunk_job_ids) if chunk_job_ids else '<chunks>'})")
                continue

            if not chunk_job_ids:
                print(f"  ✗ Skipping aggregator for {feat_label}: no chunk jobs were submitted")
                continue
            try:
                jid = submit_slurm_job(
                    agg_script,
                    job_name=agg_job_name,
                    dependencies=chunk_job_ids,
                    dep_type="afterok",
                    dry_run=False,
                )
                if jid:
                    aggregator_job_ids.append(jid)
                    all_job_ids.append(jid)
            except Exception as e:
                print(f"  ✗ Failed to submit aggregator {agg_job_name}: {e}")

    if all_job_ids:
        manifest_path = log_dir / f"classification_manifest_{timestamp}.json"
        save_job_manifest(all_job_ids, manifest_path, metadata={
            "stage": "classification",
            "space": space,
            "mode": mode,
            "clf": clf,
            "cv": cv,
            "n_permutations": n_permutations,
            "combine_features": combine_features,
            "n_chunks": n_chunks,
            "aggregator_job_ids": aggregator_job_ids,
            "features": features,
            "timestamp": timestamp,
        })
        print(f"\n✓ Submitted {len(all_job_ids)} job(s) "
              f"({len(aggregator_job_ids)} aggregator); manifest: {manifest_path}")
    elif dry_run:
        n_total = len(classifications) * n_chunks + (
            len(classifications) if n_chunks > 1 and aggregate else 0
        )
        print(f"\n[DRY RUN] Would have submitted {n_total} job(s)")


@task
def classify_aggregate(c, feature, space, clf="lda", cv="logo",
                       mode="univariate", combined=False,
                       delete_chunks=False, trial_type="alltrials",
                       config="config.yaml"):
    """Manually aggregate per-chunk classification outputs.

    Use this if --aggregate=False was used at submission, or if the afterok
    aggregator job failed and chunk files are still on disk.

    Examples:
        invoke analysis.classify-aggregate --feature=psd_alpha --space=sensor
        invoke analysis.classify-aggregate --feature=combined-10 --space=schaefer_400 --combined
    """
    python_exe = get_python_executable()
    cmd = [
        python_exe, "-m", "code.classification.aggregate_chunks",
        "--feature", feature,
        "--space", space,
        "--mode", mode,
        "--clf", clf,
        "--cv", cv,
        "--trial-type", trial_type,
        "--config", config,
    ]
    if combined:
        cmd.append("--combined")
    if delete_chunks:
        cmd.append("--delete-chunks")
    print(f"Running: {' '.join(cmd)}")
    c.run(" ".join(cmd), pty=True, env=get_env_with_pythonpath())


# ==============================================================================
# Build Namespace Collections
# ==============================================================================

# dev.check.* subcollection
check = Collection("check")
check.add_task(check_dataset, name="dataset")
check.add_task(check_qc, name="qc")
check.add_task(check_code, name="code")

# Development tasks
dev = Collection("dev")
dev.add_task(test)
dev.add_task(test_fast, name="test-fast")
dev.add_task(clean)
dev.add_task(precommit)
dev.add_collection(check)  # Nested: dev.check.*

# Environment tasks
env = Collection("env")
env.add_task(setup)
env.add_task(info)
env.add_task(validate_config, name="validate-config")
env.add_task(rebuild)

# Data download tasks
get = Collection("get")
get.add_task(get_atlases, name="atlases")

# Feature extraction tasks (nested under pipeline)
features = Collection("features")
features.add_task(fooof)
features.add_task(psd)
features.add_task(complexity)
features.add_task(extract_all, name="all")

# Pipeline tasks (includes features as subcollection)
pipeline = Collection("pipeline")
pipeline.add_task(validate_inputs, name="validate-inputs")
pipeline.add_task(bids)
pipeline.add_task(preprocess)
pipeline.add_task(preprocess_report, name="preprocess-report")
pipeline.add_task(source_recon, name="source-recon")
pipeline.add_task(atlas)
pipeline.add_collection(features)  # Nested: pipeline.features.*


@task
def slurm_jobs(c, pattern=None, state=None, user=None):
    """List your running/queued SLURM jobs, optionally filtered by name glob.

    Args:
        pattern: glob-style name filter (e.g. 'classify_*', '*chunk-0*'). Default: all.
        state: filter by SLURM state code (R, PD, CG, ...). Default: all.
        user: SLURM username to query (default: current user).

    Examples:
        invoke slurm.jobs
        invoke slurm.jobs --pattern='classify_*'
        invoke slurm.jobs --pattern='aggregate_*' --state=PD
    """
    import fnmatch
    from code.utils.slurm import get_user_jobs

    jobs = get_user_jobs(user=user)
    if pattern:
        jobs = [j for j in jobs if fnmatch.fnmatch(j["JobName"], pattern)]
    if state:
        jobs = [j for j in jobs if j["State"] == state]

    if not jobs:
        print("No matching jobs.")
        return

    print(f"{'JobID':>10}  {'State':>5}  {'Time':>10}  {'CPUs':>4}  {'Mem':>6}  Name")
    print("-" * 80)
    for j in jobs:
        print(
            f"{j['JobID']:>10}  {j['State']:>5}  {j['Time']:>10}  "
            f"{j['CPUs']:>4}  {j['Memory']:>6}  {j['JobName']}"
        )
    print(f"\n{len(jobs)} job(s)")


@task
def slurm_cancel(c, pattern=None, job_ids=None, state=None, user=None,
                 dry_run=False, yes=False):
    """Cancel SLURM jobs matching a name glob (or explicit IDs).

    Safety: by default, prints what would be cancelled and asks for confirmation.
    Pass --yes to skip the prompt, or --dry-run to print without cancelling.

    Args:
        pattern: glob-style name filter (e.g. 'classify_psd_*', '*chunk-*').
        job_ids: comma- or space-separated explicit job IDs (overrides pattern matching).
        state: optional SLURM state code filter (R, PD, CG, ...). Useful with pattern.
        user: SLURM username to query (default: current user).
        dry_run: just print the matches; don't cancel.
        yes: skip the confirmation prompt and cancel immediately.

    Examples:
        invoke slurm.cancel --pattern='classify_psd_*' --dry-run
        invoke slurm.cancel --pattern='aggregate_*' --state=PD --yes
        invoke slurm.cancel --pattern='*chunk-0of*'
        invoke slurm.cancel --job-ids='123,124,125'
    """
    import fnmatch
    from code.utils.slurm import cancel_job, get_user_jobs

    if not pattern and not job_ids:
        print("ERROR: pass --pattern or --job-ids")
        return

    targets: list = []
    if job_ids:
        ids = [s for s in job_ids.replace(",", " ").split() if s]
        # If state/pattern also given, look up names from squeue and apply filters
        if state or pattern:
            all_jobs = {j["JobID"]: j for j in get_user_jobs(user=user)}
            for jid in ids:
                if jid not in all_jobs:
                    print(f"  ! {jid}: not in queue, skipping")
                    continue
                j = all_jobs[jid]
                if state and j["State"] != state:
                    continue
                if pattern and not fnmatch.fnmatch(j["JobName"], pattern):
                    continue
                targets.append(j)
        else:
            targets = [{"JobID": jid, "JobName": "(unknown)", "State": "?"} for jid in ids]
    else:
        jobs = get_user_jobs(user=user)
        targets = [j for j in jobs if fnmatch.fnmatch(j["JobName"], pattern)]
        if state:
            targets = [j for j in targets if j["State"] == state]

    if not targets:
        print("No matching jobs.")
        return

    print(f"Targets ({len(targets)}):")
    for j in targets:
        print(f"  {j['JobID']:>10}  [{j.get('State', '?'):>3}]  {j['JobName']}")

    if dry_run:
        print(f"\n[DRY RUN] Would cancel {len(targets)} job(s)")
        return

    if not yes:
        try:
            ans = input(f"\nCancel {len(targets)} job(s)? [y/N] ").strip().lower()
        except EOFError:
            ans = ""
        if ans not in ("y", "yes"):
            print("Aborted.")
            return

    cancelled = 0
    for j in targets:
        if cancel_job(j["JobID"]):
            cancelled += 1
    print(f"\n✓ Cancelled {cancelled}/{len(targets)} job(s)")


# Analysis tasks
analysis = Collection("analysis")
analysis.add_task(stats)
analysis.add_task(classify)
analysis.add_task(classify_aggregate, name="classify-aggregate")

# Visualization tasks
viz = Collection("viz")
viz.add_task(viz_stats, name="stats")
viz.add_task(viz_maps, name="maps")
viz.add_task(spectra)
viz.add_task(behavior)

# SLURM job-management tasks
slurm = Collection("slurm")
slurm.add_task(slurm_jobs, name="jobs")
slurm.add_task(slurm_cancel, name="cancel")

# Build main namespace
namespace = Collection()
namespace.add_collection(dev)
namespace.add_collection(env)
namespace.add_collection(get)
namespace.add_collection(pipeline)
namespace.add_collection(analysis)
namespace.add_collection(viz)
namespace.add_collection(slurm)


# Default task
@task(default=True)
def help_task(c):
    """Show available tasks (default task)."""
    c.run("invoke --list")


namespace.add_task(help_task, name="help")
