"""Invoke tasks for saflow development.

This module provides common development tasks using the invoke library.
Run `invoke --list` to see all available tasks.

Usage:
    invoke test              # Run tests
    invoke lint              # Run linting
    invoke format            # Format code
    invoke clean             # Clean generated files
    invoke setup --mode=dev  # Setup environment
"""

from pathlib import Path
from invoke import task


# ==============================================================================
# Configuration
# ==============================================================================

PROJECT_ROOT = Path(__file__).parent
SAFLOW_PKG = PROJECT_ROOT / "saflow"
CODE_DIR = PROJECT_ROOT / "code"
TESTS_DIR = PROJECT_ROOT / "tests"
DOCS_DIR = PROJECT_ROOT / "docs"

# Directories to clean
CLEAN_PATTERNS = [
    "**/__pycache__",
    "**/*.pyc",
    "**/*.pyo",
    "**/*.egg-info",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "htmlcov",
    ".coverage",
    "coverage.xml",
    "build",
    "dist",
]


# ==============================================================================
# Testing
# ==============================================================================

@task
def test(c, verbose=False, coverage=True, markers=None):
    """Run tests with pytest.

    Args:
        verbose: Verbose output (-v)
        coverage: Generate coverage report (default: True)
        markers: Only run tests matching given marker expression

    Examples:
        invoke test
        invoke test --verbose
        invoke test --markers="not slow"
        invoke test --no-coverage
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
def test_unit(c):
    """Run only unit tests."""
    test(c, markers="unit")


@task
def test_integration(c):
    """Run only integration tests."""
    test(c, markers="integration")


# ==============================================================================
# Code Quality
# ==============================================================================

@task
def lint(c, fix=False):
    """Run ruff linting.

    Args:
        fix: Automatically fix issues where possible

    Examples:
        invoke lint
        invoke lint --fix
    """
    cmd = ["ruff", "check", str(SAFLOW_PKG), str(CODE_DIR)]

    if fix:
        cmd.append("--fix")

    print(f"Running: {' '.join(cmd)}")
    result = c.run(" ".join(cmd), warn=True, pty=True)

    if result.exited != 0:
        print("\n⚠️  Linting issues found. Run 'invoke lint --fix' to auto-fix.")
        return result

    print("✓ No linting issues found!")


@task
def format(c, check=False):
    """Format code with ruff.

    Args:
        check: Only check formatting, don't modify files

    Examples:
        invoke format
        invoke format --check
    """
    cmd = ["ruff", "format", str(SAFLOW_PKG), str(CODE_DIR)]

    if check:
        cmd.append("--check")

    print(f"Running: {' '.join(cmd)}")
    result = c.run(" ".join(cmd), warn=True, pty=True)

    if result.exited != 0 and check:
        print("\n⚠️  Code formatting issues found. Run 'invoke format' to fix.")
        return result

    if not check:
        print("✓ Code formatted successfully!")


@task
def typecheck(c):
    """Run mypy type checking."""
    cmd = ["mypy", str(SAFLOW_PKG), str(CODE_DIR)]

    print(f"Running: {' '.join(cmd)}")
    c.run(" ".join(cmd), warn=True, pty=True)


@task
def check(c):
    """Run all code quality checks (lint, format check, type check).

    This is useful for pre-commit or CI checks.
    """
    print("=" * 80)
    print("Running all code quality checks...")
    print("=" * 80)

    print("\n1. Linting...")
    lint_result = lint(c)

    print("\n2. Format checking...")
    format_result = format(c, check=True)

    print("\n3. Type checking...")
    typecheck(c)

    print("\n" + "=" * 80)
    if lint_result and lint_result.exited != 0:
        print("❌ Linting failed")
    elif format_result and format_result.exited != 0:
        print("❌ Format checking failed")
    else:
        print("✓ All code quality checks passed!")
    print("=" * 80)


# ==============================================================================
# Cleaning
# ==============================================================================

@task
def clean(c, bytecode=True, cache=True, coverage=True, build=True, logs=False, reports=False):
    """Clean generated files.

    Args:
        bytecode: Remove Python bytecode files (default: True)
        cache: Remove tool cache directories (default: True)
        coverage: Remove coverage reports (default: True)
        build: Remove build artifacts (default: True)
        logs: Remove log files (default: False)
        reports: Remove report files (default: False)

    Examples:
        invoke clean
        invoke clean --logs --reports  # Clean everything including logs/reports
        invoke clean --no-bytecode     # Clean everything except bytecode
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

    if reports:
        patterns.append("reports")

    print("Cleaning generated files...")
    removed_count = 0

    for pattern in patterns:
        if "**" in pattern:
            # Recursive glob pattern
            for path in PROJECT_ROOT.rglob(pattern.replace("**/", "")):
                if path.exists():
                    if path.is_file():
                        path.unlink()
                        removed_count += 1
                    elif path.is_dir():
                        c.run(f"rm -rf {path}", hide=True)
                        removed_count += 1
        else:
            # Direct path
            path = PROJECT_ROOT / pattern
            if path.exists():
                if path.is_file():
                    path.unlink()
                    print(f"  Removed: {path.relative_to(PROJECT_ROOT)}")
                    removed_count += 1
                elif path.is_dir():
                    c.run(f"rm -rf {path}", hide=True)
                    print(f"  Removed: {path.relative_to(PROJECT_ROOT)}/")
                    removed_count += 1

    print(f"✓ Cleaned {removed_count} items")


@task
def clean_all(c):
    """Clean everything including logs and reports."""
    clean(c, logs=True, reports=True)


# ==============================================================================
# Environment Setup
# ==============================================================================

@task
def setup(c, mode="basic", python="python3.9", force=False):
    """Run the setup script to create development environment.

    Args:
        mode: Installation mode (basic, dev, hpc, all)
        python: Python executable to use
        force: Force reinstall if venv exists

    Examples:
        invoke setup
        invoke setup --mode=dev
        invoke setup --mode=all --python=python3.10
        invoke setup --force
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


# ==============================================================================
# Documentation
# ==============================================================================

@task
def docs(c, serve=False, port=8000):
    """Build documentation with Sphinx.

    Args:
        serve: Serve documentation locally after building
        port: Port for local server (default: 8000)

    Examples:
        invoke docs
        invoke docs --serve
        invoke docs --serve --port=8080
    """
    docs_build = DOCS_DIR / "_build"

    print("Building documentation...")
    c.run(f"sphinx-build -b html {DOCS_DIR} {docs_build}/html", pty=True)

    print(f"✓ Documentation built in {docs_build}/html")

    if serve:
        print(f"\nServing documentation at http://localhost:{port}")
        print("Press Ctrl+C to stop server")
        c.run(f"python -m http.server {port} -d {docs_build}/html", pty=True)


# ==============================================================================
# Utility Tasks
# ==============================================================================

@task
def info(c):
    """Display project information."""
    print("=" * 80)
    print("Saflow Project Information")
    print("=" * 80)
    print(f"Project root:     {PROJECT_ROOT}")
    print(f"Saflow package:   {SAFLOW_PKG}")
    print(f"Code directory:   {CODE_DIR}")
    print(f"Tests directory:  {TESTS_DIR}")
    print(f"Docs directory:   {DOCS_DIR}")
    print()

    # Check if config exists
    config_file = PROJECT_ROOT / "config.yaml"
    config_template = PROJECT_ROOT / "config.yaml.template"

    print("Configuration:")
    print(f"  config.yaml:          {'✓ exists' if config_file.exists() else '✗ missing'}")
    print(f"  config.yaml.template: {'✓ exists' if config_template.exists() else '✗ missing'}")
    print()

    # Check if venv exists
    venv_dir = PROJECT_ROOT / "env"
    print(f"Virtual environment:  {'✓ exists' if venv_dir.exists() else '✗ missing'}")

    if venv_dir.exists():
        # Try to get Python version
        python_exe = venv_dir / "bin" / "python"
        if python_exe.exists():
            result = c.run(f"{python_exe} --version", hide=True)
            print(f"  Python version:     {result.stdout.strip()}")

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


# ==============================================================================
# Development Workflow
# ==============================================================================

@task(pre=[format, lint, test])
def precommit(c):
    """Run pre-commit checks (format, lint, test).

    This is useful to run before committing changes.
    """
    print("\n" + "=" * 80)
    print("✓ All pre-commit checks passed!")
    print("=" * 80)


@task(pre=[clean])
def rebuild(c, mode="dev"):
    """Clean and rebuild environment.

    Args:
        mode: Installation mode (basic, dev, hpc, all)
    """
    setup(c, mode=mode, force=True)


# ==============================================================================
# Helper Functions
# ==============================================================================

def get_python_executable():
    """Get the Python executable, preferring venv if it exists."""
    venv_python = PROJECT_ROOT / "env" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return "python"


# ==============================================================================
# Pipeline Tasks
# ==============================================================================

@task
def validate_inputs(c, data_root=None, verbose=False):
    """Validate that raw input data is present and complete.

    Checks for:
    - Raw MEG data directory exists
    - Behavioral logfiles directory exists
    - Required .ds files present
    - Behavioral logfiles present for expected subjects

    Args:
        data_root: Override data root from config
        verbose: Show detailed file listing

    Examples:
        invoke validate-inputs
        invoke validate-inputs --verbose
        invoke validate-inputs --data-root=/path/to/data
    """
    print("=" * 80)
    print("Validating Input Data")
    print("=" * 80)

    # Get Python executable (prefer venv)
    python_exe = get_python_executable()

    # Build command - run as module with PYTHONPATH set
    cmd = [python_exe, "-m", "code.utils.validation", "--check-inputs"]

    if data_root:
        cmd.extend(["--data-root", str(data_root)])

    if verbose:
        cmd.append("--verbose")

    # Set PYTHONPATH to project root
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    c.run(" ".join(cmd), pty=True, env=env)


@task
def bids(
    c,
    input_dir=None,
    output_dir=None,
    subjects=None,
    log_level="INFO",
    dry_run=False,
):
    """Run BIDS conversion (Stage 0).

    Converts raw MEG data (.ds files) to BIDS format and enriches
    gradCPT events with behavioral data (VTC, RT, performance, IN/OUT zones).

    Args:
        input_dir: Override raw data directory from config
        output_dir: Override BIDS output directory from config
        subjects: Process specific subjects only (space-separated)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        dry_run: Validate inputs without processing

    Examples:
        invoke bids
        invoke bids --subjects "04 05 06"
        invoke bids --log-level=DEBUG
        invoke bids --input-dir=/path/to/raw --output-dir=/path/to/bids
        invoke bids --dry-run
    """
    print("=" * 80)
    print("BIDS Conversion - Stage 0")
    print("=" * 80)

    # Get Python executable (prefer venv)
    python_exe = get_python_executable()

    # Build command - run as module with PYTHONPATH set
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

    # Set PYTHONPATH to include project root
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    c.run(" ".join(cmd), pty=True, env=env)


@task
def preprocess(
    c,
    subject,
    runs=None,
    bids_root=None,
    log_level="INFO",
    skip_existing=True,
):
    """Run MEG preprocessing (Stage 1).

    Preprocesses BIDS MEG data with filtering, AutoReject, and ICA artifact removal.
    Saves preprocessed continuous data, epochs, logs, and HTML reports.

    Args:
        subject: Subject ID to process (required)
        runs: Run numbers to process (space-separated, default: all task runs from config)
        bids_root: Override BIDS root directory from config
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        skip_existing: Skip runs already preprocessed (default: True)

    Examples:
        invoke preprocess --subject=04
        invoke preprocess --subject=04 --runs="02 03"
        invoke preprocess --subject=04 --log-level=DEBUG
        invoke preprocess --subject=04 --bids-root=/path/to/bids
        invoke preprocess --subject=04 --no-skip-existing
    """
    print("=" * 80)
    print("MEG Preprocessing - Stage 1")
    print("=" * 80)

    # Get Python executable (prefer venv)
    python_exe = get_python_executable()

    # Build command - run as module with PYTHONPATH set
    cmd = [python_exe, "-m", "code.preprocessing.run_preprocessing"]

    cmd.extend(["--subject", subject])

    if runs:
        cmd.extend(["--runs"] + runs.split())

    if bids_root:
        cmd.extend(["--bids-root", str(bids_root)])

    cmd.extend(["--log-level", log_level])

    if skip_existing:
        cmd.append("--skip-existing")

    print(f"\nRunning: {' '.join(cmd)}\n")

    # Set PYTHONPATH to include project root
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    c.run(" ".join(cmd), pty=True, env=env)


# ==============================================================================
# List Tasks
# ==============================================================================

@task(default=True)
def help(c):
    """Show available tasks (default task)."""
    c.run("invoke --list")
