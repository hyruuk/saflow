# Saflow Task Reference

This document describes all available invoke tasks for the saflow pipeline.
All tasks can be run using `invoke <task-name> [options]`.

**Quick reference**: Run `invoke --list` to see all available tasks.

---

## Table of Contents

1. [Pipeline Tasks](#pipeline-tasks) - Main analysis pipeline
2. [Testing Tasks](#testing-tasks) - Run tests
3. [Code Quality Tasks](#code-quality-tasks) - Linting, formatting, type checking
4. [Cleaning Tasks](#cleaning-tasks) - Clean generated files
5. [Environment Tasks](#environment-tasks) - Setup and manage environment
6. [Documentation Tasks](#documentation-tasks) - Build documentation
7. [Utility Tasks](#utility-tasks) - Helper tasks
8. [Workflow Tasks](#workflow-tasks) - Combined workflows

---

## Pipeline Tasks

### `invoke validate-inputs`

Validate that raw input data is present and complete before running pipeline.

**What it checks:**
- Raw MEG data directory exists (`data_root/raw/meg/`)
- Behavioral logfiles directory exists (`data_root/raw/behav/`)
- Required `.ds` files are present
- Behavioral `.mat` files are present
- All expected subjects have data
- Shows summary table of available runs per subject

**Arguments:**
- `--data-root PATH`: Override data root from config
- `--verbose`: Show detailed file listings

**Examples:**
```bash
# Basic validation
invoke validate-inputs

# Verbose output with file listings
invoke validate-inputs --verbose

# Override data location
invoke validate-inputs --data-root=/media/storage/DATA/saflow
```

**Output:**
- Colored console report with ✓/✗ indicators
- Table showing runs per subject
- List of missing subjects (if any)
- Summary of behavioral files

**Exit codes:**
- `0`: Validation passed
- `1`: Validation failed (missing data or errors)

---

### `invoke bids`

Run BIDS conversion (Stage 0 of pipeline).

**What it does:**
1. Finds all CTF MEG datasets (`.ds` files) in raw directory
2. Converts to BIDS format using mne-bids
3. For gradCPT task runs:
   - Loads behavioral logfiles
   - Computes VTC (variability of reaction times)
   - Enriches events.tsv with:
     - Trial indices
     - VTC and RT values
     - Task performance (commission_error, correct_omission, etc.)
     - IN/OUT zone classifications (50/50, 25/75, 10/90 percentile bounds)
4. Writes empty-room noise recordings
5. Saves provenance metadata (git hash, timestamp, subjects processed)

**Arguments:**
- `--input-dir PATH`: Override raw data directory from config
- `--output-dir PATH`: Override BIDS output directory from config
- `--subjects "ID1 ID2 ..."`: Process specific subjects only (space-separated)
- `--log-level LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `--dry-run`: Validate inputs without processing files

**Examples:**
```bash
# Basic usage (uses paths from config.yaml)
invoke bids

# Process specific subjects only
invoke bids --subjects "04 05 06"

# Override input/output paths
invoke bids --input-dir=/path/to/raw --output-dir=/path/to/bids

# Dry run to validate before processing
invoke bids --dry-run

# Debug mode with verbose logging
invoke bids --log-level=DEBUG
```

**Output locations:**
- BIDS dataset: `{config.paths.data_root}/bids/`
- Logs: `logs/bids/bids_conversion_YYYYMMDD_HHMMSS.log`
- Provenance: `{bids_root}/code/provenance_bids.json`

**Expected runtime:**
- ~5-10 minutes for 30 subjects (7 runs each)
- Depends on disk I/O speed

**Requirements:**
- Raw MEG data in CTF format (`.ds` files)
- Behavioral logfiles (`.mat` files)
- Valid `config.yaml` with correct paths

---

## Testing Tasks

### `invoke test`

Run full test suite with pytest.

**Arguments:**
- `--verbose`: Enable verbose output (`pytest -v`)
- `--coverage` / `--no-coverage`: Generate coverage report (default: True)
- `--markers EXPR`: Run only tests matching marker expression

**Examples:**
```bash
# Run all tests with coverage
invoke test

# Verbose output
invoke test --verbose

# Run specific markers
invoke test --markers="not slow"

# Without coverage report
invoke test --no-coverage
```

**Output:**
- Coverage report in terminal
- HTML coverage report: `htmlcov/index.html`
- XML coverage report: `coverage.xml`

---

### `invoke test-fast`

Run only fast tests (excludes tests marked with `@pytest.mark.slow`).

**Examples:**
```bash
invoke test-fast
```

---

### `invoke test-unit`

Run only unit tests (tests marked with `@pytest.mark.unit`).

**Examples:**
```bash
invoke test-unit
```

---

### `invoke test-integration`

Run only integration tests (tests marked with `@pytest.mark.integration`).

**Examples:**
```bash
invoke test-integration
```

---

## Code Quality Tasks

### `invoke lint`

Run ruff linting to check code quality.

**Arguments:**
- `--fix`: Automatically fix issues where possible

**Examples:**
```bash
# Check for issues
invoke lint

# Auto-fix issues
invoke lint --fix
```

**What it checks:**
- Code style (PEP 8)
- Common bugs (unused imports, undefined names)
- Code complexity
- Import sorting
- Documentation quality

---

### `invoke format`

Format code with ruff formatter.

**Arguments:**
- `--check`: Only check formatting without modifying files

**Examples:**
```bash
# Format all code
invoke format

# Check formatting without changes
invoke format --check
```

**What it does:**
- Applies consistent code style
- Sorts imports
- Adjusts line lengths
- Standardizes quotes

---

### `invoke typecheck`

Run mypy type checking.

**Examples:**
```bash
invoke typecheck
```

**What it checks:**
- Type hint correctness
- Type consistency
- Missing type annotations (when enabled)

---

### `invoke check`

Run all code quality checks (lint + format + typecheck).

Useful for pre-commit or CI checks.

**Examples:**
```bash
invoke check
```

**Output:**
- Summary of all checks
- Overall pass/fail status

---

## Cleaning Tasks

### `invoke clean`

Clean generated files and caches.

**Arguments:**
- `--bytecode` / `--no-bytecode`: Remove Python bytecode (default: True)
- `--cache` / `--no-cache`: Remove tool caches (default: True)
- `--coverage` / `--no-coverage`: Remove coverage reports (default: True)
- `--build` / `--no-build`: Remove build artifacts (default: True)
- `--logs`: Remove log files (default: False)
- `--reports`: Remove report files (default: False)

**Examples:**
```bash
# Clean standard generated files
invoke clean

# Clean including logs and reports
invoke clean --logs --reports

# Clean only bytecode
invoke clean --no-cache --no-coverage --no-build
```

**What it removes:**
- `__pycache__/`, `*.pyc`, `*.pyo` (bytecode)
- `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/` (caches)
- `htmlcov/`, `.coverage`, `coverage.xml` (coverage)
- `*.egg-info/`, `build/`, `dist/` (build artifacts)
- `logs/` (if `--logs` specified)
- `reports/` (if `--reports` specified)

---

### `invoke clean-all`

Clean everything including logs and reports.

Equivalent to `invoke clean --logs --reports`.

**Examples:**
```bash
invoke clean-all
```

---

## Environment Tasks

### `invoke setup`

Run the setup script to create development environment.

**Arguments:**
- `--mode MODE`: Installation mode (basic, dev, hpc, all)
- `--python PYTHON`: Python executable to use (default: python3.9)
- `--force`: Force reinstall if venv exists

**Examples:**
```bash
# Basic installation
invoke setup

# Development installation with all tools
invoke setup --mode=dev

# All optional dependencies
invoke setup --mode=all

# Use different Python version
invoke setup --mode=dev --python=python3.10

# Force reinstall
invoke setup --force
```

**Installation modes:**
- `basic`: Core dependencies only
- `dev`: Core + development tools (ruff, mypy, pytest, etc.)
- `hpc`: Core + HPC/SLURM utilities
- `all`: Everything (dev + hpc + docs)

---

### `invoke rebuild`

Clean environment and rebuild from scratch.

**Arguments:**
- `--mode MODE`: Installation mode (default: dev)

**Examples:**
```bash
# Rebuild development environment
invoke rebuild

# Rebuild with all dependencies
invoke rebuild --mode=all
```

---

## Documentation Tasks

### `invoke docs`

Build documentation with Sphinx.

**Arguments:**
- `--serve`: Serve documentation locally after building
- `--port PORT`: Port for local server (default: 8000)

**Examples:**
```bash
# Build documentation
invoke docs

# Build and serve locally
invoke docs --serve

# Serve on different port
invoke docs --serve --port=8080
```

**Output:**
- HTML documentation: `docs/_build/html/`
- Access at: `http://localhost:8000`

---

## Utility Tasks

### `invoke info`

Display project information.

**Examples:**
```bash
invoke info
```

**Shows:**
- Project root path
- Package directories
- Configuration status
- Virtual environment status
- Python version

---

### `invoke validate-config`

Validate configuration file syntax and content.

**Examples:**
```bash
invoke validate-config
```

**Checks:**
- YAML syntax is valid
- Required fields are present
- Paths are correctly formatted
- No placeholder values remain

---

## Workflow Tasks

### `invoke precommit`

Run pre-commit checks (format + lint + test).

Runs in sequence:
1. Format code
2. Lint code
3. Run tests

**Examples:**
```bash
invoke precommit
```

**Use before:**
- Committing changes
- Creating pull requests
- Pushing to shared branches

---

### `invoke help`

Show available tasks (default task).

**Examples:**
```bash
# These are equivalent
invoke
invoke help
invoke --list
```

---

## Typical Workflows

### Starting a new analysis

```bash
# 1. Validate configuration
invoke validate-config

# 2. Check input data is complete
invoke validate-inputs --verbose

# 3. Run BIDS conversion (dry run first)
invoke bids --dry-run
invoke bids

# 4. Check logs for any issues
less logs/bids/bids_conversion_*.log
```

### Before committing code

```bash
# Run all quality checks and tests
invoke precommit
```

### Setting up development environment

```bash
# First time setup
invoke setup --mode=dev

# After pulling changes
invoke rebuild --mode=dev
```

### Running specific subjects

```bash
# BIDS conversion for subjects 04, 05, 06 only
invoke bids --subjects "04 05 06"
```

---

## Task Dependencies

Some tasks run other tasks automatically:

- `invoke precommit` → format, lint, test
- `invoke rebuild` → clean, setup
- `invoke check` → lint, format (check), typecheck

---

## Exit Codes

All tasks follow standard exit code conventions:
- `0`: Success
- `1`: Failure (validation failed, tests failed, errors occurred)
- Non-zero: Error or failure

Use exit codes for scripting:
```bash
# Example: Only proceed if validation passes
if invoke validate-inputs; then
    invoke bids
else
    echo "Validation failed, fix errors first"
    exit 1
fi
```

---

## Getting Help

For any task, use `--help`:
```bash
invoke bids --help
invoke test --help
```

For overall help:
```bash
invoke --list
invoke --help
```

For pipeline questions, see:
- `README.md` - Project overview
- `PROGRESS.md` - Current implementation status
- `docs/workflow.md` - Detailed pipeline documentation
- `AGENTS.md` - Development guidelines
