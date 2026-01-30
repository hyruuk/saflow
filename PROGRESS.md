# Saflow Refactoring Progress Tracker

**Last Updated**: 2026-01-30
**Current Phase**: Phase 1 COMPLETE - Foundation & Infrastructure ✅
**Analysis Focus**: Sensor-level analysis (Phase 1), Source-level planned for Phase 2
**Status**: 🛑 CHECKPOINT - Awaiting user review before Phase 2

---

## Key Context for Resuming

### What We're Doing
Complete refactoring of cc_saflow → saflow with:
- Clean configuration system (no hardcoded paths)
- Proper logging (no print statements)
- Modern Python packaging (pyproject.toml)
- Sensor/source analysis space architecture
- BIDS-compliant organization

### Critical Paths
- **Old codebase**: `/home/hyruuk/GitHub/cocolab/cc_saflow`
- **New codebase**: `/home/hyruuk/GitHub/cocolab/saflow`
- **Data location**: `/media/hyruuk/YH_storage/DATA/saflow`

### Important Notes
- Log directories should be created by scripts as needed (not pre-created)
- Phase 1 focuses on SENSOR-LEVEL analysis only
- Source-level analysis is Phase 2 (post-refactoring)
- Architecture supports both spaces from the start

---

## Completed Tasks

### ✅ Phase 1.1: Create project structure
**Status**: COMPLETE
**What was done**:
- Created all directory structure:
  - `saflow/` package (with proper `__init__.py` file, not folder!)
  - `code/` with all subdirectories including sensor/ and source/ splits
  - `slurm/templates/`
  - `tests/`, `docs/`, `reports/` with sensor/source subdirs
  - `logs/` (empty, to be populated by scripts)
- Created `__init__.py` files for all Python modules
- Copied LICENSE from cc_saflow

**Files created**:
- Directory structure
- Multiple `__init__.py` files
- `LICENSE` (copied)

### ✅ Phase 1.2: Implement configuration system
**Status**: COMPLETE
**What was done**:
- ✅ Created `config.yaml.template` with comprehensive schema
  - Includes analysis.space configuration (sensor/source)
  - All paths with placeholders
  - BIDS configuration with exact subject list from cc_saflow
  - Preprocessing, source reconstruction, features settings
  - SLURM resource allocations
- ✅ Created `code/utils/config.py` with:
  - Configuration loading and validation
  - Placeholder detection
  - Path expansion (relative → absolute)
  - Helper functions (get_subjects, get_analysis_space, etc.)
- ✅ Created `code/utils/paths.py` with:
  - BIDS path construction helpers
  - Path getters for all pipeline stages
  - Analysis space-aware path routing
- ✅ Created `code/utils/space.py` with:
  - Analysis space validation
  - Sensor/source routing utilities
  - Source-level prerequisite checks
  - Phase 1 sensor-only enforcement
- ✅ Created working `config.yaml` for testing
- ✅ Validated YAML syntax and structure

### ✅ Phase 1.3: Implement logging system
**Status**: COMPLETE
**What was done**:
- ✅ Created `code/utils/logging_config.py` with:
  - `setup_logging()` function for configuring loggers
  - Console and file output support
  - Color-coded console output (ANSI colors)
  - Configurable log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  - Automatic timestamp addition to log files
  - Support for multiple file handlers
  - `add_file_handler()` for adding additional log files
  - `log_provenance()` for recording git hash, config, timestamp
- ✅ Features:
  - Log format and level configurable via config.yaml
  - Automatic log directory creation
  - Support for nested log directories (e.g., logs/preprocessing/subject_01/)
  - Colored console output for better readability
  - File output without ANSI codes
  - Integration with configuration system
- ✅ Tested all logging functionality:
  - Basic logging (console + file)
  - Console-only logging
  - Multiple file handlers
  - Provenance logging
  - Subdirectory logging
  - All tests passed ✓

**Files created**:
- `code/utils/logging_config.py`
- `code/__init__.py` (created to fix package imports)

### ✅ Phase 1.4: Setup dependency management
**Status**: COMPLETE
**What was done**:
- ✅ Created `pyproject.toml` with modern Python packaging (PEP 621)
  - Build system configuration (setuptools)
  - Project metadata (name, version, description, authors, keywords, classifiers)
  - Python version constraint: >=3.9,<3.13
  - Core dependencies with version constraints extracted from cc_saflow/requirements.txt:
    - MEG/EEG: mne, mne-bids, autoreject
    - Scientific computing: numpy, scipy, pandas
    - Machine learning: scikit-learn, xgboost, mlneurotools
    - Signal processing: fooof, neurokit2
    - Statistics: statsmodels, statannotations
    - Visualization: matplotlib, seaborn
    - Utilities: joblib, tqdm, rich, h5io, str2bool
    - Configuration: pyyaml, python-dotenv, click
  - Optional dependency groups:
    - `[dev]`: ruff, mypy, pre-commit, ipython, jupyter
    - `[test]`: pytest, pytest-cov, pytest-xdist, coverage
    - `[docs]`: sphinx, sphinx-rtd-theme, myst-parser
    - `[hpc]`: jinja2 (for SLURM templating)
    - `[all]`: All optional dependencies combined
- ✅ Comprehensive tool configurations:
  - **Ruff**: Linting and formatting with ~20 rule sets enabled
    - Line length: 100
    - Target Python 3.9+
    - Excludes: .git, .venv, __pycache__, build, logs, reports, data
    - isort configuration with first-party packages
  - **Pytest**: Testing configuration
    - Test discovery patterns
    - Coverage reporting (term, html, xml)
    - Custom markers for test organization (slow, integration, unit, etc.)
    - Test paths and exclusions
  - **Mypy**: Type checking configuration
    - Python 3.9 target
    - Strict equality, no implicit optional
    - Ignore missing imports for third-party packages (mne, etc.)
  - **Coverage**: Code coverage configuration
    - Source paths, omit patterns
    - Exclude lines (pragma, abstract methods, etc.)

**Key features**:
- All dependencies from cc_saflow captured with reasonable version constraints
- Separate optional dependency groups for different use cases
- Comprehensive linting/testing/type-checking configurations
- Ready for `pip install -e .[all]` for development setup
- Supports Python 3.9-3.12

**Files created**:
- `pyproject.toml`

### ✅ Phase 1.5: Create setup script
**Status**: COMPLETE
**What was done**:
- ✅ Created `setup.sh` bash script with comprehensive environment setup
  - Virtual environment creation with configurable Python version
  - Package installation with multiple modes (basic, dev, hpc, all)
  - Automatic config.yaml creation from template
  - Project directory creation (logs, reports, etc.)
  - Development tools setup (pre-commit hooks if available)
  - Installation verification
  - Color-coded output for better readability
- ✅ Features:
  - Command-line options:
    - `--dev`: Install development dependencies
    - `--hpc`: Install HPC dependencies
    - `--all`: Install all optional dependencies
    - `--python`: Specify Python executable
    - `--force`: Force reinstall if venv exists
    - `--help`: Show help message
  - Preflight checks:
    - Python version validation (requires 3.9+)
    - venv module availability check
    - Existing environment handling
  - Package installation:
    - Upgrades pip, setuptools, wheel
    - Installs saflow in editable mode
    - Installs optional dependencies based on mode
  - Configuration setup:
    - Creates config.yaml from template
    - Warns about placeholder replacement
  - Verification:
    - Tests package imports
    - Checks configuration loading
    - Provides next steps summary
  - Made executable with proper permissions
- ✅ Error handling:
  - Exits on any error (set -e)
  - Clear error messages for missing dependencies
  - User prompts for existing environments

**Key features**:
- One-command environment setup
- Flexible installation modes for different use cases
- Comprehensive verification and helpful next steps
- Production-ready with proper error handling

**Files created**:
- `setup.sh` (executable)

### ✅ Phase 1.6: Create .gitignore
**Status**: COMPLETE
**What was done**:
- ✅ Created comprehensive `.gitignore` file covering all necessary exclusions
  - Configuration files: config.yaml, .env (keeps config.yaml.template)
  - Python artifacts: __pycache__/, *.pyc, *.egg-info/, build/, dist/
  - Virtual environments: env/, venv/, ENV/
  - Testing artifacts: .pytest_cache/, .coverage, htmlcov/, .tox/
  - Tool caches: .mypy_cache/, .ruff_cache/, .pylint.d/
  - IDE files: .vscode/, .idea/, *.swp, .ipynb_checkpoints
  - Project-specific:
    - logs/ (generated by pipeline)
    - reports/ (generated outputs)
    - data/, sourcedata/, derivatives/, processed/ (external data)
    - tmp/, temp/, *.tmp
  - MEG/EEG specific:
    - .mne/ cache
    - fs_subjects/ (FreeSurfer outputs)
    - *.fif, *.ds, *.sqd (large binary files)
  - Scientific computing: *.h5, *.hdf5, *.npy, *.npz, *.pkl, *.mat
  - SLURM: slurm-*.out, *.sbatch
  - Documentation: docs/_build/
  - OS-specific: .DS_Store, Thumbs.db, .directory
  - Backup files: *.bak, *.backup, *.old, *.orig
  - Compressed archives: *.zip, *.tar, *.gz, *.rar
- ✅ Exception rules to keep important files:
  - !config.yaml.template
  - !.gitkeep
  - !README.md, !LICENSE, !CHANGELOG.md, !AGENTS.md, !PROGRESS.md, !TASKS.md
  - !tests/data/, !tests/fixtures/

**Key features**:
- Comprehensive coverage for Python/MEG project
- Organized by category with clear section headers
- Protects user-specific and generated files
- Keeps templates and documentation
- Prevents accidental commits of large data files
- Cross-platform (macOS, Windows, Linux)

**Files created**:
- `.gitignore`

### ✅ Phase 1.7: Create task runner
**Status**: COMPLETE
**What was done**:
- ✅ Created `tasks.py` with comprehensive invoke tasks for development workflow
  - **Testing tasks**:
    - `invoke test`: Run pytest with coverage
    - `invoke test-fast`: Run only fast tests (exclude slow markers)
    - `invoke test-unit`: Run only unit tests
    - `invoke test-integration`: Run only integration tests
  - **Code quality tasks**:
    - `invoke lint [--fix]`: Run ruff linting with optional auto-fix
    - `invoke format [--check]`: Format code with ruff (or check only)
    - `invoke typecheck`: Run mypy type checking
    - `invoke check`: Run all quality checks (lint + format + typecheck)
  - **Cleaning tasks**:
    - `invoke clean`: Clean generated files (bytecode, cache, coverage, build)
    - `invoke clean-all`: Clean everything including logs and reports
  - **Environment tasks**:
    - `invoke setup [--mode=MODE]`: Run setup script with specified mode
    - `invoke rebuild [--mode=MODE]`: Clean and rebuild environment
  - **Documentation tasks**:
    - `invoke docs [--serve] [--port=PORT]`: Build and optionally serve docs
  - **Utility tasks**:
    - `invoke info`: Display project information
    - `invoke validate-config`: Validate configuration file
  - **Workflow tasks**:
    - `invoke precommit`: Run format + lint + test before committing
    - `invoke help`: Show available tasks (default)
- ✅ Added `invoke>=2.2.0` to pyproject.toml [dev] dependencies
- ✅ Features:
  - All tasks have comprehensive docstrings with examples
  - Tasks support command-line arguments with sensible defaults
  - Organized into logical sections (testing, quality, cleaning, etc.)
  - Color-coded output where appropriate (✓, ❌, ⚠️)
  - Pre-task dependencies (e.g., precommit runs format, lint, test)
  - Path-agnostic (uses Path objects, works cross-platform)

**Key features**:
- Complete development workflow automation
- Reduces cognitive load for common tasks
- Consistent command interface across team members
- Easy to extend with new tasks
- Self-documenting with `invoke --list`

**Usage examples**:
```bash
invoke --list              # Show all tasks
invoke test                # Run tests with coverage
invoke lint --fix          # Lint and auto-fix issues
invoke format              # Format all code
invoke check               # Run all quality checks
invoke precommit           # Pre-commit workflow
invoke clean               # Clean generated files
invoke setup --mode=dev    # Setup dev environment
invoke info                # Show project info
```

**Files created**:
- `tasks.py`

**Files modified**:
- `pyproject.toml` (added invoke to dev dependencies)

---

## Next Steps

### ✅ PHASE 1 COMPLETE - READY FOR CHECKPOINT

**All Phase 1 sub-phases complete:**
1. ~~Phase 1.1: Create project structure~~ ✅
2. ~~Phase 1.2: Implement configuration system~~ ✅
3. ~~Phase 1.3: Implement logging system~~ ✅
4. ~~Phase 1.4: Setup dependency management~~ ✅
5. ~~Phase 1.5: Create setup script~~ ✅
6. ~~Phase 1.6: Create .gitignore~~ ✅
7. ~~Phase 1.7: Create task runner~~ ✅

**🛑 CHECKPOINT: User review of Phase 1 foundation required before proceeding to Phase 2**

### After Phase 1 Approval: Begin Phase 2
Phase 2.1: Migrate saflow package utilities from cc_saflow
- Extract reusable functions from cc_saflow into saflow/ package
- Migrate saflow/utils.py → saflow/core.py
- Migrate saflow/data.py → saflow/data.py
- Migrate saflow/neuro.py → saflow/neuro.py
- Migrate saflow/behav.py → saflow/behav.py
- Migrate saflow/stats.py → saflow/stats.py
- Migrate saflow/visualization.py → saflow/viz.py

---

## Files Created This Session

1. `/home/hyruuk/GitHub/cocolab/saflow/config.yaml.template`
2. `/home/hyruuk/GitHub/cocolab/saflow/config.yaml`
3. `/home/hyruuk/GitHub/cololab/saflow/code/utils/config.py`
4. `/home/hyruuk/GitHub/cocolab/saflow/code/utils/paths.py`
5. `/home/hyruuk/GitHub/cocolab/saflow/code/utils/space.py`
6. `/home/hyruuk/GitHub/cocolab/saflow/code/utils/logging_config.py`
7. `/home/hyruuk/GitHub/cocolab/saflow/LICENSE` (copied)
8. `/home/hyruuk/GitHub/cocolab/saflow/code/__init__.py`
9. `/home/hyruuk/GitHub/cocolab/saflow/pyproject.toml`
10. `/home/hyruuk/GitHub/cocolab/saflow/setup.sh`
11. `/home/hyruuk/GitHub/cocolab/saflow/.gitignore`
12. `/home/hyruuk/GitHub/cocolab/saflow/tasks.py`
13. Multiple other `__init__.py` files

## Path Structure Updates

Updated path organization in both config files:
- `raw:` changed from `raw/` to `sourcedata/` (relative to data_root)
- `reports:`, `logs:`, `venv:` now use relative paths (`./reports`, `./logs`, `./env`) pointing to saflow project directory
- `slurm_output:` now `./logs/slurm` (within logs directory)
- `freesurfer_subjects_dir:` changed to `fs_subjects/` (relative to data_root, not absolute path)

---

## Issues Fixed

1. **Issue**: Created `saflow/__init__.py` as folder instead of file
   - **Fixed**: Removed directory, created proper file
2. **Issue**: Pre-created log subdirectories
   - **Fixed**: Removed them (scripts will create as needed)
3. **Issue**: setup.sh Python version detection failing with parsing errors
   - **Problem**: `python --version` output varied across systems, causing arithmetic errors at line 125
   - **Fixed**: Changed to use `sys.version_info` directly, added validation for integer version numbers
4. **Issue**: setup.sh venv check failing despite venv being installed
   - **Problem**: Using `--help` flag was unreliable for checking venv availability
   - **Fixed**: Changed to direct import test: `python -c "import venv"`

---

## Dependencies to Extract from cc_saflow

When creating pyproject.toml, pin these from `cc_saflow/requirements.txt`:
- mne
- numpy
- scipy
- pandas
- matplotlib
- seaborn
- scikit-learn
- mne-bids
- autoreject
- fooof
- pyyaml
- python-dotenv
- etc.

---

## Verification Checklist (End of Phase 1)

- [x] No hardcoded paths in any code
- [x] Configuration system loads and validates correctly
- [x] Logging system works (console + file)
- [x] pyproject.toml has all pinned dependencies
- [x] setup.sh creates working environment
- [x] .gitignore comprehensive
- [x] Task runner (invoke) functional

**✅ ALL PHASE 1 VERIFICATION ITEMS COMPLETE**
