# Scientific Analysis Project Guidelines

**Universal principles for computational reproducibility, publication readiness, and collaborative science.**

---

## Core Mission

1. **Computational reproducibility** - Anyone can rerun and verify results
2. **Publication readiness** - Code and outputs meet peer review standards
3. **Collaboration efficiency** - Clear, documented, accessible to all contributors

---

## General instructions

1. **Checkpoints and feedback** - When doing long series of tasks (e.g. refactoring a codebase, building a codebase from scratch etc...), interrupt yourself regularly to ask the user for tests and feedback, and offer them to compact their context window
2. **Ask questions to the user** - Take the habit of requiring information from the user when useful or necessary. If they are not knowledgeable about something, explain what they need to know to take an informed decision

---

## 1. Code Organization

- **Well-contained, simple functions**: The code should be made mostly of functions that are doing one and only one thing, usually in no more than ~10 lines of code. If more than that is needed, that is probably because the function is trying to do several things at once and it should be split further. Of course, sometimes the functions will need to be bigger than that but that's to be avoided when possible
- **Separate utilities from scripts**: Reusable functions in `utils.py`, orchestration scripts call them
- **One script, one purpose**: Each script accomplishes a single, well-defined analysis step
- **User-facing interface**: Provide task runner (invoke, Makefile, etc.) - users shouldn't run scripts directly
- **Descriptive naming**: `snake_case` for Python; functions start with verbs (`compute_`, `load_`, `generate_`)
- **No magic values**: Extract parameters to config files or constants
- **Document assumptions**: If code assumes data structure/format/preprocessing, state it explicitly
- **Type hints for public APIs**: All user-facing functions have type annotations
- **Google-style docstrings**: For all public functions
- **Package installability**: Project must be installable as a Python package using `pyproject.toml` (not setup.py), with tool configurations (ruff, pytest, etc.)
- **Use ruff and pytest**


---

## 2. Data Organization

### Data Integrity
- **Raw data is immutable**: Never overwrite or modify original data files
- **Separate input from output**: Never mix raw and processed data
- **Document data sources**: Where data came from, version, access date
- **BIDS compliance** for neuroimaging data

### Processed Data Storage
- **Location**: All processed outputs go to `{DATA_PATH}/processed/{script_name}/`
- **BIDS structure**: Maintain BIDS organization within processed folders when applicable
- **BIDS naming**: Follow BIDS conventions for filenames (e.g., `sub-01_ses-001_task-X_run-01_desc-Y_stat-z.nii.gz`)
- **Derivative metadata**: Include `dataset_description.json` in processed folders documenting pipeline

---

## 3. Configuration & Environment

### Centralized Configuration
- **Never hardcode paths**: All paths come from `config.yaml`
- **Default locations**: Scripts look in `./data/` and saves outputs in `./reports/` by default
- **Configurable paths**: Allow override via `config.yaml` for data and environment paths
- **Template first**: Provide `config.yaml.template` with `<PLACEHOLDER>` values
- **Validate config on startup**: Check config exists and is valid before running tasks
- **Gitignore config**: Actual `config.yaml` is always gitignored
- **Document all options**: Every config parameter has inline comment

### Environment Reproducibility
- **Use pyproject.toml**: Define package metadata, dependencies (pinned to exact versions), and tool configurations
- **Pin all dependencies**: Use exact versions (`package==X.Y.Z`) in `pyproject.toml` for reproducibility
- **Document system requirements**: Python version in `requires-python` field of `pyproject.toml`
- **Virtual environment**: Default location `./env/`, configurable via `config.yaml`
- **Provide setup script**: `setup.sh` for environment creation (`pip install -e .`)

---

## 4. Logging System

### Log Organization
- **Centralized logs**: All logs go to `./logs/` directory (gitignored)
- **SLURM outputs**: Direct SLURM output files to `./logs/{script}_{date}/slurm-*.out`
- **Structured naming**: `logs/{module}/{subject}_{session}_{timestamp}.log` or similar
- **Separate log levels**: Support verbosity control (WARNING, INFO, DEBUG)

### Logging Requirements
- **Use Python logging module**: Not print statements
- **Console + file output**: Summary to console, details to log file
- **Log file headers**: Script name, timestamp, key parameters
- **Progress tracking**: Show what's being processed and completion status
- **Error context**: When errors occur, log full context for debugging
- **Summary at end**: Report successes, skips, errors

### Log Content
- **What to log**:
  - Script start/end with timestamp
  - Input/output file paths
  - Key parameters used
  - Computation steps (at INFO level)
  - Skipped items (already exist)
  - Errors with full traceback
  - Final summary counts

---

## 5. Documentation (Auto-Updated)

**When you modify functionality, automatically update without asking:**
- **README.md** - Feature list, usage examples, workflow
- **TASKS.md** - Full documentation of all tasks/scripts with arguments and examples
- **Docstrings** - Function documentation matches behavior
- **CHANGELOG.md** - Log significant changes

**Exception**: Only skip if user explicitly says "don't update docs"

### Documentation Standards
- **Every script has a header**: Purpose, inputs, outputs, key parameters
- **Document statistical choices**: Test used, correction method, assumptions
- **Link to methods**: Reference papers for algorithms/pipelines
- **Document what failed**: Negative results prevent others from repeating mistakes

---

## 6. Version Control (Git)

### Agent Responsibilities
- **Do NOT commit code yourself**
- **List what changed**: After completing work, summarize files modified/created
- **Suggest commits**: Recommend what should be committed after user testing
- **Maintain .gitignore**: Keep .gitignore up to date at all times

### What Never Goes in Git
- Raw or processed data files (use data repositories)
- Generated outputs (figures, tables, stats)
- Secrets, passwords, API keys
- Virtual environments (`env/`, `.venv/`)
- User-specific config (`config.yaml`)
- Data (`data/`), results (`reports/`), and logs (`logs/`)
- Cache files (`__pycache__/`, `.ipynb_checkpoints/`, `.DS_Store`)

### Results Versioning
- **Link results to code**: Document which git commit produced publication outputs

---

## 7. Statistical Rigor

### Required Documentation
For every statistical test, document:
- **Test used**: Specific test name
- **Multiple comparison correction**: Method (FWE, FDR, cluster-based) and threshold
- **Sample size**: N subjects/trials/observations
- **Effect sizes**: Report alongside p-values
- **Software versions**: Packages and versions used

### Neuroimaging Standards (when applicable)
- **Preprocessing pipeline documented**: Pipeline name, version, parameters
- **Atlas/parcellation specified**: Name, version, resolution
- **Smoothing kernel**: FWHM in mm
- **Statistical threshold**: Cluster-forming + extent thresholds
- **BIDS naming** for all outputs

---

## 8. Automated Analysis & Reporting

### Pipeline Requirements
- **End-to-end automation**: Single command runs full analysis (or documents manual steps)
- **Skip existing outputs**: Check for completed files, don't recompute
- **Progress tracking**: Use logging to show what's running
- **Informative errors**: Tell user what's wrong AND how to fix it

### Computational Job Management (if requested)
- **Support cluster computing**: Scripts work on local and HPC/SLURM
- **Parallel processing**: Use available cores for independent computations
- **Resource documentation**: Document typical runtime and memory needs
- **Batch processing**: Handle multiple subjects/conditions automatically

### Report Generation
- **Programmatic figures**: All plots generated by code, not manual editing
- **Consistent styling**: Define plot parameters once, reuse everywhere
- **Publication-ready**: High-resolution, proper labels, colorblind-friendly
- **Automated tables**: Generate from analysis outputs, not manual entry

---

## 9. Provenance & Reproducibility

### Track Everything
- **Parameters logged**: Save analysis parameters as JSON/YAML with outputs
- **Git commit hash**: Log current commit hash for reproducibility
- **Random seeds**: Set and document all random seeds
- **Timestamps**: Include datetime in output filenames or metadata

### Metadata Files
Save JSON sidecar with each major output:
```json
{
  "description": "Statistical map for condition X",
  "script": "code/glm/compute_glm.py",
  "git_commit": "a7f3e9c",
  "date": "2024-01-15T14:30:22",
  "parameters": {
    "parameter1": "value1",
    "parameter2": 42
  },
  "software": {
    "package1": "1.2.3",
    "python": "3.9.16"
  }
}
```

### Environment Logging
- **Log package versions**: Save environment snapshot with results
- **Document hardware**: GPU, RAM if relevant to reproducibility

---

## 10. Validation & Testing

### Input Validation
- **Check file existence**: Fail early with clear messages
- **Validate data format**: Check structure before processing
- **Sensible defaults**: Provide defaults for optional parameters
- **User-friendly errors**: What's wrong + how to fix it

### Quality Checks
- **Automated QC plots**: Generate diagnostic visualizations
- **Sanity checks**: Results in expected range? Flag anomalies
- **Test on subset**: Verify pipeline on small data before full run

---

## 11. Collaboration & Sharing

### Code Sharing
- **Clear README**: Installation, configuration, basic usage
- **Include LICENSE**: Recommend open-source licenses (MIT, GPL, BSD) but do not enforce them
- **CITATION.cff**: Proper attribution information

### Data Sharing
- **Use repositories**: OpenNeuro (BIDS), OSF, Zenodo, Dryad
- **Anonymize**: Scan for PHI/PII and remove before sharing
- **DOI for datasets**: Permanent, citable identifiers when available

### Publication Standards
- **Code availability**: Link to repository in manuscript
- **Data availability**: How to access data
- **Reproducibility**: Provide commands for key findings

---

## Workflow Checklist

Before asking user to commit or publish:

**Code:**
- [ ] No hardcoded paths - all from config or defaults
- [ ] Functions have docstrings and type hints
- [ ] Descriptive naming (no `x`, `df2`, `tmp`)
- [ ] Utilities separated from scripts
- [ ] User-facing tasks provided
- [ ] Logging implemented (not print statements)
- [ ] Dead code removed

**Documentation:**
- [ ] README.md up to date
- [ ] TASKS.md up to date
- [ ] CHANGELOG.md up to date

**Reproducibility:**
- [ ] Package defined in pyproject.toml
- [ ] Dependencies pinned in pyproject.toml (exact versions)
- [ ] Random seeds set (unless required otherwise)
- [ ] Parameters + git hash + script path logged with outputs
- [ ] Config template provided/updated

**Data:**
- [ ] Raw data separate from derivatives
- [ ] Processed data in {DATA_PATH}/processed/{script}/
- [ ] BIDS compliance (if neuroimaging)
- [ ] Metadata saved with outputs

**Version Control:**
- [ ] .gitignore up to date
- [ ] No data/secrets/large files in git
- [ ] Changes summarized for user to commit

---

## Critical Don'ts

- **NEVER** hardcode paths or parameters
- **NEVER** modify raw data - always create derivatives
- **NEVER** commit data, secrets, or absolute paths
- **NEVER** skip documentation updates when functionality changes
- **NEVER** use unclear variable names (`x`, `tmp`, `data2`)
- **NEVER** report p-values without correction method
- **NEVER** generate results without logging git hash + parameters + script path
- **NEVER** commit code on behalf of user
- **NEVER** use print() for logging - use logging module

---

## Project Template

```
project/
├── data/                           # Default data path (configurable in config.yaml)
│   ├── raw/                        # Original, immutable source data
│   ├── derivatives/                # Preprocessed data (e.g., fMRIPrep outputs)
│   └── processed/                  # Analysis outputs organized by script
│       └── {script_name}/          # Each analysis script has its own folder
│           ├── {subject}/          # Maintain BIDS structure within
│           │   └── {session}/
│           └── dataset_description.json
├── code/                           # All analysis code
│   ├── {module}/                   # Analysis modules (e.g. glm, mvpa, etc.)
│   │   ├── {analysis_script}.py
│   │   └── utils.py
│   └── utils/                      # Shared utilities across modules
├── reports/                        # Final outputs for publication/sharing
│   ├── figures/
│   ├── tables/
│   └── statistics/
├── logs/                           # All execution logs and SLURM outputs (gitignored)
│   ├── {module}/                   # Organized by module
│   └── {script}_{date}/            # SLURM batch logs
│       └── slurm-*.out
├── env/ or .venv/                  # Default venv path (configurable, gitignored)
├── config/                         # Configuration files
├── docs/                           # Documentation
├── config.yaml.template            # Config template with <PLACEHOLDER> values
├── config.yaml                     # User config (gitignored)
├── pyproject.toml                  # Package metadata, pinned dependencies, tool configs
├── setup.sh                        # Environment setup script
├── tasks.py                        # Task runner (invoke/Makefile)
├── .gitignore
├── README.md
├── TASKS.md
├── CHANGELOG.md
└── LICENSE
```

### Data Organization Example (Neuroimaging)
```
data/
├── raw/                            # BIDS raw dataset
│   ├── sub-01/
│   │   └── ses-001/
│   │       ├── anat/
│   │       └── func/
│   └── dataset_description.json
├── derivatives/                    # Preprocessed (e.g., fMRIPrep)
│   └── fmriprep/
│       ├── sub-01/
│       └── dataset_description.json
└── processed/                      # Analysis outputs
    ├── glm_run_level/              # From compute_glm_run_level.py
    │   ├── sub-01/
    │   │   └── ses-001/
    │   │       ├── z_maps/
    │   │       │   └── sub-01_ses-001_task-X_run-01_contrast-Y_stat-z.nii.gz
    │   │       └── beta_maps/
    │   └── dataset_description.json
    └── mvpa_session_level/         # From compute_mvpa_session.py
        ├── sub-01/
        │   └── ses-001/
        │       └── sub-01_ses-001_task-X_desc-accuracy_stats.json
        └── dataset_description.json
```

---

**These are living guidelines. Adapt to project needs, and always prioritize reproducibility and transparency.**
