# Saflow Pipeline Task Reference

This document describes the invoke tasks for running the saflow MEG analysis pipeline.

**Quick reference**: Run `invoke --list` to see all available tasks.

---

## Task Namespaces

Tasks are organized into namespaces:

| Namespace | Description |
|-----------|-------------|
| `pipeline.*` | Main pipeline stages (BIDS, preprocess, source-recon) |
| `features.*` | Feature extraction (FOOOF, PSD, complexity) |
| `analysis.*` | Statistical analysis and classification |
| `dev.*` | Development tools (tests, linting, cleaning) |
| `dev.check.*` | Data quality and validation checks |
| `env.*` | Environment setup and management |
| `viz.*` | Visualization and figures |

---

## Pipeline Tasks

### `invoke pipeline.validate-inputs`

Validate that raw input data is present and complete before running pipeline.

**What it checks:**
- Raw MEG data directory exists (`data_root/sourcedata/meg/`)
- Behavioral logfiles directory exists (`data_root/sourcedata/behav/`)
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
invoke pipeline.validate-inputs

# Verbose output with file listings
invoke pipeline.validate-inputs --verbose
```

---

### `invoke pipeline.bids`

Run BIDS conversion (Stage 0 of pipeline).

**What it does:**
1. Finds all CTF MEG datasets (`.ds` files) in raw directory
2. Converts to BIDS format using mne-bids
3. For gradCPT task runs:
   - Loads behavioral logfiles (runs 1-6, ignoring run 0 and typos)
   - Computes VTC (variability of reaction times)
   - Enriches events.tsv with trial indices, VTC, RT, and performance
4. Writes empty-room noise recordings
5. Saves provenance metadata

**Run Matching:**
- MEG run 02 → behavioral run 1
- MEG run 03 → behavioral run 2
- ... up to MEG run 07 → behavioral run 6
- Run 0 (practice) and files with unexpected run numbers are ignored

**Arguments:**
- `--input-dir PATH`: Override raw data directory from config
- `--output-dir PATH`: Override BIDS output directory from config
- `--subjects "ID1 ID2 ..."`: Process specific subjects only (space-separated)
- `--log-level LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `--dry-run`: Validate inputs without processing files

**Examples:**
```bash
# Basic usage (uses paths from config.yaml)
invoke pipeline.bids

# Process specific subjects only
invoke pipeline.bids --subjects "04 05 06"

# Dry run to validate before processing
invoke pipeline.bids --dry-run
```

---

### `invoke pipeline.preprocess`

Run MEG preprocessing (Stage 1 of pipeline).

**What it does:**
1. Loads raw BIDS data
2. Applies gradient compensation (grade 3)
3. Applies bandpass filtering (0.1-200 Hz) and notch filtering (60 Hz)
4. Creates epochs for all events
5. Runs AutoReject to identify bad epochs for ICA
6. Fits ICA and removes ECG/EOG artifacts
7. Runs AutoReject with interpolation on ICA-cleaned data
8. Saves preprocessed data, epochs, reports, and metadata

**Arguments:**
- `--subject ID`: Subject ID to process
- `--runs "R1 R2 ..."`: Run numbers to process (default: all task runs)
- `--log-level LEVEL`: Set logging level
- `--skip-existing` / `--no-skip-existing`: Skip/reprocess existing files
- `--slurm`: Submit jobs to SLURM cluster
- `--dry-run`: Generate SLURM scripts without submitting

**Examples:**
```bash
# Local execution
invoke pipeline.preprocess --subject=04
invoke pipeline.preprocess --subject=04 --runs="02 03"

# SLURM execution
invoke pipeline.preprocess --subject=04 --slurm
invoke pipeline.preprocess --slurm  # All subjects
```

---

### `invoke pipeline.source-recon`

Run source reconstruction (Stage 2 of pipeline).

**What it does:**
1. Computes coregistration (MEG ↔ MRI)
2. Sets up source space and BEM model
3. Computes forward solution
4. Applies inverse operator
5. Morphs to fsaverage template

**Arguments:**
- `--subject ID`: Subject ID to process
- `--runs "R1 R2 ..."`: Run numbers to process
- `--input-type TYPE`: `continuous` or `epochs`
- `--processing STATE`: `clean`, `ica`, or `icaar`
- `--slurm`: Submit to SLURM
- `--dry-run`: Generate scripts without submitting

**Examples:**
```bash
invoke pipeline.source-recon --subject=04 --runs="02 03"
invoke pipeline.source-recon --subject=04 --slurm
```

---

## Feature Extraction Tasks

### `invoke features.fooof`

Extract FOOOF aperiodic parameters and corrected PSDs.

**Arguments:**
- `--subject ID`: Subject to process (default: all)
- `--space SPACE`: `sensor`, `source`, or `atlas`
- `--slurm`: Submit to SLURM

**Examples:**
```bash
invoke features.fooof --subject=04
invoke features.fooof --space=source --slurm
```

---

### `invoke features.psd`

Extract power spectral density features.

**Arguments:**
- `--subject ID`: Subject to process (default: all)
- `--space SPACE`: `sensor`, `source`, or `atlas`
- `--slurm`: Submit to SLURM

---

### `invoke features.complexity`

Extract complexity and entropy measures.

**Arguments:**
- `--subject ID`: Subject to process (default: all)
- `--space SPACE`: `sensor`, `source`, or `atlas`
- `--slurm`: Submit to SLURM

---

### `invoke features.all`

Extract all feature types (FOOOF, PSD, complexity).

**Arguments:**
- `--subject ID`: Subject to process (default: all)
- `--space SPACE`: `sensor`, `source`, or `atlas`
- `--slurm`: Submit to SLURM

---

## Analysis Tasks

### `invoke analysis.statistics`

Run group-level statistical analysis (IN vs OUT attentional states).

**Arguments:**
- `--feature-type TYPE`: Feature to analyze (e.g., `fooof_exponent`, `psd_alpha`)
- `--space SPACE`: Analysis space (`sensor`, `source`, `atlas`)
- `--test TEST`: Statistical test (`paired_ttest`, `permutation`)
- `--corrections CORR`: Correction methods (`fdr`, `bonferroni`, `tmax`)
- `--slurm`: Submit to SLURM

**Examples:**
```bash
invoke analysis.statistics --feature-type=fooof_exponent --space=sensor
invoke analysis.statistics --feature-type=psd_alpha --corrections="fdr bonferroni"
```

---

### `invoke analysis.classify`

Run classification analysis (decode IN vs OUT from neural features).

**Arguments:**
- `--features FEAT`: Feature type(s) to use
- `--clf CLF`: Classifier (`lda`, `svm`, `rf`, `xgboost`)
- `--cv CV`: Cross-validation strategy (`logo`, `stratified`)
- `--space SPACE`: Analysis space
- `--slurm`: Submit to SLURM

**Examples:**
```bash
invoke analysis.classify --features=fooof_exponent --clf=lda
invoke analysis.classify --features="psd_alpha psd_theta" --clf=svm
```

---

## Development Tasks

### `invoke dev.check.dataset`

Check dataset completeness - which files exist for each subject across all pipeline stages.

**What it shows:**
- Raw MEG runs (8 expected: runs 01-08)
- Behavioral files (6 expected: runs 1-6, with extras flagged)
- MRI availability
- BIDS conversion status
- Preprocessing status
- Feature extraction status

**Arguments:**
- `--verbose`: Show detailed missing data information

**Examples:**
```bash
invoke dev.check.dataset
invoke dev.check.dataset --verbose
```

**Output format:**
```
Subj   | MEG Runs | Behav    | MRI  | BIDS | Preproc | FOOOF | PSD | Cmplx
04     | 8/8      | 6/6      | Y    | 6/6  | 6/6     | Y     | Y   | Y
14     | 8/8      | 5/6 (+2) | Y    | -    | -       | -     | -   | -
```

The `(+N)` notation indicates extra files (run 0 practice files, duplicates, or typos).

---

### `invoke dev.check.qc`

Run data quality checks on BIDS MEG data.

**What it checks:**
- Inter-stimulus intervals (ISI)
- Response detection and rates
- Channel quality
- Event validation
- Data integrity

**Arguments:**
- `--subject ID`: Check specific subject (default: all subjects)
- `--runs RUNS`: Specific runs to check
- `--output-dir DIR`: Output directory for reports
- `--verbose`: Detailed output

**Examples:**
```bash
invoke dev.check.qc                    # Check all subjects
invoke dev.check.qc --subject=04       # Check specific subject
invoke dev.check.qc --verbose          # Detailed output
```

---

### `invoke dev.check.code`

Run code quality checks (linting, formatting, type checking).

---

### `invoke dev.test`

Run tests with pytest.

**Arguments:**
- `--verbose`: Enable verbose output
- `--coverage` / `--no-coverage`: Generate coverage report

---

### `invoke dev.test-fast`

Run only fast tests (excludes tests marked as slow).

---

### `invoke dev.clean`

Clean generated files (caches, build artifacts).

---

### `invoke dev.precommit`

Run pre-commit checks (code quality + tests).

---

## Environment Tasks

### `invoke env.info`

Display project information and configuration status.

---

### `invoke env.setup`

Run setup script to create development environment.

---

### `invoke env.rebuild`

Clean and rebuild the development environment.

---

### `invoke env.validate-config`

Validate configuration file syntax and required fields.

---

## Visualization Tasks

### `invoke viz.behavior`

Generate behavioral analysis figure (VTC, RT distributions, etc.).

---

## SLURM Integration (HPC)

Pipeline tasks support distributed execution on HPC clusters via SLURM.

### Using SLURM

Add `--slurm` to any pipeline task to submit jobs to the cluster:

```bash
# Process all subjects in parallel
invoke pipeline.preprocess --slurm

# Process specific subject
invoke pipeline.preprocess --subject=04 --slurm

# Dry run: generate scripts without submitting
invoke pipeline.preprocess --slurm --dry-run
```

### Configuration

SLURM settings are in `config.yaml`:

```yaml
computing:
  slurm:
    enabled: true
    account: def-kjerbi
    partition: standard
    email: user@example.com

    preprocessing:
      cpus: 12
      mem: 32G
      time: "12:00:00"
```

### Job Management

```bash
squeue -u $USER              # View your jobs
sacct -j JOBID               # Check job status
scancel JOBID                # Cancel job
scancel -u $USER             # Cancel all jobs
```

---

## Notes

- All tasks support `--help` for detailed options
- Logs are saved to `logs/{stage}/`
- Use `--skip-existing` (default) to avoid reprocessing
- Configuration is loaded from `config.yaml`
