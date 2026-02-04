# Saflow Pipeline Task Reference

This document describes the invoke tasks for running the saflow MEG analysis pipeline.

**Quick reference**: Run `invoke --list` to see all available tasks.

---

## Task Namespaces

Tasks are organized into namespaces:

| Namespace | Description |
|-----------|-------------|
| `dev.check.*` | Data validation (dataset, qc, code) |
| `dev.*` | Development tasks (test, clean, precommit) |
| `env.*` | Environment setup and management |
| `get.*` | Data downloads (atlases) |
| `pipeline.*` | Main pipeline stages (BIDS, preprocess, source-recon, atlas) |
| `pipeline.features.*` | Feature extraction (psd, fooof, complexity, all) |
| `analysis.*` | Statistical analysis (statistics, classify) |
| `analysis.stats.*` | Dedicated statistics tasks (complexity, fooof) |
| `viz.*` | Visualization (stats, behavior) |

---

## Data Download Tasks

### `invoke get.atlases`

Download FreeSurfer atlas annotation files required for source-level parcellation.

**What it downloads:**
- Destrieux atlas (aparc.a2009s): 148 ROIs
- Schaefer 100 parcels (7 Networks)
- Schaefer 200 parcels (7 Networks)
- Schaefer 400 parcels (7 Networks)

Files are downloaded to `{data_root}/{freesurfer_subjects_dir}/fsaverage/label/` as specified in `config.yaml`.

**Arguments:** None

**Examples:**
```bash
invoke get.atlases
```

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
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-root` | PATH | config | Override data root from config |
| `--verbose` | flag | false | Show detailed file listings |

**Examples:**
```bash
invoke pipeline.validate-inputs
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
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input-dir` | PATH | config | Override raw data directory from config |
| `--output-dir` | PATH | config | Override BIDS output directory from config |
| `--subjects` | string | all | Space-separated subject IDs (e.g., "04 05 06") |
| `--log-level` | choice | INFO | DEBUG, INFO, WARNING, ERROR |
| `--dry-run` | flag | false | Validate inputs without processing files |

**Examples:**
```bash
invoke pipeline.bids
invoke pipeline.bids --subjects "04 05 06"
invoke pipeline.bids --dry-run
invoke pipeline.bids --log-level DEBUG
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
7. Runs second AutoReject pass (fit only) for bad epoch detection
8. Saves preprocessed data, epochs, reports, and metadata

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--subject` | string | required* | Subject ID to process (required for local) |
| `--runs` | string | all | Space-separated run numbers (e.g., "02 03 04") |
| `--bids-root` | PATH | config | Override BIDS root directory |
| `--log-level` | choice | INFO | DEBUG, INFO, WARNING, ERROR |
| `--skip-existing` | flag | true | Skip if output files exist (default) |
| `--crop` | float | none | Crop to first N seconds (for testing) |
| `--skip-second-ar` | flag | false | Skip second AutoReject pass |
| `--slurm` | flag | false | Submit jobs to SLURM cluster |
| `--dry-run` | flag | false | Generate SLURM scripts without submitting |

*Required for local execution; optional with `--slurm` (processes all subjects)

**Examples:**
```bash
# Local execution
invoke pipeline.preprocess --subject=04
invoke pipeline.preprocess --subject=04 --runs="02 03"
invoke pipeline.preprocess --subject=04 --crop=50  # Quick test

# SLURM execution
invoke pipeline.preprocess --slurm              # All subjects
invoke pipeline.preprocess --subject=04 --slurm # Single subject
invoke pipeline.preprocess --slurm --dry-run    # Preview jobs
```

---

### `invoke pipeline.source-recon`

Run source reconstruction (Stage 2 of pipeline).

**What it does:**
1. Loads preprocessed continuous data
2. Computes coregistration (MEG ↔ MRI)
3. Sets up source space and BEM model
4. Computes forward solution
5. Computes noise covariance from empty-room recording
6. Applies inverse operator (dSPM method)
7. Morphs source estimates to fsaverage template
8. Saves morphed source estimates (.h5 format)

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--subject` | string | all | Subject ID (processes all if not specified) |
| `--runs` | string | all | Space-separated run numbers |
| `--bids-root` | PATH | config | Override BIDS root directory |
| `--log-level` | choice | INFO | DEBUG, INFO, WARNING, ERROR |
| `--skip-existing` | flag | true | Skip if output files exist |
| `--slurm` | flag | false | Submit jobs to SLURM cluster |
| `--dry-run` | flag | false | Generate SLURM scripts without submitting |

**Examples:**
```bash
# Local execution
invoke pipeline.source-recon                    # All subjects
invoke pipeline.source-recon --subject=04       # Single subject
invoke pipeline.source-recon --subject=04 --runs="02 03"

# SLURM execution
invoke pipeline.source-recon --slurm
invoke pipeline.source-recon --slurm --dry-run
```

---

### `invoke pipeline.atlas`

Apply atlas parcellation to morphed source estimates (Stage 2b).

**What it does:**
1. Loads morphed source estimates from fsaverage space
2. Applies cortical parcellations to extract ROI time series
3. Averages source activity within each parcel/region
4. Saves ROI-level time series (.npz format) with metadata

**Default atlases:**
- aparc.a2009s (Destrieux, 148 ROIs)
- schaefer_100 (Schaefer 100 parcels, 7 Networks)
- schaefer_200 (Schaefer 200 parcels, 7 Networks)
- schaefer_400 (Schaefer 400 parcels, 7 Networks)

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--subject` | string | all | Subject ID (processes all if not specified) |
| `--runs` | string | all | Space-separated run numbers |
| `--atlases` | string | config | Space-separated atlas names (e.g., "aparc.a2009s schaefer_100") |
| `--skip-existing` | flag | true | Skip if output files exist |
| `--slurm` | flag | false | Submit jobs to SLURM cluster |
| `--dry-run` | flag | false | Generate SLURM scripts without submitting |

**Examples:**
```bash
# Local execution
invoke pipeline.atlas                           # All subjects, all atlases
invoke pipeline.atlas --subject=04              # Single subject
invoke pipeline.atlas --subject=04 --atlases="aparc.a2009s"

# SLURM execution
invoke pipeline.atlas --slurm
invoke pipeline.atlas --slurm --dry-run
```

---

## Feature Extraction Tasks

All feature extraction tasks are under `pipeline.features.*`.

### `invoke pipeline.features.psd`

Extract power spectral density features using Welch's method.

**What it computes:**
- Welch PSD estimates per trial/epoch
- Band power for configured frequency bands
- Saves with IN/OUT classification metadata

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--subject` | string | required* | Subject ID to process |
| `--runs` | string | all | Space-separated run numbers |
| `--space` | string | sensor | Analysis space: `sensor`, or atlas name (e.g., `aparc.a2009s`) |
| `--skip-existing` | flag | true | Skip if output files exist |
| `--slurm` | flag | false | Submit jobs to SLURM cluster |
| `--dry-run` | flag | false | Generate SLURM scripts without submitting |

*Required for local execution

**Examples:**
```bash
invoke pipeline.features.psd --subject=04
invoke pipeline.features.psd --subject=04 --space=aparc.a2009s
invoke pipeline.features.psd --slurm
```

---

### `invoke pipeline.features.fooof`

Extract FOOOF (specparam) aperiodic parameters and corrected PSDs.

**What it computes:**
- Aperiodic parameters (exponent, offset, knee if enabled)
- Goodness of fit metrics (r_squared, error)
- Aperiodic-corrected PSDs (periodic component only)

FOOOF parameters (freq_range, aperiodic_mode, etc.) are configured in `config.yaml`.

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--subject` | string | required* | Subject ID to process |
| `--runs` | string | all | Space-separated run numbers |
| `--space` | string | sensor | Analysis space: `sensor`, or atlas name |
| `--skip-existing` | flag | true | Skip if output files exist |
| `--slurm` | flag | false | Submit jobs to SLURM cluster |
| `--dry-run` | flag | false | Generate SLURM scripts without submitting |

*Required for local execution

**Examples:**
```bash
invoke pipeline.features.fooof --subject=04
invoke pipeline.features.fooof --subject=04 --space=schaefer_100
invoke pipeline.features.fooof --slurm
```

---

### `invoke pipeline.features.complexity`

Extract complexity and entropy measures.

**What it computes:**
- Lempel-Ziv Complexity (LZC)
- Entropy measures: permutation, spectral, sample, approximate, SVD
- Fractal dimensions: Higuchi, Petrosian, Katz, DFA

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--subject` | string | required* | Subject ID to process |
| `--runs` | string | all | Space-separated run numbers |
| `--space` | string | sensor | Analysis space: `sensor`, or atlas name |
| `--complexity-type` | string | "lzc entropy fractal" | Space-separated types to compute |
| `--overwrite` | flag | false | Overwrite existing files |
| `--slurm` | flag | false | Submit jobs to SLURM cluster |
| `--dry-run` | flag | false | Generate SLURM scripts without submitting |

*Required for local execution

**Examples:**
```bash
invoke pipeline.features.complexity --subject=04
invoke pipeline.features.complexity --subject=04 --complexity-type="lzc entropy"
invoke pipeline.features.complexity --slurm
```

---

### `invoke pipeline.features.all`

Extract all feature types (PSD, FOOOF, complexity) in sequence.

**Order:** PSD → FOOOF → Complexity (FOOOF depends on PSD)

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--subject` | string | required* | Subject ID to process |
| `--runs` | string | all | Space-separated run numbers |
| `--space` | string | sensor | Analysis space: `sensor`, or atlas name |
| `--overwrite` | flag | false | Overwrite existing files |
| `--slurm` | flag | false | Submit jobs to SLURM cluster |
| `--dry-run` | flag | false | Generate SLURM scripts without submitting |

*Required for local execution

**Examples:**
```bash
invoke pipeline.features.all --subject=04
invoke pipeline.features.all --subject=04 --space=aparc.a2009s
invoke pipeline.features.all --slurm
```

---

## Analysis Tasks

### `invoke analysis.statistics`

Run group-level statistical analysis (IN vs OUT attentional states).

**Feature types:**
- FOOOF: `fooof_exponent`, `fooof_offset`, `fooof_knee`, `fooof_r_squared`
- PSD: `psd_delta`, `psd_theta`, `psd_alpha`, `psd_lobeta`, `psd_hibeta`, `psd_gamma1`, etc.
- Complexity: `complexity` (uses dedicated script)

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--feature-type` | string | required | Feature to analyze |
| `--space` | string | sensor | Analysis space: `sensor`, or atlas name |
| `--test` | string | paired_ttest | Statistical test: `paired_ttest`, `independent_ttest`, `permutation` |
| `--corrections` | string | "fdr bonferroni" | Space-separated correction methods |
| `--alpha` | float | 0.05 | Significance threshold |
| `--n-permutations` | int | 10000 | Number of permutations (for permutation tests) |
| `--visualize` | flag | false | Generate visualization figures |
| `--slurm` | flag | false | Submit to SLURM (not yet implemented) |
| `--dry-run` | flag | false | Preview without running |

**Examples:**
```bash
invoke analysis.statistics --feature-type=fooof_exponent
invoke analysis.statistics --feature-type=psd_alpha --visualize
invoke analysis.statistics --feature-type=psd_theta --test=permutation --n-permutations=5000
invoke analysis.statistics --feature-type=complexity
```

---

### `invoke analysis.stats.complexity`

Run paired t-tests on complexity measures (IN vs OUT).

**What it analyzes:**
- LZC (Lempel-Ziv Complexity)
- Entropy measures (permutation, spectral, sample, approximate, SVD)
- Fractal dimensions (Higuchi, Petrosian, Katz, DFA)

**Outputs:**
- Topographic figure: `reports/figures/complexity_ttest_{correction}.png`
- Numerical results: `{data_root}/features/statistics_{space}/complexity_ttest_results.npz`

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--space` | string | sensor | Analysis space |
| `--correction` | string | fdr | Correction method: `fdr`, `bonferroni`, `tmax`, `none` |
| `--alpha` | float | 0.05 | Significance threshold |
| `--n-permutations` | int | 1000 | Number of permutations (for permutation correction) |

**Examples:**
```bash
invoke analysis.stats.complexity
invoke analysis.stats.complexity --correction=bonferroni
invoke analysis.stats.complexity --correction=tmax --n-permutations=5000
invoke analysis.stats.complexity --correction=none --alpha=0.01
```

---

### `invoke analysis.stats.fooof`

Run paired t-tests on FOOOF parameters (IN vs OUT).

**What it analyzes:**
- Aperiodic exponent (1/f slope)
- Aperiodic offset
- Model fit (r_squared)

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--space` | string | sensor | Analysis space |
| `--alpha` | float | 0.05 | Significance threshold |

**Examples:**
```bash
invoke analysis.stats.fooof
invoke analysis.stats.fooof --alpha=0.01
```

---

### `invoke analysis.classify`

Run classification analysis (decode IN vs OUT from neural features).

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--features` | string | required | Space-separated feature types |
| `--clf` | string | lda | Classifier: `lda`, `svm`, `rf`, `logistic` |
| `--cv` | string | logo | Cross-validation: `logo` (leave-one-group-out), `stratified`, `group` |
| `--space` | string | sensor | Analysis space |
| `--n-permutations` | int | 1000 | Permutations for significance testing |
| `--no-balance` | flag | false | Disable class balancing within subjects |
| `--visualize` | flag | false | Generate visualization figures |
| `--slurm` | flag | false | Submit to SLURM (not yet implemented) |
| `--dry-run` | flag | false | Preview without running |

**Examples:**
```bash
invoke analysis.classify --features=fooof_exponent
invoke analysis.classify --features="fooof_exponent psd_alpha" --clf=svm
invoke analysis.classify --features=psd_theta --cv=stratified --visualize
```

---

## Visualization Tasks

### `invoke viz.stats`

Visualize saved statistical results.

Loads previously computed statistics and displays summary. The statistics must have been computed first using `analysis.stats.*` tasks.

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--feature-type` | string | complexity | Feature type to visualize |
| `--space` | string | sensor | Analysis space |
| `--alpha` | float | 0.05 | Significance threshold for display |
| `--show` | flag | false | Open figure in viewer |

**Examples:**
```bash
invoke viz.stats
invoke viz.stats --feature-type=complexity --show
invoke viz.stats --feature-type=fooof
```

---

### `invoke viz.behavior`

Generate behavioral analysis figure (VTC, RT distributions, IN/OUT zones).

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--subject` | string | 07 | Subject ID |
| `--run` | string | 4 | Run number |
| `--inout-bounds` | string | "25 75" | Space-separated percentile bounds for IN/OUT zones |
| `--output` | PATH | none | Output file path |
| `--verbose` | flag | false | Verbose output |

**Examples:**
```bash
invoke viz.behavior
invoke viz.behavior --subject=04 --run=3
invoke viz.behavior --inout-bounds="10 90"
invoke viz.behavior --output=reports/figures/behavior_sub04.png
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
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--verbose` | flag | false | Show detailed missing data information |

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
- Inter-stimulus intervals (ISI) consistency and outliers
- Response detection and rates
- Reaction times
- Channel quality (flat, noisy, saturated)
- Event/trigger counts and timing
- Data integrity (sampling rate, duration)

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--subject` | string | all | Subject ID (checks all if not specified) |
| `--runs` | string | all | Space-separated run numbers |
| `--output-dir` | PATH | reports/qc | Output directory for reports |
| `--verbose` | flag | false | Detailed output |

**Examples:**
```bash
invoke dev.check.qc                         # Check all subjects
invoke dev.check.qc --subject=04            # Check specific subject
invoke dev.check.qc --subject=04 --runs="02 03 04"
invoke dev.check.qc --verbose
```

---

### `invoke dev.check.code`

Run code quality checks (linting, formatting, type checking).

**What it runs:**
1. Ruff linting (style, errors, complexity)
2. Ruff formatting (code style consistency)
3. Mypy type checking (type annotations)

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--fix` | flag | false | Auto-fix issues where possible |

**Examples:**
```bash
invoke dev.check.code
invoke dev.check.code --fix
```

---

### `invoke dev.test`

Run tests with pytest.

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--verbose` | flag | false | Enable verbose output |
| `--coverage` | flag | true | Generate coverage report |
| `--markers` | string | none | Pytest markers to filter tests (e.g., "not slow") |

**Examples:**
```bash
invoke dev.test
invoke dev.test --verbose
invoke dev.test --markers="not slow"
invoke dev.test --no-coverage
```

---

### `invoke dev.test-fast`

Run only fast tests (excludes tests marked as slow).

**Arguments:** None

**Examples:**
```bash
invoke dev.test-fast
```

---

### `invoke dev.clean`

Clean generated files (caches, build artifacts).

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--bytecode` | flag | true | Clean Python bytecode files |
| `--cache` | flag | true | Clean cache directories |
| `--coverage` | flag | true | Clean coverage reports |
| `--build` | flag | true | Clean build artifacts |
| `--logs` | flag | false | Clean log files |

**Examples:**
```bash
invoke dev.clean
invoke dev.clean --logs
invoke dev.clean --no-bytecode --no-cache
```

---

### `invoke dev.precommit`

Run pre-commit checks (code quality checks + tests).

**Arguments:** None

**Examples:**
```bash
invoke dev.precommit
```

---

## Environment Tasks

### `invoke env.setup`

Run setup script to create development environment.

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | string | basic | Setup mode: `basic`, `dev`, `all` |
| `--python` | string | python3.9 | Python executable to use |
| `--force` | flag | false | Force recreation of environment |

**Examples:**
```bash
invoke env.setup
invoke env.setup --mode=dev
invoke env.setup --mode=all --force
invoke env.setup --python=python3.11
```

---

### `invoke env.info`

Display project information and configuration status.

**Arguments:** None

**Examples:**
```bash
invoke env.info
```

---

### `invoke env.validate-config`

Validate configuration file syntax and required fields.

**Arguments:** None

**Examples:**
```bash
invoke env.validate-config
```

---

### `invoke env.rebuild`

Clean and rebuild the development environment.

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | string | dev | Setup mode for rebuild |

**Examples:**
```bash
invoke env.rebuild
invoke env.rebuild --mode=all
```

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

### Supported Tasks

| Task | SLURM Support |
|------|---------------|
| `pipeline.preprocess` | Yes |
| `pipeline.source-recon` | Yes |
| `pipeline.atlas` | Yes |
| `pipeline.features.psd` | Yes |
| `pipeline.features.fooof` | Yes |
| `pipeline.features.complexity` | Yes |
| `pipeline.features.all` | Yes |
| `analysis.statistics` | Not yet |
| `analysis.classify` | Not yet |

### Configuration

SLURM settings are in `config.yaml`:

```yaml
computing:
  slurm:
    enabled: true
    account: def-kjerbi
    partition: ""  # Empty to let Slurm auto-select

    preprocessing:
      cpus: 12
      mem: 32G
      time: "12:00:00"

    source_reconstruction:
      cpus: 1
      mem: 64G
      time: "2:00:00"

    features:
      cpus: 12
      mem: 32G
      time: "6:00:00"
```

### Job Management

```bash
squeue -u $USER              # View your jobs
sacct -j JOBID               # Check job status
scancel JOBID                # Cancel job
scancel -u $USER             # Cancel all jobs
```

### Job Manifests

When submitting SLURM jobs, manifests are saved to `logs/slurm/{stage}/` with:
- Job IDs
- Submission timestamp
- Subjects and runs processed
- Stage metadata

---

## Notes

- All tasks support `--help` for detailed options (e.g., `invoke pipeline.preprocess --help`)
- Logs are saved to `logs/{stage}/`
- Use `--skip-existing` (default) to avoid reprocessing completed files
- Configuration is loaded from `config.yaml`
- Use `invoke --list` to see all available tasks
