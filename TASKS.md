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
| `analysis.*` | Group-level analysis (stats, classify, classify-multifeature) |
| `analysis.networks.*` | Yeo-network analysis (stats agg, coherence, classify, importance) |
| `viz.*` | Visualization (stats, maps, spectra, stats-classif-panel, behavior, auto) |
| `viz.networks.*` | Network visualizations (composite story panel) |
| `slurm.*` | SLURM job management (jobs, cancel) |

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
4. Creates epochs for stimulus events only (Freq+Rare, no Resp)
5. Runs AutoReject (AR1) to identify bad epochs for ICA
6. Fits ICA and removes ECG/EOG artifacts, applies to continuous raw
7. Re-epochs from ICA-cleaned continuous (1:1 mapping with AR1)
8. Runs AutoReject (AR2) in fit-only mode to flag bad epochs
9. Saves continuous Raw + BAD annotations and canonical ICA epochs (`*_proc-ica_epo.fif`)

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--subject` | string | required* | Subject ID to process (required for local) |
| `--runs` | string | all | Space-separated run numbers (e.g., "02 03 04") |
| `--bids-root` | PATH | config | Override BIDS root directory |
| `--log-level` | choice | INFO | DEBUG, INFO, WARNING, ERROR |
| `--skip-existing` | flag | true | Skip if output files exist (default) |
| `--crop` | float | none | Crop to first N seconds (for testing) |
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

### `invoke pipeline.preprocess-report`

Generate aggregate preprocessing QC reports from existing per-run params JSON sidecars.

**What it does:**
- Per-subject report: aggregates each subject's runs into a single HTML/JSON summary (`sub-XX_preprocessing-summary.{html,json}`).
- Dataset-level report: aggregates all subjects into a group HTML/JSON with interactive Plotly distributions (`group_preprocessing-summary.{html,json}`). When `--dataset` is passed, subject-level reports are regenerated first.

The per-run preprocessing task already generates a subject-level report at the end; use this task to refresh subject or group summaries on demand.

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--subject` | string | none | Subject ID for a single-subject report (mutually exclusive with `--dataset`) |
| `--dataset` | flag | false | Generate the group dataset report (also regenerates all subject reports) |

**Examples:**
```bash
invoke pipeline.preprocess-report --subject=04
invoke pipeline.preprocess-report --dataset
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

All feature extraction tasks share a common `--n-events-window` knob that controls the windowing of consecutive stimulus trials:

- `--n-events-window=1`: single-trial mode (one PSD per epoch).
- `--n-events-window=8` (default): cc_saflow-compatible sliding window — 8 consecutive stim trials per Welch window. This is the canonical setting used by the analysis/visualization tasks downstream.

The chosen window determines both which Welch profile is loaded and the `welch{N}` desc suffix written into output filenames.

### `invoke pipeline.features.psd`

Extract power spectral density features using Welch's method.

**What it computes:**
- Welch PSD estimates per trial/window
- Band power for configured frequency bands
- Saves with IN/OUT classification metadata and channel names

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--subject` | string | all | Subject ID (processes all if not specified) |
| `--runs` | string | all | Space-separated run numbers |
| `--space` | string | sensor | Analysis space: `sensor`, or atlas name (e.g., `aparc.a2009s`) |
| `--n-events-window` | int | 8 | Trials per Welch window (1 = single-trial, 8 = cc_saflow default) |
| `--skip-existing` | flag | true | Skip if output files exist |
| `--slurm` | flag | false | Submit jobs to SLURM cluster (array job, one task per subject×run) |
| `--dry-run` | flag | false | Generate SLURM scripts without submitting |

**Examples:**
```bash
invoke pipeline.features.psd --subject=04
invoke pipeline.features.psd --subject=04 --space=aparc.a2009s
invoke pipeline.features.psd --slurm
invoke pipeline.features.psd --n-events-window=1   # single-trial mode
```

---

### `invoke pipeline.features.fooof`

Extract specparam aperiodic parameters and corrected PSDs, preserving the
existing FOOOF-compatible task name and `fooof_*` output feature names.

**What it computes:**
- Aperiodic parameters (exponent, offset; fixed aperiodic mode only)
- Goodness of fit metrics (r_squared, error)
- Aperiodic-corrected PSDs (periodic component only)

Specparam fitting parameters (`freq_range`, `aperiodic_mode`, etc.) are
configured in `config.yaml`. `aperiodic_mode` must be `fixed`; other modes
are intentionally rejected during config validation. FOOOF loads the Welch
PSDs that match `--n-events-window`, so make sure
`pipeline.features.psd --n-events-window=N` ran first with the same N.

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--subject` | string | all | Subject ID (processes all if not specified) |
| `--runs` | string | all | Space-separated run numbers |
| `--space` | string | sensor | Analysis space: `sensor`, or atlas name |
| `--n-events-window` | int | 8 | Trials per Welch window (must match PSD window) |
| `--skip-existing` | flag | true | Skip if output files exist |
| `--slurm` | flag | false | Submit jobs to SLURM cluster |
| `--dry-run` | flag | false | Generate SLURM scripts without submitting |

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
| `--subject` | string | all | Subject ID (processes all if not specified) |
| `--runs` | string | all | Space-separated run numbers |
| `--space` | string | sensor | Analysis space: `sensor`, or atlas name |
| `--complexity-type` | string | "lzc entropy fractal" | Space-separated types to compute |
| `--n-events-window` | int | 8 | Trials per epoch window |
| `--overwrite` | flag | false | Overwrite existing files |
| `--slurm` | flag | false | Submit jobs to SLURM cluster |
| `--dry-run` | flag | false | Generate SLURM scripts without submitting |

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
| `--subject` | string | all | Subject ID (processes all if not specified) |
| `--runs` | string | all | Space-separated run numbers |
| `--space` | string | sensor | Analysis space: `sensor`, or atlas name |
| `--n-events-window` | int | 8 | Trials per Welch window |
| `--overwrite` | flag | false | Overwrite existing files |
| `--slurm` | flag | false | Submit jobs to SLURM cluster |
| `--dry-run` | flag | false | Generate SLURM scripts without submitting |

**Examples:**
```bash
invoke pipeline.features.all --subject=04
invoke pipeline.features.all --subject=04 --space=aparc.a2009s
invoke pipeline.features.all --slurm
```

---

## Analysis Tasks

`analysis.stats` and `analysis.classify` share the same `--features` and `--trial-type` interfaces so they can be called identically.

**`--features` accepts (default: `all`):**
- A single feature name: `fooof_exponent`, `psd_alpha`, `complexity_lzc_median`, ...
- A space-separated list: `"fooof_exponent psd_alpha"`
- A shortcut name (expands to the family below):
  - `psds` → `psd_<band>` for every band in `config.yaml`
  - `psds_corrected` → `psd_corrected_<band>` for every band
  - `fooof` → `fooof_exponent`, `fooof_offset`, `fooof_r_squared`
  - `complexity` → all 10 complexity sub-metrics (LZC, entropies, fractals)
  - `all` → union of `fooof` + `psds` + `psds_corrected` + `complexity`

**`--trial-type` accepts (default: `all`):**
- A single type: `alltrials`, `correct`, `rare`, `lapse`, `correct_commission`.
- The shortcut `all` runs three variants in a single invocation: `alltrials`, `correct` (baseline), and `lapse`. Each is written to its own `_type-<...>` result file.

**`--analysis-level` accepts (default: `both`):**
- `epoch`: per-epoch / per-trial test or classification.
- `average`: one IN + one OUT row per subject — subject-spectrum (pool → FOOOF) for `psd_*` and `fooof_*`, per-subject median/mean for `complexity_*`.
- `both`: run both levels, saving one result file per level (`_level-epoch` / `_level-average`).

By default each feature is processed independently (single-feature framework). `analysis.classify-multifeature` is a separate task for joint multi-feature models.

---

### `invoke analysis.stats`

Run group-level statistical analysis (IN vs OUT attentional states) on one or more features.

Complexity features are routed through the same `run_group_statistics` schema as PSD/FOOOF so a single run can mix families freely. With no arguments, runs every feature through every trial-type variant at every analysis level.

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--features` | string | all | Single feature, space-separated list, or shortcut (`psds`, `psds_corrected`, `fooof`, `complexity`, `all`) |
| `--space` | string | sensor | Analysis space: `sensor`, or atlas name (e.g. `schaefer_400`, `aparc.a2009s`) |
| `--test` | string | paired_ttest | `paired_ttest` or `independent_ttest` |
| `--correction` | string | fdr | `none`, `fdr`, `bonferroni`, or `tmax` (FWER via permutation) |
| `--alpha` | float | 0.05 | Significance threshold |
| `--n-permutations` | int | 10000 | Permutations for `tmax` |
| `--n-jobs` | int | 1 | Parallel jobs (`-1` = all cores) |
| `--analysis-level` | string | both | `epoch`, `average`, or `both` (writes one file per level) |
| `--aggregate` | string | median | Per-subject reducer for `level=average` on `complexity_*` (`median` or `mean`) |
| `--trial-type` | string | all | `alltrials`, `correct`, `rare`, `lapse`, `correct_commission`, or `all` (runs 3 variants) |
| `--zoning` | string | per-run | IN/OUT zoning policy (matches feature-extraction `zoning`) |
| `--n-events-window` | int | 8 | Trials per Welch window (must match feature extraction) |
| `--visualize` | flag | false | Generate visualization figures alongside results |
| `--continue-on-error` / `--no-continue-on-error` | flag | true | Keep going if a feature family fails |
| `--slurm` | flag | false | Submit as SLURM array (one task per trial-type) |
| `--slurm-time` / `--slurm-mem` / `--slurm-cpus` | string/int | from config | Per-job resource overrides |
| `--dry-run` | flag | false | Preview SLURM submissions without running |

**Examples:**
```bash
# Default: every feature × every trial-type × every level
invoke analysis.stats

# Single feature
invoke analysis.stats --features=fooof_exponent

# Whole family
invoke analysis.stats --features=psds
invoke analysis.stats --features=fooof --space=schaefer_400

# Every feature on disk (single-feature framework: each tested independently)
invoke analysis.stats --features=all --space=schaefer_400 --n-jobs=4

# Per-epoch only (skip subject-level paired test)
invoke analysis.stats --features="psd_alpha psd_theta" --analysis-level=epoch

# Lapse-only trials, FDR correction
invoke analysis.stats --features=fooof --trial-type=lapse --correction=fdr

# SLURM (one job per trial-type variant)
invoke analysis.stats --features=all --slurm
```

---

### `invoke analysis.classify`

Run classification analysis (decode IN vs OUT from neural features) on one or more features.

With no arguments, runs every feature through every trial-type variant at every analysis level.

**Spatial mode:**
- `univariate` (default): per-channel/ROI classifier + shared-permutation t-max correction.
- `multivariate`: pool spatial dim into one feature vector, single `permutation_test_score`.

**Analysis level & CV:**
- `--analysis-level=epoch`: per-epoch classification. Default CV resolves to `logo` (leave-one-subject-out).
- `--analysis-level=average`: subject-level (1 IN + 1 OUT row per subject). Default CV resolves to `group` (GroupKFold, k=6).
- `--analysis-level=both` (default): run both levels, one scores file per level.
- `--cv=auto` (default) resolves from the level; `--cv` (`logo`|`stratified`|`group`) forces a specific splitter.

**Standardization:**
- `--standardize=auto` (default): per-subject z-scoring for epoch-level trial classification (avoids the between-subject baseline pinning balanced-accuracy at 0.5), no standardization otherwise.
- Explicit values: `per-subject`, `none`, or any sklearn-compatible scaler key.

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--features` | string | all | Single feature, space-separated list, or shortcut (`psds`, `psds_corrected`, `fooof`, `complexity`, `all`) |
| `--clf` | string | logistic | Classifier: `lda`, `svm`, `rf`, `logistic` |
| `--cv` | string | auto | Cross-validation: `auto`, `logo`, `stratified`, `group` |
| `--space` | string | sensor | Analysis space (`sensor` or atlas name) |
| `--mode` | string | univariate | `univariate` (per-channel + tmax) or `multivariate` (pool spatial dim) |
| `--analysis-level` | string | both | `epoch`, `average`, or `both` |
| `--standardize` | string | auto | `auto`, `per-subject`, `none`, ... |
| `--trial-type` | string | all | `alltrials`, `correct`, `rare`, `lapse`, `correct_commission`, or `all` |
| `--zoning` | string | per-run | IN/OUT zoning policy |
| `--n-events-window` | int | 8 | Trials per Welch window (must match feature extraction) |
| `--n-permutations` | int | 1000 | Permutations for significance testing |
| `--no-balance` | flag | false | Disable class balancing within subjects |
| `--n-jobs` | int | -1 | Parallel jobs (`-1` = all cores) |
| `--combine-features` | flag | false | Stack all selected features into a single model (otherwise: one model per feature) |
| `--importances` | flag | false | Save RF feature importances (use with `--clf=rf`) |
| `--label` | string | auto | Output label override (only with `--combine-features`) |
| `--n-chunks` | int | 1 | Spatial-dim chunking (univariate + SLURM only) |
| `--aggregate` / `--no-aggregate` | flag | true | Auto-merge per-chunk SLURM outputs via afterok aggregator job |
| `--delete-chunks` | flag | false | Delete per-chunk files after successful aggregation |
| `--continue-on-error` / `--no-continue-on-error` | flag | false | Keep going if a feature fails |
| `--seed` | int | 42 | Random seed for permutations |
| `--slurm` | flag | false | Submit each classification as its own SLURM job (or array) |
| `--slurm-time` / `--slurm-mem` / `--slurm-cpus` | string/int | from config | Per-job resource overrides |
| `--dry-run` | flag | false | Preview SLURM submissions without running |

**Examples:**
```bash
# Default: every feature × every trial-type × every level
invoke analysis.classify

# Single feature, per-channel LDA + tmax
invoke analysis.classify --features=fooof_exponent --clf=lda

# Whole family — one classification per feature
invoke analysis.classify --features=psds --space=sensor

# Per-epoch only with explicit LOSO CV
invoke analysis.classify --features=fooof_exponent --analysis-level=epoch --cv=logo

# Combine all complexity metrics into one RF model + save importances
invoke analysis.classify --features=complexity --space=schaefer_400 \
    --clf=rf --combine-features --importances

# Combine + multivariate: one big RF over (n_features × n_spatial)
invoke analysis.classify --features=all --combine-features \
    --mode=multivariate --clf=rf --importances

# Fan out to SLURM — one job per feature, chunk the spatial dim for big atlases
invoke analysis.classify --features=all --slurm
invoke analysis.classify --features=psds --slurm --n-chunks=8
```

---

### `invoke analysis.classify-multifeature`

Multi-feature classification (IN vs OUT) along one or all axes of the (trials × spatial × features) tensor.

Loads all selected features as a single stacked tensor and runs decoding along one or more axes:

| Axis | Classifier configuration | Output |
|------|--------------------------|--------|
| `per-spatial` | One classifier per sensor/ROI, all features as input | `scores(n_spatial,)`, `importance(n_spatial, n_feat)` |
| `per-feature` | One classifier per feature, all spatial units as input | `scores(n_features,)`, `importance(n_features, n_sp)` |
| `per-cell` | One classifier per (sensor, feature) pair | `scores(n_spatial, n_features)` |
| `joint` | One classifier on flattened `n_spatial * n_features` | scalar score, `importance(n_spatial, n_feat)` |
| `all` (default) | Run all four axes, one file per axis | bundled aggregate JSON |

**Feature-importance backends (`--importance`):**
- `permutation` (default): CV-fold permutation importance, model-agnostic, averaged across folds.
- `coef`: signed coefficients (linear models only: `lda`, `logistic`, `svm`).
- `tree`: `feature_importances_` (RF only).
- `none`: skip importance.

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--features` | string | all | Same conventions as `analysis.classify` |
| `--label` | string | auto | Output label (default: `combined-<N>`) |
| `--clf` | string | logistic | Classifier choice |
| `--cv` | string | logo | Cross-validation scheme |
| `--space` | string | sensor | Analysis space |
| `--axis` | string | all | `per-spatial`, `per-feature`, `per-cell`, `joint`, or `all` |
| `--importance` | string | permutation | Importance backend |
| `--importance-n-repeats` | int | 5 | Permutation-importance repeats per CV fold |
| `--n-permutations` | int | 1000 | Significance-test permutations |
| `--per-feature-scale` / `--no-per-feature-scale` | flag | true | Standardize each feature independently before stacking |
| `--no-balance` | flag | false | Disable per-subject class balancing |
| `--trial-type` | string | alltrials | Single trial-type only (no `all` shortcut for now) |
| `--zoning` | string | per-run | IN/OUT zoning policy |
| `--n-events-window` | int | 8 | Trials per Welch window |
| `--standardize` | string | per-subject | Pre-stacking scaler |
| `--analysis-level` | string | epoch | `epoch` or `average` |
| `--n-chunks` / `--chunk-idx` | int | 1 / 0 | Chunked execution for `--axis=per-cell` |
| `--keep-bad-trials` | flag | false | Skip the AR1/AR2-bad mask |
| `--output-dir` | PATH | from config | Override results directory |
| `--aggregate` / `--no-aggregate` | flag | true | Auto-bundle outputs when `--axis=all` |
| `--slurm` | flag | false | Submit each axis (and `per-cell` chunks) as SLURM jobs with an afterok bundle aggregator |
| `--slurm-time` / `--slurm-mem` / `--slurm-cpus` | string/int | from config | Per-job resource overrides |
| `--dry-run` | flag | false | Preview SLURM submissions without running |

**Examples:**
```bash
# All four axes, all features, sensor space, logreg + permutation importance
invoke analysis.classify-multifeature

# Per-feature axis only on schaefer_400, RF + tree importance
invoke analysis.classify-multifeature --axis=per-feature \
    --space=schaefer_400 --clf=rf --importance=tree

# Joint axis with bundled aggregation
invoke analysis.classify-multifeature --axis=joint --features=all

# Fan out per-cell to SLURM with chunking
invoke analysis.classify-multifeature --axis=per-cell --space=schaefer_400 \
    --n-chunks=8 --slurm
```

---

### `invoke analysis.classify-multifeature-aggregate`

Bundle per-axis multi-feature classification outputs (from `--axis=all`) into a single file.

Useful when `--aggregate=False` was used at submission, or when the afterok aggregator job failed and per-axis files are still on disk.

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--label` | string | required | Bundle label used at submission |
| `--space` | string | required | Analysis space |
| `--clf` / `--cv` / `--importance` | string | logistic / logo / permutation | Match the original submission |
| `--trial-type` | string | alltrials | Trial-type filter |
| `--analysis-level` | string | epoch | Level filter |
| `--axes` | string | all on disk | Restrict to a subset (e.g. `"per-feature joint"`) |
| `--output-dir` | PATH | from config | Override results directory |
| `--config` | PATH | config.yaml | Config file path |

**Examples:**
```bash
invoke analysis.classify-multifeature-aggregate --label=combined-19 --space=sensor
invoke analysis.classify-multifeature-aggregate --label=all --space=schaefer_400 \
    --axes="joint per-feature"
```

---

### `invoke analysis.classify-aggregate`

Manually aggregate per-chunk classification outputs from `analysis.classify --n-chunks=N --slurm` runs.

Use this when `--aggregate=False` was used at submission, or when the afterok aggregator job failed and chunk files are still on disk.

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--feature` | string | required | Feature name or `combined-<N>` label |
| `--space` | string | required | Analysis space |
| `--clf` / `--cv` / `--mode` | string | lda / logo / univariate | Match the original submission |
| `--combined` | flag | false | Set when aggregating a `--combine-features` run |
| `--trial-type` | string | alltrials | Trial-type filter |
| `--delete-chunks` | flag | false | Delete per-chunk files after merging |
| `--config` | PATH | config.yaml | Config file path |

**Examples:**
```bash
invoke analysis.classify-aggregate --feature=psd_alpha --space=sensor
invoke analysis.classify-aggregate --feature=combined-10 --space=schaefer_400 --combined
```

---

## Network Analysis Tasks (`analysis.networks.*`)

Yeo-network-restricted analysis layer that aggregates per-parcel results to Yeo 7/17 networks. Requires Schaefer (or compatible Yeo-tagged) atlas space and that the underlying parcel-level stats/classification have already been computed.

Outputs land under `<results>/statistics_<space>/group/networks/` and `<results>/classification_<space>/group_mf/networks/`.

### `invoke analysis.networks.aggregate-stats`

Aggregate per-parcel statistical results to Yeo networks.

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--space` | string | schaefer_400 | Schaefer/Yeo-tagged atlas |
| `--trial-type` | string | all | Trial-type filter |
| `--correction` | string | fdr | P-value correction used in input stats |
| `--yeo` | int | 7 | Yeo network resolution (`7` or `17`) |
| `--alpha` | float | 0.05 | Significance threshold |
| `--inout-token` | string | 2575 | IN/OUT bounds token (e.g., `2575` for the 25/75 percentile split) |

**Examples:**
```bash
invoke analysis.networks.aggregate-stats --space=schaefer_400
invoke analysis.networks.aggregate-stats --space=schaefer_400 --yeo=17 --correction=tmax
```

---

### `invoke analysis.networks.coherence`

Compute within- and between-network coherence of IN-OUT contrasts at the network level.

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--space` | string | schaefer_400 | Atlas |
| `--trial-type` | string | all | Trial-type filter |
| `--feature` | string | none | Restrict to one feature (default: all) |
| `--yeo` | int | 7 | Yeo network resolution |
| `--aggregate` | string | median | Per-network reducer (`median` or `mean`) |

**Examples:**
```bash
invoke analysis.networks.coherence --space=schaefer_400
invoke analysis.networks.coherence --space=schaefer_400 --feature=fooof_exponent
```

---

### `invoke analysis.networks.classify`

Yeo-network-restricted IN-vs-OUT classification across three scopes.

**Scopes:**
- `per-family`: one classifier per (network × feature-family) cell.
- `per-feature`: one classifier per (network × feature) cell.
- `joint`: one classifier per network using all features at once.
- `all` (default): run all three scopes.

The underlying script's default scoring is `roc_auc` (was `balanced_accuracy` before 2026-05-23) to match the AUC labels in `viz.networks.panel`. Pass `--scoring=...` to the underlying script directly if you need a different metric.

With `--slurm`, submits one SLURM array task per `(scope × trial-type × network)` cell (up to `3 × 3 × {yeo}`). A second array depends on the first (afterok) to merge the per-network partials into the combined bundle expected by `viz.networks.panel`.

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--space` | string | schaefer_400 | Atlas |
| `--scope` | string | all | `per-family`, `per-feature`, `joint`, or `all` |
| `--trial-type` | string | all | Trial-type filter (`all` runs alltrials + correct + lapse) |
| `--yeo` | int | 7 | Yeo network resolution |
| `--clf` | string | logistic | Classifier |
| `--cv` | string | logo | Cross-validation |
| `--n-permutations` | int | 1000 | Significance permutations |
| `--n-jobs` | int | -1 | Parallel jobs |
| `--families` | string | from script | Override default family list for `per-family` scope |
| `--per-feature-features` | string | from script | Override default feature list for `per-feature` scope |
| `--subjects` | string | all | Restrict to a subject subset |
| `--aggregate` / `--no-aggregate` | flag | true | Auto-merge partials after the classify array completes |
| `--delete-partials` | flag | false | Delete per-network partials after merge |
| `--slurm` | flag | false | Submit array jobs (classify + aggregator) |
| `--slurm-time` / `--slurm-mem` / `--slurm-cpus` | string/int | from config | Per-job resource overrides |
| `--dry-run` | flag | false | Preview SLURM submissions without running |

**Examples:**
```bash
invoke analysis.networks.classify --space=schaefer_400
invoke analysis.networks.classify --space=schaefer_400 --scope=joint
invoke analysis.networks.classify --slurm
invoke analysis.networks.classify --slurm --dry-run
```

---

### `invoke analysis.networks.classify-aggregate`

Merge per-network classification partials into combined bundles. Use after a SLURM run if the afterok aggregator job failed and partials remain on disk.

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--space` / `--scope` / `--trial-type` / `--yeo` / `--clf` / `--cv` | various | match submission | Selectors |
| `--delete-partials` | flag | false | Delete partials after merging |

**Examples:**
```bash
invoke analysis.networks.classify-aggregate --space=schaefer_400
invoke analysis.networks.classify-aggregate --space=schaefer_400 --delete-partials
```

---

### `invoke analysis.networks.importance`

Aggregate joint-axis multi-feature permutation importance to networks. Requires that `analysis.classify-multifeature --axis=joint --importance=permutation` has been executed for the matching `(space, clf, cv, trial-type)`.

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--space` | string | schaefer_400 | Atlas |
| `--label` | string | all | Multi-feature label (default: `all`) |
| `--trial-type` | string | all | Trial-type filter |
| `--yeo` | int | 7 | Yeo network resolution |
| `--clf` / `--cv` | string | logistic / logo | Match the multifeature submission |
| `--importance` | string | permutation | Importance backend |
| `--analysis-level` | string | epoch | Level of the multifeature run |
| `--input` | PATH | auto-discovered | Override path to the joint-axis scores file |

**Examples:**
```bash
invoke analysis.networks.importance --space=schaefer_400
invoke analysis.networks.importance --input=/path/to/feature-all_..._scores.npz --yeo=7
```

---

### `invoke analysis.networks.all`

Run the full network-layer pipeline end-to-end: stats aggregation → coherence → network-restricted classification → joint-axis importance aggregation.

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--space` | string | schaefer_400 | Atlas |
| `--trial-type` | string | all | Trial-type filter |
| `--yeo` | int | 7 | Yeo network resolution |
| `--correction` | string | fdr | Stats correction |
| `--clf` / `--cv` | string | logistic / logo | Classifier settings |
| `--n-permutations` | int | 1000 | Significance permutations |
| `--label` | string | all | Multi-feature label for `networks_importance` |

**Examples:**
```bash
invoke analysis.networks.all --space=schaefer_400
invoke analysis.networks.all --space=schaefer_400 --yeo=17
```

---

## Visualization Tasks

### `invoke viz.stats`

Visualize saved statistical results as topographic (sensor) or surface (source/atlas) maps.

Loads previously computed statistics produced by `analysis.stats`. Sensor space renders topomaps of t-values; source/atlas spaces render inflated brain surfaces.

By convention (for contrast colormaps): **red = OUT > IN** (positive t-values), **blue = IN > OUT** (negative).

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--feature-type` | string | fooof_exponent | Feature to visualize (e.g., `fooof_exponent`, `psd_alpha`) |
| `--space` | string | sensor | Analysis space (`sensor`, `source`, or atlas name) |
| `--alpha` | float | 0.05 | Significance threshold for marking |
| `--trial-type` | string | alltrials | Trial-type filter |
| `--cmap` | string | from config | Override colormap |
| `--show` | flag | false | Display the figure interactively |
| `--save` / `--no-save` | flag | true | Write figure to `reports/figures/` |

**Examples:**
```bash
invoke viz.stats --feature-type=fooof_exponent --show
invoke viz.stats --feature-type=psd_alpha --space=schaefer_400
invoke viz.stats --feature-type=psd --space=aparc.a2009s
invoke viz.stats --feature-type=psd --cmap=coolwarm
```

---

### `invoke viz.maps`

Unified rows-of-maps visualization for stats + classification results, across sensor and atlas spaces.

Auto-discovers result files matching the chosen metric and renders one row of topomaps (sensor) or inflated-brain panels (source/atlas) per feature family. Prints the exact command to run when nothing is found.

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--metric` | string | required | `tval`, `contrast`, or `balanced_accuracy` |
| `--space` | string | sensor | `sensor`, `source`, or atlas name |
| `--feature` | string | none | Space-separated feature names (e.g., `"psd_alpha psd_theta"`) |
| `--feature-set` | string | none | Shortcut family (`psds`, `psds_corrected`, `fooof`, `complexity`, `all`) |
| `--family` | string | none | Filter rendering to one family when features span multiple |
| `--clf` / `--cv` / `--mode` | string | lda / logo / univariate | Classification result filters |
| `--test` | string | paired_ttest | Statistics test filter |
| `--trial-type` | string | alltrials | Trial-type variant to plot |
| `--correction` | string | auto | `auto`, `tmax`, `fdr_bh`, `bonferroni`, `uncorrected` |
| `--alpha` | float | 0.05 | Significance threshold |
| `--cmap` | string | from metric | Override colormap |
| `--output-subdir` | string | classification | Subfolder under `reports/figures/` |

**Examples:**
```bash
invoke viz.maps --metric=balanced_accuracy --space=sensor --feature-set=psds
invoke viz.maps --metric=balanced_accuracy --space=schaefer_400 --feature-set=all
invoke viz.maps --metric=tval --space=sensor --feature=fooof_exponent
invoke viz.maps --metric=contrast --space=sensor --feature-set=psds --trial-type=lapse
```

---

### `invoke viz.spectra`

Reproduce Figure 3 C–F: the FOOOF spectral decomposition panel.

Selects the most significant sensor/region from the FOOOF exponent group-statistics map, then renders, at that unit, a four-panel plot of:

| Panel | Content |
|-------|---------|
| C | Raw spectrum (PSD) — IN vs OUT |
| D | Aperiodic component — IN vs OUT |
| E | Corrected spectrum (PSDc) — IN vs OUT |
| F | Periodic components — IN vs OUT |

Curves are group means ± SEM unless `--subject` pins a single subject (the manuscript's example-subject case).

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--space` | string | sensor | `sensor` or atlas name (e.g., `schaefer_400`) |
| `--stat-feature` | string | fooof_exponent | Statistics map used to pick the unit |
| `--select-by` | string | corrected | Rank by `corrected` or `uncorrected` p-values |
| `--subject` | string | none | Restrict spectra to one subject (default: group average) |
| `--n-events-window` | int | 8 | Trials per Welch window (welch desc suffix) |
| `--show` | flag | false | Display the figure interactively |
| `--save` / `--no-save` | flag | true | Save to `reports/figures/statistics/` |

**Examples:**
```bash
invoke viz.spectra
invoke viz.spectra --space=sensor --select-by=uncorrected
invoke viz.spectra --subject=07
```

---

### `invoke viz.stats-classif-panel`

Render the paper-ready stats + classification multi-panel figure (Fig. 3 from cc_saflow).

A single composite figure with letter labels A–J:

| Panel | Content |
|-------|---------|
| A | Per-band t-values for raw PSD (7 spatial maps) |
| B | Per-band AUC for raw PSD (7 spatial maps) |
| C | Raw spectrum (PSD) — IN vs OUT line plot |
| D | Aperiodic component — IN vs OUT line plot |
| E | Corrected spectrum (PSDc) — IN vs OUT line plot |
| F | Periodic components — IN vs OUT line plot |
| G | FOOOF t-values (exponent, offset, R²) (3 spatial maps) |
| H | FOOOF AUC (exponent, offset, R²) (3 spatial maps) |
| I | Per-band t-values for corrected PSD (7 spatial maps) |
| J | Per-band AUC for corrected PSD (7 spatial maps) |

For `--space=sensor`, the spatial panels are MNE topomaps; for an atlas space (e.g. `schaefer_400`, `aparc.a2009s`) they become 2×2 inflated-brain composites (left/right × lateral/medial), drawn with `aspect="equal"` so they aren't squeezed. When a panel has zero significant parcels, the cortical surface is still rendered (no overlay) instead of falling back to a "no data" placeholder. Each spatial panel's xlabel reports the band/parameter and the number of significant units (e.g. `Theta (4–8Hz) · n=12 sig`).

Significance is computed within each spatial map independently per feature — never pooled across bands or metrics. The spectral lines (C–F) are picked at the sensor/region with the largest |t| on the FOOOF-exponent map.

**Default mixes analysis modes:**
- Stats rows (A, G, I): subject-level pooled-mean (`level-average`) + FDR.
- Classification rows (B, H, J): single-epoch (`level-epoch`) + tmax + `cv-logo`.
- Classification scoring metric is AUC (`roc_auc`); the loader warns if any input file was scored with something else.

**Requires:** Both `analysis.stats` and `analysis.classify` must have been run for the chosen `--space` and `--trial-type` at the required granularity (raw + corrected PSD bands and the three FOOOF parameters).

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--space` | string | sensor | Analysis space: `sensor` (topomaps) or an atlas name like `schaefer_400` / `aparc.a2009s` (inflated-brain panels). Vertex-level `source` is not wired |
| `--trial-type` | string | alltrials | `alltrials`, `correct`, `lapse` |
| `--stats-correction` | string | fdr | Stats mask: `fdr`, `tmax`, `bonferroni`, `uncorrected` (applied per-feature) |
| `--stats-level` | string | average | Stats granularity: `average` (subject-level pooled) or `epoch` (single-trial) |
| `--classif-correction` | string | tmax | Classification mask correction |
| `--classif-level` | string | epoch | Classification granularity |
| `--classif-cv` | string | auto | Classification CV token. Auto: `logo` for level=epoch, `group` for level=average |
| `--clf` | string | logistic | Classifier used in the classification results to load |
| `--alpha` | float | 0.05 | Significance threshold for the mask |
| `--n-events-window` | int | 8 | Trials per Welch window (welch desc suffix) |
| `--correction` | string | none | (Legacy) sets both `--stats-correction` and `--classif-correction` |
| `--cv` | string | none | (Legacy) alias for `--classif-cv` |
| `--output` | PATH | none | Override output path. Default: `reports/figures/stats_classif_panel_space-<space>_type-<trial>_stats-<lvl>-<corr>_classif-<lvl>-<corr>.png` |
| `--config` | PATH | config.yaml | Config file path |

**Examples:**
```bash
invoke viz.stats-classif-panel --space=schaefer_400
invoke viz.stats-classif-panel --trial-type=correct
invoke viz.stats-classif-panel --classif-level=average      # all-average panel
invoke viz.stats-classif-panel --correction=tmax            # tmax on both halves (legacy)
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

### `invoke viz.networks.panel`

Render the composite Yeo-network story panel as a single PNG (four tiers: parcel maps, network-level stats, network-restricted classification, joint-axis importance).

Reads from `results/statistics_<space>/`, `results/statistics_<space>/group/networks/`, `results/classification_<space>/group/`, and `results/classification_<space>/group_mf/networks/`. Sections without inputs degrade to "no data" placeholders, which is useful while pipelines are still running. Tier-1 brain composites use `aspect="equal"` (no squeeze) and still render the cortical surface when zero parcels are significant. Each Tier-1 brain gets a `<feature display>\n(n=X sig)` xlabel mirroring the topomap convention.

**Default mixes analysis modes:**
- Stats rows (Tier 1A + Tier 2): `level-average` + FDR.
- Classification rows (Tier 1B + Tier 3 + Tier 4): `level-epoch` + tmax + `cv-logo`.
- Classification scoring metric is AUC (`roc_auc`).

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--space` | string | schaefer_400 | Atlas (Schaefer or aparc.a2009s) |
| `--trial-type` | string | correct | `alltrials`, `correct`, or `lapse` |
| `--yeo` | int | 7 | Yeo network resolution (`7` or `17`) |
| `--stats-correction` | string | fdr | Stats mask: `fdr`, `tmax`, `bonferroni`, `uncorrected` |
| `--stats-level` | string | average | Stats granularity: `average` or `epoch` |
| `--classif-correction` | string | tmax | Classification mask correction |
| `--classif-level` | string | epoch | Classification granularity |
| `--classif-cv` | string | auto | Classification CV token. Auto: `logo` for level=epoch, `group` for level=average |
| `--alpha` | float | 0.05 | Significance threshold |
| `--clf` | string | logistic | Classification result filter |
| `--mf-label` | string | all | Multi-feature label for the importance tier |
| `--no-yeo-overlay` | flag | false | Skip Tier-1 Yeo outlines (faster) |
| `--correction` | string | none | (Legacy) sets both `--stats-correction` and `--classif-correction` |
| `--cv` | string | none | (Legacy) alias for `--classif-cv` |
| `--output` | PATH | from config | Override output PNG path |
| `--config` | PATH | config.yaml | Config file path |

**Examples:**
```bash
invoke viz.networks.panel --space=schaefer_400 --trial-type=correct
invoke viz.networks.panel --space=schaefer_400 --trial-type=lapse --yeo=17
invoke viz.networks.panel --no-yeo-overlay   # faster, skip Tier-1 outlines
invoke viz.networks.panel --classif-level=average    # subject-level classification panel
```

---

### `invoke viz.auto`

Scan the results folder and render every available visualization.

Walks `<results>/` for `statistics_<space>` and `classification_<space>` folders, discovers which metrics, spaces, trial-types, and (for classification) clf/cv/mode combinations actually have result files on disk, and runs `viz.maps` once per discovered combination. Every filter defaults to "discover and render all" — pass a value to restrict to it.

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--results-dir` | PATH | from config | Override the results root |
| `--metric` | string | all | Restrict to one metric |
| `--space` | string | all | Restrict to one space |
| `--trial-type` | string | all | Restrict to one trial-type |
| `--clf` / `--cv` / `--mode` | string | all | Restrict classification combinations |
| `--test` | string | all | Restrict statistics test |
| `--feature-set` | string | all | Feature family set forwarded to `viz.maps` |
| `--correction` | string | auto | Forwarded to `viz.maps` |
| `--alpha` | float | 0.05 | Significance threshold |
| `--cmap` | string | from metric | Override colormap |
| `--dry-run` | flag | false | Print the planned commands without running |
| `--continue-on-error` / `--no-continue-on-error` | flag | true | Keep going if a render fails |

**Examples:**
```bash
invoke viz.auto                  # everything found on disk
invoke viz.auto --dry-run        # preview the plan
invoke viz.auto --space=sensor   # only sensor maps
invoke viz.auto --metric=tval    # only t-value statistics maps
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
| `--python` | string | auto | Python executable to use (`python3.12`/`python3.11`) |
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

Most pipeline stages now use SLURM **job arrays** for fan-out (one array task per subject×run or per parameter cell), so a single invocation submits one array job rather than many independent jobs.

### Supported Tasks

| Task | SLURM Support |
|------|---------------|
| `pipeline.preprocess` | Yes (array) |
| `pipeline.source-recon` | Yes (array) |
| `pipeline.atlas` | Yes (array) |
| `pipeline.features.psd` | Yes (array) |
| `pipeline.features.fooof` | Yes (array) |
| `pipeline.features.complexity` | Yes (array) |
| `pipeline.features.all` | Yes (array) |
| `analysis.stats` | Yes (one array task per trial-type) |
| `analysis.classify` | Yes (`--slurm`, chunkable with `--n-chunks`) |
| `analysis.classify-multifeature` | Yes (one job per axis, `per-cell` chunkable) |
| `analysis.networks.classify` | Yes (array per scope×trial-type×network + afterok aggregator) |

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

## SLURM Management Tasks (`slurm.*`)

Helper tasks for inspecting and cancelling SLURM jobs without leaving the invoke CLI.

### `invoke slurm.jobs`

List your running/queued SLURM jobs, optionally filtered by name glob and/or state.

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--pattern` | string | all | Glob-style name filter (e.g., `'classify_*'`, `'*chunk-0*'`) |
| `--state` | string | all | SLURM state code (`R`, `PD`, `CG`, ...) |
| `--user` | string | current | SLURM username to query |

**Examples:**
```bash
invoke slurm.jobs
invoke slurm.jobs --pattern='classify_*'
invoke slurm.jobs --pattern='aggregate_*' --state=PD
```

---

### `invoke slurm.cancel`

Cancel SLURM jobs matching a name glob (or explicit IDs).

Safety: by default, prints what would be cancelled and asks for confirmation. Pass `--yes` to skip the prompt, or `--dry-run` to print without cancelling.

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--pattern` | string | none | Glob-style name filter |
| `--job-ids` | string | none | Comma- or space-separated explicit job IDs (overrides pattern) |
| `--state` | string | all | SLURM state code (useful with `--pattern`) |
| `--user` | string | current | SLURM username to query |
| `--dry-run` | flag | false | Print matches without cancelling |
| `--yes` | flag | false | Skip the confirmation prompt |

**Examples:**
```bash
invoke slurm.cancel --pattern='classify_psd_*' --dry-run
invoke slurm.cancel --pattern='aggregate_*' --state=PD --yes
invoke slurm.cancel --pattern='*chunk-0of*'
invoke slurm.cancel --job-ids='123,124,125'
```

---

## Notes

- All tasks support `--help` for detailed options (e.g., `invoke pipeline.preprocess --help`)
- Logs are saved to `logs/{stage}/`
- Use `--skip-existing` (default) to avoid reprocessing completed files
- Configuration is loaded from `config.yaml`
- Use `invoke --list` to see all available tasks
