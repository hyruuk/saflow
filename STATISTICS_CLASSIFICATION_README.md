# Statistics & Classification Modules - Implementation Documentation

## Overview

This document describes the newly implemented **Statistics** and **Classification** modules for the saflow MEG pipeline. These modules enable group-level statistical analysis and machine learning-based decoding of IN vs OUT attentional states.

**Status**: Phase 1-3 Complete (Core Statistics and Core Classification)
**Date**: 2026-01-31

---

## Table of Contents

1. [Module Architecture](#module-architecture)
2. [Statistics Module](#statistics-module)
3. [Classification Module](#classification-module)
4. [Configuration](#configuration)
5. [Usage Examples](#usage-examples)
6. [Output Structure](#output-structure)
7. [Testing](#testing)
8. [Next Steps](#next-steps)

---

## Module Architecture

### Pipeline Position

```
BIDS → Preprocessing → Source Reconstruction → Feature Extraction → [Statistics & Classification]
```

Both modules operate on extracted features (FOOOF, PSD, complexity metrics) and compare IN vs OUT attentional states defined by VTC-based trial classification.

### File Structure

```
code/
├── statistics/
│   ├── __init__.py
│   ├── run_group_statistics.py      # Main CLI script (400 lines)
│   ├── effect_sizes.py               # Effect size computations (100 lines)
│   ├── corrections.py                # Multiple comparison corrections (150 lines)
│   └── visualize_statistics.py       # Plotting functions (200 lines)
│
└── classification/
    ├── __init__.py
    ├── run_classification.py         # Main CLI script (500 lines)
    ├── classifiers.py                # Classifier definitions (200 lines)
    ├── feature_selection.py          # Feature selection (future)
    ├── cross_validation.py           # CV strategies (future)
    └── visualize_classification.py   # Plotting functions (future)
```

---

## Statistics Module

### Purpose

Perform group-level statistical comparisons between IN and OUT attentional states with:
- Multiple statistical tests (paired t-test, independent t-test, permutation)
- Multiple comparison corrections (FDR, Bonferroni, Holm-Bonferroni, tmax)
- Effect size measures (Cohen's d, Hedges' g, eta-squared)

### Features Implemented

#### Core Functionality (`run_group_statistics.py`)

- **`load_all_features()`**: Load features from all subjects using existing `code.utils.data_loading`
- **`run_statistical_test()`**: Run statistical tests using existing `code.utils.statistics`
- **`compute_all_effect_sizes()`**: Compute multiple effect size measures
- **`apply_corrections()`**: Apply multiple comparison corrections
- **`save_statistical_results()`**: Save results with provenance metadata (git hash, timestamps)

#### Effect Sizes (`effect_sizes.py`)

- **Cohen's d**: Standardized mean difference
  - Formula: `d = (mean_OUT - mean_IN) / pooled_std`
  - Interpretation: Small (0.2), Medium (0.5), Large (0.8)

- **Hedges' g**: Bias-corrected Cohen's d for small samples
  - Formula: `g = d * J`, where `J = 1 - 3/(4*df - 1)`
  - More accurate for n < 20

- **Eta-squared**: Proportion of variance explained
  - Formula: `η² = SS_between / SS_total`
  - Range: 0 to 1, Interpretation: Small (0.01), Medium (0.06), Large (0.14)

#### Corrections (`corrections.py`)

- **FDR (False Discovery Rate)**:
  - Benjamini-Hochberg (BH): Assumes independence or positive dependence
  - Benjamini-Yekutieli (BY): More conservative, handles arbitrary dependence

- **Bonferroni**: Controls family-wise error rate (FWER)
  - Adjusted threshold: `α/m` where m = number of tests
  - Very conservative, use for confirmatory analyses

- **Holm-Bonferroni**: Step-down procedure, more powerful than Bonferroni

- **Tmax**: Maximum statistic correction using permutation distributions
  - Requires permutation scores from classification
  - Controls FWER while accounting for correlations

#### Visualization (`visualize_statistics.py`)

Currently implemented (placeholders for future integration with `code.utils.visualization`):
- Contrast topomaps
- P-value maps with significance markers
- Effect size topomaps
- Correction method comparison plots
- Effect size histograms
- P-value distribution plots

### Command-Line Interface

```bash
python -m code.statistics.run_group_statistics \
    --feature-type fooof_exponent \
    --space sensor \
    --test paired_ttest \
    --correction fdr bonferroni \
    --n-permutations 10000 \
    --alpha 0.05 \
    --visualize
```

**Arguments**:
- `--feature-type`: Feature to analyze (e.g., `fooof_exponent`, `psd_alpha`, `lzc`)
- `--space`: Analysis space (`sensor`, `source`, `atlas`)
- `--test`: Statistical test (`paired_ttest`, `independent_ttest`, `permutation`)
- `--correction`: Correction method(s) - can specify multiple
- `--n-permutations`: Number of permutations (default: 10000)
- `--alpha`: Significance threshold (default: 0.05)
- `--visualize`: Generate plots (default: False)

### Invoke Task

```bash
# Basic usage
invoke statistics --feature-type=fooof_exponent

# Advanced usage
invoke statistics \
    --feature-type=psd_alpha \
    --space=sensor \
    --test=paired_ttest \
    --corrections="fdr bonferroni" \
    --alpha=0.05 \
    --visualize
```

---

## Classification Module

### Purpose

Decode IN vs OUT attentional states from neural features using machine learning:
- Multiple classifiers (LDA, SVM, Random Forest, Logistic Regression)
- Cross-validation strategies (LeaveOneGroupOut, StratifiedKFold, GroupKFold)
- Permutation testing for significance
- Multivariate classification (combine multiple features)

### Features Implemented

#### Core Functionality (`run_classification.py`)

- **`load_classification_data()`**: Load and combine multiple feature types
  - Supports multivariate classification (e.g., FOOOF + PSD + complexity)
  - Flattens spatial dimensions for sklearn compatibility
  - Optional class balancing within subjects

- **`get_cv_strategy()`**: Get cross-validation splitter
  - **LOGO** (LeaveOneGroupOut): Leave-one-subject-out, ensures generalization
  - **StratifiedKFold**: Maintains class balance, allows within-subject splits
  - **GroupKFold**: Keeps subjects together, useful for nested CV

- **`run_classification_with_cv()`**: Run classification with CV and permutation testing
  - Cross-validated ROC AUC scores
  - Permutation testing for statistical significance
  - Confusion matrix and accuracy
  - Per-fold predictions

- **`save_classification_results()`**: Save results with provenance metadata

#### Classifiers (`classifiers.py`)

##### Linear Discriminant Analysis (LDA)
- **Best for**: High-dimensional data, interpretable linear decision boundary
- **Parameters**: `solver='svd'`, optional `shrinkage` for regularization
- **Use case**: Default choice, fast and effective

##### Support Vector Machine (SVM)
- **Best for**: Non-linear classification, robust to outliers
- **Parameters**: `C` (regularization), `kernel` ('linear', 'rbf', 'poly'), `gamma`
- **Use case**: When linear separation insufficient

##### Random Forest (RF)
- **Best for**: Complex non-linear patterns, feature importance
- **Parameters**: `n_estimators`, `max_depth`, `max_features`
- **Use case**: When interpretability not critical, robust to overfitting

##### Logistic Regression
- **Best for**: Probabilistic outputs, simple linear classifier
- **Parameters**: `C` (inverse regularization), `penalty` ('l1', 'l2')
- **Use case**: Simple baseline, interpretable coefficients

#### Hyperparameter Grids

Each classifier has:
- **Default parameters**: Sensible defaults for immediate use
- **Search grids**: For GridSearchCV hyperparameter tuning (future)

### Command-Line Interface

```bash
# Single feature
python -m code.classification.run_classification \
    --features fooof_exponent \
    --clf lda \
    --cv logo \
    --space sensor

# Multivariate
python -m code.classification.run_classification \
    --features fooof_exponent psd_alpha psd_theta \
    --clf svm \
    --cv logo \
    --n-permutations 1000 \
    --visualize
```

**Arguments**:
- `--features`: Feature type(s) - space-separated for multivariate
- `--clf`: Classifier (`lda`, `svm`, `rf`, `logistic`)
- `--cv`: Cross-validation strategy (`logo`, `stratified`, `group`)
- `--space`: Analysis space (`sensor`, `source`, `atlas`)
- `--n-permutations`: Permutations for significance (default: 1000)
- `--no-balance`: Disable class balancing
- `--visualize`: Generate plots (default: False)

### Invoke Task

```bash
# Basic usage
invoke classify --features=fooof_exponent

# Multivariate classification
invoke classify --features="fooof_exponent psd_alpha psd_theta"

# Advanced usage
invoke classify \
    --features=psd_alpha \
    --clf=svm \
    --cv=logo \
    --n-permutations=1000 \
    --visualize
```

---

## Configuration

### Statistics Configuration (`config.yaml`)

```yaml
statistics:
  test: paired_ttest
  corrections:
    - fdr
    - bonferroni
  fdr_method: bh
  n_permutations: 10000
  alpha: 0.05
  features:
    - fooof_exponent
    - psd_alpha
    - psd_theta
  effect_sizes:
    - cohens_d
    - hedges_g
    - eta_squared
  visualize: true
```

### Classification Configuration (`config.yaml`)

```yaml
classification:
  classifier: lda
  cv_strategy: logo
  n_splits: 5

  feature_selection:
    enabled: false
    method: univariate
    n_features: 100

  hyperparameter_tuning:
    enabled: false
    method: grid
    cv_splits: 3

  n_permutations: 1000
  balance_classes: true
  scoring: roc_auc
  save_models: false
  visualize: true
```

---

## Usage Examples

### Statistics Workflow

#### Example 1: Basic FOOOF Exponent Analysis

```bash
# Run paired t-test with FDR correction
invoke statistics --feature-type=fooof_exponent

# Expected output:
# - processed/statistics_sensor/group/feature-fooof_exponent_inout-2575_test-paired_ttest_contrast.npz
# - processed/statistics_sensor/group/feature-fooof_exponent_inout-2575_test-paired_ttest_tvals.npz
# - processed/statistics_sensor/group/feature-fooof_exponent_inout-2575_test-paired_ttest_pvals.npz
# - processed/statistics_sensor/group/feature-fooof_exponent_inout-2575_test-paired_ttest_pvals-corrected-fdr.npz
# - processed/statistics_sensor/group/feature-fooof_exponent_inout-2575_test-paired_ttest_pvals-corrected-bonferroni.npz
# - processed/statistics_sensor/group/feature-fooof_exponent_inout-2575_test-paired_ttest_effectsize-cohens_d.npz
# - processed/statistics_sensor/group/feature-fooof_exponent_inout-2575_test-paired_ttest_metadata.json
```

#### Example 2: Multiple Features

```bash
# Analyze alpha power with visualization
invoke statistics \
    --feature-type=psd_alpha \
    --corrections="fdr bonferroni" \
    --visualize

# Analyze LZC complexity
invoke statistics \
    --feature-type=lzc \
    --test=paired_ttest
```

### Classification Workflow

#### Example 1: Single Feature LDA

```bash
# Basic LDA classification with LOGO CV
invoke classify --features=fooof_exponent

# Expected output:
# - processed/classification_sensor/group/feature-fooof_exponent_inout-2575_clf-lda_cv-logo_scores.npz
# - processed/classification_sensor/group/feature-fooof_exponent_inout-2575_clf-lda_cv-logo_predictions.npz
# - processed/classification_sensor/group/feature-fooof_exponent_inout-2575_clf-lda_cv-logo_metadata.json
```

#### Example 2: Multivariate SVM

```bash
# Combine FOOOF + PSD features with SVM
invoke classify \
    --features="fooof_exponent psd_alpha psd_theta" \
    --clf=svm \
    --cv=logo \
    --n-permutations=1000

# Random Forest with visualization
invoke classify \
    --features="psd_alpha psd_beta" \
    --clf=rf \
    --visualize
```

### Python API Usage

```python
from pathlib import Path
import numpy as np
from code.statistics.run_group_statistics import load_all_features, run_statistical_test
from code.statistics.effect_sizes import compute_cohens_d
from code.statistics.corrections import apply_fdr_correction
from code.classification.run_classification import load_classification_data, run_classification_with_cv
from code.classification.classifiers import get_classifier

# Load configuration
import yaml
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Statistics: Load and analyze features
X, y, groups, metadata = load_all_features(
    feature_type='fooof_exponent',
    space='sensor',
    inout_bounds=(25, 75),
    config=config,
)

contrast, tvals, pvals = run_statistical_test(X, y, groups, test_type='paired_ttest')
cohens_d = compute_cohens_d(X, y, groups)
fdr_pvals = apply_fdr_correction(pvals, alpha=0.05, method='bh')

print(f"Significant features (FDR): {np.sum(fdr_pvals < 0.05)}")

# Classification: Load and classify
X_clf, y_clf, groups_clf, metadata_clf = load_classification_data(
    feature_types=['fooof_exponent'],
    space='sensor',
    inout_bounds=(25, 75),
    config=config,
    balance=True,
)

clf = get_classifier('lda')
from sklearn.model_selection import LeaveOneGroupOut
cv = LeaveOneGroupOut()

results = run_classification_with_cv(X_clf, y_clf, groups_clf, clf, cv, n_permutations=1000)
print(f"ROC AUC: {results['mean_score']:.3f} ± {results['std_score']:.3f}")
print(f"Permutation p-value: {results['perm_pvalue']:.4f}")
```

---

## Output Structure

### Statistics Output

```
processed/statistics_{space}/
├── group/
│   ├── feature-{type}_inout-{bounds}_test-{method}_contrast.npz
│   ├── feature-{type}_inout-{bounds}_test-{method}_tvals.npz
│   ├── feature-{type}_inout-{bounds}_test-{method}_pvals.npz
│   ├── feature-{type}_inout-{bounds}_test-{method}_pvals-corrected-{correction}.npz
│   ├── feature-{type}_inout-{bounds}_test-{method}_effectsize-{name}.npz
│   └── feature-{type}_inout-{bounds}_test-{method}_metadata.json
└── plots/
    ├── feature-{type}_inout-{bounds}_contrast_topomap.png
    ├── feature-{type}_inout-{bounds}_pvals_topomap.png
    └── feature-{type}_inout-{bounds}_effectsize_topomap.png
```

### Classification Output

```
processed/classification_{space}/
├── group/
│   ├── feature-{type}_inout-{bounds}_clf-{name}_cv-{strategy}_scores.npz
│   ├── feature-{type}_inout-{bounds}_clf-{name}_cv-{strategy}_predictions.npz
│   └── feature-{type}_inout-{bounds}_clf-{name}_cv-{strategy}_metadata.json
└── plots/
    ├── feature-{type}_inout-{bounds}_clf-{name}_roc_curve.png
    ├── feature-{type}_inout-{bounds}_clf-{name}_confusion_matrix.png
    └── feature-{type}_inout-{bounds}_clf-{name}_feature_importance.png
```

### Metadata Format

Both modules save comprehensive metadata in JSON format:

```json
{
  "feature_type": "fooof_exponent",
  "inout_bounds": [25, 75],
  "test_type": "paired_ttest",
  "timestamp": "2026-01-31T12:00:00",
  "git_hash": "abc123...",
  "data_metadata": {
    "n_subjects": 32,
    "n_trials": 640,
    "n_in": 320,
    "n_out": 320,
    "n_features": 5,
    "n_spatial": 272
  },
  "results": {
    "mean_score": 0.658,
    "perm_pvalue": 0.0123
  }
}
```

---

## Testing

### Unit Tests (To Be Implemented)

```bash
# Test statistics module
pytest code/tests/test_statistics_loading.py
pytest code/tests/test_effect_sizes.py
pytest code/tests/test_corrections.py

# Test classification module
pytest code/tests/test_classifiers.py
pytest code/tests/test_classification_pipeline.py
```

### Integration Tests

```bash
# Test full statistics pipeline
invoke statistics --feature-type=fooof_exponent

# Test full classification pipeline
invoke classify --features=fooof_exponent

# Verify outputs exist
ls processed/statistics_sensor/group/
ls processed/classification_sensor/group/
```

### Manual Validation

1. **Check data loading**:
   ```python
   from code.statistics.run_group_statistics import load_all_features
   import yaml
   config = yaml.safe_load(open('config.yaml'))
   X, y, groups, meta = load_all_features('fooof_exponent', 'sensor', (25, 75), config)
   print(X.shape, y.shape, len(np.unique(groups)))
   ```

2. **Verify effect sizes**:
   ```python
   from code.statistics.effect_sizes import compute_cohens_d
   d = compute_cohens_d(X, y, groups)
   print(f"Cohen's d range: [{np.nanmin(d):.3f}, {np.nanmax(d):.3f}]")
   ```

3. **Test classifiers**:
   ```python
   from code.classification.classifiers import get_classifier
   for clf_name in ['lda', 'svm', 'rf', 'logistic']:
       clf = get_classifier(clf_name)
       print(f"{clf_name}: {clf}")
   ```

---

## Next Steps

### Phase 4: Advanced Classification (In Progress)

- [ ] **Feature selection** (`code/classification/feature_selection.py`):
  - Univariate feature selection (SelectKBest)
  - Recursive Feature Elimination (RFE)
  - PCA dimensionality reduction
  - Feature selection within CV loop

- [ ] **Hyperparameter tuning**:
  - Nested cross-validation
  - GridSearchCV integration
  - Parameter grids for each classifier

- [ ] **Visualization** (`code/classification/visualize_classification.py`):
  - ROC curves with confidence intervals
  - Confusion matrices
  - Feature importance plots
  - CV score distributions

### Phase 5: Integration & SLURM

- [ ] **SLURM templates**:
  - `slurm/templates/statistics.sh.j2`
  - `slurm/templates/classification.sh.j2`

- [ ] **SLURM task functions**:
  - Update `invoke statistics --slurm`
  - Update `invoke classify --slurm`

- [ ] **Batch processing**:
  - Process multiple features in parallel
  - Submit job arrays for parameter sweeps

### Future Enhancements

- [ ] **Cluster-based permutation testing** (MNE integration)
- [ ] **Spatiotemporal statistics** (time-resolved analysis)
- [ ] **Searchlight analysis** (local pattern decoding)
- [ ] **Model comparison** (Bayesian model selection)
- [ ] **Cross-decoding** (generalization across features/conditions)

---

## Design Principles

### 1. Reuse Existing Code

Both modules leverage existing utilities:
- `code.utils.data_loading.load_features()`: Feature loading
- `code.utils.data_loading.balance_dataset()`: Class balancing
- `code.utils.statistics.subject_contrast()`: Paired t-tests
- `code.utils.statistics.simple_contrast()`: Independent t-tests
- `code.utils.statistics.apply_tmax()`: Tmax correction
- `code.utils.visualization.grid_topoplot()`: Topomap plotting (future)

### 2. Config-Driven

All parameters come from `config.yaml`:
- INOUT bounds from `analysis.inout_bounds`
- Data paths from `paths.*`
- Default test parameters from `statistics.*` and `classification.*`
- Subject lists from `bids.subjects`

### 3. Provenance Tracking

All outputs include:
- Git commit hash (`get_git_hash()`)
- Timestamp (ISO 8601 format)
- Input parameters (test type, classifier, CV strategy)
- Data metadata (n_subjects, n_trials, n_features)

### 4. SLURM-Ready Architecture

Scripts designed for both local and HPC execution:
- CLI arguments mirror config parameters
- Invoke tasks abstract SLURM submission
- Resource allocations in `config.yaml`

### 5. Extensibility

Modular design allows easy addition of:
- New statistical tests (add to `run_statistical_test()`)
- New correction methods (add to `corrections.py`)
- New classifiers (add to `classifiers.py`)
- New feature selection methods (add to `feature_selection.py`)

---

## Known Limitations

1. **Visualization**: Placeholder implementations need integration with existing `code.utils.visualization`
2. **Feature selection**: Not yet implemented
3. **Hyperparameter tuning**: Not yet implemented
4. **SLURM execution**: Not yet implemented for statistics/classification
5. **Cluster-based permutation**: Requires MNE-Python integration

---

## References

### Statistical Methods

- **FDR**: Benjamini & Hochberg (1995), Benjamini & Yekutieli (2001)
- **Cohen's d**: Cohen (1988), Lakens (2013)
- **Tmax**: Blair & Karniski (1993)

### Classification

- **LDA**: Fisher (1936), Hastie et al. (2009)
- **SVM**: Vapnik (1995), Cortes & Vapnik (1995)
- **Random Forest**: Breiman (2001)
- **Cross-validation**: Stone (1974), Arlot & Celisse (2010)

### Neuroscience Applications

- **MEG decoding**: King & Dehaene (2014), Cichy et al. (2014)
- **Mind wandering**: Esterman et al. (2013), Christoff et al. (2009)
- **VTC framework**: Bastian & Sackur (2013)

---

## Contact & Support

For questions or issues:
1. Check this documentation
2. Review existing code in `code.utils.statistics` and `code.utils.data_loading`
3. Consult the main README.md for pipeline overview
4. Check configuration in `config.yaml`

---

**Implementation Date**: 2026-01-31
**Version**: 1.0 (Phase 1-3 Complete)
**Authors**: Claude Code Implementation
