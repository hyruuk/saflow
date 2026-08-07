# Corrected paper-panel methods notes

## Behavioral labeling

VTC is filtered independently within each run with a Gaussian kernel using
reflected boundaries and FWHM 9 trials
(`sigma = FWHM / sqrt(8 ln 2)`). IN and OUT thresholds are the run-specific
25th and 75th percentiles. A neural window is assigned a state only when all
eight contributing trials have that state; mixed windows and windows
containing any final AR2 bad-trial flag are excluded. This boundary correction
may change labels close to run boundaries relative to historical zero-padded
outputs.

Matched lapse models use the eighth/anchor rare target. Correct omission and
commission error windows contain the same preceding seven trials. Broad
legacy correct/lapse selectors are exploratory.

## Feature policy

The paper bands are Theta (4–8 Hz), Alpha (8–12 Hz), Low Beta (12–20 Hz),
High Beta (20–30 Hz), Gamma 1 (30–60 Hz), Gamma 2 (60–90 Hz), and Gamma 3
(90–120 Hz). Delta is excluded. Panel 1 uses raw PSD, FOOOF exponent/offset/R²,
and corrected PSD. Panels 2–3 omit raw PSD. Complexity is exploratory.

All paper maps use 400 ordered cortical parcels from
Schaefer-400/7Networks after excluding medial-wall labels.

## Panel 1

IN and OUT windows are aggregated within run and then within subject.
Spatial inference uses paired tests and Benjamini–Hochberg FDR separately
within each feature. Exports include subject counts, window counts, paired
Cohen’s dz, uncorrected p-values, and corrected p-values.

Epoch-level state decoding uses outer leave-one-subject-out validation with
AUC as the primary metric. Balanced accuracy and confusion matrices are
secondary. Within-subject label permutations are synchronized across parcels;
the maximum parcel AUC controls family-wise error within each feature,
preserving the established t-max behavior.

## Panel 2

The three prespecified models are state, matched lapse within IN, and matched
lapse within OUT. Ridge logistic regression is class-balanced. Median
imputation, standardization, and C selection occur only within outer training
data, using subject-grouped inner folds.

State nulls use run-wise circular VTC shifts farther than 24 trials from zero,
then rebuild strict labels once per run/permutation. Lapse nulls permute
matched outcomes within subject/run/state. Joint models, all standalone and
grouped feature tests, and all parcel-reliance tests form three separate
synchronized maximum-statistic families. Reliance is predictive, not causal.

## Panel 3

The four cells are IN-correct omission, IN-commission error,
OUT-correct omission, and OUT-commission error. The primary contrast is
`(lapse-correct)_OUT - (lapse-correct)_IN`; all four prespecified simple
effects are reported regardless of interaction significance.

Network modulation requires at least five windows in every cell. FOOOF and
corrected-PSD families are corrected separately across contrasts, features,
and Yeo-7 networks with synchronized sign flips. The all-available
random-intercept state×outcome model is secondary.

DMN–DAN association is estimated within subject/run/cell, Fisher-z
transformed, and combined across runs with `n-3` weights. Primary coupling
requires ten windows per cell and is corrected across ten features and all
prespecified contrasts. All other Yeo-7 pairs are exploratory.

## Reproducibility

Every analysis records the immutable ID, full Git state and dirty flag,
resolved configuration, Python/package snapshot, cell commands, deterministic
seeds, permutation intervals, SLURM job/dependency/resource metadata, input
roots, exclusions, and figure sidecars. Exact versions are captured in each
analysis `environment.json`; the supported interpreter range is Python
3.11–3.12.
