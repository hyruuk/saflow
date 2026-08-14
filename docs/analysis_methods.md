# Corrected analysis methods notes

## Behavioral labeling

VTC is filtered independently within each run with a Gaussian kernel using
reflected boundaries and FWHM 9 trials
(`sigma = FWHM / sqrt(8 ln 2)`). IN and OUT thresholds are the run-specific
25th and 75th percentiles. A neural window is assigned a state only when all
eight contributing trials have that state; mixed windows and windows
containing any final AR2 bad-trial flag are excluded from feature modulation
and multifeature decoding. This boundary correction
may change labels close to run boundaries relative to historical zero-padded
outputs.

Matched lapse models use the eighth/anchor rare target. Correct omission and
commission error windows contain the same preceding seven trials. Broad
legacy correct/lapse selectors are exploratory.

## Feature policy

The paper bands are Theta (4–8 Hz), Alpha (8–12 Hz), Low Beta (12–20 Hz),
High Beta (20–30 Hz), Gamma 1 (30–60 Hz), Gamma 2 (60–90 Hz), and Gamma 3
(90–120 Hz). Delta is excluded. The feature-modulation analysis uses raw PSD, FOOOF exponent/offset,
and corrected PSD. The multifeature-decoding and network-dynamics analyses omit raw PSD. Complexity is exploratory.

All paper maps use 400 ordered cortical parcels from
Schaefer-400/7Networks after excluding medial-wall labels.

## Feature modulation analysis

IN and OUT windows are aggregated within run and then within subject.
Spatial inference uses paired tests and Benjamini–Hochberg FDR separately
within each feature. Exports include subject counts, window counts, paired
Cohen’s dz, uncorrected p-values, and corrected p-values.

Epoch-level state decoding uses outer leave-one-subject-out validation with
AUC as the primary metric. Balanced accuracy and confusion matrices are
secondary. Within-subject label permutations are synchronized across parcels;
the maximum parcel AUC controls family-wise error within each feature,
preserving the established t-max behavior.

## Multifeature decoding analysis

The primary multifeature endpoint is Schaefer-400 IN-versus-OUT state decoding.
It combines FOOOF exponent and offset with seven aperiodic-corrected frequency
bands (3,600 predictors). FOOOF fit quality (R²) is excluded because it is a
model-quality diagnostic rather than a neurophysiological feature. The primary
classifier is class-balanced fixed ridge with a prespecified penalty, outer
leave-one-subject-out validation, training-only median imputation and scaling,
and pooled held-out AUC. Run-wise circular VTC shifts provide the state null.
Inputs are materialized once, each permutation is checkpointed independently,
and no hyperparameter selection is repeated inside the null loop.
The workflow writes to the canonical active `main/fast_state/` directory; a
fresh preparation replaces that branch after users archive any result they
intend to retain.

The matched lapse-within-IN and lapse-within-OUT models are no longer part of
the primary Panel 2 analysis.

State nulls use run-wise circular VTC shifts farther than 24 trials from zero,
then rebuild strict labels once per run/permutation. Any later feature or
parcel reliance analysis is secondary and predictive, not causal.

## Network dynamics analysis

The four cells are IN-correct omission, IN-commission error,
OUT-correct omission, and OUT-commission error. The primary contrast is
`(lapse-correct)_OUT - (lapse-correct)_IN`; all four prespecified simple
effects are reported regardless of interaction significance.

Network modulation requires at least five windows in every cell. FOOOF and
corrected-PSD families are corrected separately across contrasts, features,
and Yeo-7 networks with synchronized sign flips. The all-available
random-intercept state×outcome model is secondary.

For coupling only, clean windows use an opposite-state-free definition:
IN/MID mixtures containing at least one IN and no OUT are IN, while OUT/MID
mixtures containing at least one OUT and no IN are OUT. All-MID windows,
windows containing both extremes, and windows containing any AR2-bad trial
are excluded. This retains the directional state distinction without requiring
all eight trials to be extreme; it yielded 30/32 subjects with at least five
windows in every cell in the diagnostic cohort snapshot.

For every subject/cell, DMN and DAN values are mean-centered separately within
run and then pooled across runs. One Pearson association is estimated from the
pooled residuals and Fisher-z transformed. Run centering prevents between-run
baseline differences from inducing the pooled association while allowing rare
outcomes distributed across runs to contribute. Primary coupling requires five
pooled windows per cell and is corrected across nine features and all
prespecified contrasts. All other Yeo-7 pairs are exploratory.

## Reproducibility

Every analysis records the immutable ID, full Git state and dirty flag,
resolved configuration, Python/package snapshot, cell commands, deterministic
seeds, permutation intervals, SLURM job/dependency/resource metadata, input
roots, exclusions, and figure sidecars. Exact versions are captured in each
analysis `environment.json`; the supported interpreter range is Python
3.11–3.12.
