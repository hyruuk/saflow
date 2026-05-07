# Analysis workflow

This document describes the two parallel analysis paths and the rules that
govern trial filtering, aggregation, and FOOOF correction. The same trial
filter (drop `bad_ar2 == True`) and the same per-subject IN/OUT split (VTC
percentile bounds computed on **all** trials, applied after the filter)
are used everywhere — only the aggregation step differs between paths.

For the upstream filter mechanics, see
[`docs/preprocessing_workflow.md`](preprocessing_workflow.md). The bad-channel
interpolation that happens *during* preprocessing eliminates the trial-level
INTERP category that previously needed a second filter here.

## Two-path overview

```mermaid
flowchart LR
    subgraph features [Per-trial feature extraction]
        F0[Continuous cleaned raw<br/>BAD_AR2 annotations] --> F1[Segment per stimulus event<br/>tag bad_ar2 from annotations]
        F1 --> F2[Welch PSD per trial<br/>welch_psds_<space>/]
        F1 --> F3[Per-trial FOOOF self-correction<br/>fooof_<space>/, welch_psds_corrected_<space>/]
        F1 --> F4[Per-trial complexity<br/>complexity_<space>/]
    end

    F2 --> P1[Path 1: subject-spectrum]
    F2 --> P2C[Path 2: single-trial classification]
    F3 --> P2C
    F4 --> P1B[Path 1B: subject-trial-median<br/>(complexity)]
    F4 --> P2C

    subgraph path1 [Path 1 — subject-averaged stats]
        P1 --> A1[Per (subj, run, cond):<br/>median PSD over good trials]
        A1 --> A2[Fit FOOOF on the aggregated PSD]
        A2 --> A3[Derive psd_band /<br/>psd_corrected_band /<br/>fooof_exp/off/r²]
        A3 --> A4[Mean across runs]
        A4 --> A5[Paired t-test across subjects<br/>+ tmax / FDR / Bonferroni]
        P1B --> B1[Per (subj, run, cond):<br/>median complexity over good trials]
        B1 --> B2[Mean across runs]
        B2 --> A5
    end

    subgraph path2 [Path 2 — single-trial classification]
        P2C --> C1[Per-trial features<br/>good trials only]
        C1 --> C2[Per-subject IN/OUT split<br/>balance within subject]
        C2 --> C3[LOSO classification<br/>+ tmax / FDR]
    end

    classDef path1 fill:#d4edda,stroke:#155724,color:#000
    classDef path2 fill:#cfe2ff,stroke:#084298,color:#000
    class A1,A2,A3,A4,A5,B1,B2 path1
    class C1,C2,C3 path2
```

## Path 1 — subject-averaged stats (default for stats)

Used for: `psd_<band>`, `psd_corrected_<band>`, `fooof_exponent`,
`fooof_offset`, `fooof_r_squared`, complexity metrics.

Aggregation (PSD-derived families):

1. **Per `(subject, run, condition)`** — keep only good trials; compute
   the **median** PSD across them.
2. **FOOOF fit** on that median PSD (one fit per channel, per
   `(subject, run, condition)`).
3. Derive the requested feature:
   - `psd_<band>`: average the median PSD over the band's frequencies.
   - `psd_corrected_<band>`: subtract the aperiodic fit from the median
     PSD in log-space, then average over the band.
   - `fooof_exponent` / `_offset` / `_r_squared`: read directly off the fit.
4. **Mean across runs** of those per-`(subj, run, cond)` values → one
   value per `(subject, condition)` per channel.
5. **Paired t-test** across subjects (OUT − IN). Multiple-comparison
   correction via tmax (default), FDR or Bonferroni; tmax permutation
   uses the same paired diffs as the test stat (FWER-correct).

Aggregation (complexity): identical except no FOOOF fit — features are
already per-trial scalars. Use `--analysis-mode subject-trial-median`.

Why this path: per-trial FOOOF fits on 0.852 s windows are noisy (alpha
band has ~4 frequency bins). Fitting on a clean run-averaged spectrum
gives a much better aperiodic estimate. Within a subject, IN and OUT are
corrected by *different* aperiodic baselines — this is by design, and it
isolates the periodic IN/OUT effect from any IN/OUT difference in the
1/f baseline.

> Caveat: because the aperiodic fit depends on the trial's condition,
> `psd_corrected_*` from Path 1 carries condition information by
> construction. Don't feed it to a classifier — it would learn the label
> via the correction itself rather than the underlying neural signal.
> Path 2 features are leakage-free.

## Path 2 — single-trial classification (default for classification)

Used for: `psd_<band>`, `psd_corrected_<band>` (per-trial self-corrected),
`fooof_exponent` / `_offset` / `_r_squared` (per-trial fits), complexity
metrics. Each feature is already at trial granularity from the
extraction step — the loader just filters good trials and forwards them.

Mechanics:

1. Drop trials with `bad_ar2 == True`.
2. Per-subject VTC percentile cuts on all trials → IN / OUT / MID
   labels; MID dropped, IN/OUT kept.
3. Balance IN vs OUT counts within subject (default; `--no-balance` to
   skip).
4. Leave-one-subject-out cross-validation (default), with permutation-
   based significance testing across spatial units.

Per-trial corrected PSDs are produced by self-correction in
`compute_fooof.py` (each trial subtracts its *own* aperiodic), so they
do not leak the condition label and are safe to classify on.

## What feeds what

| Feature folder | Path 1 reads | Path 2 reads |
|---|---|---|
| `welch_psds_<space>/` | yes — re-aggregates and re-fits | yes — band mean per trial |
| `welch_psds_corrected_<space>/` | no — recomputes corrected PSDs from welch | yes — per-trial self-corrected |
| `fooof_<space>/` | no — refits aperiodic on aggregated spectrum | yes — per-trial aperiodic params |
| `complexity_<space>/` | yes (subject-trial-median sub-mode) | yes |

Path 1 only ever needs raw welch PSDs for the PSD/FOOOF families. The
precomputed `welch_psds_corrected_<space>/` and `fooof_<space>/` folders
exist for Path 2 and don't need to be regenerated when changing
`inout_bounds` or the FOOOF config.

## Trial filter, IN/OUT split

Same rule for both paths:

- **VTC percentile thresholds** (`config.analysis.inout_bounds`, default
  `[25, 75]`): computed per subject over **all** trials, including those
  flagged `bad_ar2`. This anchors the percentile cut to the full
  distribution and accepts a small class imbalance after the bad filter.
- **Bad-trial filter**: drop `bad_ar2 == True` after masking. Default
  on; `--keep-bad-trials` flips it.
- **Task filter**: keep only `task == "correct_commission"` for IN/OUT
  trials.
- **MID zone**: trials whose VTC sits between the cuts are excluded
  from IN-vs-OUT comparisons.

## Provenance per analysis run

Every `*_results.npz` carries a sibling `*_metadata.json` with:

- `analysis_mode`: `subject-spectrum` / `subject-trial-median` / `single-trials`.
- `aggregate`: per-subject statistic for sub-mode (`median` / `mean`).
- `test`: `paired_ttest` or `independent_ttest` or `ttest_ind` (legacy).
- `drop_bad_trials`, `bad_ar2_metadata_present`, `n_bad_excluded`.
- `inout_bounds`, `n_subjects`, `n_in`, `n_out`.
- `per_subject`: `{sub-XX: {n_total, n_in, n_out, n_mid, n_bad_in, n_bad_out, ...}}`.
- `fooof_freq_range`, `fooof_params` (Path 1 only).
- Provenance: git hash of stats script, hash of script file, git hashes
  of every input feature file consumed.

The output filename includes a `path-<mode>` tag, so Path 1 and Path 2
results for the same feature_type land in different files.

## CLI

Stats — Path 1 (default) on PSD / corrected / FOOOF features:

```
python -m code.statistics.run_group_statistics \
    --feature-type psd_alpha psd_corrected_alpha fooof_exponent \
    --space sensor --correction tmax
```

Stats — complexity (must use trial-median sub-mode):

```
python -m code.statistics.run_group_statistics \
    --feature-type complexity_lzc_median complexity_entropy_permutation \
    --space sensor --analysis-mode subject-trial-median \
    --aggregate median --correction tmax
```

Classification (Path 2):

```
python -m code.classification.run_classification \
    --feature psd_corrected_alpha --space sensor \
    --mode univariate --n-permutations 1000
```

Both default to dropping `bad_ar2` trials. Pass `--keep-bad-trials` to
disable.
