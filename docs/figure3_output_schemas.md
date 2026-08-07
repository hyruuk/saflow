# Paper-panel output contracts

The corrected paper workflow uses schema version `1.0.0`. Real and synthetic
bundles obey the same schemas; `data_mode` is always either `real` or
`synthetic`. Every artifact also records the immutable analysis ID, Git state,
configuration hash, inputs, and software environment.

The authoritative paper branch has seven ordered frequency bands: Theta
(4–8 Hz), Alpha (8–12 Hz), Low Beta (12–20 Hz), High Beta (20–30 Hz),
Gamma 1 (30–60 Hz), Gamma 2 (60–90 Hz), and Gamma 3 (90–120 Hz). Compatibility
keys are `theta`, `alpha`, `lobeta`, `hibeta`, `gamma1`, `gamma2`, and
`gamma3`. Delta is not part of any corrected paper schema.

The versioned catalog covers:

- `labels`: exact alignment keys, eight contributing trials, strict state,
  matched rare-target outcome, and any-constituent bad-trial rejection.
- `maps`: ordered features/parcels, contrasts, and inferential statistics.
- `decoding`: models, held-out probabilities, metrics, and grouped predictive
  reliance.
- `factorial_networks`: Yeo-7 order, four cells, contrasts, and complete-case
  eligibility.
- `coupling`: network pairs, four cells, Fisher-z estimates, and contrasts.
- `compact_export`: tables and render-ready arrays, excluding subject-level
  matrices and resumable chunks.
- `figure`: panel, path, DPI, data mode, and render parameters.
- `dag_manifest`: immutable nodes, dependencies, array cells, and provenance.

The machine-readable catalog is written to
`<analysis-id>/manifests/schemas.json` during preflight. The dry-run graph is
written to `<analysis-id>/manifests/dag.json`. Its `submission_plan` records
the resource class, array size, stable dry-run job identifier, and typed
dependencies for every retained node. `aftercorr` edges are rejected unless
both arrays have the same subject/run index mapping.

Scientific node cells use their actual dimensions: Panel 1 features and
feature×chunk intervals, Panel 2 models and synchronized decoding chunks, and
Panel 3 features. They never reuse the subject/run array mapping. Scheduler
cell status JSON records are stored under `manifests/cells/<node>/`.

Panel 1 render bundles must provide raw-PSD modulation and decoding arrays,
IN/OUT raw, aperiodic, corrected, and periodic spectra, FOOOF modulation and
decoding arrays, and corrected-PSD modulation and decoding arrays. This
preserves the established narrative:
A/B raw PSD, C–F raw/aperiodic/corrected/periodic spectra, G/H FOOOF, and I/J
corrected PSD. A map-only result is not considered render-ready and cannot be
labeled as a real composite.
