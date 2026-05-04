"""Source reconstruction utilities.

This module contains helper functions for:
- Coregistration (MEG to MRI space)
- Source space setup
- BEM model creation
- Forward solution computation
- Noise covariance estimation
- Inverse operator creation
- Source estimate morphing
"""

import contextlib
import fcntl
import logging
import os
import os.path as op
import shutil
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import mne
import numpy as np
from mne import SourceEstimate
from mne.minimum_norm import apply_inverse_epochs, apply_inverse_raw, make_inverse_operator
from mne_bids import BIDSPath

logger = logging.getLogger(__name__)


def create_output_paths(
    subject: str,
    run: str,
    bids_root: Path,
    derivatives_root: Path,
) -> Dict[str, Path]:
    """Create path objects for all source reconstruction inputs/outputs.

    Note: Source reconstruction uses preprocessed files (not raw BIDS) and
    pre-computed noise covariance from preprocessing. This avoids dependencies
    on raw BIDS files and empty-room recordings.

    Args:
        subject: Subject ID (e.g., "04")
        run: Run number (e.g., "02")
        bids_root: Path to BIDS dataset root (not used, kept for API compatibility)
        derivatives_root: Path to derivatives directory

    Returns:
        Dictionary mapping names to Path/BIDSPath objects:
            - 'preproc': Preprocessed continuous data (BIDSPath)
            - 'epoch': Preprocessed epochs (BIDSPath)
            - 'noise_cov': Pre-computed noise covariance from preprocessing (Path)
            - 'trans': Coregistration transform (BIDSPath)
            - 'fwd': Forward solution (BIDSPath)
            - 'stc': Source estimate (continuous) (BIDSPath)
            - 'morph': Morphed source estimate (BIDSPath)
    """
    logger.debug(f"Creating output paths for sub-{subject}, run-{run}")

    # Input: preprocessed continuous data
    preproc_bidspath = BIDSPath(
        subject=subject,
        task="gradCPT",
        run=run,
        datatype="meg",
        suffix="meg",
        processing="clean",
        root=derivatives_root / "preprocessed",
    )

    # Input: preprocessed epochs
    epoch_bidspath = BIDSPath(
        subject=subject,
        task="gradCPT",
        run=run,
        datatype="meg",
        processing="ica",
        root=derivatives_root / "epochs",
    )

    # Input: pre-computed noise covariance from preprocessing (under subject directory)
    noise_cov_path = (
        derivatives_root
        / "noise_covariance"
        / f"sub-{subject}"
        / "meg"
        / f"sub-{subject}_task-noise_cov.fif"
    )

    # Output: coregistration transform
    trans_bidspath = BIDSPath(
        subject=subject,
        task="gradCPT",
        run=run,
        datatype="meg",
        processing="trans",
        suffix="trans",
        extension=".fif",
        root=derivatives_root / "trans",
        check=False,  # Allow non-standard suffix for derivatives
    )
    trans_bidspath.mkdir(exist_ok=True)

    # Output: forward solution
    fwd_bidspath = BIDSPath(
        subject=subject,
        task="gradCPT",
        run=run,
        datatype="meg",
        processing="forward",
        extension=".fif",
        root=derivatives_root / "fwd",
        check=False,
    )
    fwd_bidspath.mkdir(exist_ok=True)

    # Output: source estimates
    stc_bidspath = BIDSPath(
        subject=subject,
        task="gradCPT",
        run=run,
        datatype="meg",
        processing="clean",
        description="sources",
        root=derivatives_root / "minimum-norm-estimate",
    )
    stc_bidspath.mkdir(exist_ok=True)

    # Output: morphed sources
    morph_bidspath = BIDSPath(
        subject=subject,
        task="gradCPT",
        run=run,
        datatype="meg",
        processing="clean",
        description="morphed",
        root=derivatives_root / "morphed_sources",
    )
    morph_bidspath.mkdir(exist_ok=True)

    return {
        "preproc": preproc_bidspath,
        "epoch": epoch_bidspath,
        "noise_cov": noise_cov_path,
        "trans": trans_bidspath,
        "fwd": fwd_bidspath,
        "stc": stc_bidspath,
        "morph": morph_bidspath,
    }


def _run_coreg_fit(
    info: mne.Info,
    fs_subject: str,
    subjects_dir: Path,
    scale_mode: Optional[str] = None,
) -> mne.coreg.Coregistration:
    """Run the standard ICP coregistration pipeline.

    Args:
        info: MEG Info (sensor + digitization).
        fs_subject: FreeSurfer subject name to fit against.
        subjects_dir: FreeSurfer subjects directory.
        scale_mode: None (no scaling), "uniform", or "3-axis".

    Returns:
        Fitted Coregistration object.
    """
    coreg = mne.coreg.Coregistration(
        info, fs_subject, subjects_dir, fiducials="auto"
    )
    if scale_mode is not None:
        coreg.set_scale_mode(scale_mode)
    coreg.fit_fiducials(verbose=False)
    coreg.fit_icp(n_iterations=6, nasion_weight=2.0, verbose=False)
    coreg.omit_head_shape_points(distance=5.0 / 1000)
    coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=False)
    return coreg


@contextlib.contextmanager
def _subject_anatomy_lock(subjects_dir: Path, fs_subject: str) -> Iterator[None]:
    """Serialize anatomy bootstrap across concurrent processes.

    Uses fcntl.flock on a per-subject lock file. The lock is auto-released
    when the file descriptor closes (i.e. on process exit), so crashed jobs
    do not leave stale locks. Required because parallel SLURM jobs for the
    same subject otherwise race in mne.scale_mri (which calls os.makedirs
    without exist_ok and trips over each other's partial dirs).
    """
    subjects_dir = Path(subjects_dir)
    subjects_dir.mkdir(parents=True, exist_ok=True)
    lock_path = subjects_dir / f".{fs_subject}.anatomy.lock"
    with open(lock_path, "w") as lock_f:
        logger.debug(f"Waiting for anatomy lock: {lock_path}")
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        logger.debug(f"Acquired anatomy lock: {lock_path}")
        try:
            yield
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def _has_head_bem(subject_path: Path) -> bool:
    """Return True if the subject dir has a head BEM (bem/*-head*.fif)."""
    bem_dir = subject_path / "bem"
    if not bem_dir.exists():
        return False
    return any(bem_dir.glob("*-head*.fif"))


def _has_real_mri(subject_path: Path) -> bool:
    """Return True if the directory looks like a real FreeSurfer recon-all output.

    Distinguishing rule: a real recon-all has mri/T1.mgz AND does NOT have the
    'MRI scaling parameters.cfg' marker that mne.scale_mri writes.
    """
    if not (subject_path / "mri" / "T1.mgz").exists():
        return False
    if (subject_path / "MRI scaling parameters.cfg").exists():
        return False
    return True


def resolve_fs_subject(subject: str, subjects_dir: Path) -> str:
    """Return the FreeSurfer subject name to use for this BIDS subject.

    - 'sub-{subject}' if a real recon-all MRI lives at that path.
    - 'sub-{subject}_scaled' otherwise (template-based subject).

    The pipeline owns and may rebuild '*_scaled' dirs; it never modifies a real
    MRI directory beyond running BEM generation inside it.
    """
    real_path = Path(subjects_dir) / f"sub-{subject}"
    if real_path.exists() and _has_real_mri(real_path):
        return f"sub-{subject}"
    return f"sub-{subject}_scaled"


def _ensure_real_mri_bem(fs_subject: str, subjects_dir: Path) -> None:
    """Generate BEM surfaces + head FIF for a real-MRI subject if missing.

    Uses MNE's wrappers around FreeSurfer's mri_watershed and mkheadsurf — both
    require a working FreeSurfer installation on PATH (e.g. `module load
    freesurfer` on HPC).
    """
    subject_path = Path(subjects_dir) / fs_subject
    if _has_head_bem(subject_path):
        logger.info(f"Real-MRI BEM already present for {fs_subject}")
        return

    logger.info(
        f"Real MRI present but BEM missing for {fs_subject}. "
        "Running watershed BEM + scalp surfaces..."
    )
    (subject_path / "bem").mkdir(exist_ok=True)
    mne.bem.make_watershed_bem(
        subject=fs_subject,
        subjects_dir=subjects_dir,
        overwrite=True,
        verbose=False,
    )
    mne.bem.make_scalp_surfaces(
        subject=fs_subject,
        subjects_dir=subjects_dir,
        force=True,
        overwrite=True,
        verbose=False,
    )
    if not _has_head_bem(subject_path):
        raise RuntimeError(
            f"BEM generation completed for {fs_subject} but no head FIF "
            f"appeared in {subject_path / 'bem'}. Check FreeSurfer install."
        )
    logger.info(f"Generated BEM for {fs_subject}")


def ensure_subject_anatomy(
    subject: str,
    subjects_dir: Path,
    info: mne.Info,
    uniform_residual_threshold_mm: float = 5.0,
) -> Tuple[str, Optional[mne.transforms.Transform]]:
    """Ensure the subject has a usable FreeSurfer anatomy directory.

    Resolution rule:
      - If 'sub-{subject}/' is a real MRI: use it. Generate BEM via watershed
        if missing. Never delete or overwrite.
      - Otherwise: use 'sub-{subject}_scaled/'. Build it from fsaverage via
        mne.scale_mri if missing or incomplete. The pipeline owns this dir and
        may rmtree+rebuild incomplete copies.

    Returns:
        (fs_subject_name, precomputed_trans). The trans is returned only when a
        new scaled template was just fitted; callers should save it as the
        run's trans file.
    """
    fs_subject = resolve_fs_subject(subject, subjects_dir)
    subject_path = Path(subjects_dir) / fs_subject
    is_scaled = fs_subject.endswith("_scaled")

    # Real MRI branch: never rmtree, just make sure BEM is there.
    if not is_scaled:
        if _has_head_bem(subject_path):
            logger.info(f"Real-MRI anatomy ready: {fs_subject}")
            return fs_subject, None
        with _subject_anatomy_lock(subjects_dir, fs_subject):
            if _has_head_bem(subject_path):
                logger.info(
                    f"Real-MRI BEM built by concurrent job, reusing: {fs_subject}"
                )
                return fs_subject, None
            _ensure_real_mri_bem(fs_subject, subjects_dir)
            return fs_subject, None

    # Scaled-template branch.
    if subject_path.exists() and _has_head_bem(subject_path):
        logger.info(f"Scaled template ready: {fs_subject}")
        return fs_subject, None

    fsaverage_path = Path(subjects_dir) / "fsaverage"
    if not _has_head_bem(fsaverage_path):
        raise RuntimeError(
            f"fsaverage is missing or incomplete in {subjects_dir} "
            f"(no head BEM at {fsaverage_path}/bem/*-head*.fif). "
            "Run scripts/download_fsaverage.sh (or "
            "`mne.datasets.fetch_fsaverage(subjects_dir)`) before source recon."
        )

    with _subject_anatomy_lock(subjects_dir, fs_subject):
        if subject_path.exists() and _has_head_bem(subject_path):
            logger.info(
                f"Scaled template built by concurrent job, reusing: {fs_subject}"
            )
            return fs_subject, None

        if subject_path.exists():
            logger.warning(
                f"Stale scaled-template dir {subject_path} (no head BEM). "
                "Removing and rebuilding."
            )
            shutil.rmtree(subject_path)

        logger.info(
            f"Fitting scaled fsaverage template for {fs_subject}..."
        )

        coreg = _run_coreg_fit(info, "fsaverage", subjects_dir, scale_mode="uniform")
        dists_mm = coreg.compute_dig_mri_distances() * 1000.0
        median_err = float(np.median(dists_mm))
        logger.info(f"Uniform scaling fit: median dig-MRI distance {median_err:.2f} mm")

        if median_err > uniform_residual_threshold_mm:
            logger.warning(
                f"Uniform scaling residuals too large ({median_err:.2f} mm > "
                f"{uniform_residual_threshold_mm} mm). Retrying with 3-axis scaling."
            )
            coreg = _run_coreg_fit(info, "fsaverage", subjects_dir, scale_mode="3-axis")
            dists_mm = coreg.compute_dig_mri_distances() * 1000.0
            logger.info(
                f"3-axis scaling fit: median dig-MRI distance {float(np.median(dists_mm)):.2f} mm"
            )

        logger.info(f"Scale factors: {coreg.scale}")
        mne.scale_mri(
            subject_from="fsaverage",
            subject_to=fs_subject,
            scale=coreg.scale,
            subjects_dir=subjects_dir,
            overwrite=False,
            labels=True,
            annot=True,
            skip_fiducials=True,
            verbose=False,
        )
        logger.info(f"Created scaled template: {subject_path}")
        return fs_subject, coreg.trans


def compute_coregistration(
    preproc_path: BIDSPath,
    trans_path: BIDSPath,
    fs_subject: str,
    subjects_dir: Path,
) -> mne.transforms.Transform:
    """Rigid-body coregistration against an existing FreeSurfer subject dir.

    Use ensure_subject_anatomy first to determine the resolved fs_subject
    name and guarantee its anatomy is in place.
    """
    logger.info("Computing coregistration...")

    raw = mne.io.read_raw_fif(str(preproc_path.fpath), preload=False, verbose=False)
    info = raw.info

    logger.info(f"Using FreeSurfer subject: {fs_subject}")

    coreg = _run_coreg_fit(info, fs_subject, subjects_dir, scale_mode=None)

    trans_fpath = str(trans_path.fpath)
    os.makedirs(op.dirname(trans_fpath), exist_ok=True)
    mne.write_trans(trans_fpath, coreg.trans, overwrite=True)
    logger.info(f"Saved coregistration: {trans_fpath}")

    return coreg.trans


def setup_source_space(
    fs_subject: str,
    subjects_dir: Path,
    spacing: str = "oct6",
) -> mne.SourceSpaces:
    """Create source space for the resolved FreeSurfer subject."""
    logger.info(f"Setting up source space for {fs_subject} (spacing={spacing})...")

    src = mne.setup_source_space(
        fs_subject,
        spacing=spacing,
        add_dist="patch",
        subjects_dir=subjects_dir,
        verbose=False,
    )

    logger.info(f"Source space created: {len(src[0]['vertno'])} + {len(src[1]['vertno'])} vertices")
    return src


def create_bem_model(
    fs_subject: str,
    subjects_dir: Path,
    conductivity: Tuple[float, ...] = (0.3,),
) -> mne.bem.ConductorModel:
    """Create BEM solution for the resolved FreeSurfer subject."""
    logger.info(f"Creating BEM model for {fs_subject}...")

    model = mne.make_bem_model(
        subject=fs_subject,
        ico=5,
        conductivity=conductivity,
        subjects_dir=subjects_dir,
        verbose=False,
    )

    bem = mne.make_bem_solution(model, verbose=False)
    logger.info("BEM solution created")

    return bem


def compute_forward_solution(
    preproc_path: BIDSPath,
    trans_path: BIDSPath,
    src: mne.SourceSpaces,
    bem: mne.bem.ConductorModel,
    n_jobs: int = -1,
) -> mne.Forward:
    """Compute forward solution.

    Args:
        preproc_path: BIDSPath to preprocessed data
        trans_path: BIDSPath to transformation
        src: Source spaces
        bem: BEM solution
        n_jobs: Number of parallel jobs

    Returns:
        Forward solution
    """
    logger.info("Computing forward solution...")

    preproc_fpath = str(preproc_path.fpath)
    trans_fpath = str(trans_path.fpath)

    fwd = mne.make_forward_solution(
        preproc_fpath,
        trans=trans_fpath,
        src=src,
        bem=bem,
        meg=True,
        eeg=False,
        mindist=0.0,
        n_jobs=n_jobs,
        verbose=False,
    )

    logger.info(f"Forward solution computed: {fwd['nsource']} sources, {fwd['nchan']} channels")
    return fwd


def compute_noise_covariance(noise_path: BIDSPath) -> mne.Covariance:
    """Compute noise covariance from empty-room recording.

    Args:
        noise_path: BIDSPath to empty-room noise recording

    Returns:
        Noise covariance matrix
    """
    logger.info("Computing noise covariance from empty-room recording...")

    noise_raw = mne.io.read_raw_fif(str(noise_path.fpath), preload=True, verbose=False)

    noise_cov = mne.compute_raw_covariance(
        noise_raw,
        method=["shrunk", "empirical"],
        rank=None,
        verbose=False,
    )

    logger.info(f"Noise covariance computed: rank={noise_cov['data'].shape[0]}")
    return noise_cov


def apply_inverse_continuous(
    preproc_path: BIDSPath,
    fwd: mne.Forward,
    noise_cov: mne.Covariance,
    method: str = "dSPM",
    snr: float = 3.0,
    loose: float = 0.2,
    depth: float = 0.8,
) -> SourceEstimate:
    """Apply inverse operator to continuous data.

    Args:
        preproc_path: BIDSPath to preprocessed continuous data
        fwd: Forward solution
        noise_cov: Noise covariance
        method: Inverse method ("MNE", "dSPM", "sLORETA")
        snr: Signal-to-noise ratio
        loose: Loose orientation constraint (0-1)
        depth: Depth weighting (0-1)

    Returns:
        Source estimate (continuous)
    """
    logger.info(f"Applying inverse solution (method={method}, SNR={snr})...")

    # Load preprocessed data
    preproc = mne.io.read_raw_fif(preproc_path, preload=True, verbose=False)
    info = preproc.info

    # Create inverse operator
    lambda2 = 1.0 / snr**2
    inverse_operator = make_inverse_operator(
        info, fwd, noise_cov, loose=loose, depth=depth, verbose=False
    )

    # Apply inverse to continuous data
    stc = apply_inverse_raw(
        preproc,
        inverse_operator,
        lambda2=lambda2,
        method=method,
        pick_ori=None,
        verbose=False,
    )

    logger.info(f"Inverse applied: {stc.data.shape[0]} sources, {stc.data.shape[1]} time points")
    return stc


def apply_inverse_to_epochs(
    epoch_path: BIDSPath,
    fwd: mne.Forward,
    noise_cov: mne.Covariance,
    method: str = "dSPM",
    snr: float = 3.0,
    loose: float = 0.2,
    depth: float = 0.8,
) -> List[SourceEstimate]:
    """Apply inverse operator to epochs.

    Args:
        epoch_path: BIDSPath to preprocessed epochs
        fwd: Forward solution
        noise_cov: Noise covariance
        method: Inverse method
        snr: Signal-to-noise ratio
        loose: Loose orientation constraint
        depth: Depth weighting

    Returns:
        List of source estimates (one per epoch)
    """
    logger.info("Applying inverse solution to epochs...")

    # Load epochs
    epochs = mne.read_epochs(epoch_path, preload=True, verbose=False)
    info = epochs.info

    # Create inverse operator
    lambda2 = 1.0 / snr**2
    inverse_operator = make_inverse_operator(
        info, fwd, noise_cov, loose=loose, depth=depth, verbose=False
    )

    # Apply inverse to all epochs
    stcs = apply_inverse_epochs(
        epochs,
        inverse_operator,
        lambda2=lambda2,
        method=method,
        pick_ori=None,
        verbose=False,
    )

    logger.info(f"Inverse applied to {len(stcs)} epochs")
    return stcs


def morph_to_fsaverage(
    stcs: List[SourceEstimate],
    fwd: mne.Forward,
    fs_subject: str,
    subjects_dir: Path,
) -> List[SourceEstimate]:
    """Morph source estimates from the resolved subject to fsaverage."""
    logger.info("Morphing source estimates to fsaverage...")

    # Ensure fsaverage source space exists
    fsaverage_fpath = op.join(subjects_dir, "fsaverage", "bem", "fsaverage-oct-6-src.fif")
    if not op.exists(fsaverage_fpath):
        logger.info("Creating fsaverage source space...")
        src = mne.setup_source_space(
            "fsaverage",
            spacing="oct6",
            subjects_dir=subjects_dir,
            add_dist=False,
            verbose=False,
        )
        src_fname = op.join(subjects_dir, "fsaverage", "bem", "fsaverage-oct-6-src.fif")
        mne.write_source_spaces(src_fname, src, overwrite=True)

    # Load fsaverage source space
    src_to = mne.read_source_spaces(fsaverage_fpath, verbose=False)

    # Morph each source estimate
    morphed_stcs = []
    for idx, stc in enumerate(stcs):
        logger.debug(f"Morphing source estimate {idx+1}/{len(stcs)}")

        morph = mne.compute_source_morph(
            fwd["src"],
            subject_from=fs_subject,
            src_to=src_to,
            subject_to="fsaverage",
            subjects_dir=subjects_dir,
            verbose=False,
        ).apply(stc)

        # Convert to float32 to save space
        morphed_stc = SourceEstimate(
            data=np.float32(morph.data),
            vertices=morph.vertices,
            tmin=morph.tmin,
            tstep=morph.tstep,
            subject="fsaverage",
        )

        morphed_stcs.append(morphed_stc)

    logger.info(f"Morphed {len(morphed_stcs)} source estimates to fsaverage")
    return morphed_stcs


def save_source_estimate(
    stc: SourceEstimate,
    output_path: BIDSPath,
    epoch_idx: Optional[int] = None,
) -> Path:
    """Save source estimate to disk.

    Args:
        stc: Source estimate
        output_path: BIDSPath for output
        epoch_idx: Epoch index (for epoched data)

    Returns:
        Path to saved file
    """
    if epoch_idx is not None:
        filename = f"{str(output_path.fpath)}_epoch{epoch_idx}"
        logger.debug(f"Saving source estimate for epoch {epoch_idx}")
    else:
        filename = str(output_path.fpath)
        logger.debug("Saving continuous source estimate")

    stc.save(filename, ftype="h5", overwrite=True)
    saved_path = Path(f"{filename}-stc.h5")

    return saved_path


def has_real_mri(subject: str, subjects_dir: Path) -> bool:
    """Return True if sub-XX has a real recon-all MRI on disk.

    See _has_real_mri for the detection rule.
    """
    return _has_real_mri(Path(subjects_dir) / f"sub-{subject}")
