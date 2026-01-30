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

import logging
import os
import os.path as op
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np
from mne import SourceEstimate
from mne.minimum_norm import apply_inverse_epochs, apply_inverse_raw, make_inverse_operator
from mne_bids import BIDSPath, read_raw_bids

logger = logging.getLogger(__name__)


def create_output_paths(
    subject: str,
    run: str,
    bids_root: Path,
    derivatives_root: Path,
) -> Dict[str, BIDSPath]:
    """Create BIDSPath objects for all source reconstruction inputs/outputs.

    Args:
        subject: Subject ID (e.g., "04")
        run: Run number (e.g., "02")
        bids_root: Path to BIDS dataset root
        derivatives_root: Path to derivatives directory

    Returns:
        Dictionary mapping names to BIDSPath objects:
            - 'raw': Raw BIDS data
            - 'preproc': Preprocessed continuous data
            - 'epoch': Preprocessed epochs
            - 'noise': Empty-room noise recording
            - 'trans': Coregistration transform
            - 'fwd': Forward solution
            - 'noise_cov': Noise covariance matrix
            - 'stc': Source estimate (continuous)
            - 'morph': Morphed source estimate
    """
    logger.debug(f"Creating output paths for sub-{subject}, run-{run}")

    # Input: raw BIDS data
    raw_bidspath = BIDSPath(
        subject=subject,
        task="gradCPT",
        run=run,
        datatype="meg",
        suffix="meg",
        extension=".ds",
        root=bids_root,
    )

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

    # Find date for noise file
    try:
        raw = read_raw_bids(raw_bidspath, verbose=False)
        er_date = raw.info["meas_date"].strftime("%Y%m%d")
        logger.debug(f"Found empty-room date: {er_date}")
    except Exception as e:
        logger.warning(f"Could not read raw data to get empty-room date: {e}")
        er_date = "20000101"  # Fallback

    # Input: noise recording
    noise_bidspath = BIDSPath(
        subject="emptyroom",
        session=er_date,
        task="noise",
        datatype="meg",
        root=bids_root,
    )

    # Output: coregistration transform
    trans_bidspath = BIDSPath(
        subject=subject,
        task="gradCPT",
        run=run,
        datatype="meg",
        processing="trans",
        root=derivatives_root / "trans",
    )
    trans_bidspath.mkdir(exist_ok=True)

    # Output: forward solution
    fwd_bidspath = BIDSPath(
        subject=subject,
        task="gradCPT",
        run=run,
        datatype="meg",
        processing="forward",
        root=derivatives_root / "fwd",
    )
    fwd_bidspath.mkdir(exist_ok=True)

    # Output: noise covariance
    noise_cov_bidspath = BIDSPath(
        subject="emptyroom",
        session=er_date,
        task="noise",
        datatype="meg",
        root=derivatives_root / "noise_cov",
    )
    noise_cov_bidspath.mkdir(exist_ok=True)

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
        "raw": raw_bidspath,
        "preproc": preproc_bidspath,
        "epoch": epoch_bidspath,
        "noise": noise_bidspath,
        "trans": trans_bidspath,
        "fwd": fwd_bidspath,
        "noise_cov": noise_cov_bidspath,
        "stc": stc_bidspath,
        "morph": morph_bidspath,
    }


def compute_coregistration(
    raw_path: BIDSPath,
    trans_path: BIDSPath,
    subject: str,
    subjects_dir: Path,
    mri_available: bool = False,
) -> mne.transforms.Transform:
    """Compute coregistration between MEG and MRI coordinate systems.

    Uses MNE's automatic coregistration with ICP fitting.

    Args:
        raw_path: BIDSPath to raw data
        trans_path: BIDSPath to save transformation
        subject: Subject ID
        subjects_dir: FreeSurfer subjects directory
        mri_available: Whether individual MRI is available

    Returns:
        Transform object with MEG-to-MRI transformation
    """
    logger.info("Computing coregistration...")

    raw = read_raw_bids(raw_path, verbose=False)
    info = raw.info

    # Use individual MRI if available, otherwise fsaverage
    fs_subject = f"sub-{subject}" if mri_available else "fsaverage"
    logger.info(f"Using FreeSurfer subject: {fs_subject}")

    # Automatic coregistration with fiducials
    coreg = mne.coreg.Coregistration(
        info, fs_subject, subjects_dir, fiducials="auto"
    )

    # First pass: coarse fit
    logger.debug("Coregistration: First pass (coarse fit)")
    coreg.fit_icp(n_iterations=6, nasion_weight=2.0, verbose=False)

    # Remove outliers
    logger.debug("Coregistration: Removing outlier head shape points")
    coreg.omit_head_shape_points(distance=5.0 / 1000)

    # Second pass: fine fit
    logger.debug("Coregistration: Second pass (fine fit)")
    coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=False)

    # Save transformation
    trans_fpath = str(trans_path.fpath) + ".fif"
    os.makedirs(op.dirname(trans_fpath), exist_ok=True)
    mne.write_trans(trans_fpath, coreg.trans, overwrite=True)
    logger.info(f"Saved coregistration: {trans_fpath}")

    return coreg.trans


def setup_source_space(
    subject: str,
    subjects_dir: Path,
    mri_available: bool = False,
    spacing: str = "oct6",
) -> mne.SourceSpaces:
    """Create source space.

    Args:
        subject: Subject ID
        subjects_dir: FreeSurfer subjects directory
        mri_available: Whether individual MRI is available
        spacing: Source space spacing (e.g., "oct6")

    Returns:
        Source spaces object
    """
    fs_subject = f"sub-{subject}" if mri_available else "fsaverage"
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
    subject: str,
    subjects_dir: Path,
    mri_available: bool = False,
    conductivity: Tuple[float, ...] = (0.3,),
) -> mne.bem.ConductorModel:
    """Create BEM (Boundary Element Model) solution.

    Args:
        subject: Subject ID
        subjects_dir: FreeSurfer subjects directory
        mri_available: Whether individual MRI is available
        conductivity: Conductivity values (single-layer for MEG)

    Returns:
        BEM solution
    """
    fs_subject = f"sub-{subject}" if mri_available else "fsaverage"
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

    noise_raw = mne.io.read_raw_ctf(noise_path, preload=True, verbose=False)

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
    subject: str,
    subjects_dir: Path,
    mri_available: bool = False,
) -> List[SourceEstimate]:
    """Morph source estimates to fsaverage template.

    Args:
        stcs: List of source estimates
        fwd: Forward solution
        subject: Subject ID
        subjects_dir: FreeSurfer subjects directory
        mri_available: Whether individual MRI is available

    Returns:
        List of morphed source estimates
    """
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

    # Determine subject for morphing
    fs_subject = f"sub-{subject}" if mri_available else "fsaverage"

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


def check_mri_availability(subject: str, subjects_dir: Path) -> bool:
    """Check if individual MRI is available for subject.

    Args:
        subject: Subject ID
        subjects_dir: FreeSurfer subjects directory

    Returns:
        True if individual MRI exists, False otherwise
    """
    subject_path = subjects_dir / f"sub-{subject}"
    available = subject_path.exists()

    if available:
        logger.info(f"Individual MRI found for sub-{subject}")
    else:
        logger.info(f"No individual MRI for sub-{subject}, using fsaverage")

    return available
