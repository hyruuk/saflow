"""Classification analysis for decoding IN vs OUT attentional states.

This script performs machine learning classification to decode attentional
states (IN vs OUT) from neural features using various classifiers and
cross-validation strategies.

Usage:
    python -m code.classification.run_classification \
        --features fooof_exponent \
        --clf lda \
        --cv logo \
        --space sensor
"""

import argparse
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, GroupKFold

from code.classification.classifiers import get_classifier
from code.utils.data_loading import load_features, balance_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: Optional[Path] = None) -> Dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. Defaults to ./config.yaml

    Returns:
        Configuration dictionary.
    """
    if config_path is None:
        config_path = Path("config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")
    return config


def get_git_hash() -> str:
    """Get current git commit hash for provenance tracking.

    Returns:
        Git commit hash, or 'unknown' if not in a git repo.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def inout_bounds_to_string(bounds: Tuple[int, int]) -> str:
    """Convert INOUT bounds to string format for filenames.

    Args:
        bounds: Tuple of (lower_percentile, upper_percentile).

    Returns:
        String like "2575" or "5050".
    """
    return f"{bounds[0]}{bounds[1]}"


def get_feature_folder(
    config: Dict,
    feature_type: str,
    space: str,
) -> Path:
    """Get the path to feature folder based on feature type and space.

    Args:
        config: Configuration dictionary.
        feature_type: Feature type (e.g., 'fooof_exponent', 'psd_alpha', 'lzc').
        space: Analysis space ('sensor', 'source', 'atlas').

    Returns:
        Path to feature folder.
    """
    data_root = Path(config["paths"]["data_root"])
    processed = data_root / config["paths"]["processed"]

    # Map feature types to folder names
    if feature_type.startswith("fooof_"):
        folder_name = f"features_fooof_{space}"
    elif feature_type.startswith("psd_"):
        folder_name = f"features_psd_{space}"
    elif feature_type.startswith("lzc"):
        folder_name = f"features_lzc_{space}"
    else:
        folder_name = f"features_{feature_type}_{space}"

    return processed / folder_name


def load_classification_data(
    feature_types: List[str],
    space: str,
    inout_bounds: Tuple[int, int],
    config: Dict,
    balance: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Load and optionally combine multiple feature types for classification.

    Args:
        feature_types: List of feature types to load and combine.
        space: Analysis space ('sensor', 'source', 'atlas').
        inout_bounds: Tuple of (lower_percentile, upper_percentile).
        config: Configuration dictionary.
        balance: Whether to balance classes. Defaults to True.

    Returns:
        Tuple containing:
        - X: Feature data, shape (n_trials, n_total_features) - flattened for sklearn
        - y: Labels (0=IN, 1=OUT), shape (n_trials,)
        - groups: Subject indices, shape (n_trials,)
        - metadata: Dictionary with loading info
    """
    logger.info(f"Loading {len(feature_types)} feature type(s) for classification")

    all_X = []
    y = None
    groups = None
    all_metadata = []

    inout_str = f"INOUT_{inout_bounds[0]}{inout_bounds[1]}"

    for feature_type in feature_types:
        logger.info(f"Loading feature: {feature_type}")

        # Get feature folder
        feature_folder = get_feature_folder(config, feature_type, space)

        if not feature_folder.exists():
            raise FileNotFoundError(f"Feature folder not found: {feature_folder}")

        # Determine feature key for load_features
        if feature_type.startswith("fooof_"):
            feature_name = feature_type.replace("fooof_", "")
            feature_key = "slope" if feature_name == "exponent" else feature_name
        elif feature_type.startswith("psd_"):
            feature_key = "psd"
        elif feature_type.startswith("lzc"):
            feature_key = "lzc"
        else:
            feature_key = feature_type

        # Load features
        X_feat, y_feat, groups_feat, metadata = load_features(
            feature_folder=feature_folder,
            feature=feature_key,
            splitby="inout",
            inout=inout_str,
            remove_errors=True,
            get_task=["correct_commission"],
        )

        # Ensure consistent labels and groups across features
        if y is None:
            y = y_feat
            groups = groups_feat
        else:
            # Verify consistency
            if not np.array_equal(y, y_feat):
                logger.warning("Inconsistent labels across features")
            if not np.array_equal(groups, groups_feat):
                logger.warning("Inconsistent groups across features")

        # Flatten spatial dimension: (n_features, n_trials, n_spatial) -> (n_trials, n_features*n_spatial)
        X_flat = X_feat.transpose(1, 0, 2).reshape(X_feat.shape[1], -1)
        all_X.append(X_flat)
        all_metadata.append(metadata)

        logger.info(
            f"  Loaded {X_flat.shape[1]} features from {len(np.unique(groups_feat))} subjects"
        )

    # Concatenate features
    X = np.concatenate(all_X, axis=1)

    logger.info(
        f"Combined features: {X.shape[0]} trials × {X.shape[1]} total features "
        f"(IN: {np.sum(y == 0)}, OUT: {np.sum(y == 1)})"
    )

    # Balance dataset if requested
    if balance:
        logger.info("Balancing dataset...")
        # Reshape for balance_dataset: (n_features, n_trials, 1)
        X_reshaped = X.T[:, :, np.newaxis]
        X_balanced, y_balanced, groups_balanced = balance_dataset(
            X_reshaped, y, groups
        )
        # Reshape back: (n_trials, n_features)
        X = X_balanced.squeeze(-1).T
        y = y_balanced
        groups = groups_balanced

        logger.info(
            f"Balanced to {len(y)} trials (IN: {np.sum(y == 0)}, OUT: {np.sum(y == 1)})"
        )

    # Compile metadata
    metadata_out = {
        "feature_types": feature_types,
        "space": space,
        "inout_bounds": inout_bounds,
        "n_subjects": len(np.unique(groups)),
        "n_trials": len(y),
        "n_features": X.shape[1],
        "n_in": int(np.sum(y == 0)),
        "n_out": int(np.sum(y == 1)),
        "balanced": balance,
    }

    # Validate sufficient trials
    if metadata_out["n_in"] < 20 or metadata_out["n_out"] < 20:
        logger.warning(
            f"Low trial counts: IN={metadata_out['n_in']}, OUT={metadata_out['n_out']}. "
            "Results may be unreliable."
        )

    return X, y, groups, metadata_out


def get_cv_strategy(cv_name: str, n_splits: int = 5, groups: Optional[np.ndarray] = None) -> object:
    """Get cross-validation splitter.

    Args:
        cv_name: CV strategy name ('logo', 'stratified', 'group').
        n_splits: Number of splits for K-fold strategies.
        groups: Group labels for group-based CV.

    Returns:
        Scikit-learn CV splitter object.
    """
    if cv_name == "logo":
        # Leave-One-Group-Out (leave-one-subject-out)
        cv = LeaveOneGroupOut()
        logger.info("Using LeaveOneGroupOut cross-validation")

    elif cv_name == "stratified":
        # Stratified K-Fold (maintains class balance)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        logger.info(f"Using StratifiedKFold cross-validation (n_splits={n_splits})")

    elif cv_name == "group":
        # Group K-Fold (keeps groups together)
        cv = GroupKFold(n_splits=n_splits)
        logger.info(f"Using GroupKFold cross-validation (n_splits={n_splits})")

    else:
        raise ValueError(f"Unknown CV strategy: {cv_name}")

    return cv


def run_classification_with_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    clf: object,
    cv: object,
    n_permutations: int = 1000,
) -> Dict:
    """Run classification with cross-validation and permutation testing.

    Args:
        X: Feature data, shape (n_trials, n_features).
        y: Labels (0=IN, 1=OUT), shape (n_trials,).
        groups: Subject indices, shape (n_trials,).
        clf: Scikit-learn classifier instance.
        cv: Cross-validation splitter.
        n_permutations: Number of permutations for significance testing.

    Returns:
        Dictionary containing:
        - scores: CV scores
        - predictions: Predicted labels for each fold
        - perm_scores: Permutation scores
        - confusion_matrix: Confusion matrix
        - roc_auc: ROC AUC score
    """
    from sklearn.metrics import (
        confusion_matrix,
        roc_auc_score,
        accuracy_score,
    )
    from sklearn.model_selection import cross_val_score, permutation_test_score

    logger.info("Running cross-validated classification...")

    # Cross-validation scores
    if hasattr(cv, 'split'):
        # For group-based CV, pass groups
        if isinstance(cv, (LeaveOneGroupOut, GroupKFold)):
            scores = cross_val_score(
                clf, X, y, cv=cv, groups=groups, scoring="roc_auc", n_jobs=-1
            )
        else:
            scores = cross_val_score(
                clf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1
            )
    else:
        raise ValueError("CV object must have a split() method")

    mean_score = np.mean(scores)
    std_score = np.std(scores)

    logger.info(
        f"CV scores: {mean_score:.3f} ± {std_score:.3f} "
        f"(range: [{np.min(scores):.3f}, {np.max(scores):.3f}])"
    )

    # Get predictions for confusion matrix
    y_pred = np.zeros_like(y)
    if isinstance(cv, (LeaveOneGroupOut, GroupKFold)):
        for train_idx, test_idx in cv.split(X, y, groups):
            clf.fit(X[train_idx], y[train_idx])
            y_pred[test_idx] = clf.predict(X[test_idx])
    else:
        for train_idx, test_idx in cv.split(X, y):
            clf.fit(X[train_idx], y[train_idx])
            y_pred[test_idx] = clf.predict(X[test_idx])

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    logger.info(f"Confusion matrix:\n{cm}")

    # Overall accuracy
    accuracy = accuracy_score(y, y_pred)
    logger.info(f"Overall accuracy: {accuracy:.3f}")

    # Permutation testing for significance
    logger.info(f"Running permutation test with {n_permutations} permutations...")

    if isinstance(cv, (LeaveOneGroupOut, GroupKFold)):
        score, perm_scores, pvalue = permutation_test_score(
            clf,
            X,
            y,
            groups=groups,
            cv=cv,
            n_permutations=n_permutations,
            scoring="roc_auc",
            n_jobs=-1,
        )
    else:
        score, perm_scores, pvalue = permutation_test_score(
            clf,
            X,
            y,
            cv=cv,
            n_permutations=n_permutations,
            scoring="roc_auc",
            n_jobs=-1,
        )

    logger.info(
        f"Permutation test: score={score:.3f}, p-value={pvalue:.4f} "
        f"(perm mean={np.mean(perm_scores):.3f})"
    )

    results = {
        "scores": scores,
        "mean_score": float(mean_score),
        "std_score": float(std_score),
        "predictions": y_pred,
        "confusion_matrix": cm,
        "accuracy": float(accuracy),
        "perm_scores": perm_scores,
        "perm_pvalue": float(pvalue),
        "roc_auc": float(mean_score),  # Same as mean CV score when scoring='roc_auc'
    }

    return results


def save_classification_results(
    output_dir: Path,
    feature_types: List[str],
    inout_bounds: Tuple[int, int],
    clf_name: str,
    cv_name: str,
    results: Dict,
    metadata: Dict,
    config: Dict,
) -> None:
    """Save classification results with provenance metadata.

    Args:
        output_dir: Directory to save results.
        feature_types: List of feature types used.
        inout_bounds: IN/OUT bounds.
        clf_name: Classifier name.
        cv_name: Cross-validation strategy name.
        results: Classification results dictionary.
        metadata: Metadata from data loading.
        config: Configuration dictionary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    inout_str = inout_bounds_to_string(inout_bounds)
    features_str = "+".join(feature_types)
    base_name = f"feature-{features_str}_inout-{inout_str}_clf-{clf_name}_cv-{cv_name}"

    # Save scores
    scores_file = output_dir / f"{base_name}_scores.npz"
    np.savez_compressed(
        scores_file,
        scores=results["scores"],
        perm_scores=results["perm_scores"],
    )
    logger.info(f"Saved scores to {scores_file}")

    # Save predictions and confusion matrix
    predictions_file = output_dir / f"{base_name}_predictions.npz"
    np.savez_compressed(
        predictions_file,
        predictions=results["predictions"],
        confusion_matrix=results["confusion_matrix"],
    )
    logger.info(f"Saved predictions to {predictions_file}")

    # Save metadata
    metadata_out = {
        "feature_types": feature_types,
        "inout_bounds": list(inout_bounds),
        "classifier": clf_name,
        "cv_strategy": cv_name,
        "timestamp": datetime.now().isoformat(),
        "git_hash": get_git_hash(),
        "data_metadata": metadata,
        "results": {
            "mean_score": results["mean_score"],
            "std_score": results["std_score"],
            "accuracy": results["accuracy"],
            "roc_auc": results["roc_auc"],
            "perm_pvalue": results["perm_pvalue"],
            "n_permutations": len(results["perm_scores"]),
        },
    }

    metadata_file = output_dir / f"{base_name}_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata_out, f, indent=2)
    logger.info(f"Saved metadata to {metadata_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run classification analysis for IN vs OUT states"
    )
    parser.add_argument(
        "--features",
        type=str,
        nargs="+",
        required=True,
        help="Feature type(s) (e.g., fooof_exponent psd_alpha)",
    )
    parser.add_argument(
        "--clf",
        type=str,
        default="lda",
        choices=["lda", "svm", "rf", "logistic"],
        help="Classifier type",
    )
    parser.add_argument(
        "--cv",
        type=str,
        default="logo",
        choices=["logo", "stratified", "group"],
        help="Cross-validation strategy",
    )
    parser.add_argument(
        "--space",
        type=str,
        default="sensor",
        choices=["sensor", "source", "atlas"],
        help="Analysis space",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=1000,
        help="Number of permutations for significance testing",
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Disable class balancing",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Generate visualization plots",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(Path(args.config))

    # Get INOUT bounds from config
    inout_bounds = tuple(config["analysis"]["inout_bounds"])

    # Set up output directory
    data_root = Path(config["paths"]["data_root"])
    output_dir = data_root / config["paths"]["processed"] / f"classification_{args.space}" / "group"

    logger.info("=" * 80)
    logger.info("CLASSIFICATION ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Features: {args.features}")
    logger.info(f"Classifier: {args.clf}")
    logger.info(f"CV strategy: {args.cv}")
    logger.info(f"Space: {args.space}")
    logger.info(f"IN/OUT bounds: {inout_bounds}")
    logger.info(f"Balance classes: {not args.no_balance}")
    logger.info("=" * 80)

    # Load data
    logger.info("Loading data...")
    X, y, groups, metadata = load_classification_data(
        feature_types=args.features,
        space=args.space,
        inout_bounds=inout_bounds,
        config=config,
        balance=not args.no_balance,
    )

    # Get classifier
    logger.info("Initializing classifier...")
    clf = get_classifier(args.clf)

    # Get CV strategy
    logger.info("Setting up cross-validation...")
    cv = get_cv_strategy(args.cv, groups=groups)

    # Run classification
    logger.info("Running classification...")
    results = run_classification_with_cv(
        X=X,
        y=y,
        groups=groups,
        clf=clf,
        cv=cv,
        n_permutations=args.n_permutations,
    )

    # Save results
    logger.info("Saving results...")
    save_classification_results(
        output_dir=output_dir,
        feature_types=args.features,
        inout_bounds=inout_bounds,
        clf_name=args.clf,
        cv_name=args.cv,
        results=results,
        metadata=metadata,
        config=config,
    )

    # Visualization
    if args.visualize:
        logger.info("Generating visualizations...")
        logger.info("Visualization not yet implemented")

    logger.info("=" * 80)
    logger.info("CLASSIFICATION COMPLETE")
    logger.info(f"Mean ROC AUC: {results['mean_score']:.3f} ± {results['std_score']:.3f}")
    logger.info(f"Permutation p-value: {results['perm_pvalue']:.4f}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
