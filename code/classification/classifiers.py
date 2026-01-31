"""Classifier definitions and utilities for classification analysis.

This module provides:
- Classifier instantiation with default parameters
- Hyperparameter grids for tuning
- Classifier-specific utilities

Supported classifiers:
- LDA: Linear Discriminant Analysis
- SVM: Support Vector Machine
- RF: Random Forest
- Logistic: Logistic Regression
"""

import logging
from typing import Dict, Optional

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


def get_classifier(
    clf_name: str,
    random_state: int = 42,
    **kwargs,
) -> object:
    """Get classifier instance with default or custom parameters.

    Args:
        clf_name: Classifier name ('lda', 'svm', 'rf', 'logistic').
        random_state: Random seed for reproducibility.
        **kwargs: Additional classifier-specific parameters.

    Returns:
        Scikit-learn classifier instance.

    Examples:
        >>> clf = get_classifier('lda')
        >>> clf = get_classifier('svm', C=1.0, kernel='rbf')
    """
    if clf_name == "lda":
        clf = get_lda(**kwargs)

    elif clf_name == "svm":
        clf = get_svm(random_state=random_state, **kwargs)

    elif clf_name == "rf":
        clf = get_random_forest(random_state=random_state, **kwargs)

    elif clf_name == "logistic":
        clf = get_logistic_regression(random_state=random_state, **kwargs)

    else:
        raise ValueError(
            f"Unknown classifier: {clf_name}. "
            f"Supported: lda, svm, rf, logistic"
        )

    logger.info(f"Initialized {clf_name} classifier: {clf}")
    return clf


def get_lda(
    solver: str = "svd",
    shrinkage: Optional[str] = None,
    **kwargs,
) -> LinearDiscriminantAnalysis:
    """Get Linear Discriminant Analysis classifier.

    LDA is a simple, fast, and interpretable linear classifier that works
    well for high-dimensional data with few samples.

    Args:
        solver: Solver to use ('svd', 'lsqr', 'eigen').
        shrinkage: Shrinkage parameter ('auto', None, or float).
        **kwargs: Additional parameters for LinearDiscriminantAnalysis.

    Returns:
        LDA classifier instance.

    Examples:
        >>> clf = get_lda()
        >>> clf = get_lda(shrinkage='auto', solver='lsqr')
    """
    # SVD solver doesn't support shrinkage
    if shrinkage is not None and solver == "svd":
        solver = "lsqr"
        logger.debug(f"Changed solver to 'lsqr' to support shrinkage={shrinkage}")

    clf = LinearDiscriminantAnalysis(
        solver=solver,
        shrinkage=shrinkage,
        **kwargs,
    )

    return clf


def get_svm(
    C: float = 1.0,
    kernel: str = "rbf",
    gamma: str = "scale",
    probability: bool = True,
    random_state: int = 42,
    **kwargs,
) -> SVC:
    """Get Support Vector Machine classifier.

    SVM is a powerful non-linear classifier that can handle high-dimensional
    data. Use probability=True to enable predict_proba for ROC AUC scoring.

    Args:
        C: Regularization parameter.
        kernel: Kernel type ('linear', 'rbf', 'poly', 'sigmoid').
        gamma: Kernel coefficient ('scale', 'auto', or float).
        probability: Enable probability estimates (needed for ROC AUC).
        random_state: Random seed.
        **kwargs: Additional parameters for SVC.

    Returns:
        SVM classifier instance.

    Examples:
        >>> clf = get_svm()
        >>> clf = get_svm(C=10.0, kernel='linear')
    """
    clf = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        probability=probability,
        random_state=random_state,
        **kwargs,
    )

    return clf


def get_random_forest(
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str = "sqrt",
    random_state: int = 42,
    n_jobs: int = -1,
    **kwargs,
) -> RandomForestClassifier:
    """Get Random Forest classifier.

    Random Forest is an ensemble method that builds multiple decision trees
    and combines their predictions. It's robust to overfitting and handles
    high-dimensional data well.

    Args:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of trees (None = unlimited).
        min_samples_split: Minimum samples required to split a node.
        min_samples_leaf: Minimum samples required at a leaf node.
        max_features: Number of features to consider for best split.
        random_state: Random seed.
        n_jobs: Number of jobs for parallel processing (-1 = all cores).
        **kwargs: Additional parameters for RandomForestClassifier.

    Returns:
        Random Forest classifier instance.

    Examples:
        >>> clf = get_random_forest()
        >>> clf = get_random_forest(n_estimators=200, max_depth=10)
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs,
    )

    return clf


def get_logistic_regression(
    C: float = 1.0,
    penalty: str = "l2",
    solver: str = "lbfgs",
    max_iter: int = 1000,
    random_state: int = 42,
    n_jobs: int = -1,
    **kwargs,
) -> LogisticRegression:
    """Get Logistic Regression classifier.

    Logistic Regression is a simple, interpretable linear classifier that
    provides probabilistic outputs.

    Args:
        C: Inverse of regularization strength (smaller = stronger).
        penalty: Regularization type ('l1', 'l2', 'elasticnet', 'none').
        solver: Optimization algorithm ('lbfgs', 'liblinear', 'saga').
        max_iter: Maximum iterations for convergence.
        random_state: Random seed.
        n_jobs: Number of jobs for parallel processing (-1 = all cores).
        **kwargs: Additional parameters for LogisticRegression.

    Returns:
        Logistic Regression classifier instance.

    Examples:
        >>> clf = get_logistic_regression()
        >>> clf = get_logistic_regression(C=10.0, penalty='l1', solver='saga')
    """
    clf = LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs,
    )

    return clf


def get_default_hyperparameters(clf_name: str) -> Dict:
    """Get default hyperparameters for a classifier.

    Args:
        clf_name: Classifier name.

    Returns:
        Dictionary of default hyperparameters.

    Examples:
        >>> params = get_default_hyperparameters('lda')
    """
    defaults = {
        "lda": {
            "solver": "svd",
            "shrinkage": None,
        },
        "svm": {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "probability": True,
        },
        "rf": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "max_features": "sqrt",
        },
        "logistic": {
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 1000,
        },
    }

    if clf_name not in defaults:
        raise ValueError(f"Unknown classifier: {clf_name}")

    return defaults[clf_name]


def get_hyperparameter_grid(clf_name: str) -> Dict:
    """Get hyperparameter search grid for GridSearchCV.

    Args:
        clf_name: Classifier name.

    Returns:
        Dictionary of hyperparameter ranges for grid search.

    Examples:
        >>> grid = get_hyperparameter_grid('svm')
        >>> from sklearn.model_selection import GridSearchCV
        >>> clf = GridSearchCV(get_classifier('svm'), grid)
    """
    grids = {
        "lda": {
            "solver": ["svd", "lsqr", "eigen"],
            "shrinkage": [None, "auto", 0.1, 0.5, 0.9],
        },
        "svm": {
            "C": [0.1, 1.0, 10.0, 100.0],
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        },
        "rf": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "max_features": ["sqrt", "log2", None],
        },
        "logistic": {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "saga"],
        },
    }

    if clf_name not in grids:
        raise ValueError(f"Unknown classifier: {clf_name}")

    return grids[clf_name]


def get_classifier_instance(clf_name: str, params: Dict, random_state: int = 42) -> object:
    """Instantiate classifier with specific parameters.

    Args:
        clf_name: Classifier name.
        params: Dictionary of classifier parameters.
        random_state: Random seed.

    Returns:
        Classifier instance with specified parameters.

    Examples:
        >>> params = {'C': 10.0, 'kernel': 'linear'}
        >>> clf = get_classifier_instance('svm', params)
    """
    return get_classifier(clf_name, random_state=random_state, **params)
