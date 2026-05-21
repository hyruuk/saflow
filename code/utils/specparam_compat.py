"""Compatibility helpers for spectral parameterization model classes."""

from __future__ import annotations

from typing import Any

import numpy as np


def load_spectral_model_classes() -> tuple[Any, Any]:
    """Load specparam model classes.

    Returns:
        Tuple of ``(SpectralModel, SpectralGroupModel)`` classes from
        ``specparam``.
    """
    from specparam import SpectralGroupModel, SpectralModel

    return SpectralModel, SpectralGroupModel


def load_spectral_model() -> Any:
    """Load the specparam single-spectrum model class.

    Returns:
        The installed ``specparam.SpectralModel`` class.
    """
    spectral_model_class, _ = load_spectral_model_classes()
    return spectral_model_class


def load_spectral_group_model() -> Any:
    """Load the specparam group model class.

    Returns:
        The installed ``specparam.SpectralGroupModel`` class.
    """
    _, spectral_group_model_class = load_spectral_model_classes()
    return spectral_group_model_class


def get_group_model(model_group: Any, index: int) -> Any:
    """Return one fitted model from a specparam group model.

    Args:
        model_group: Fitted ``SpectralGroupModel``.
        index: Model index to extract.

    Returns:
        A fitted ``SpectralModel`` for the requested index.
    """
    return model_group.get_model(index)


def model_has_fit(model: Any) -> bool:
    """Return whether a specparam model has a fitted result."""
    return bool(model.results.has_model)


def get_aperiodic_params(model: Any) -> np.ndarray:
    """Return aperiodic parameters as ``[offset, (knee), exponent]``."""
    return np.asarray(model.get_params("aperiodic"), dtype=float)


def get_r_squared(model: Any) -> float:
    """Return the spectral model R-squared goodness-of-fit."""
    return float(model.get_metrics("gof", "rsquared"))


def get_fit_freqs(model: Any) -> np.ndarray:
    """Return the frequency bins used inside the model fit range."""
    return np.asarray(model.data.freqs, dtype=float)


def get_peak_fit(model: Any) -> np.ndarray:
    """Return the summed Gaussian peak fit in log10 space."""
    return np.asarray(model.results.model._peak_fit, dtype=float)


def get_modeled_spectrum(model: Any) -> np.ndarray:
    """Return the full fitted spectrum in log10 space."""
    return np.asarray(model.results.model.modeled_spectrum, dtype=float)
