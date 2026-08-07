"""Synchronized permutation workers and family-wise decoding correction."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from code.classification.multifeature_scientific import (
    NestedRidgeConfig,
    max_statistic_pvalues,
    run_primary_analysis,
)


@dataclass(frozen=True)
class DecodingModelInput:
    """Aligned input for one prespecified decoding model."""

    tensor: np.ndarray
    labels: np.ndarray
    groups: np.ndarray


def run_decoding_permutation_chunk(
    model_inputs: Mapping[str, DecodingModelInput],
    label_generators: Mapping[
        str, Callable[[np.random.Generator, np.ndarray], np.ndarray]
    ],
    *,
    model_order: Sequence[str],
    config: NestedRidgeConfig,
    n_permutations: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Run synchronized null fits for joint, feature, and parcel families."""
    ordered = [model_inputs[name] for name in model_order]
    if set(model_order) != set(model_inputs) or set(model_order) != set(label_generators):
        raise ValueError("model inputs and label generators must match model order")
    feature_count = ordered[0].tensor.shape[2]
    parcel_count = ordered[0].tensor.shape[1]
    if any(item.tensor.shape[1:] != (parcel_count, feature_count) for item in ordered):
        raise ValueError("all models must share feature and parcel order")
    joint = np.empty((n_permutations, len(model_order)))
    standalone = np.empty((n_permutations, len(model_order), feature_count))
    feature = np.empty_like(standalone)
    parcel = np.empty((n_permutations, len(model_order), parcel_count))
    generator = np.random.default_rng(seed)
    for permutation in range(n_permutations):
        for model_index, name in enumerate(model_order):
            item = model_inputs[name]
            labels = label_generators[name](generator, item.labels)
            result = run_primary_analysis(
                item.tensor,
                labels,
                item.groups,
                config,
                n_permutations=0,
            )
            joint[permutation, model_index] = result["joint"]["metrics"]["roc_auc"]
            standalone[permutation, model_index] = [
                fitted["metrics"]["roc_auc"]
                for fitted in result["standalone-feature"]
            ]
            feature[permutation, model_index] = result["feature-contribution"][
                "mean_delta_auc"
            ]
            parcel[permutation, model_index] = result["region-contribution"][
                "mean_delta_auc"
            ]
    return {
        "joint_auc": joint,
        "standalone_feature_auc": standalone,
        "feature_contribution": feature,
        "parcel_contribution": parcel,
    }


def correct_decoding_families(
    observed: Mapping[str, np.ndarray], null: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Correct joint, feature, and parcel tests as three prespecified families."""
    required = (
        "joint_auc",
        "standalone_feature_auc",
        "feature_contribution",
        "parcel_contribution",
    )
    if any(name not in observed or name not in null for name in required):
        raise ValueError("observed and null decoding families are incomplete")
    corrected = {}
    corrected["joint_auc_p_fwer"] = max_statistic_pvalues(
        np.asarray(observed["joint_auc"]).ravel(),
        np.asarray(null["joint_auc"]).reshape(len(null["joint_auc"]), -1),
    ).reshape(np.asarray(observed["joint_auc"]).shape)
    feature_observed = np.concatenate(
        [
            np.asarray(observed["standalone_feature_auc"]).ravel(),
            np.asarray(observed["feature_contribution"]).ravel(),
        ]
    )
    feature_null = np.concatenate(
        [
            np.asarray(null["standalone_feature_auc"]).reshape(
                len(null["standalone_feature_auc"]), -1
            ),
            np.asarray(null["feature_contribution"]).reshape(
                len(null["feature_contribution"]), -1
            ),
        ],
        axis=1,
    )
    corrected["feature_family_p_fwer"] = max_statistic_pvalues(
        feature_observed, feature_null
    )
    corrected["parcel_family_p_fwer"] = max_statistic_pvalues(
        np.asarray(observed["parcel_contribution"]).ravel(),
        np.asarray(null["parcel_contribution"]).reshape(
            len(null["parcel_contribution"]), -1
        ),
    ).reshape(np.asarray(observed["parcel_contribution"]).shape)
    return corrected
