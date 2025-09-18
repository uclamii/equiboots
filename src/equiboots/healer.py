import numpy as np
from typing import Optional, Dict, Tuple, Union
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
)


def find_group_thresholds(
    y_true: Union[np.ndarray, list],
    y_prob: Union[np.ndarray, list],
    group_vec: Union[np.ndarray, list],
    reference_group: Union[str, int],
    ref_metrics: Optional[Dict[str, float]] = None,
    threshold_range: Tuple[float, float] = (0.1, 0.9),
    n_steps: int = 100,
    default_threshold: float = 0.5,
) -> Dict[Union[str, int], float]:
    """
    Find per-group probability thresholds that minimize the sum of absolute
    differences from a reference group's metrics: accuracy, precision, recall, specificity.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary ground truth labels, 0 or 1.
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.
    group_vec : array-like of shape (n_samples,)
        Group labels, e.g., race or sex for each sample.
    reference_group : hashable
        The group used as the baseline.
    ref_metrics : dict or None
        If provided, must have keys 'accuracy', 'precision', 'recall', 'specificity'.
        If None, they are computed from the reference group at default_threshold.
    threshold_range : tuple(float, float)
        Inclusive range of thresholds to search.
    n_steps : int
        Number of thresholds to evaluate between the range endpoints.
    default_threshold : float
        Used for the reference group's fixed threshold and as the initial best value.

    Returns
    -------
    dict
        Mapping group -> optimal threshold.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    group_vec = np.asarray(group_vec).ravel()

    if not np.isin(reference_group, group_vec).any():
        raise ValueError(f"reference_group '{reference_group}' not found in group_vec")

    # Compute reference metrics if not provided
    if ref_metrics is None:
        ref_mask = group_vec == reference_group
        ref_true = y_true[ref_mask]
        ref_preds = (y_prob[ref_mask] >= default_threshold).astype(int)
        ref_metrics = {
            "accuracy": accuracy_score(ref_true, ref_preds),
            "precision": precision_score(ref_true, ref_preds),
            "recall": recall_score(ref_true, ref_preds),
            "specificity": recall_score(ref_true, ref_preds, pos_label=0),
        }

    required = {"accuracy", "precision", "recall", "specificity"}
    missing = required.difference(ref_metrics.keys())
    if missing:
        raise ValueError(f"ref_metrics missing keys: {sorted(missing)}")

    thresholds = {}
    unique_groups = np.unique(group_vec)

    for g in unique_groups:
        if g == reference_group:
            thresholds[g] = default_threshold
            continue

        mask = group_vec == g
        probs = y_prob[mask]
        true = y_true[mask]

        best_t = default_threshold
        best_diff = float("inf")

        for t in np.linspace(threshold_range[0], threshold_range[1], n_steps):
            preds = (probs >= t).astype(int)
            acc = accuracy_score(true, preds)
            prec = precision_score(true, preds, zero_division=0)
            rec = recall_score(true, preds, zero_division=0)
            spec = recall_score(
                true, preds, pos_label=0, zero_division=0
            )  # specificity

            diff = (
                abs(acc - ref_metrics["accuracy"])
                + abs(prec - ref_metrics["precision"])
                + abs(rec - ref_metrics["recall"])
                + abs(spec - ref_metrics["specificity"])
            )
            if diff < best_diff:
                best_diff = diff
                best_t = t

        thresholds[g] = best_t

    return thresholds


# This function applies custom thresholds for each group based on their group label
def grouped_threshold_predict(
    y_prob, group_labels, group_thresholds, default_threshold=0.5
):
    """
    Convert predicted probabilities to class labels using a threshold per group.
    """
    predictions = np.zeros_like(
        y_prob, dtype=int
    )  # Initialize array of 0s for predicted labels
    for group in np.unique(
        group_labels
    ):  # Loop over each unique group (e.g., each race)
        idx = group_labels == group  # Get indices where group label matches
        threshold = group_thresholds.get(
            group, default_threshold
        )  # Use custom or default threshold
        predictions[idx] = (y_prob[idx] >= threshold).astype(
            int
        )  # Apply threshold to get predictions
    return predictions
