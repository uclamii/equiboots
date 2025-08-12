import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    confusion_matrix,
    r2_score,
    explained_variance_score,
    mean_squared_log_error,
)

from sklearn.calibration import calibration_curve

# --- Root-Mean-Squared Error fallback for old sklearns ------------------------
try:
    from sklearn.metrics import root_mean_squared_error  # ≥ 1.4
except ImportError:  # < 1.4

    def root_mean_squared_error(y_true, y_pred, **kwargs):
        # mean_squared_error(..., squared=False) --> RMSE
        return mean_squared_error(y_true, y_pred, squared=False, **kwargs)


# ------------------------------------------------------------------------------

from sklearn.preprocessing import MultiLabelBinarizer
from typing import Optional, List, Dict, Tuple


def binary_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Evaluate binary classification metrics.

    Parameters:
    - y_true: np.ndarray
        Array of true binary labels.
    - y_pred: np.ndarray
        Array of predicted binary class labels.
    - y_proba: Optional[np.ndarray]
        Array of predicted probabilities for the positive class.

    Returns:
    - metrics: Dict[str, float]
        Dictionary with various metrics such as accuracy, precision, recall,
        f1 score, TP, FP, FN, TN rates, prevalence, predicted prevalence, and
        optionally ROC AUC, log loss, and Brier score.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    prevalence = np.mean(y_true)
    predicted_prevalence = np.mean(y_pred)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "Specificity": recall_score(y_true, y_pred, pos_label=0),
        "TP Rate": tp / (tp + fn) if tp + fn > 0 else 0,
        "FP Rate": fp / (fp + tn) if fp + tn > 0 else 0,
        "FN Rate": fn / (tp + fn) if tp + fn > 0 else 0,
        "TN Rate": tn / (fp + tn) if fp + tn > 0 else 0,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Prevalence": prevalence,
        "Predicted Prevalence": predicted_prevalence,
    }

    if y_proba is not None:
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
        metrics.update(
            {
                "ROC AUC": roc_auc_score(y_true, y_proba),
                "Average Precision Score": average_precision_score(y_true, y_proba),
                "Log Loss": log_loss(y_true, y_proba),
                "Brier Score": brier_score_loss(y_true, y_proba),
                "Calibration AUC": calibration_auc(prob_pred, prob_true),
            }
        )
    return metrics


def multi_class_prevalence(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> Tuple[List[float], List[float]]:
    """
    Calculate prevalence and predicted prevalence for multi-class classification.

    Parameters:
    - y_true: np.ndarray
        Array of true class labels.
    - y_pred: np.ndarray
        Array of predicted class labels.
    - n_classes: int
        Number of classes.

    Returns:
    - Tuple[List[float], List[float]]
        Lists of prevalence and predicted prevalence for each class.
    """
    prevalence = [np.mean(y_true == i) for i in range(n_classes)]
    predicted_prevalence = [np.mean(y_pred == i) for i in range(n_classes)]
    return prevalence, predicted_prevalence


def multi_class_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    n_classes: Optional[int] = None,
    average: str = "weighted",
) -> Dict[str, List[float]]:
    """
    Evaluate multi-class classification metrics.

    Parameters:
    - y_true: np.ndarray
        Array of true class labels.
    - y_pred: np.ndarray
        Array of predicted class labels.
    - y_proba: Optional[np.ndarray]
        Matrix of predicted probabilities for each class.
    - n_classes: Optional[int]
        Number of classes. Required if y_proba is provided.
    - average: str
        Defines the type of averaging performed on the data.

    Returns:
    - metrics: Dict[str, List[float]]
        Dictionary with metrics like accuracy, precision, recall, f1 score,
        TP, FP, FN, TN rates, prevalence, predicted prevalence, and optionally
        ROC AUC, log loss, and Brier score.
    """
    # only for average precision required input format
    mlb = MultiLabelBinarizer()
    y_true_ap = mlb.fit_transform(y_true.reshape(-1, 1))

    prevalence, predicted_prevalence = multi_class_prevalence(y_true, y_pred, n_classes)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=average),
        "Recall": recall_score(y_true, y_pred, average=average),
        "F1 Score": f1_score(y_true, y_pred, average=average),
        "Prevalence": prevalence,
        "Predicted Prevalence": predicted_prevalence,
    }

    if y_proba is not None:
        metrics.update(
            {
                "ROC AUC": roc_auc_score(
                    y_true, y_proba, average=average, multi_class="ovr"
                ),
                "Average Precision Score": average_precision_score(
                    y_true_ap, y_proba, average=average
                ),
                "Log Loss": log_loss(y_true, y_proba),
            }
        )

    return metrics


def calculate_bootstrap_stats(
    group_boot_metrics: List[Dict], metric: str
) -> pd.DataFrame:
    """
    Calculate mean and 95% CI for a specific metric across all groups and bootstrap samples.

    Parameters:
    group_boot_metrics: List of nested dictionaries containing bootstrap samples
    metric: String name of the metric to analyze (e.g., 'Accuracy', 'Precision')

    Returns:
    DataFrame with columns: group, mean, ci_lower, ci_upper, std
    """

    # Get all group names from first sample
    groups = list(group_boot_metrics[0].keys())

    results = []

    for group in groups:
        # Extract all values for this metric and group across bootstrap samples
        values = []
        for sample in group_boot_metrics:
            if group in sample and metric in sample[group]:
                values.append(sample[group][metric])

        if values:  # If we have data for this group
            values = np.array(values)
            mean_val = np.mean(values)
            std_val = np.std(values)
            ci_lower = np.percentile(values, 2.5)
            ci_upper = np.percentile(values, 97.5)

            results.append(
                {
                    "group": group,
                    "mean": mean_val,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "std": std_val,
                    "n_samples": len(values),
                }
            )

    return pd.DataFrame(results)


def multi_label_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    average: str = "weighted",
) -> Dict[str, List[float]]:
    """
    Evaluate multi-label classification metrics.

    Parameters:
    - y_true: List[List[int]]
        True class labels for each sample.
    - y_pred: List[List[int]]
        Predicted class labels for each sample.
    - y_proba: Optional[np.ndarray]
        Probability estimates for each class.
    - average: str, optional, default='weighted'
        Strategy to average metric scores.

    Returns:
    - metrics: dict
        Dictionary containing the calculated metrics, such as accuracy,
        precision, recall, f1 score, and optionally ROC AUC, average precision
        score, log loss, and confusion matrix rates.
    """

    # Calculate unique row prevalence and predicted prevalence
    unique_true, true_counts = np.unique(y_true, axis=0, return_counts=True)
    unique_pred, pred_counts = np.unique(y_pred, axis=0, return_counts=True)

    prevalence = [count / len(y_true) for count in true_counts]
    predicted_prevalence = [count / len(y_pred) for count in pred_counts]
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=average),
        "Recall": recall_score(y_true, y_pred, average=average),
        "F1 Score": f1_score(y_true, y_pred, average=average),
        "Prevalence": prevalence,
        "Prevalence multi-labels": unique_true.tolist(),
        "Predicted Prevalence": predicted_prevalence,
        "Predicted Prevalence multi-labels": unique_pred.tolist(),
    }

    if y_proba is not None:
        metrics.update(
            {
                "ROC AUC": roc_auc_score(y_true, y_proba, average=average),
                "Average Precision Score": average_precision_score(
                    y_true, y_proba, average=average
                ),
                "Log Loss": log_loss(y_true, y_proba),
            }
        )

    return metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate regression metrics.

    Parameters:
    - y_true: np.ndarray
        Array of true values, with shape (n_samples,).
    - y_pred: np.ndarray
        Array of predicted values, with shape (n_samples,).

    Returns:
    - metrics: Dict[str, float]
        Dictionary containing the calculated metrics. Keys include
        'Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error',
        'R^2 Score', 'Explained Variance', and 'Mean Squared Log Error'.
    """
    metrics = {
        "Mean Absolute Error": mean_absolute_error(y_true, y_pred),
        "Mean Squared Error": mean_squared_error(y_true, y_pred),
        "Root Mean Squared Error": root_mean_squared_error(y_true, y_pred),
        # "Root Mean Squared Error": mean_squared_error(y_true, y_pred, squared=False),
        "R^2 Score": r2_score(y_true, y_pred),
        "Explained Variance": explained_variance_score(y_true, y_pred),
        "Mean Squared Log Error": mean_squared_log_error(y_true, y_pred),
        "Residual Mean": np.mean(y_true - y_pred),
    }

    return metrics


def metrics_dataframe(metrics_data: List[Dict[str, Dict[str, float]]]) -> pd.DataFrame:
    """
    Transform a list of metrics dictionaries into a flattened DataFrame.

    Returns
    -------
        A DataFrame with columns for each metric and an 'attribute_value' column
        indicating the group.
    """
    melted = pd.DataFrame(metrics_data).melt()
    df = melted["value"].apply(pd.Series).assign(attribute_value=melted["variable"])
    return df


def area_trap(curve1, curve2):
    y_diff = np.abs(curve1[:, 1] - curve2[:, 1])
    x = curve1[:, 0]
    return np.trapz(y_diff, x)


def calibration_auc(mean_pred: np.ndarray, frac_pos: np.ndarray) -> float:
    """
    Compute the area between a calibration curve and the 45° diagonal
    via the trapezoidal rule.

    mean_pred: 1D array of bin centers (x)
    frac_pos:  1D array of observed fraction positives per bin (y)
    """

    x = np.concatenate(([0.0], mean_pred, [1.0]))  ## common x-axis including 0 and 1

    ## two y-vectors
    y_curve = np.concatenate(([0.0], frac_pos, [1.0]))  ## calibration curve
    y_diag = x.copy()  # perfect-calibration line y=x

    ## Pack into (n+2)x2 arrays
    curve = np.column_stack((x, y_curve))
    diagonal = np.column_stack((x, y_diag))

    return area_trap(curve, diagonal)  ## Integrate |y_curve - y_diag| dx


####################################### toy examples


def binary_classification_example():
    # Generate random true binary labels and predicted probabilities
    y_true = np.random.randint(0, 2, size=100)
    y_proba = np.random.rand(100)
    y_pred = (y_proba > 0.5).astype(int)

    # Calculate metrics
    metrics = binary_classification_metrics(y_true, y_pred, y_proba)
    print("Binary Classification Metrics:\n", metrics)


def multi_class_classification_example():
    n_samples = 100
    n_classes = 3
    y_true = np.random.randint(0, n_classes, size=n_samples)
    y_proba = np.random.rand(n_samples, n_classes)
    y_proba /= y_proba.sum(axis=1, keepdims=True)
    y_pred = np.argmax(y_proba, axis=1)
    metrics = multi_class_classification_metrics(y_true, y_pred, y_proba, n_classes)
    print("\nMulti-Class Classification Metrics:\n", metrics)


def multi_label_classification_example():
    # Generate random true multi-class labels and predicted probabilities
    n_classes = 3
    n_samples = 100
    y_true = [
        np.random.choice(
            range(n_classes), size=np.random.randint(1, n_classes + 1), replace=False
        )
        for _ in range(n_samples)
    ]
    y_proba = np.random.rand(100, n_classes)
    y_proba /= y_proba.sum(axis=1, keepdims=True)  # Normalize to sum to 1
    y_pred = (y_proba > 0.5).astype(int)

    mlb = MultiLabelBinarizer()
    y_true = mlb.fit_transform(y_true)

    # Calculate metrics
    metrics = multi_label_classification_metrics(y_true, y_pred, y_proba)
    print("\nMulti-Label Classification Metrics:\n", metrics)


def regression_example():
    # Generate random true and predicted values for regression
    y_true = np.random.rand(100) * 100  # True values between 0 and 100
    y_pred = np.random.rand(100) * 100  # Predicted values between 0 and 100

    # Calculate metrics
    metrics = regression_metrics(y_true, y_pred)
    print("\nRegression Metrics:\n", metrics)


if __name__ == "__main__":
    binary_classification_example()
    multi_class_classification_example()
    multi_label_classification_example()
    regression_example()
