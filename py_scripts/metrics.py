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
    # root_mean_squared_error,
    r2_score,
    explained_variance_score,
    mean_squared_log_error,
)

from sklearn.preprocessing import MultiLabelBinarizer


def binary_classification_metrics(y_true, y_pred, y_proba=None):
    """
    Evaluate binary classification metrics.

    Parameters:
    - y_true: array-like of shape (n_samples,)
        True binary labels.
    - y_pred: array-like of shape (n_samples,)
        Predicted binary class labels.
    - y_proba: array-like of shape (n_samples,), optional, default=None
        Probability estimates for the positive class, used for ROC AUC, Average Precision, etc.

    Returns:
    - metrics: dict
        Dictionary containing the calculated metrics. Keys include 'Accuracy',
        'Precision', 'Recall', 'F1 Score', and optionally 'ROC AUC',
        'Average Precision Score', 'Log Loss', 'Brier Score' if `y_proba` is provided.
    """
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
    }

    if y_proba is not None:
        metrics.update(
            {
                "ROC AUC": roc_auc_score(y_true, y_proba),
                "Average Precision Score": average_precision_score(y_true, y_proba),
                "Log Loss": log_loss(y_true, y_proba),
                "Brier Score": brier_score_loss(y_true, y_proba),
            }
        )

    return metrics


def multi_label_classification_metrics(
    y_true, y_pred, y_proba=None, average="weighted"
):
    """
    Evaluate multi-label classification metrics.

    Parameters:
    - y_true: array-like of shape (n_samples,)
        True class labels.
    - y_pred: array-like of shape (n_samples,)
        Predicted class labels.
    - y_proba: array-like of shape (n_samples, n_classes), optional, default=None
        Probability estimates for each class, used for ROC AUC, Average Precision, etc.
    - average: str, optional, default='weighted'
        This parameter is required for multiclass/multilabel targets.

    Returns:
    - metrics: dict
        Dictionary containing the calculated metrics. Keys include 'Accuracy',
        'Precision', 'Recall', 'F1 Score', and optionally 'ROC AUC',
        'Average Precision Score', 'Log Loss', and 'Brier Score' if `y_proba` is provided.
    """

    mlb = MultiLabelBinarizer()
    y_true = mlb.fit_transform(y_true)
    y_pred = mlb.transform(y_pred)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=average),
        "Recall": recall_score(y_true, y_pred, average=average),
        "F1 Score": f1_score(y_true, y_pred, average=average),
    }

    if y_proba is not None:
        metrics.update(
            {
                "ROC AUC": roc_auc_score(
                    y_true, y_proba, average=average, multi_class="ovr"
                ),
                "Average Precision Score": average_precision_score(
                    y_true, y_proba, average=average
                ),
                "Log Loss": log_loss(y_true, y_proba),
            }
        )

    return metrics


def regression_metrics(y_true, y_pred):
    """
    Evaluate regression metrics.

    Parameters:
    - y_true: array-like of shape (n_samples,)
        True values.
    - y_pred: array-like of shape (n_samples,)
        Predicted values.

    Returns:
    - metrics: dict
        Dictionary containing the calculated metrics. Keys include 'Mean Absolute Error',
        'Mean Squared Error', 'Root Mean Squared Error', 'R^2 Score',
        'Explained Variance', and 'Mean Squared Log Error'.
    """
    metrics = {
        "Mean Absolute Error": mean_absolute_error(y_true, y_pred),
        "Mean Squared Error": mean_squared_error(y_true, y_pred),
        # "Root Mean Squared Error": root_mean_squared_error(y_true, y_pred),
        "Root Mean Squared Error": mean_squared_error(y_true, y_pred, squared=False),
        "R^2 Score": r2_score(y_true, y_pred),
        "Explained Variance": explained_variance_score(y_true, y_pred),
        "Mean Squared Log Error": mean_squared_log_error(y_true, y_pred),
    }

    return metrics


####################################### toy examples

import numpy as np


def binary_classification_example():
    # Generate random true binary labels and predicted probabilities
    y_true = np.random.randint(0, 2, size=100)
    y_proba = np.random.rand(100)
    y_pred = (y_proba > 0.5).astype(int)

    # Calculate metrics
    metrics = binary_classification_metrics(y_true, y_pred, y_proba)
    print("Binary Classification Metrics:\n", metrics)


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
    multi_label_classification_example()
    regression_example()
