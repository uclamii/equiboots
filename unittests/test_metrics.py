import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from src.equiboots import metrics

from src.equiboots.metrics import (
    binary_classification_metrics,
    multi_class_prevalence,
    multi_class_classification_metrics,
    multi_label_classification_metrics,
    regression_metrics,
)


def test_binary_classification_example_executes():
    metrics.binary_classification_example()


def test_multi_class_classification_example_executes():
    metrics.multi_class_classification_example()


def test_multi_label_classification_example_executes():
    metrics.multi_label_classification_example()


def test_regression_example_executes():
    metrics.regression_example()


def test_binary_classification_metrics():
    y_true = np.array([0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.7, 0.8, 0.3, 0.9])
    y_pred = (y_proba > 0.5).astype(int)

    result = binary_classification_metrics(y_true, y_pred, y_proba)
    assert isinstance(result, dict)
    assert "Accuracy" in result
    assert result["TP Rate"] >= 0 and result["FP Rate"] >= 0


def test_multi_class_prevalence():
    y_true = np.array([0, 1, 2, 1, 0])
    y_pred = np.array([0, 2, 2, 1, 0])
    prevalence, predicted_prevalence = multi_class_prevalence(
        y_true,
        y_pred,
        3,
    )
    assert len(prevalence) == 3
    assert len(predicted_prevalence) == 3
    assert np.isclose(sum(prevalence), 1.0)
    assert np.isclose(sum(predicted_prevalence), 1.0)


def test_multi_class_classification_metrics():
    y_true = np.array([0, 1, 2, 1, 0])
    y_pred = np.array([0, 2, 2, 1, 0])
    y_proba = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.1, 0.2, 0.7],
            [0.05, 0.1, 0.85],
            [0.2, 0.7, 0.1],
            [0.85, 0.1, 0.05],
        ]
    )
    result = multi_class_classification_metrics(
        y_true,
        y_pred,
        y_proba,
        n_classes=3,
    )
    assert "Accuracy" in result
    assert "ROC AUC" in result
    assert "Average Precision Score" in result


def test_multi_label_classification_metrics():
    mlb = MultiLabelBinarizer()
    y_true = mlb.fit_transform([[0], [1], [2], [0, 2], [1, 2]])
    y_pred = mlb.transform([[0], [1], [2], [2], [1]])
    y_proba = np.array(
        [
            [0.9, 0.1, 0.2],
            [0.2, 0.8, 0.3],
            [0.1, 0.4, 0.9],
            [0.5, 0.3, 0.8],
            [0.2, 0.7, 0.6],
        ]
    )
    result = multi_label_classification_metrics(y_true, y_pred, y_proba)
    assert "Accuracy" in result
    assert "ROC AUC" in result
    assert isinstance(result["Prevalence multi-labels"], list)


def test_regression_metrics():
    y_true = np.array([3.0, 5.0, 2.0, 7.0])
    y_pred = np.array([2.5, 5.0, 4.0, 8.0])
    result = regression_metrics(y_true, y_pred)

    assert "Mean Absolute Error" in result
    assert "Root Mean Squared Error" in result
    assert "Mean Squared Log Error" in result
    assert result["R^2 Score"] <= 1.0


def test_main_executes_examples(monkeypatch):
    # Monkeypatch print to avoid clutter
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)
    import runpy

    runpy.run_module("src.equiboots.metrics", run_name="__main__")
