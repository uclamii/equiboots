import builtins
import equiboots


def test_metadata_attributes():
    assert hasattr(equiboots, "__version__")
    assert isinstance(equiboots.__version__, str)
    assert len(equiboots.__version__) > 0  # Ensure not empty
    assert "Leonid Shpaner" in equiboots.__author__


def test_docstring_exists():
    assert equiboots.__doc__ is not None
    assert "fairness-aware model evaluation" in equiboots.__doc__


def test_imports_available():
    # Smoke test to ensure submodules are accessible
    assert hasattr(equiboots, "binary_classification_metrics")
    assert hasattr(equiboots, "eq_plot_roc_auc")
    assert hasattr(equiboots, "equiboots_logo")
    assert hasattr(equiboots, "EquiBoots")


def test_custom_help_override(monkeypatch, capsys):
    # Monkeypatch sys.modules so custom_help detects the module
    import sys

    monkeypatch.setitem(sys.modules, "equiboots", equiboots)

    # Call help with no argument (should trigger ASCII + doc)
    builtins.help(equiboots)
    captured = capsys.readouterr()

    assert "EquiBoots is particularly useful" in captured.out
    assert "equiboots" in captured.out.lower()


def test_custom_help_on_object_falls_back(monkeypatch, capsys):
    import sys

    monkeypatch.setitem(sys.modules, "equiboots", equiboots)

    # Call help on a function — should use original_help (not print logo or docstring)
    help(equiboots.binary_classification_metrics)
    captured = capsys.readouterr()

    # This should *not* include the custom docstring banner
    assert "fairness-aware model evaluation" not in captured.out
    assert "binary_classification_metrics" in captured.out
