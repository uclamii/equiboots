import builtins
import equiboots


def test_metadata_attributes():
    assert hasattr(equiboots, "__version__")
    assert equiboots.__version__ == "0.0.0a"
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
