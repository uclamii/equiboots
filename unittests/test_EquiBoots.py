import pandas as pd
import numpy as np
import inspect
import pytest
import warnings
from src.equiboots import EquiBoots


# Synthetic dataset fixture
@pytest.fixture
def equiboots_fixture():
    np.random.seed(42)
    y_prob = np.random.rand(100)
    y_pred = (y_prob > 0.5).astype(int)
    y_true = np.random.randint(0, 2, size=100)
    fairness_df = pd.DataFrame(
        {
            "race": np.random.choice(["white", "black", "asian"], 100),
            "sex": np.random.choice(["M", "F"], 100),
        }
    )

    eq = EquiBoots(
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred,
        fairness_df=fairness_df,
        fairness_vars=["race", "sex"],
        reference_groups=["white", "M"],
        task="binary_classification",
        bootstrap_flag=False,
    )
    return eq


def test_init_sets_attributes(equiboots_fixture):
    eq = equiboots_fixture
    assert isinstance(eq.fairness_vars, list)
    assert eq.task == "binary_classification"
    assert eq.reference_groups["race"] == "white"


def test_set_fix_seeds(equiboots_fixture):
    eq = equiboots_fixture
    eq.set_fix_seeds([11, 22, 33])
    assert eq.seeds == [11, 22, 33]


def test_invalid_task_raises():
    with pytest.raises(ValueError):
        EquiBoots(
            y_true=np.array([1, 0]),
            y_prob=np.array([0.9, 0.1]),
            y_pred=np.array([1, 0]),
            fairness_df=pd.DataFrame(
                {"race": ["white", "black"], "sex": ["M", "F"]},
            ),
            fairness_vars=["race"],
            task="invalid_task",
        )


def test_check_fairness_vars_type():
    with pytest.raises(ValueError):
        EquiBoots(
            y_true=np.array([1, 0]),
            y_prob=np.array([0.9, 0.1]),
            y_pred=np.array([1, 0]),
            fairness_df=pd.DataFrame(
                {"race": ["white", "black"], "sex": ["M", "F"]},
            ),
            fairness_vars=None,
        )


def test_get_metrics(equiboots_fixture):
    eq = equiboots_fixture
    eq.grouper(groupings_vars=["race"])
    data = eq.slicer("race")
    metrics = eq.get_metrics(data)
    assert isinstance(metrics, dict)
    assert all(isinstance(val, dict) for val in metrics.values())


def test_calculate_disparities(equiboots_fixture):
    eq = equiboots_fixture
    eq.grouper(groupings_vars=["race"])
    data = eq.slicer("race")
    metrics = eq.get_metrics(data)
    disparities = eq.calculate_disparities(metrics, "race")
    assert isinstance(disparities, dict)
    assert all(isinstance(val, dict) for val in disparities.values())


test_functions = [
    obj
    for name, obj in globals().items()
    if inspect.isfunction(obj) and name.startswith("test_")
]
test_names = [fn.__name__ for fn in test_functions]

df = pd.DataFrame(test_names, columns=["Test Function"])
print(df)


def test_bootstrap_grouper_returns_list():
    np.random.seed(42)
    y_prob = np.random.rand(100)
    y_pred = (y_prob > 0.5).astype(int)
    y_true = np.random.randint(0, 2, size=100)
    fairness_df = pd.DataFrame(
        {
            "race": np.random.choice(["white", "black", "asian"], 100),
            "sex": np.random.choice(["M", "F"], 100),
        }
    )

    eq = EquiBoots(
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred,
        fairness_df=fairness_df,
        fairness_vars=["race"],
        reference_groups=["white"],
        task="binary_classification",
        bootstrap_flag=True,
        num_bootstraps=3,
        boot_sample_size=20,
    )

    eq.grouper(groupings_vars=["race"])
    assert isinstance(eq.groups, list)
    assert len(eq.groups) == 3


def test_stratify_by_outcome_regression_raises():
    y_true = np.random.rand(100)
    y_pred = np.random.rand(100)
    y_prob = np.random.rand(100)
    fairness_df = pd.DataFrame(
        {
            "race": np.random.choice(["white", "black"], 100),
        }
    )

    with pytest.raises(ValueError):
        EquiBoots(
            y_true=y_true,
            y_prob=y_prob,
            y_pred=y_pred,
            fairness_df=fairness_df,
            fairness_vars=["race"],
            task="regression",
            bootstrap_flag=True,
            stratify_by_outcome=True,
        ).grouper(groupings_vars=["race"])


def test_bootstrap_slicer_returns_list():
    np.random.seed(42)
    y_prob = np.random.rand(100)
    y_pred = (y_prob > 0.5).astype(int)
    y_true = np.random.randint(0, 2, size=100)
    fairness_df = pd.DataFrame(
        {
            "race": np.random.choice(["white", "black"], 100),
        }
    )

    eq = EquiBoots(
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred,
        fairness_df=fairness_df,
        fairness_vars=["race"],
        bootstrap_flag=True,
        num_bootstraps=2,
        boot_sample_size=20,
    )
    eq.grouper(groupings_vars=["race"])
    sliced = eq.slicer("race")
    assert isinstance(sliced, list)
    assert all(isinstance(d, dict) for d in sliced)


def test_bootstrap_calculate_disparities_returns_list():
    np.random.seed(42)
    y_prob = np.random.rand(100)
    y_pred = (y_prob > 0.5).astype(int)
    y_true = np.random.randint(0, 2, size=100)
    fairness_df = pd.DataFrame(
        {
            "race": np.random.choice(["white", "black"], 100),
        }
    )

    eq = EquiBoots(
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred,
        fairness_df=fairness_df,
        fairness_vars=["race"],
        reference_groups=["white"],
        task="binary_classification",
        bootstrap_flag=True,
        num_bootstraps=2,
        boot_sample_size=20,
    )
    eq.grouper(groupings_vars=["race"])
    sliced = eq.slicer("race")
    metrics = eq.get_metrics(sliced)
    disparities = eq.calculate_disparities(metrics, "race")
    assert isinstance(disparities, list)
    assert all(isinstance(d, dict) for d in disparities)


def test_bootstrap_stratify_by_outcome_binary():
    y_true = np.random.randint(0, 2, 100)
    y_prob = np.random.rand(100)
    y_pred = (y_prob > 0.5).astype(int)
    fairness_df = pd.DataFrame({"race": np.random.choice(["white", "black"], 100)})

    eq = EquiBoots(
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred,
        fairness_df=fairness_df,
        fairness_vars=["race"],
        reference_groups=["white"],
        task="binary_classification",
        bootstrap_flag=True,
        stratify_by_outcome=True,
        num_bootstraps=2,
        boot_sample_size=20,
    )
    eq.grouper(groupings_vars=["race"])
    assert isinstance(eq.groups, list)
    assert len(eq.groups) == 2


def test_sample_group_unbalanced():
    y_true = np.random.randint(0, 2, 100)
    y_prob = np.random.rand(100)
    y_pred = (y_prob > 0.5).astype(int)
    fairness_df = pd.DataFrame({"race": ["white"] * 100})

    eq = EquiBoots(
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred,
        fairness_df=fairness_df,
        fairness_vars=["race"],
        task="binary_classification",
        bootstrap_flag=True,
        balanced=False,
        num_bootstraps=1,
        boot_sample_size=50,
    )
    group = fairness_df[fairness_df["race"] == "white"].index
    result = eq.sample_group(group, 1, 0, 50, [42], balanced=False)
    assert len(result) > 0


def test_check_group_size_below_min(equiboots_fixture):
    eq = equiboots_fixture
    eq.group_min_size = 10  # Set minimum group size
    group = pd.Index([1, 2, 3])  # Group with fewer than 10 samples
    cat = "black"
    var = "race"

    result = eq.check_group_size(group, cat, var)
    assert not result
    assert cat in eq.groups_below_min_size[var]


def test_check_group_size_above_min(equiboots_fixture):
    eq = equiboots_fixture
    eq.group_min_size = 2  # Set minimum group size
    group = pd.Index([1, 2, 3])  # Group with more than 2 samples
    cat = "black"
    var = "race"

    result = eq.check_group_size(group, cat, var)
    assert result
    assert cat not in eq.groups_below_min_size[var]


def test_check_group_empty(equiboots_fixture):
    eq = equiboots_fixture
    sampled_group = np.array([])  # Empty sampled group
    cat = "black"
    var = "race"

    result = eq.check_group_empty(sampled_group, cat, var)
    assert result is False  # Ensure no exception is raised


def test_calculate_disparities_warns_on_zero_ref_value():
    eq = EquiBoots(
        y_true=np.array([1, 0]),
        y_prob=np.array([0.9, 0.1]),
        y_pred=np.array([1, 0]),
        fairness_df=pd.DataFrame({"race": ["white", "black"]}),
        fairness_vars=["race"],
        reference_groups=["white"],
        task="binary_classification",
    )

    # Create a fake metric dict with a zero reference value
    metric_dict = {
        "white": {"accuracy": 0.0},  # reference group with zero value
        "black": {"accuracy": 0.8},
    }

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # catch all warnings
        disparities = eq.calculate_groups_disparities(metric_dict, "race")

        # Make sure the warning was triggered
        assert any("Reference metric value is zero" in str(warn.message) for warn in w)
        assert disparities["black"]["accuracy_ratio"] == -1


def test_default_reference_group_selection():
    df = pd.DataFrame({"race": ["black"] * 60 + ["white"] * 40})
    eq = EquiBoots(
        y_true=np.random.randint(0, 2, 100),
        y_prob=np.random.rand(100),
        y_pred=(np.random.rand(100) > 0.5).astype(int),
        fairness_df=df,
        fairness_vars=["race"],
        task="binary_classification",
    )
    assert eq.reference_groups["race"] == "black"


def test_groups_slicer_regression():
    y_true = np.random.rand(100)
    y_pred = np.random.rand(100)
    df = pd.DataFrame({"race": ["white"] * 100})
    eq = EquiBoots(
        y_true=y_true,
        y_prob=y_pred,  # not used for regression
        y_pred=y_pred,
        fairness_df=df,
        fairness_vars=["race"],
        task="regression",
    )
    eq.grouper(groupings_vars=["race"])
    sliced = eq.slicer("race")
    assert isinstance(sliced, dict)
    assert "white" in sliced
    assert "y_true" in sliced["white"]


def test_get_groups_metrics_multiclass():
    y_true = np.random.choice([0, 1, 2], 100)
    y_pred = np.random.choice([0, 1, 2], 100)
    y_prob = np.random.dirichlet(np.ones(3), size=100)  # rows will sum to 1.0
    df = pd.DataFrame({"race": ["white"] * 100})

    eq = EquiBoots(
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred,
        fairness_df=df,
        fairness_vars=["race"],
        reference_groups=["white"],
        task="multi_class_classification",
    )
    eq.grouper(["race"])
    sliced = eq.slicer("race")
    metrics = eq.get_metrics(sliced)
    assert "white" in metrics


def test_set_fix_seeds_invalid_type_raises():
    eq = EquiBoots(
        y_true=np.array([1, 0]),
        y_prob=np.array([0.5, 0.5]),
        y_pred=np.array([1, 0]),
        fairness_df=pd.DataFrame({"race": ["white", "black"]}),
        fairness_vars=["race"],
        task="binary_classification",
    )
    with pytest.raises(ValueError):
        eq.set_fix_seeds([1, "bad_seed", 3])


def test_stratify_invalid_task():
    eb = EquiBoots(
        y_true=np.array([0, 1]),
        y_prob=np.array([0.5, 0.6]),
        y_pred=np.array([0, 1]),
        fairness_df=pd.DataFrame({"race": ["A", "B"]}),
        fairness_vars=["race"],
        task="regression",
    )
    with pytest.raises(ValueError):
        eb.check_classification_task("regression")


def test_check_fairness_vars_none():
    with pytest.raises(ValueError):
        EquiBoots(
            y_true=np.array([0, 1]),
            y_prob=np.array([0.5, 0.6]),
            y_pred=np.array([0, 1]),
            fairness_df=pd.DataFrame({"race": ["A", "B"]}),
            fairness_vars=None,
        )


def test_check_fairness_vars_not_list():
    with pytest.raises(ValueError):
        EquiBoots(
            y_true=np.array([0, 1]),
            y_prob=np.array([0.5, 0.6]),
            y_pred=np.array([0, 1]),
            fairness_df=pd.DataFrame({"race": ["A", "B"]}),
            fairness_vars="race",
        )


def test_reference_group_zero_warning():
    eb = EquiBoots(
        y_true=np.array([0, 1]),
        y_prob=np.array([0.5, 0.6]),
        y_pred=np.array([0, 1]),
        fairness_df=pd.DataFrame({"race": ["A", "B"]}),
        fairness_vars=["race"],
    )
    metrics_dict = {
        "A": {"acc": 0.0},
        "B": {"acc": 0.9},
    }
    eb.reference_groups = {"race": "A"}
    disparities = eb.calculate_groups_disparities(metrics_dict, var_name="race")
    assert disparities["B"]["acc_ratio"] == -1


def set_fix_seeds(self, seeds: list) -> None:
    if not all(isinstance(seed, int) for seed in seeds):
        raise ValueError("All seeds must be integers.")
    self.seeds = seeds


def test_set_fix_seeds_validation():
    eb = EquiBoots(
        y_true=np.array([0, 1]),
        y_prob=np.array([0.5, 0.6]),
        y_pred=np.array([0, 1]),
        fairness_df=pd.DataFrame({"race": ["A", "B"]}),
        fairness_vars=["race"],
    )
    with pytest.raises(ValueError):
        eb.set_fix_seeds([1, "not_int"])


def test_check_classification_task_raises():
    eb = EquiBoots(
        y_true=np.array([0, 1]),
        y_prob=np.array([0.5, 0.6]),
        y_pred=np.array([0, 1]),
        fairness_df=pd.DataFrame({"race": ["A", "B"]}),
        fairness_vars=["race"],
        task="regression",
    )
    with pytest.raises(ValueError):
        eb.check_classification_task("regression")


def test_groups_slicer_skips_small_group(equiboots_fixture):
    eq = equiboots_fixture
    eq.groups = {
        "race": {
            "categories": ["small", "large"],
            "indices": {
                "small": np.array([1, 2]),
                "large": np.arange(10, 60),
            },
        }
    }
    eq.groups_below_min_size = {"race": {"small"}}
    result = eq.groups_slicer(eq.groups, "race")
    assert "small" not in result
    assert "large" in result
