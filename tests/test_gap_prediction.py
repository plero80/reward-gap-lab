import numpy as np
import pytest

from reward_gap.gap_prediction import RidgePredictor, metrics, select_cutoff, select_ridge


def test_perfect_and_reversed_ranking():
    actual = [-2., -1., 1., 2.]
    perfect = metrics(actual, actual, theta=0, cutoff=0)
    assert perfect["mae"] == perfect["rmse"] == 0
    assert perfect["r2"] == perfect["auroc"] == perfect["average_precision"] == 1
    reverse = metrics(actual, [2., 1., -1., -2.], theta=0, cutoff=0)
    assert reverse["auroc"] == 0
    assert reverse["average_precision"] == pytest.approx((1/3 + 2/4) / 2)


def test_tied_baseline_and_strict_actual_threshold():
    result = metrics([0, 0, 1, 2], [0, 0, 0, 0], theta=0, cutoff=0)
    assert result["auroc"] == .5
    assert result["average_precision"] == result["positive_prevalence"] == .5
    assert result["precision"] == .5 and result["recall"] == 1


def test_partial_ties_receive_half_credit_in_auroc():
    result = metrics([-1, 1, -1, 1], [0, .5, .5, 1], theta=0, cutoff=.5)
    assert result["auroc"] == .875
    assert result["average_precision"] == pytest.approx(5/6)


def test_absent_classes_are_explicit_and_json_safe():
    result = metrics([0, 0], [1, 2], theta=1, cutoff=None)
    assert result["auroc"] is result["average_precision"] is result["recall"] is result["r2"] is None
    assert result["precision"] == result["predicted_positives"] == 0
    assert select_cutoff([0, 0], [1, 2], 1) is None
    all_positive = metrics([2, 3], [1, 2], theta=1, cutoff=0)
    assert all_positive["auroc"] is None
    assert all_positive["average_precision"] == 1


def test_validation_cutoff_and_regression_error():
    assert select_cutoff([-1, 2, 3], [.1, .8, .9], theta=0) == .8
    result = metrics([0, 2], [1, 0], theta=1, cutoff=.5)
    assert result["mae"] == 1.5 and result["rmse"] == pytest.approx(np.sqrt(2.5))


def test_ridge_learns_intercept_in_high_dimension_without_test_labels():
    x = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 1, 1]], dtype=float)
    y = 2 + x @ np.array([1, 2, 0, 0])
    model = select_ridge(x, y, x, y, [.000001, 10.])
    assert model.alpha == .000001
    np.testing.assert_allclose(model.predict(x), y, atol=1e-5)
    restored = RidgePredictor(**model.to_dict())
    np.testing.assert_array_equal(restored.predict(x), model.predict(x))


def test_bad_metric_arrays_are_rejected():
    with pytest.raises(ValueError):
        metrics([0], [float("nan")], theta=0, cutoff=0)
    with pytest.raises(ValueError):
        metrics([0, 1], [0], theta=0, cutoff=0)
