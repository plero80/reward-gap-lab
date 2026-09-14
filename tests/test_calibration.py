from dataclasses import replace

import pytest

from reward_gap.calibration import CalibrationError, FrozenCalibration, ScoreScale
from reward_gap.scorers import ScoreBatch


def batches():
    proxy = ScoreBatch(("a", "b"), (2., 6.), (10, 10), "proxy", "proxy", "v1")
    judge = ScoreBatch(("a", "b"), (10., 14.), (10, 10), "judge", "judge", "j1")
    return proxy, judge


def test_population_fit_round_trip_and_no_batch_refit(tmp_path):
    proxy, judge = batches()
    calibration = FrozenCalibration.fit(proxy, judge, calibration_id="cal-1")
    assert calibration.proxy.mean == 4.
    assert calibration.proxy.std == 2.
    assert calibration.normalize_proxy(proxy) == (-1., 1.)
    assert calibration.normalize_judge(judge) == (-1., 1.)
    changed = replace(proxy, scores=(8., 12.))
    assert calibration.normalize_proxy(changed) == (2., 4.)
    path = calibration.save(tmp_path / "calibration.json")
    assert FrozenCalibration.load(path) == calibration


@pytest.mark.parametrize("scores", [(1., 1.), (float("nan"), 2.), (True, 2.), (1.,)])
def test_bad_fit_scores(scores):
    proxy, judge = batches()
    with pytest.raises(CalibrationError):
        FrozenCalibration.fit(replace(proxy, scores=scores), judge, calibration_id="cal")


def test_fit_requires_aligned_roles_and_ids():
    proxy, judge = batches()
    for bad in [replace(judge, role="proxy"), replace(judge, prompt_ids=("b", "a"))]:
        with pytest.raises(CalibrationError, match="aligned"):
            FrozenCalibration.fit(proxy, bad, calibration_id="cal")


@pytest.mark.parametrize("mean,std", [(0., 0.), (0., -1.), (float("inf"), 1.), (0., True)])
def test_invalid_scale(mean, std):
    with pytest.raises(CalibrationError):
        ScoreScale("proxy", "v1", mean, std)


def test_invalid_saved_calibration(tmp_path):
    path = tmp_path / "bad.json"
    for raw in ["bad json", "[]", '{"schema_version": 2}']:
        path.write_text(raw, encoding="utf-8")
        with pytest.raises(CalibrationError):
            FrozenCalibration.load(path)
