"""Gap approximation and detection diagnostics with explicit undefined cases."""

import numpy as np

from reward_gap.gap_prediction import metrics as base_metrics


def _ranks(values):
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def _correlation(left, right):
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def gap_metrics(actual, predicted, *, theta):
    if len(actual) == 0 and len(predicted) == 0:
        return {"count": 0, "mae": None, "rmse": None, "r2": None, "auroc": None,
                "average_precision": None, "precision": None, "recall": None,
                "positive_prevalence": None, "positives": 0, "predicted_positives": 0,
                "detector_cutoff": theta, "mse": None, "pearson": None, "spearman": None,
                "true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0}
    result = base_metrics(actual, predicted, theta=theta, cutoff=theta, comparison=">")
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    positive, detected = actual > theta, predicted > theta
    result.update(mse=float(np.mean((actual - predicted) ** 2)),
                  pearson=_correlation(actual, predicted),
                  spearman=_correlation(_ranks(actual), _ranks(predicted)),
                  true_positive=int(np.sum(positive & detected)), false_positive=int(np.sum(~positive & detected)),
                  true_negative=int(np.sum(~positive & ~detected)), false_negative=int(np.sum(positive & ~detected)))
    if not detected.any():
        result["precision"] = None
    return result
