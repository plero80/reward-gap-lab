"""Small frozen-feature baselines and tie-aware RQ1 metrics."""

from dataclasses import asdict, dataclass

import numpy as np


def _array(values, *, ndim=1):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != ndim or not result.size or not np.isfinite(result).all():
        raise ValueError("Expected a nonempty finite array with the requested dimensions")
    return result


@dataclass(frozen=True)
class RidgePredictor:
    coefficients: list[float]
    intercept: float
    alpha: float

    @classmethod
    def fit(cls, features, targets, alpha):
        x, y = _array(features, ndim=2), _array(targets)
        if len(x) != len(y) or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Ridge requires aligned rows and positive regularization")
        # Center using training rows only; an unpenalized intercept and SVD work
        # even when there are fewer labeled examples than embedding dimensions.
        center, mean = x.mean(axis=0), y.mean()
        u, s, vt = np.linalg.svd(x - center, full_matrices=False)
        weights = vt.T @ ((s / (s * s + alpha)) * (u.T @ (y - mean)))
        return cls(weights.tolist(), float(mean - center @ weights), float(alpha))

    def predict(self, features):
        return _array(features, ndim=2) @ np.asarray(self.coefficients) + self.intercept

    def to_dict(self):
        return asdict(self)


def select_ridge(train_x, train_y, validation_x, validation_y, alphas):
    """Choose regularization by validation target MSE; never refit on validation."""
    candidates = [RidgePredictor.fit(train_x, train_y, alpha) for alpha in sorted(alphas)]
    if not candidates:
        raise ValueError("Provide at least one ridge alpha")
    return min(candidates, key=lambda model: float(np.mean((model.predict(validation_x) - validation_y) ** 2)))


def _ranking(labels, scores):
    order = np.argsort(-scores, kind="stable")
    y, p = labels[order], scores[order]
    ends = np.r_[np.flatnonzero(p[:-1] != p[1:]), len(p) - 1]
    true_positives = np.cumsum(y)[ends]
    false_positives = 1 + ends - true_positives
    return p[ends], true_positives, false_positives


def select_cutoff(actual, predicted, theta):
    """Maximize validation F1 for g_hat >= cutoff; prefer higher cutoffs on ties.

    None means no-positive predictions when validation has no positives. This
    explicit rule can be serialized without nonstandard JSON Infinity values.
    """
    g, p = _array(actual), _array(predicted)
    if g.shape != p.shape:
        raise ValueError("Actual and predicted gaps must align")
    y = g > theta
    if not y.any():
        return None
    cutoffs, tp, fp = _ranking(y, p)
    f1 = 2 * tp / (tp + fp + y.sum())
    return float(cutoffs[np.argmax(f1)])


def metrics(actual, predicted, *, theta, cutoff):
    g, p = _array(actual), _array(predicted)
    if g.shape != p.shape or not np.isfinite(theta):
        raise ValueError("Metrics require aligned gaps and a finite threshold")
    error = p - g
    y = g > theta
    positives = int(y.sum())
    negative_count = len(y) - positives
    selected = np.zeros(len(y), dtype=bool) if cutoff is None else p >= cutoff
    tp = int((selected & y).sum())
    _, rank_tp, rank_fp = _ranking(y, p)
    auc = ap = None
    if positives:
        recall = rank_tp / positives
        precision = rank_tp / (rank_tp + rank_fp)
        ap = float(np.sum(np.diff(np.r_[0., recall]) * precision))
        if negative_count:
            tpr, fpr = np.r_[0., recall], np.r_[0., rank_fp / negative_count]
            auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    total_variance = float(np.sum((g - g.mean()) ** 2))
    return {"count": len(g), "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "r2": float(1 - np.sum(error ** 2) / total_variance) if total_variance > 0 else None,
            "auroc": auc, "average_precision": ap,
            "precision": tp / int(selected.sum()) if selected.any() else 0.,
            "recall": tp / positives if positives else None,
            "positive_prevalence": positives / len(g), "positives": positives,
            "predicted_positives": int(selected.sum()), "detector_cutoff": cutoff}
