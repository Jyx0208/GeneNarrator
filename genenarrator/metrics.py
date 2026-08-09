"""Evaluation metrics (Methods): C-index, bootstrap CI, log-rank, silhouette."""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from sklearn.metrics import silhouette_score


def c_index(
    time: Sequence[float],
    risk: Sequence[float],
    event: Sequence[int],
) -> float:
    """Harrell's C-index (risk = 1 / predicted median survival)."""
    return float(concordance_index(time, -np.asarray(risk), event))


def bootstrap_ci(
    time: Sequence[float],
    risk: Sequence[float],
    event: Sequence[int],
    *,
    n_bootstrap: int = 500,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Percentile bootstrap 95% CI for the C-index (manuscript: B = 500)."""
    time = np.asarray(time, dtype=float)
    risk = np.asarray(risk, dtype=float)
    event = np.asarray(event, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(time)
    stats = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        stats[i] = c_index(time[idx], risk[idx], event[idx])
    lo, hi = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)
    return {
        "c_index": float(c_index(time, risk, event)),
        "ci_low": float(np.percentile(stats, lo)),
        "ci_high": float(np.percentile(stats, hi)),
        "n_bootstrap": n_bootstrap,
    }


def logrank_pvalue(
    time: Sequence[float], event: Sequence[int], group: Sequence[int]
) -> float:
    """Log-rank test between risk strata (two-sided P)."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    group = np.asarray(group, dtype=bool)
    result = logrank_test(
        time[group], time[~group], event[group], event[~group]
    )
    return float(result.p_value)


def silhouette_index(features: np.ndarray, labels: Sequence[int]) -> float:
    """Cohort-label separability (batch-effect proxy) in a feature space."""
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2 or len(labels) < 3:
        return float("nan")
    return float(silhouette_score(features, labels))


def concordance_compare(
    time: Sequence[float],
    risk_a: Sequence[float],
    risk_b: Sequence[float],
    event: Sequence[int],
    *,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> Dict[str, float]:
    """Paired bootstrap comparison of two models' C-index (two-sided P)."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    risk_a = np.asarray(risk_a, dtype=float)
    risk_b = np.asarray(risk_b, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(time)
    diffs = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diffs[i] = c_index(time[idx], risk_a[idx], event[idx]) - c_index(
            time[idx], risk_b[idx], event[idx]
        )
    delta = float(c_index(time, risk_a, event) - c_index(time, risk_b, event))
    if np.all(diffs == 0.0):
        p_value = 1.0
    else:
        p_value = float(2.0 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)))
    return {
        "delta_c_index": delta,
        "p_two_sided": min(max(p_value, 0.0), 1.0),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
    }
