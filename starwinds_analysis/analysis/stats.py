"""THIS FILE contains small weighted-statistics primitives.

It provides reusable numerical helpers (weighted mean/std/quantile).
It should remain pure math with no dataset or plotting dependencies.
"""

from __future__ import annotations

import numpy as np

# Weighted mean and standard deviation over finite values.
# Used in: `test/test_shell_analysis.py`, `starwinds_analysis/analysis/shell_summary.py`,
#   `starwinds_analysis/analysis/stats.py`
def weighted_mean_std(values, weights=None):
    """
    Weighted mean and standard deviation over finite values.
    """
    v = np.array(values)
    if weights is None:
        w = np.ones_like(v, dtype=float)
    else:
        w = np.array(weights)
        if w.shape != v.shape:
            w = np.broadcast_to(w, v.shape)

    mask = np.isfinite(v) & np.isfinite(w) & (w >= 0)
    if not np.any(mask):
        return np.nan, np.nan

    v = v[mask]
    w = w[mask]
    wsum = float(np.sum(w))
    if wsum <= 0:
        return np.nan, np.nan

    mean = float(np.average(v, weights=w))
    var = float(np.average((v - mean) ** 2, weights=w))
    return mean, float(np.sqrt(var))

# Weighted quantiles for 1D data.
# Used in: `test/test_shell_analysis.py`, `starwinds_analysis/analysis/shell_summary.py`,
#   `starwinds_analysis/analysis/stats.py`
def weighted_quantile(values, quantiles, weights=None):
    """
    Weighted quantiles for 1D data.
    """
    v = np.array(values).ravel()
    q = np.array(quantiles)

    if weights is None:
        w = np.ones_like(v, dtype=float)
    else:
        w = np.array(weights).ravel()
        if w.shape != v.shape:
            raise ValueError("weights must have the same shape as values")

    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        out = np.full_like(q, np.nan, dtype=float)
        return float(out) if out.ndim == 0 else out

    v = v[mask]
    w = w[mask]
    order = np.argsort(v)
    v = v[order]
    w = w[order]

    cdf = np.cumsum(w, dtype=float)
    cdf /= cdf[-1]
    q = np.clip(q, 0.0, 1.0)

    out = v[np.searchsorted(cdf, q, side="left")]
    return float(out) if np.ndim(quantiles) == 0 else out

# Weighted quantiles + mean/std summary for 1D samples.
# Used in: `test/test_shell_analysis.py`, `starwinds_analysis/physics/orbit_local.py`,
#   `starwinds_analysis/physics/orbit_surface.py`, `starwinds_analysis/physics/orbit_pressure.py`
def summarize_samples(values, *, quantiles=(0.0, 0.25, 0.5, 0.75, 1.0), weights=None):
    """
    Weighted quantiles + mean/std summary for 1D samples.
    """
    v = np.array(values)
    qv = weighted_quantile(v, quantiles, weights=weights)
    mean, std = weighted_mean_std(v, weights=weights)
    return {
        "quantiles": np.array(quantiles),
        "values": np.array(qv),
        "mean": float(mean),
        "std": float(std),
    }
