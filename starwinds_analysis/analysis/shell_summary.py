"""THIS FILE contains weighted shell-band summary helpers.

It aggregates already-computed shell series (means, quantiles, weighted summaries).
It should not sample datasets or define new physical quantities.
"""

from __future__ import annotations

import numpy as np

from starwinds_analysis.analysis.stats import weighted_mean_std, weighted_quantile

def boxcar_shell_weights(radii_r, *, rmin: float | None = None, rmax: float | None = None):
    """
    Boxcar weights over shell radii in units of body radii.
    Used by: `test/test_shell_analysis.py`, `starwinds_analysis/analysis/shell_summary.py`
    """
    r = np.array(radii_r)
    w = np.ones_like(r, dtype=float)
    if rmin is not None:
        w = np.where(r >= float(rmin), w, 0.0)
    if rmax is not None:
        w = np.where(r <= float(rmax), w, 0.0)
    w = np.where(np.isfinite(r), w, 0.0)
    return w

def summarize_shell_series(
    radii_r,
    values,
    *,
    coverage=None,
    weights=None,
    rmin: float | None = None,
    rmax: float | None = None,
    quantiles=(0.0, 0.25, 0.5, 0.75, 1.0),
):
    """
    Weighted summary (mean/std/quantiles) for a 1D shell profile series.
    Used by: `test/test_shell_analysis.py`, `starwinds_analysis/analysis/shell_summary.py`
    """
    r = np.array(radii_r).ravel()
    v = np.array(values).ravel()
    if r.shape != v.shape:
        raise ValueError("radii and values must have the same shape")

    if weights is None:
        w = boxcar_shell_weights(r, rmin=rmin, rmax=rmax)
    else:
        w = np.array(weights).ravel()
        if w.shape != v.shape:
            raise ValueError("weights must have the same shape as values")

    if coverage is not None:
        c = np.array(coverage).ravel()
        if c.shape != v.shape:
            raise ValueError("coverage must have the same shape as values")
        w = w * np.clip(c, 0.0, np.inf)

    mean, std = weighted_mean_std(v, w)
    qvals = weighted_quantile(v, quantiles, w)

    active = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if np.any(active):
        r_active = r[active]
        rmin_eff = float(np.min(r_active))
        rmax_eff = float(np.max(r_active))
        n_active = int(np.count_nonzero(active))
        weight_sum = float(np.sum(w[active]))
    else:
        rmin_eff = np.nan
        rmax_eff = np.nan
        n_active = 0
        weight_sum = 0.0

    return {
        "rmin [R]": rmin_eff,
        "rmax [R]": rmax_eff,
        "n_active": n_active,
        "weight_sum": weight_sum,
        "mean": float(mean),
        "std": float(std),
        "quantiles": np.array(quantiles).tolist(),
        "values": np.array(qvals).tolist(),
    }

def summarize_shell_diagnostics_band(
    diagnostics,
    *,
    rmin: float | None = None,
    rmax: float | None = None,
    weights=None,
    quantiles=(0.0, 0.25, 0.5, 0.75, 1.0),
):
    """
    Summarize all 1D shell-profile series in a diagnostics bundle over a shell-radius band.
    Used by: `test/test_shell_analysis.py`, `starwinds_analysis/pipelines/quicklook2d.py`
    """
    out = {}
    for name, profile in diagnostics.items():
        if not isinstance(profile, dict):
            continue
        radii = profile.get("radius [R]")
        if radii is None:
            continue
        coverage = profile.get("coverage [none]")
        per_profile = {}
        for key, value in profile.items():
            if key in {"radius [R]", "height [R]", "coverage [none]", "shell_samples"}:
                continue
            arr = np.array(value)
            if arr.ndim != 1:
                continue
            if arr.shape != np.array(radii).shape:
                continue
            per_profile[key] = summarize_shell_series(
                radii,
                arr,
                coverage=coverage,
                weights=weights,
                rmin=rmin,
                rmax=rmax,
                quantiles=quantiles,
            )
        if per_profile:
            out[name] = per_profile
    return out

