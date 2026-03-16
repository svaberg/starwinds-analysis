"""Weighted shell-band summary helpers.
"""

# It aggregates already-computed shell series (means, quantiles, weighted summaries).
# It should not sample datasets or define new physical quantities.


from __future__ import annotations

import logging

import numpy as np

from batwind.analysis.stats import weighted_mean_std
from batwind.analysis.stats import weighted_quantile

log = logging.getLogger(__name__)

def boxcar_shell_weights(radii_r, *, rmin: float | None = None, rmax: float | None = None):
    """
    Boxcar weights over shell radii in units of body radii.
    Used by: `test/test_shell_analysis.py`, `batwind/analysis/shell_summary.py`
    """
    r = np.array(radii_r)
    w = np.ones_like(r, dtype=float)
    if rmin is not None:
        w = np.where(r >= float(rmin), w, 0.0)
    if rmax is not None:
        w = np.where(r <= float(rmax), w, 0.0)
    w = np.where(np.isfinite(r), w, 0.0)
    log.debug("boxcar_shell_weights active=%d/%d", int(np.count_nonzero(w > 0)), w.size)
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
    Used by: `test/test_shell_analysis.py`, `batwind/analysis/shell_summary.py`
    """
    r = np.array(radii_r).ravel()
    v = np.array(values).ravel()
    if r.shape != v.shape:
        log.error("summarize_shell_series failed: radii shape %s values shape %s", r.shape, v.shape)
        raise ValueError("radii and values must have the same shape")

    if weights is None:
        w = boxcar_shell_weights(r, rmin=rmin, rmax=rmax)
    else:
        w = np.array(weights).ravel()
        if w.shape != v.shape:
            log.error("summarize_shell_series failed: weights shape %s values shape %s", w.shape, v.shape)
            raise ValueError("weights must have the same shape as values")

    if coverage is not None:
        c = np.array(coverage).ravel()
        if c.shape != v.shape:
            log.error("summarize_shell_series failed: coverage shape %s values shape %s", c.shape, v.shape)
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
        log.warning("summarize_shell_series: no active weighted samples")

    log.info("summarize_shell_series done n_active=%d", n_active)
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
    Used by: `test/test_shell_analysis.py`, `batwind/pipelines/slice.py`, `batwind/pipelines/volume.py`
    """
    out = {}
    log.info("summarize_shell_diagnostics_band start")
    for name, profile in diagnostics.items():
        if not isinstance(profile, dict):
            log.debug("summarize_shell_diagnostics_band skipping %s (not a dict)", name)
            continue
        radii = profile.get("radius [R]")
        if radii is None:
            log.debug("summarize_shell_diagnostics_band skipping %s (no radius [R])", name)
            continue
        coverage = profile.get("coverage [none]")
        per_profile = {}
        for key, value in profile.items():
            if key in {"radius [R]", "height [R]", "coverage [none]", "shell_samples"}:
                continue
            arr = np.array(value)
            if arr.ndim != 1:
                log.debug("summarize_shell_diagnostics_band skipping %s/%s (ndim=%d)", name, key, arr.ndim)
                continue
            if arr.shape != np.array(radii).shape:
                log.debug(
                    "summarize_shell_diagnostics_band skipping %s/%s shape=%s radii=%s",
                    name,
                    key,
                    arr.shape,
                    np.array(radii).shape,
                )
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
    log.info("summarize_shell_diagnostics_band done groups=%d", len(out))
    return out
