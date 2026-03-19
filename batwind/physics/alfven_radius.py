"""Alfven-radius diagnostics from shell-sampled Alfven Mach number."""

# This module computes r_A maps and summary statistics from shell SmartDs data.
# The geometric target is the 3D M_A=1 isosurface; shell sampling approximates it
# through first outward radial crossings on a directional grid.

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def alfven_radius_map(
    shell_ds,
    *,
    mach_field: str = "M_A [none]",
    radius_field: str = "R [R]",
    level: float = 1.0,
):
    """
    Compute first outward radial crossing radius where `M_A` goes `< level -> > level`.
    Used by: `test/test_alfven_radius.py`, `examples/alfven_radius_shell.ipynb`
    """
    log.info("alfven_radius_map...")
    radius = np.array(shell_ds[radius_field])
    mach = np.array(shell_ds[mach_field])
    if radius.shape != mach.shape:
        log.error("alfven_radius_map failed: radius shape=%s mach shape=%s", radius.shape, mach.shape)
        raise ValueError("radius and mach fields must have the same shape")
    if radius.ndim < 2:
        log.error("alfven_radius_map failed: radius ndim=%d", radius.ndim)
        raise ValueError("shell fields must have radial and angular dimensions")
    if radius.shape[0] < 2:
        log.error("alfven_radius_map failed: n_r=%d", radius.shape[0])
        raise ValueError("need at least two radial shells to detect a crossing")

    n_r = radius.shape[0]
    flat_r = radius.reshape(n_r, -1)
    flat_m = mach.reshape(n_r, -1)
    out = np.full(flat_r.shape[1], np.nan)

    target = float(level)
    log.debug("alfven_radius_map level=%g n_r=%d n_columns=%d", target, n_r, flat_r.shape[1])
    crossings = 0
    for col in range(flat_r.shape[1]):
        r_col = flat_r[:, col]
        m_col = flat_m[:, col]
        pair_ok = (
            np.isfinite(r_col[:-1])
            & np.isfinite(r_col[1:])
            & np.isfinite(m_col[:-1])
            & np.isfinite(m_col[1:])
        )
        crossing = pair_ok & (m_col[:-1] < target) & (m_col[1:] > target)
        idx = np.flatnonzero(crossing)
        if idx.size > 0:
            i = int(idx[0])
            r0 = r_col[i]
            r1 = r_col[i + 1]
            m0 = m_col[i]
            m1 = m_col[i + 1]
            out[col] = r0 + (target - m0) * (r1 - r0) / (m1 - m0)
            crossings += 1

    log.debug("alfven_radius_map complete crossings=%d/%d", crossings, flat_r.shape[1])
    return out.reshape(radius.shape[1:])


def projected_solid_angle_weights(
    shell_ds,
    *,
    area_field: str = "dA [m^2]",
    radius_field: str = "R [m]",
):
    """
    Compute projected angular weights dOmega = dA / R^2 from shell geometry.
    Used by: `test/test_alfven_radius.py`, `examples/alfven_radius_shell.ipynb`
    """
    log.debug("projected_solid_angle_weights...")
    area = np.array(shell_ds[area_field])
    radius = np.array(shell_ds[radius_field])
    if area.shape != radius.shape:
        log.error("projected_solid_angle_weights failed: area shape=%s radius shape=%s", area.shape, radius.shape)
        raise ValueError("area and radius fields must have the same shape")
    if area.ndim < 2:
        log.error("projected_solid_angle_weights failed: area ndim=%d", area.ndim)
        raise ValueError("shell fields must include angular dimensions")

    if area.ndim == 2:
        area_angular = area
        radius_angular = radius
    else:
        # Shell samplers keep one radial axis + angular axes; angular grids are shared
        # across radii, so one radial slice is sufficient for dOmega.
        area_angular = area[0]
        radius_angular = radius[0]

    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.divide(
            area_angular,
            np.square(radius_angular),
            out=np.full_like(area_angular, np.nan, dtype=float),
            where=np.isfinite(area_angular) & np.isfinite(radius_angular) & (radius_angular != 0),
        )
    non_finite = int(np.count_nonzero(~np.isfinite(out)))
    if non_finite > 0:
        log.warning("projected_solid_angle_weights output has %d/%d non-finite values", non_finite, out.size)
    return out


def summarize_alfven_radius(
    radius_map,
    polar_map,
    *,
    weights=None,
):
    """
    Summarize min/max/mean/mean-cyl and coverage for an Alfven-radius map.
    Used by: `test/test_alfven_radius.py`, `examples/alfven_radius_shell.ipynb`
    """
    log.info("summarize_alfven_radius...")
    radius = np.array(radius_map)
    polar = np.array(polar_map)
    if radius.shape != polar.shape:
        log.error("summarize_alfven_radius failed: radius shape=%s polar shape=%s", radius.shape, polar.shape)
        raise ValueError("radius_map and polar_map must have the same shape")

    if weights is None:
        weight = np.ones_like(radius, dtype=float)
    else:
        weight = np.array(weights)
        if weight.shape != radius.shape:
            log.error("summarize_alfven_radius failed: weight shape=%s radius shape=%s", weight.shape, radius.shape)
            raise ValueError("weights must have the same shape as radius_map")

    valid_weight = (
        np.isfinite(radius)
        & np.isfinite(polar)
        & np.isfinite(weight)
        & (weight > 0)
    )
    sum_weight = np.sum(np.where(valid_weight, weight, 0.0))
    total_weight = np.sum(
        np.where(np.isfinite(weight) & (weight > 0), weight, 0.0)
    )

    if sum_weight == 0:
        log.warning("summarize_alfven_radius: zero active weight")
        return np.nan, np.nan, np.nan, np.nan, np.nan

    rf = radius[valid_weight]
    pf = polar[valid_weight]
    wf = weight[valid_weight]

    avg = np.sum(rf * wf) / sum_weight
    avg_cyl = np.sum(rf * np.sin(pf) * wf) / sum_weight
    coverage = sum_weight / total_weight if total_weight > 0 else np.nan

    result = (
        float(np.min(rf)),
        float(np.max(rf)),
        float(avg),
        float(avg_cyl),
        float(coverage),
    )
    log.debug("summarize_alfven_radius complete coverage=%g", result[4])
    return result
