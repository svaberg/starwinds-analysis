"""Alfven-radius diagnostics from shell-sampled Alfven Mach number."""

# This module computes r_A maps and summary statistics from shell SmartDs data.
# The geometric target is the 3D M_A=1 isosurface; shell sampling approximates it
# through first outward radial crossings on a directional grid.

from __future__ import annotations

import numpy as np


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
    radius = np.array(shell_ds(radius_field))
    mach = np.array(shell_ds(mach_field))
    if radius.shape != mach.shape:
        raise ValueError("radius and mach fields must have the same shape")
    if radius.ndim < 2:
        raise ValueError("shell fields must have radial and angular dimensions")
    if radius.shape[0] < 2:
        raise ValueError("need at least two radial shells to detect a crossing")

    n_r = radius.shape[0]
    flat_r = radius.reshape(n_r, -1)
    flat_m = mach.reshape(n_r, -1)
    out = np.full(flat_r.shape[1], np.nan)

    target = float(level)
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
            if m1 == m0:
                out[col] = 0.5 * (r0 + r1)
            else:
                out[col] = r0 + (target - m0) * (r1 - r0) / (m1 - m0)

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
    area = np.array(shell_ds(area_field))[0]
    radius = np.array(shell_ds(radius_field))[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.divide(
            area,
            np.square(radius),
            out=np.full_like(area, np.nan, dtype=float),
            where=np.isfinite(area) & np.isfinite(radius) & (radius != 0),
        )


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
    radius = np.array(radius_map)
    polar = np.array(polar_map)
    if radius.shape != polar.shape:
        raise ValueError("radius_map and polar_map must have the same shape")

    if weights is None:
        weight = np.ones_like(radius, dtype=float)
    else:
        weight = np.array(weights)
        if weight.shape != radius.shape:
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
        return np.nan, np.nan, np.nan, np.nan, np.nan

    rf = radius[valid_weight]
    pf = polar[valid_weight]
    wf = weight[valid_weight]

    avg = np.sum(rf * wf) / sum_weight
    avg_cyl = np.sum(rf * np.sin(pf) * wf) / sum_weight
    coverage = sum_weight / total_weight if total_weight > 0 else np.nan

    return (
        float(np.min(rf)),
        float(np.max(rf)),
        float(avg),
        float(avg_cyl),
        float(coverage),
    )
