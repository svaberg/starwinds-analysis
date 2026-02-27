"""THIS FILE contains structured 2D slice resampling helpers (for example XZ from 3D).

It resamples onto structured planes and returns SmartDs-compatible datasets.
It should not own plotting behavior.
"""

from __future__ import annotations

import numpy as np

def structured_quad_corners(nx: int, nz: int):
    """
    Quad connectivity for a row-major `(nz, nx)` point grid.
    Used by: `test/test_slices_analysis.py`, `starwinds_analysis/analysis/slices.py`
    """
    if nx < 2 or nz < 2:
        raise ValueError("nx and nz must be >= 2")

    corners = np.empty(((nx - 1) * (nz - 1), 4), dtype=int)
    k = 0
    for iz in range(nz - 1):
        row0 = iz * nx
        row1 = (iz + 1) * nx
        for ix in range(nx - 1):
            corners[k] = [row0 + ix, row0 + ix + 1, row1 + ix + 1, row1 + ix]
            k += 1
    return corners

def infer_range(values, *, symmetric: bool = False, padding_frac: float = 0.0):
    """
    Infer a plotting/resampling range from data with optional symmetry/padding.
    Used by: `test/test_slices_analysis.py`, `starwinds_analysis/analysis/slices.py`
    """
    v = np.array(values)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("No finite values to infer range from")
    lo = float(np.min(v))
    hi = float(np.max(v))
    if symmetric:
        m = max(abs(lo), abs(hi))
        lo, hi = -m, m
    if padding_frac:
        pad = (hi - lo) * float(padding_frac)
        lo -= pad
        hi += pad
    return lo, hi

def resample_structured_xz_slice(
    smart_ds,
    *,
    y_value: float = 0.0,
    x_range=None,
    z_range=None,
    nx: int = 200,
    nz: int = 200,
    fields=None,
    method: str = "nearest",
    fill_value: float = np.nan,
    symmetric_ranges: bool = False,
    padding_frac: float = 0.0,
):
    """
    Resample a 3D dataset onto a structured XZ plane and return a new `SmartDs`.
    Used by: `test/test_slices_analysis.py`, `starwinds_analysis/pipelines/slice.py`, `starwinds_analysis/pipelines/volume.py`
    """
    if nx < 2 or nz < 2:
        raise ValueError("nx and nz must be >= 2")

    x = np.array(smart_ds.variable("X [R]"))
    z = np.array(smart_ds.variable("Z [R]"))

    if x_range is None:
        x_range = infer_range(x, symmetric=symmetric_ranges, padding_frac=padding_frac)
    if z_range is None:
        z_range = infer_range(z, symmetric=symmetric_ranges, padding_frac=padding_frac)

    x1d = np.linspace(float(x_range[0]), float(x_range[1]), int(nx))
    z1d = np.linspace(float(z_range[0]), float(z_range[1]), int(nz))
    xx, zz = np.meshgrid(x1d, z1d, indexing="xy")
    yy = np.full_like(xx, float(y_value), dtype=float)

    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    corners = structured_quad_corners(nx=int(nx), nz=int(nz))

    if fields is None:
        fields = tuple(smart_ds.variables)

    sliced = smart_ds.resample(
        points,
        coordinate_fields=("X [R]", "Y [R]", "Z [R]"),
        fields=fields,
        method=method,
        fill_value=fill_value,
        corners=corners,
        zone=f"{smart_ds.zone} (XZ slice y={y_value:g})",
    )
    return sliced

