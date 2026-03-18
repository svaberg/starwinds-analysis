"""Spherical shell sampling and shell-integration primitives.
"""

from __future__ import annotations

import math

import numpy as np

from batwind.algorithms.sphere_sampling import PolarAzimuthalGrid
from batwind.algorithms.sphere_sampling import fibonacci_sphere
from batwind.data.field_names import unit_from_brackets


def infer_cartesian_axis_radii(
    smart_ds,
    *,
    axis: str = "x",
    coord_fields=("X [R]", "Y [R]", "Z [R]"),
    atol: float = 1e-12,
    positive_only: bool = True,
    r_min: float | None = None,
    r_max: float | None = None,
):
    axis_key = str(axis).lower()
    axis_idx = {"x": 0, "y": 1, "z": 2}.get(axis_key)
    if axis_idx is None:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    coords = [np.array(smart_ds.variable(name)).ravel() for name in coord_fields]
    if len(coords) != 3:
        raise ValueError("coord_fields must have length 3")
    x, y, z = coords
    arrays = (x, y, z)
    values = arrays[axis_idx]

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    for i, arr in enumerate(arrays):
        if i == axis_idx:
            continue
        mask &= np.isclose(arr, 0.0, atol=float(atol), rtol=0.0)

    vals = np.array(values[mask])
    if positive_only:
        vals = vals[vals > 0.0]
    else:
        vals = np.abs(vals)
    vals = vals[np.isfinite(vals)]
    radii = np.unique(np.sort(vals))

    if r_min is not None:
        radii = radii[radii >= float(r_min)]
    if r_max is not None:
        radii = radii[radii <= float(r_max)]
    if radii.size == 0:
        raise ValueError("Could not infer any axis-aligned radii from the dataset points")
    return radii


def sample_spherical_shells(
    smart_ds,
    radii,
    *,
    fields=None,
    coordinate_fields=("X [R]", "Y [R]", "Z [R]"),
    n_polar: int = 24,
    n_azimuth: int = 48,
    polar_edges=None,
    azimuthal_edges=None,
    method: str = "nearest",
    fill_value: float = np.nan,
    length_unit_to_m: float | None = None,
):
    radii = np.atleast_1d(np.array(radii))
    if radii.ndim != 1:
        raise ValueError("radii must be 1D")
    if np.any(radii <= 0):
        raise ValueError("radii must be > 0")

    if polar_edges is None:
        polar_edges = np.linspace(0.0, math.pi, n_polar + 1)
    if azimuthal_edges is None:
        azimuthal_edges = np.linspace(-math.pi, math.pi, n_azimuth + 1)

    ang_grid = PolarAzimuthalGrid(polar_edges, azimuthal_edges)
    theta = np.array(ang_grid.polar_centres).T
    phi = np.array(ang_grid.azimuthal_centres).T
    ntheta, nphi = theta.shape

    sample_points = np.stack(
        [
            np.transpose(
                ang_grid.centres_cartesian(radius=float(radius)),
                (1, 0, 2),
            )
            for radius in radii
        ],
        axis=0,
    )
    area = np.stack(
        [ang_grid.cell_area(radius=float(radius)) for radius in radii],
        axis=0,
    )
    if sample_points.shape != (radii.size, ntheta, nphi, 3):
        raise ValueError(
            "Angular center arrays do not match Cartesian center grid shape "
            f"{sample_points.shape} for theta={theta.shape}, phi={phi.shape}"
        )
    if area.shape != (radii.size, ntheta, nphi):
        raise ValueError(
            "Angular center arrays do not match cell-area shape "
            f"{area.shape} for theta={theta.shape}, phi={phi.shape}"
        )

    if fields is None:
        fields_arg = None
    else:
        requested_fields = tuple(dict.fromkeys(fields))
        fields_arg = smart_ds.base_fields_for_resample(requested_fields)

    resampled = smart_ds.resample(
        sample_points,
        coordinate_fields=coordinate_fields,
        fields=fields_arg,
        method=method,
        fill_value=fill_value,
        title=f"{getattr(smart_ds, 'title', 'dataset')} (shell samples (grid))",
        zone="shell-samples",
    )

    if length_unit_to_m is not None:
        area = area * float(length_unit_to_m) ** 2

    x_name, y_name, z_name = coordinate_fields
    length_unit = unit_from_brackets(x_name) or "R"
    r_name = f"R [{length_unit}]"
    polar_name = "polar [rad]"
    azimuth_name = "azimuth [rad]"
    area_unit = "m^2" if length_unit_to_m is not None else f"{length_unit}^2"
    area_name = f"dA [{area_unit}]"

    r_field = np.broadcast_to(radii[:, None, None], (radii.size, ntheta, nphi)).copy()
    polar_field = np.broadcast_to(theta[None, :, :], (radii.size, ntheta, nphi)).copy()
    azimuth_field = np.broadcast_to(phi[None, :, :], (radii.size, ntheta, nphi)).copy()
    area_field = np.array(area)

    return resampled.append_fields(
        {
            r_name: r_field,
            polar_name: polar_field,
            azimuth_name: azimuth_field,
            area_name: area_field,
        },
        zone_suffix="shell-grid-structured",
    )


def sample_spherical_shells_fibonacci(
    smart_ds,
    radii,
    *,
    fields=None,
    coordinate_fields=("X [R]", "Y [R]", "Z [R]"),
    n_points: int = 512,
    randomize: bool = False,
    method: str = "nearest",
    fill_value: float = np.nan,
    length_unit_to_m: float | None = None,
):
    radii = np.atleast_1d(np.array(radii))
    if radii.ndim != 1:
        raise ValueError("radii must be 1D")
    if np.any(radii <= 0):
        raise ValueError("radii must be > 0")
    n_points = int(n_points)
    if n_points < 8:
        raise ValueError("n_points must be >= 8")

    unit = np.array(fibonacci_sphere(n_points, randomize=randomize))
    xhat = unit[:, 0][:, None]
    yhat = unit[:, 1][:, None]
    zhat = unit[:, 2][:, None]
    theta = np.arccos(np.clip(zhat, -1.0, 1.0))
    phi = np.arctan2(yhat, xhat)

    xyz_unit = np.stack((xhat, yhat, zhat), axis=-1)
    sample_points = radii[:, None, None, None] * xyz_unit[None, :, :, :]

    if fields is None:
        fields_arg = None
    else:
        requested_fields = tuple(dict.fromkeys(fields))
        fields_arg = smart_ds.base_fields_for_resample(requested_fields)

    resampled = smart_ds.resample(
        sample_points,
        coordinate_fields=coordinate_fields,
        fields=fields_arg,
        method=method,
        fill_value=fill_value,
        title=f"{getattr(smart_ds, 'title', 'dataset')} (shell samples (fibonacci))",
        zone="shell-samples",
    )

    area_unit = (4.0 * math.pi) / float(n_points)
    area = (radii[:, None, None] ** 2) * area_unit
    if length_unit_to_m is not None:
        area = area * float(length_unit_to_m) ** 2

    x_name, y_name, z_name = coordinate_fields
    length_unit = unit_from_brackets(x_name) or "R"
    r_name = f"R [{length_unit}]"
    polar_name = "polar [rad]"
    azimuth_name = "azimuth [rad]"
    area_unit_name = "m^2" if length_unit_to_m is not None else f"{length_unit}^2"
    area_name = f"dA [{area_unit_name}]"

    r_field = np.broadcast_to(radii[:, None, None], (radii.size, n_points, 1)).copy()
    polar_field = np.broadcast_to(theta[None, :, :], (radii.size, n_points, 1)).copy()
    azimuth_field = np.broadcast_to(phi[None, :, :], (radii.size, n_points, 1)).copy()
    area_field = np.broadcast_to(area, (radii.size, n_points, 1)).copy()

    return resampled.append_fields(
        {
            r_name: r_field,
            polar_name: polar_field,
            azimuth_name: azimuth_field,
            area_name: area_field,
        },
        zone_suffix="shell-fibonacci-structured",
    )


def integrate_shell_scalar(values, area, *, ignore_nan: bool = True):
    v = np.array(values)
    a = np.array(area)
    if v.shape != a.shape:
        a = np.broadcast_to(a, v.shape)

    if not ignore_nan:
        if not np.all(np.isfinite(v)) or not np.all(np.isfinite(a)):
            raise ValueError("Non-finite values found; pass ignore_nan=True to ignore them")
        sum_val = np.sum(v * a, axis=(-2, -1))
        coverage = np.ones_like(sum_val, dtype=float)
        return sum_val, coverage

    mask = np.isfinite(v) & np.isfinite(a)
    sum_val = np.sum(np.where(mask, v * a, 0.0), axis=(-2, -1))
    sum_area = np.sum(np.where(mask, a, 0.0), axis=(-2, -1))
    total_area = np.sum(np.where(np.isfinite(a), a, 0.0), axis=(-2, -1))

    with np.errstate(invalid="ignore", divide="ignore"):
        coverage = np.divide(
            sum_area,
            total_area,
            out=np.full_like(sum_area, np.nan, dtype=float),
            where=total_area > 0,
        )
    return sum_val, coverage
