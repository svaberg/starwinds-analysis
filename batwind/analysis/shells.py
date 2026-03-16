"""Spherical shell sampling and shell-integration primitives.
"""

# It is the foundation layer for shell-based analyses (sampling geometry, integration, coverage).


from __future__ import annotations

import math
import logging

import numpy as np

from batwind.algorithms.sphere_sampling import PolarAzimuthalGrid
from batwind.algorithms.sphere_sampling import fibonacci_sphere
from batwind.data.field_names import unit_from_brackets
log = logging.getLogger(__name__)

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
    """
    Infer available shell radii from points lying on a Cartesian axis.
    Used by: `test/test_shell_analysis.py`
    """
    log.info("infer_cartesian_axis_radii start axis=%s", axis)
    axis_key = str(axis).lower()
    axis_idx = {"x": 0, "y": 1, "z": 2}.get(axis_key)
    if axis_idx is None:
        log.error("infer_cartesian_axis_radii failed: axis=%s", axis)
        raise ValueError("axis must be 'x', 'y', or 'z'")

    coords = [np.array(smart_ds.variable(name)).ravel() for name in coord_fields]
    if len(coords) != 3:
        log.error("infer_cartesian_axis_radii failed: coord_fields=%s", coord_fields)
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
        log.error("infer_cartesian_axis_radii failed: no radii inferred")
        raise ValueError("Could not infer any axis-aligned radii from the dataset points")
    log.info("infer_cartesian_axis_radii done n_radii=%d", radii.size)
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
    """
    Resample fields onto spherical shell cell centers.
    Used by: `test/test_shell_magnetic_analysis.py`, `test/test_shell_analysis.py`,
      `test/test_shell_resample_smartds_spec.py`, `examples/smartds_quicklook_profiles.ipynb`,
      `examples/smartds_shell_mass_flux.ipynb` (+1 more)
    """
    log.info("sample_spherical_shells start method=%s", method)
    radii = np.atleast_1d(np.array(radii))
    if radii.ndim != 1:
        log.error("sample_spherical_shells failed: radii ndim=%d", radii.ndim)
        raise ValueError("radii must be 1D")
    if np.any(radii <= 0):
        log.error("sample_spherical_shells failed: non-positive radii present")
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
        log.error("sample_spherical_shells failed: sample_points shape=%s", sample_points.shape)
        raise ValueError(
            "Angular center arrays do not match Cartesian center grid shape "
            f"{sample_points.shape} for theta={theta.shape}, phi={phi.shape}"
        )
    if area.shape != (radii.size, ntheta, nphi):
        log.error("sample_spherical_shells failed: area shape=%s", area.shape)
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

    shell_ds = resampled.append_fields(
        {
            r_name: r_field,
            polar_name: polar_field,
            azimuth_name: azimuth_field,
            area_name: area_field,
        },
        zone_suffix="shell-grid-structured",
    )
    log.info("sample_spherical_shells done n_radii=%d n_theta=%d n_phi=%d", radii.size, ntheta, nphi)
    return shell_ds

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
    """
    Resample fields onto equal-area Fibonacci sphere points on each shell.
    Used by: `test/test_shell_analysis.py`, `examples/smartds_shell_mass_flux.ipynb`,
      `batwind/analysis/shells.py`
    """
    log.info("sample_spherical_shells_fibonacci start method=%s", method)
    radii = np.atleast_1d(np.array(radii))
    if radii.ndim != 1:
        log.error("sample_spherical_shells_fibonacci failed: radii ndim=%d", radii.ndim)
        raise ValueError("radii must be 1D")
    if np.any(radii <= 0):
        log.error("sample_spherical_shells_fibonacci failed: non-positive radii present")
        raise ValueError("radii must be > 0")
    n_points = int(n_points)
    if n_points < 8:
        log.error("sample_spherical_shells_fibonacci failed: n_points=%d", n_points)
        raise ValueError("n_points must be >= 8")

    unit = np.array(fibonacci_sphere(n_points, randomize=randomize))
    xhat = unit[:, 0][:, None]
    yhat = unit[:, 1][:, None]
    zhat = unit[:, 2][:, None]
    theta = np.arccos(np.clip(zhat, -1.0, 1.0))
    phi = np.arctan2(yhat, xhat)

    xyz_unit = np.stack((xhat, yhat, zhat), axis=-1)  # (npts, 1, 3)
    xyz = radii[:, None, None, None] * xyz_unit[None, :, :, :]  # (nr, npts, 1, 3)
    sample_points = xyz

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

    shell_ds = resampled.append_fields(
        {
            r_name: r_field,
            polar_name: polar_field,
            azimuth_name: azimuth_field,
            area_name: area_field,
        },
        zone_suffix="shell-fibonacci-structured",
    )
    log.info("sample_spherical_shells_fibonacci done n_radii=%d n_points=%d", radii.size, n_points)
    return shell_ds

def integrate_shell_scalar(values, area, *, ignore_nan: bool = True):
    """
    Integrate scalar values over shell surfaces and return sum + coverage.
    Used by: `test/test_shell_magnetic_analysis.py`, `test/test_shell_analysis.py`,
      `examples/smartds_inner_boundary_magnetic_zdi.ipynb`,
      `examples/smartds_quicklook_profiles.ipynb`, `examples/smartds_shell_mass_flux.ipynb` (+3 more)
    """
    v = np.array(values)
    a = np.array(area)
    if v.shape != a.shape:
        log.debug("integrate_shell_scalar broadcasting area from %s to %s", a.shape, v.shape)
        a = np.broadcast_to(a, v.shape)

    if not ignore_nan:
        if not np.all(np.isfinite(v)) or not np.all(np.isfinite(a)):
            log.error("integrate_shell_scalar failed: non-finite values with ignore_nan=False")
            raise ValueError("Non-finite values found; pass ignore_nan=True to ignore them")
        sum_val = np.sum(v * a, axis=(-2, -1))
        coverage = np.ones_like(sum_val, dtype=float)
        log.info("integrate_shell_scalar done (strict finite path)")
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
    low_coverage = int(np.count_nonzero(np.isfinite(coverage) & (coverage < 1.0)))
    if low_coverage > 0:
        log.warning("integrate_shell_scalar coverage below 1 for %d shells", low_coverage)
    log.info("integrate_shell_scalar done")
    return sum_val, coverage
