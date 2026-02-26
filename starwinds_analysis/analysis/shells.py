"""THIS FILE contains spherical shell sampling and shell-integration primitives.

It is the foundation layer for shell-based analyses (sampling geometry, integration, coverage).
Temporary field-resolution helpers live here for now, but should migrate into SmartDs.
"""

# TODO(debt): `resolve_*` field/unit helpers in this module are a known smell; callers
# should request SI quantities directly from SmartDs/griblet.
# TODO(debt): `SphericalShellSamples` is a custom container kept for compatibility.
# The preferred direction is structured `SmartDs` resampling + shared metadata fields.

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from starwinds_readplt.dataset import Dataset

from starwinds_analysis.algorithms.sphere_sampling import PolarAzimuthalGrid, fibonacci_sphere


@dataclass
class SphericalShellSamples:
    """
    Samples on one or more spherical shells.

    Shapes:
    - `radii`: `(nr,)`
    Grid sampler:
    - `theta`, `phi`: `(ntheta, nphi)`
    - `x`, `y`, `z`, `area`: `(nr, ntheta, nphi)`
    - each `fields[name]`: `(nr, ntheta, nphi)`

    Fibonacci sampler:
    - `theta`, `phi`: `(npts, 1)`
    - `x`, `y`, `z`, `area`: `(nr, npts, 1)`
    - each `fields[name]`: `(nr, npts, 1)`
    """

    radii: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    area: np.ndarray
    fields: dict[str, np.ndarray]


def _resample_shell_points(
    smart_ds,
    sample_points,
    *,
    fields,
    coordinate_fields,
    method,
    fill_value,
    title_suffix="shell samples",
):
    if fields is None:
        fields_arg = None
    else:
        fields_arg = tuple(dict.fromkeys(fields))
    return smart_ds.resample(
        sample_points,
        coordinate_fields=coordinate_fields,
        fields=fields_arg,
        method=method,
        fill_value=fill_value,
        title=f"{getattr(smart_ds, 'title', 'dataset')} ({title_suffix})",
        zone="shell-samples",
    )


def _field_unit_from_brackets(name: str) -> str | None:
    text = str(name)
    i = text.rfind("[")
    j = text.rfind("]")
    if i == -1 or j == -1 or j <= i:
        return None
    return text[i + 1 : j].strip() or None


def _append_fields_to_smart_ds(smart_ds, extra_fields: dict[str, np.ndarray], *, zone_suffix: str):
    if not extra_fields:
        return smart_ds

    base_points = np.array(smart_ds.raw.points)
    if base_points.ndim < 2:
        raise ValueError("Expected SmartDs raw points to have shape (..., nvars)")
    base_shape = base_points.shape[:-1]

    arrays = []
    names = []
    for name, values in extra_fields.items():
        arr = np.array(values, dtype=float)
        if arr.shape != base_shape:
            raise ValueError(
                f"Extra field '{name}' shape {arr.shape} does not match dataset grid shape {base_shape}"
            )
        arrays.append(arr[..., None])
        names.append(name)

    new_points = np.concatenate([base_points, *arrays], axis=-1)
    new_dataset = Dataset(
        new_points,
        smart_ds.raw.corners,
        smart_ds.raw.aux,
        smart_ds.raw.title,
        list(smart_ds.raw.variables) + names,
        f"{smart_ds.raw.zone} ({zone_suffix})",
    )
    return type(smart_ds)(
        new_dataset,
        field_functions=smart_ds._field_functions,
        aliases=smart_ds._aliases,
        cache_enabled=smart_ds._cache_enabled,
        computation_graph=smart_ds._computation_graph,
        include_aux_in_loader=smart_ds._include_aux_in_loader,
    )


def _attach_shell_compat_view(
    shell_ds,
    *,
    radii,
    theta,
    phi,
    x_name,
    y_name,
    z_name,
    sampled_field_names,
    area,
):
    # Compatibility shim for existing shell-analysis code while APIs are migrated.
    shell_ds.radii = np.array(radii, dtype=float)
    shell_ds.theta = np.array(theta, dtype=float)
    shell_ds.phi = np.array(phi, dtype=float)
    shell_ds.x = np.array(shell_ds.variable(x_name), dtype=float)
    shell_ds.y = np.array(shell_ds.variable(y_name), dtype=float)
    shell_ds.z = np.array(shell_ds.variable(z_name), dtype=float)
    shell_ds.area = np.array(area, dtype=float)
    shell_ds.fields = {
        name: np.array(shell_ds.variable(name), dtype=float) for name in tuple(sampled_field_names)
    }
    return shell_ds


# TODO this is too permissive and hacky.
# TODO if RBODY exists, which it often does, it is in solar units, same as X [R], etc.
def infer_body_radius_m(smart_ds, body_radius_m: float | None = None) -> float:
    if body_radius_m is not None:
        return float(body_radius_m)

    aux = getattr(smart_ds, "aux", {})
    for key in ("Star_radius_m", "Planet_radius_m", "BODY_RADIUS_M", "RBODY [m]", "RBODY_M"):
        if key in aux:
            try:
                return float(aux[key])
            except Exception:
                pass
    

    # If a graph exposes RBODY [m], use it.
    try:
        return float(smart_ds.variable("RBODY [m]"))
    except Exception:
        pass

    raise ValueError(
        "Could not infer body radius in meters. Pass body_radius_m explicitly."
    )


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

    This is useful for choosing shell radii that align with native BATSRUS radial
    sampling (e.g. points with `y=z=0` on the positive `x` axis).
    """
    axis_key = str(axis).lower()
    axis_idx = {"x": 0, "y": 1, "z": 2}.get(axis_key)
    if axis_idx is None:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    coords = [np.array(smart_ds.variable(name), dtype=float).ravel() for name in coord_fields]
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

    vals = np.array(values[mask], dtype=float)
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
    """
    Resample fields onto spherical shell cell centers.

    `radii` and `coordinate_fields` are assumed to use the same length unit.
    If `length_unit_to_m` is provided, shell areas are computed in `m^2`; otherwise in
    the square of the shell coordinate unit.

    Returns a NEW structured `SmartDs` whose variables include:
    - sampled bound coordinates (e.g. `X [R]`, `Y [R]`, `Z [R]`)
    - requested sampled fields (or all parent raw fields if `fields is None`)
    - free spherical coordinates (`R [unit]`, `theta [rad]`, `phi [rad]`)

    A temporary compatibility view (`.radii`, `.theta`, `.phi`, `.x`, `.y`, `.z`,
    `.area`, `.fields`) is attached for existing shell-analysis code.
    """
    radii = np.atleast_1d(np.array(radii, dtype=float))
    if radii.ndim != 1:
        raise ValueError("radii must be 1D")
    if np.any(radii <= 0):
        raise ValueError("radii must be > 0")

    if polar_edges is None:
        polar_edges = np.linspace(0.0, math.pi, n_polar + 1)
    if azimuthal_edges is None:
        azimuthal_edges = np.linspace(-math.pi, math.pi, n_azimuth + 1)

    ang_grid = PolarAzimuthalGrid(polar_edges, azimuthal_edges)
    theta = np.array(ang_grid.polar_centres, dtype=float)
    phi = np.array(ang_grid.azimuthal_centres, dtype=float)
    area_unit_sphere = np.array(ang_grid.cell_solid_angle, dtype=float)
    if theta.shape != area_unit_sphere.shape:
        if theta.T.shape != area_unit_sphere.shape or phi.T.shape != area_unit_sphere.shape:
            raise ValueError(
                "Angular center arrays do not match cell-area shape "
                f"{area_unit_sphere.shape}: theta={theta.shape}, phi={phi.shape}"
            )
        theta = theta.T
        phi = phi.T

    sin_theta = np.sin(theta)
    xhat = sin_theta * np.cos(phi)
    yhat = sin_theta * np.sin(phi)
    zhat = np.cos(theta)

    ntheta, nphi = theta.shape
    xyz_unit = np.stack((xhat, yhat, zhat), axis=-1)  # (ntheta, nphi, 3)
    xyz = radii[:, None, None, None] * xyz_unit[None, :, :, :]
    sample_points = xyz

    resampled = _resample_shell_points(
        smart_ds,
        sample_points,
        fields=fields,
        coordinate_fields=coordinate_fields,
        method=method,
        fill_value=fill_value,
        title_suffix="shell samples (grid)",
    )

    if fields is None:
        sampled_field_names = tuple(
            name for name in resampled.variables if name not in tuple(coordinate_fields)
        )
    else:
        sampled_field_names = tuple(dict.fromkeys(fields))

    area = (radii[:, None, None] ** 2) * area_unit_sphere[None, :, :]
    if length_unit_to_m is not None:
        area = area * float(length_unit_to_m) ** 2

    x_name, y_name, z_name = coordinate_fields
    length_unit = _field_unit_from_brackets(x_name) or "R"
    r_name = f"R [{length_unit}]"
    theta_name = "theta [rad]"
    phi_name = "phi [rad]"
    area_unit = "m^2" if length_unit_to_m is not None else f"{length_unit}^2"
    area_name = f"dA [{area_unit}]"

    r_field = np.broadcast_to(radii[:, None, None], (radii.size, ntheta, nphi)).copy()
    theta_field = np.broadcast_to(theta[None, :, :], (radii.size, ntheta, nphi)).copy()
    phi_field = np.broadcast_to(phi[None, :, :], (radii.size, ntheta, nphi)).copy()
    area_field = np.array(area, dtype=float)

    shell_ds = _append_fields_to_smart_ds(
        resampled,
        {
            r_name: r_field,
            theta_name: theta_field,
            phi_name: phi_field,
            area_name: area_field,
        },
        zone_suffix="shell-grid-structured",
    )

    # Attach compatibility attributes expected by existing shell-analysis code.
    return _attach_shell_compat_view(
        shell_ds,
        radii=radii,
        theta=theta,
        phi=phi,
        x_name=x_name,
        y_name=y_name,
        z_name=z_name,
        sampled_field_names=sampled_field_names,
        area=area,
    )


def sample_spherical_shells_fibonacci(
    smart_ds,
    radii,
    *,
    fields=(),
    coordinate_fields=("X [R]", "Y [R]", "Z [R]"),
    n_points: int = 512,
    randomize: bool = False,
    method: str = "nearest",
    fill_value: float = np.nan,
    length_unit_to_m: float | None = None,
):
    """
    Resample fields onto equal-area Fibonacci sphere points on each shell.

    The returned arrays use shape `(nr, n_points, 1)` so they remain compatible with
    existing shell integrations that sum over the last two axes.
    """
    radii = np.atleast_1d(np.array(radii, dtype=float))
    if radii.ndim != 1:
        raise ValueError("radii must be 1D")
    if np.any(radii <= 0):
        raise ValueError("radii must be > 0")
    n_points = int(n_points)
    if n_points < 8:
        raise ValueError("n_points must be >= 8")

    unit = np.array(fibonacci_sphere(n_points, randomize=randomize), dtype=float)
    xhat = unit[:, 0][:, None]
    yhat = unit[:, 1][:, None]
    zhat = unit[:, 2][:, None]
    theta = np.arccos(np.clip(zhat, -1.0, 1.0))
    phi = np.arctan2(yhat, xhat)

    xyz_unit = np.stack((xhat, yhat, zhat), axis=-1)  # (npts, 1, 3)
    xyz = radii[:, None, None, None] * xyz_unit[None, :, :, :]  # (nr, npts, 1, 3)
    sample_points = xyz.reshape(-1, 3)

    resampled = _resample_shell_points(
        smart_ds,
        sample_points,
        fields=fields,
        coordinate_fields=coordinate_fields,
        method=method,
        fill_value=fill_value,
        title_suffix="shell samples (fibonacci)",
    )

    field_arrays = {}
    for name in tuple(dict.fromkeys(fields)):
        field_arrays[name] = np.array(resampled.variable(name), dtype=float).reshape(
            radii.size, n_points, 1
        )

    area_unit = (4.0 * math.pi) / float(n_points)
    area = (radii[:, None, None] ** 2) * area_unit
    if length_unit_to_m is not None:
        area = area * float(length_unit_to_m) ** 2

    return SphericalShellSamples(
        radii=radii,
        theta=theta,
        phi=phi,
        x=xyz[..., 0],
        y=xyz[..., 1],
        z=xyz[..., 2],
        area=np.broadcast_to(area, (radii.size, n_points, 1)).copy(),
        fields=field_arrays,
    )


def sample_spherical_shells_by_strategy(
    smart_ds,
    radii,
    *,
    fields=(),
    coordinate_fields=("X [R]", "Y [R]", "Z [R]"),
    n_polar: int = 24,
    n_azimuth: int = 48,
    sampling: str = "fibonacci",
    fibonacci_randomize: bool = False,
    method: str = "nearest",
    fill_value: float = np.nan,
    length_unit_to_m: float | None = None,
):
    """
    Sample spherical shells using either the structured grid or Fibonacci sampler.

    This centralizes the sampling dispatch used by shell-based analysis helpers.
    """
    common_kwargs = dict(
        smart_ds=smart_ds,
        radii=radii,
        fields=fields,
        coordinate_fields=coordinate_fields,
        method=method,
        fill_value=fill_value,
        length_unit_to_m=length_unit_to_m,
    )
    if sampling == "fibonacci":
        return sample_spherical_shells_fibonacci(
            **common_kwargs,
            n_points=max(8, int(n_polar) * int(n_azimuth)),
            randomize=fibonacci_randomize,
        )
    if sampling == "grid":
        return sample_spherical_shells(
            **common_kwargs,
            n_polar=n_polar,
            n_azimuth=n_azimuth,
        )
    raise ValueError("sampling must be 'fibonacci' or 'grid'")


def integrate_shell_scalar(values, area):
    """
    Integrate scalar values over shell surfaces with NaN-safe area weighting.

    Returns `(integral, coverage)` for each shell radius, where `coverage` is the
    finite-area fraction represented by finite values.
    """
    v = np.array(values, dtype=float)
    a = np.array(area, dtype=float)
    if v.shape != a.shape:
        a = np.broadcast_to(a, v.shape)

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


def shell_profile_radius_height(shells):
    radii = np.array(shells.radii, dtype=float)
    return {
        "radius [R]": radii,
        "height [R]": radii - 1.0,
    }

__all__ = [
    "SphericalShellSamples",
    "infer_body_radius_m",
    "infer_cartesian_axis_radii",
    "integrate_shell_scalar",
    "shell_profile_radius_height",
    "sample_spherical_shells_by_strategy",
    "sample_spherical_shells",
    "sample_spherical_shells_fibonacci",
]
