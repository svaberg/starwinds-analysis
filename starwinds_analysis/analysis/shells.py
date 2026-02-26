"""THIS FILE contains spherical shell sampling and shell-integration primitives.

It is the foundation layer for shell-based analyses (sampling geometry, integration, coverage).
"""

from __future__ import annotations

import math

import numpy as np
from starwinds_readplt.dataset import Dataset

from starwinds_analysis.algorithms.sphere_sampling import PolarAzimuthalGrid, fibonacci_sphere

# Resample requested fields onto explicit shell points and return a shell SmartDs.
# Used in: `starwinds_analysis/analysis/shells.py`
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

# Extract the unit substring from a bracketed field name like `X [R]`.
# Used in: `starwinds_analysis/analysis/shells.py`
def _field_unit_from_brackets(name: str) -> str | None:
    text = str(name)
    i = text.rfind("[")
    j = text.rfind("]")
    if i == -1 or j == -1 or j <= i:
        return None
    return text[i + 1 : j].strip() or None

# Attach derived arrays (free coords/areas/etc.) to a resampled shell SmartDs.
# Used in: `starwinds_analysis/analysis/shells.py`
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
        arr = np.array(values)
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

# TODO this is too permissive and hacky.
# TODO if RBODY exists, which it often does, it is in solar units, same as X [R], etc.
# Infer the body radius in meters from args/aux so shell/orbit lengths can be converted to SI.
# Used in: `starwinds_analysis/physics/orbit_local.py`,
#   `starwinds_analysis/physics/orbit_surface.py`, `starwinds_analysis/physics/fluxes.py`,
#   `starwinds_analysis/physics/mass_loss.py`, `starwinds_analysis/physics/torque.py` (+1 more)
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

# Infer available shell radii from points lying on a Cartesian axis.
# Used in: `test/test_shell_analysis.py`
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

# Resample fields onto spherical shell cell centers.
# Used in: `test/test_shell_magnetic_analysis.py`, `test/test_shell_analysis.py`,
#   `test/test_shell_resample_smartds_spec.py`, `examples/smartds_quicklook_profiles.ipynb`,
#   `examples/smartds_shell_mass_flux.ipynb` (+1 more)
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

    No shell-specific custom container is created; callers should request fields
    directly from the returned structured `SmartDs`.
    """
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
    theta = np.array(ang_grid.polar_centres)
    phi = np.array(ang_grid.azimuthal_centres)
    area_unit_sphere = np.array(ang_grid.cell_solid_angle)
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
    area_field = np.array(area)

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

    return shell_ds

# Resample fields onto equal-area Fibonacci sphere points on each shell.
# Used in: `test/test_shell_analysis.py`, `examples/smartds_shell_mass_flux.ipynb`,
#   `starwinds_analysis/analysis/shells.py`
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

    Returns a NEW structured `SmartDs` whose arrays have shape `(nr, n_points, 1)`.
    As with the grid sampler, no shell-specific custom container is created; callers
    should request fields directly from the returned structured `SmartDs`.
    """
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

    xyz_unit = np.stack((xhat, yhat, zhat), axis=-1)  # (npts, 1, 3)
    xyz = radii[:, None, None, None] * xyz_unit[None, :, :, :]  # (nr, npts, 1, 3)
    sample_points = xyz

    resampled = _resample_shell_points(
        smart_ds,
        sample_points,
        fields=fields,
        coordinate_fields=coordinate_fields,
        method=method,
        fill_value=fill_value,
        title_suffix="shell samples (fibonacci)",
    )

    area_unit = (4.0 * math.pi) / float(n_points)
    area = (radii[:, None, None] ** 2) * area_unit
    if length_unit_to_m is not None:
        area = area * float(length_unit_to_m) ** 2

    x_name, y_name, z_name = coordinate_fields
    length_unit = _field_unit_from_brackets(x_name) or "R"
    r_name = f"R [{length_unit}]"
    theta_name = "theta [rad]"
    phi_name = "phi [rad]"
    area_unit_name = "m^2" if length_unit_to_m is not None else f"{length_unit}^2"
    area_name = f"dA [{area_unit_name}]"

    r_field = np.broadcast_to(radii[:, None, None], (radii.size, n_points, 1)).copy()
    theta_field = np.broadcast_to(theta[None, :, :], (radii.size, n_points, 1)).copy()
    phi_field = np.broadcast_to(phi[None, :, :], (radii.size, n_points, 1)).copy()
    area_field = np.broadcast_to(area, (radii.size, n_points, 1)).copy()

    shell_ds = _append_fields_to_smart_ds(
        resampled,
        {
            r_name: r_field,
            theta_name: theta_field,
            phi_name: phi_field,
            area_name: area_field,
        },
        zone_suffix="shell-fibonacci-structured",
    )
    return shell_ds

# Sample spherical shells using either the structured grid or Fibonacci sampler.
# Used in: `starwinds_analysis/physics/fluxes.py`, `starwinds_analysis/physics/mass_loss.py`,
#   `starwinds_analysis/physics/torque.py`
def sample_spherical_shells_by_strategy(
    smart_ds,
    radii,
    *,
    fields=None,
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

# Integrate scalar values over shell surfaces with NaN-safe area weighting.
# Used in: `test/test_shell_magnetic_analysis.py`, `test/test_shell_analysis.py`,
#   `examples/smartds_inner_boundary_magnetic_zdi.ipynb`,
#   `examples/smartds_quicklook_profiles.ipynb`, `examples/smartds_shell_mass_flux.ipynb` (+3 more)
def integrate_shell_scalar(values, area):
    """
    Integrate scalar values over shell surfaces with NaN-safe area weighting.

    Returns `(integral, coverage)` for each shell radius, where `coverage` is the
    finite-area fraction represented by finite values.
    """
    v = np.array(values)
    a = np.array(area)
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

# Build standard radius/height profile arrays from a shell SmartDs.
# Used in: `starwinds_analysis/physics/fluxes.py`, `starwinds_analysis/physics/mass_loss.py`,
#   `starwinds_analysis/physics/torque.py`
def shell_profile_radius_height(shells):
    if hasattr(shells, "has_field") and shells.has_field("R [R]"):
        r_field = np.array(shells("R [R]"))
        if r_field.ndim >= 2:
            radii = np.nanmean(r_field.reshape(r_field.shape[0], -1), axis=1)
        else:
            radii = np.array(r_field)
    else:
        raise ValueError("shell_profile_radius_height expects a shell SmartDs with 'R [R]'")
    return {
        "radius [R]": radii,
        "height [R]": radii - 1.0,
    }

