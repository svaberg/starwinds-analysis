from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from starwinds_analysis.algorithms.sphere_sampling import PolarAzimuthalGrid


@dataclass
class SphericalShellSamples:
    """
    Samples on one or more spherical shells.

    Shapes:
    - `radii`: `(nr,)`
    - `theta`, `phi`: `(ntheta, nphi)`
    - `x`, `y`, `z`, `area`: `(nr, ntheta, nphi)`
    - each `fields[name]`: `(nr, ntheta, nphi)`
    """

    radii: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    area: np.ndarray
    fields: dict[str, np.ndarray]


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


def sample_spherical_shells(
    smart_ds,
    radii,
    *,
    fields=(),
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
    If `length_unit_to_m` is provided, returned areas are in `m^2`; otherwise they
    are in the square of the shell coordinate unit.
    """
    radii = np.atleast_1d(np.asarray(radii, dtype=float))
    if radii.ndim != 1:
        raise ValueError("radii must be 1D")
    if np.any(radii <= 0):
        raise ValueError("radii must be > 0")

    if polar_edges is None:
        polar_edges = np.linspace(0.0, math.pi, n_polar + 1)
    if azimuthal_edges is None:
        azimuthal_edges = np.linspace(-math.pi, math.pi, n_azimuth + 1)

    ang_grid = PolarAzimuthalGrid(polar_edges, azimuthal_edges)
    theta = np.asarray(ang_grid.polar_centres, dtype=float)
    phi = np.asarray(ang_grid.azimuthal_centres, dtype=float)
    area_unit_sphere = np.asarray(ang_grid.cell_area, dtype=float)
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
    sample_points = xyz.reshape(-1, 3)

    resampled = smart_ds.resample(
        sample_points,
        coordinate_fields=coordinate_fields,
        fields=tuple(dict.fromkeys(fields)),
        method=method,
        fill_value=fill_value,
        title=f"{getattr(smart_ds, 'title', 'dataset')} (shell samples)",
        zone="shell-samples",
    )

    field_arrays = {}
    for name in tuple(dict.fromkeys(fields)):
        field_arrays[name] = np.asarray(resampled.variable(name), dtype=float).reshape(
            radii.size, ntheta, nphi
        )

    area = (radii[:, None, None] ** 2) * area_unit_sphere[None, :, :]
    if length_unit_to_m is not None:
        area = area * float(length_unit_to_m) ** 2

    return SphericalShellSamples(
        radii=radii,
        theta=theta,
        phi=phi,
        x=xyz[..., 0],
        y=xyz[..., 1],
        z=xyz[..., 2],
        area=area,
        fields=field_arrays,
    )


def integrate_shell_scalar(values, area):
    """
    Integrate scalar values over shell surfaces with NaN-safe area weighting.

    Returns `(integral, coverage)` for each shell radius, where `coverage` is the
    finite-area fraction represented by finite values.
    """
    v = np.asarray(values, dtype=float)
    a = np.asarray(area, dtype=float)
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


def resolve_field_with_scale(smart_ds, candidates):
    """
    Pick the first available field among `(name, scale_to_target)`.
    """
    for name, scale in candidates:
        try:
            if smart_ds.has_field(name):
                return name, float(scale)
        except Exception:
            continue
    names = ", ".join(name for name, _ in candidates)
    raise KeyError(f"None of the candidate fields are available: {names}")


def resolve_batsrus_density_si(smart_ds):
    amu_kg = 1.66053906660e-27
    return resolve_field_with_scale(
        smart_ds,
        [
            ("Rho [kg/m^3]", 1.0),
            ("Rho [g/cm^3]", 1e3),
            ("Rho [amu/cm^3]", amu_kg * 1e6),
        ],
    )


def resolve_batsrus_vector_xyz_si(smart_ds, prefix: str):
    if prefix == "U":
        unit_candidates = [("m/s", 1.0), ("km/s", 1e3)]
    elif prefix == "B":
        unit_candidates = [("T", 1.0), ("Gauss", 1e-4), ("G", 1e-4), ("nT", 1e-9)]
    else:
        raise ValueError(f"Unsupported vector prefix '{prefix}'")

    for unit, scale in unit_candidates:
        names = [f"{prefix}_{c} [{unit}]" for c in "xyz"]
        if all(smart_ds.has_field(name) for name in names):
            return tuple(names), float(scale)

    tried = ", ".join(f"{prefix}_x [{u}]/..." for u, _ in unit_candidates)
    raise KeyError(f"Could not resolve SI-compatible vector '{prefix}' components. Tried: {tried}")


__all__ = [
    "SphericalShellSamples",
    "infer_body_radius_m",
    "integrate_shell_scalar",
    "resolve_batsrus_density_si",
    "resolve_batsrus_vector_xyz_si",
    "resolve_field_with_scale",
    "sample_spherical_shells",
]
