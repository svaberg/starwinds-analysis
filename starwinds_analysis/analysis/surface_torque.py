"""THIS FILE contains explicit-surface torque density terms and surface torque integration.

It is the core torque machinery for arbitrary explicit surfaces (points/normals/areas).
Higher-level shell/orbit workflows should call into this file rather than reimplementing torque terms.
"""

from __future__ import annotations

import math

import numpy as np

from starwinds_analysis.analysis.shells import (
    infer_body_radius_m,
    integrate_shell_scalar,
    resolve_batsrus_density_si,
    resolve_batsrus_vector_xyz_si,
    sample_spherical_shells_by_strategy,
    shell_profile_radius_height,
)


MU0 = 4.0e-7 * math.pi


def rotational_frame_velocity(u_xyz_m_s, xyz_m, angvel_rad_s):
    """
    Convert inertial velocity `u` to rotating-frame velocity `V = u - Omega x r`
    for rotation about +z with scalar angular speed `angvel_rad_s`.
    """
    u = np.asarray(u_xyz_m_s, dtype=float)
    xyz = np.asarray(xyz_m, dtype=float)
    if u.shape != xyz.shape or u.shape[-1] != 3:
        raise ValueError("u_xyz_m_s and xyz_m must have the same shape (..., 3)")
    omega = float(angvel_rad_s)
    v = np.array(u, copy=True)
    v[..., 0] = u[..., 0] + omega * xyz[..., 1]
    v[..., 1] = u[..., 1] - omega * xyz[..., 0]
    v[..., 2] = u[..., 2]
    return v


def normalize_surface_normals(normals_xyz):
    n = np.asarray(normals_xyz, dtype=float)
    if n.shape[-1] != 3:
        raise ValueError("normals_xyz must have shape (..., 3)")
    nmag = np.sqrt(np.sum(n * n, axis=-1, keepdims=True))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.divide(n, nmag, out=np.full_like(n, np.nan), where=nmag > 0)
    return out


def radial_surface_normals(xyz):
    xyz = np.asarray(xyz, dtype=float)
    if xyz.shape[-1] != 3:
        raise ValueError("xyz must have shape (..., 3)")
    return normalize_surface_normals(xyz)


def surface_torque_density_terms(
    *,
    xyz_m,
    normals_xyz,
    area_m2,
    rho_kg_m3,
    u_xyz_m_s,
    b_xyz_t,
    pressure_pa=None,
    angvel_rad_s: float = 0.0,
    use_rotating_frame: bool = True,
):
    """
    Mestel/Vidotto-like z-angular-momentum flux terms on an explicit surface.

    Returns per-area torque-density terms (`T1..T4`, `total`) with units `N/m`.
    Integrate with `integrate_surface_torque_terms(...)`.

    Term definitions (about +z):
    - `T1`: magnetic stress term
    - `T2`: pressure + magnetic pressure geometric term
    - `T3`: corotation source-like transport term
    - `T4`: kinetic (dynamic) term in the chosen velocity frame

    For spherical surfaces with radial normals and `angvel_rad_s=0`, `T2=T3=0` and
    `T1+T4` reduces to the existing spherical-shell torque expression.
    """
    xyz = np.asarray(xyz_m, dtype=float)
    n = normalize_surface_normals(normals_xyz)
    area = np.asarray(area_m2, dtype=float)
    rho = np.asarray(rho_kg_m3, dtype=float)
    u = np.asarray(u_xyz_m_s, dtype=float)
    b = np.asarray(b_xyz_t, dtype=float)

    if xyz.shape != u.shape or xyz.shape != b.shape or xyz.shape != n.shape:
        raise ValueError("xyz_m, normals_xyz, u_xyz_m_s, and b_xyz_t must match shape (..., 3)")
    if xyz.shape[-1] != 3:
        raise ValueError("vector inputs must have shape (..., 3)")

    scalar_shape = xyz.shape[:-1]
    if area.shape != scalar_shape:
        area = np.broadcast_to(area, scalar_shape)
    if rho.shape != scalar_shape:
        rho = np.broadcast_to(rho, scalar_shape)

    if pressure_pa is None:
        p = np.zeros(scalar_shape, dtype=float)
    else:
        p = np.asarray(pressure_pa, dtype=float)
        if p.shape != scalar_shape:
            p = np.broadcast_to(p, scalar_shape)

    omega = float(angvel_rad_s)
    v = rotational_frame_velocity(u, xyz, omega) if use_rotating_frame else u

    x = xyz[..., 0]
    y = xyz[..., 1]

    bx = b[..., 0]
    by = b[..., 1]
    bz = b[..., 2]

    vx = v[..., 0]
    vy = v[..., 1]
    vz = v[..., 2]

    nx = n[..., 0]
    ny = n[..., 1]
    nz = n[..., 2]

    bdotn = bx * nx + by * ny + bz * nz
    vdotn = vx * nx + vy * ny + vz * nz
    b2 = bx * bx + by * by + bz * bz
    magnetic_pressure = b2 / (2.0 * MU0)

    t1 = (-x * by + y * bx) * (bdotn / MU0)
    t2 = (x * ny - y * nx) * (p + magnetic_pressure)
    t3 = omega * (x * x + y * y) * rho * vdotn
    t4 = (x * vy - y * vx) * rho * vdotn
    total = t1 + t2 + t3 + t4

    # Coverage can be derived from this mask consistently for all terms.
    mask = (
        np.isfinite(area)
        & np.isfinite(rho)
        & np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(nx)
        & np.isfinite(ny)
        & np.isfinite(nz)
        & np.isfinite(bx)
        & np.isfinite(by)
        & np.isfinite(bz)
        & np.isfinite(vx)
        & np.isfinite(vy)
        & np.isfinite(vz)
        & np.isfinite(p)
    )

    return {
        "area [m^2]": area,
        "mask": mask,
        "B_dot_n [T]": bdotn,
        "V_dot_n [m/s]": vdotn,
        "T1_magnetic [N/m]": t1,
        "T2_pressure [N/m]": t2,
        "T3_corotation [N/m]": t3,
        "T4_dynamic [N/m]": t4,
        "total [N/m]": total,
    }


def integrate_surface_torque_terms(terms):
    """
    Integrate per-area torque-density terms from `surface_torque_density_terms(...)`.
    """
    area = np.asarray(terms["area [m^2]"], dtype=float)
    out = {}
    coverages = []
    component_keys = (
        "T1_magnetic [N/m]",
        "T2_pressure [N/m]",
        "T3_corotation [N/m]",
        "T4_dynamic [N/m]",
    )
    component_integrals = []
    for key in component_keys:
        integral, cov = integrate_shell_scalar(np.asarray(terms[key], dtype=float), area)
        out[key.replace("[N/m]", "[Nm]")] = np.asarray(integral, dtype=float)
        component_integrals.append(np.asarray(integral, dtype=float))
        coverages.append(np.asarray(cov, dtype=float))
    # Define the total as the sum of integrated terms, so bookkeeping is exact even
    # when individual terms have different finite masks.
    out["total [Nm]"] = np.sum(np.stack(component_integrals, axis=0), axis=0)
    if "total [N/m]" in terms:
        _direct_total, cov_total = integrate_shell_scalar(np.asarray(terms["total [N/m]"], dtype=float), area)
        out["total_direct [Nm]"] = np.asarray(_direct_total, dtype=float)
        coverages.append(np.asarray(cov_total, dtype=float))
    if coverages:
        out["coverage [none]"] = np.min(np.stack(coverages, axis=0), axis=0)
    return out


def surface_torque_terms_on_shell_samples(
    shells,
    *,
    rho_kg_m3,
    u_xyz_m_s,
    b_xyz_t,
    pressure_pa=None,
    angvel_rad_s: float = 0.0,
    body_radius_m: float = 1.0,
    normals_xyz=None,
    use_rotating_frame: bool = True,
):
    """
    Convenience wrapper for explicit-surface torque terms on shell samples.
    """
    xyz_r = np.stack((shells.x, shells.y, shells.z), axis=-1)
    xyz_m = xyz_r * float(body_radius_m)
    normals = radial_surface_normals(xyz_m) if normals_xyz is None else normals_xyz
    return surface_torque_density_terms(
        xyz_m=xyz_m,
        normals_xyz=normals,
        area_m2=shells.area,
        rho_kg_m3=rho_kg_m3,
        u_xyz_m_s=u_xyz_m_s,
        b_xyz_t=b_xyz_t,
        pressure_pa=pressure_pa,
        angvel_rad_s=angvel_rad_s,
        use_rotating_frame=use_rotating_frame,
    )


def surface_torque_vs_radius(
    smart_ds,
    radii,
    *,
    body_radius_m: float | None = None,
    coordinate_fields=("X [R]", "Y [R]", "Z [R]"),
    n_polar: int = 24,
    n_azimuth: int = 48,
    sampling: str = "fibonacci",
    fibonacci_randomize: bool = False,
    method: str = "nearest",
    fill_value: float = np.nan,
    include_pressure_term: bool = True,
    angvel_rad_s: float = 0.0,
):
    """
    Explicit-surface torque profile on spherical shells using general T1..T4 terms.

    This is a non-VTK route and a bridge toward arbitrary explicit surfaces.
    """
    body_radius_m = infer_body_radius_m(smart_ds, body_radius_m=body_radius_m)
    rho_name, rho_scale = resolve_batsrus_density_si(smart_ds)
    (ux_name, uy_name, uz_name), u_scale = resolve_batsrus_vector_xyz_si(smart_ds, "U")
    (bx_name, by_name, bz_name), b_scale = resolve_batsrus_vector_xyz_si(smart_ds, "B")

    fields = [rho_name, ux_name, uy_name, uz_name, bx_name, by_name, bz_name]
    p_name = p_scale = None
    if include_pressure_term:
        for _name, _scale in (("P [Pa]", 1.0), ("P [dyne/cm^2]", 0.1)):
            if smart_ds.has_field(_name):
                p_name, p_scale = _name, float(_scale)
                fields.append(p_name)
                break

    shells = sample_spherical_shells_by_strategy(
        smart_ds,
        radii,
        fields=tuple(fields),
        coordinate_fields=coordinate_fields,
        n_polar=n_polar,
        n_azimuth=n_azimuth,
        sampling=sampling,
        fibonacci_randomize=fibonacci_randomize,
        method=method,
        fill_value=fill_value,
        length_unit_to_m=body_radius_m,
    )

    rho = rho_scale * shells.fields[rho_name]
    u_xyz = u_scale * np.stack(
        [shells.fields[ux_name], shells.fields[uy_name], shells.fields[uz_name]], axis=-1
    )
    b_xyz = b_scale * np.stack(
        [shells.fields[bx_name], shells.fields[by_name], shells.fields[bz_name]], axis=-1
    )
    p = None if p_name is None else p_scale * shells.fields[p_name]

    terms = surface_torque_terms_on_shell_samples(
        shells,
        rho_kg_m3=rho,
        u_xyz_m_s=u_xyz,
        b_xyz_t=b_xyz,
        pressure_pa=p,
        angvel_rad_s=angvel_rad_s,
        body_radius_m=body_radius_m,
        use_rotating_frame=True,
    )
    ints = integrate_surface_torque_terms(terms)

    return {
        **shell_profile_radius_height(shells),
        "T1_magnetic [Nm]": np.asarray(ints["T1_magnetic [Nm]"], dtype=float),
        "T2_pressure [Nm]": np.asarray(ints["T2_pressure [Nm]"], dtype=float),
        "T3_corotation [Nm]": np.asarray(ints["T3_corotation [Nm]"], dtype=float),
        "T4_dynamic [Nm]": np.asarray(ints["T4_dynamic [Nm]"], dtype=float),
        "total [Nm]": np.asarray(ints["total [Nm]"], dtype=float),
        "coverage [none]": np.asarray(ints["coverage [none]"], dtype=float),
        "shell_samples": shells,
        "surface_terms": terms,
        "sampling": sampling,
    }


__all__ = [
    "MU0",
    "rotational_frame_velocity",
    "normalize_surface_normals",
    "radial_surface_normals",
    "surface_torque_density_terms",
    "integrate_surface_torque_terms",
    "surface_torque_terms_on_shell_samples",
    "surface_torque_vs_radius",
]
