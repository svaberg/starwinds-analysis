"""THIS FILE contains torque-related formulas and temporary torque workflows.

It groups local torque-density formulas plus current shell/surface torque wrappers in
one place. The wrappers are still debt and should be pushed toward generic reduction
primitives later.
"""

from __future__ import annotations

import logging

import numpy as np

from starwinds_analysis.analysis.shells import infer_body_radius_m
from starwinds_analysis.analysis.shells import integrate_shell_scalar
from starwinds_analysis.analysis.shells import sample_spherical_shells_by_strategy
from starwinds_analysis.constants import MU0

log = logging.getLogger(__name__)

# TODO(debt): This file still carries the quantity-specific `surface_torque_vs_radius`
# wrapper in a deep layer.


def spherical_wind_torque_density_terms(
    *,
    rho_kg_m3,
    u_radial_m_s,
    u_azimuthal_m_s,
    b_radial_t,
    b_azimuthal_t,
    cylindrical_radius_m,
):
    """
    Spherical-shell wind torque-density terms about +z.
    Used by: no external call sites found
    """
    # TODO(griblet): These local spherical torque-density terms should be available
    # via SmartDs/griblet for SI fields, instead of being recomputed in shell/orbit
    # diagnostics.
    magnetic = -cylindrical_radius_m * b_azimuthal_t * b_radial_t / MU0
    dynamic = cylindrical_radius_m * rho_kg_m3 * u_azimuthal_m_s * u_radial_m_s
    return magnetic, dynamic


def rotational_frame_velocity(u_xyz_m_s, xyz_m, angvel_rad_s):
    """
    Convert inertial velocity `u` to rotating-frame velocity `V = u - Omega x r`
    Used by: `starwinds_analysis/physics/torque.py`
    """
    u = np.array(u_xyz_m_s)
    xyz = np.array(xyz_m)
    if u.shape != xyz.shape or u.shape[-1] != 3:
        raise ValueError("u_xyz_m_s and xyz_m must have the same shape (..., 3)")
    omega = float(angvel_rad_s)
    v = np.array(u, copy=True)
    v[..., 0] = u[..., 0] + omega * xyz[..., 1]
    v[..., 1] = u[..., 1] - omega * xyz[..., 0]
    v[..., 2] = u[..., 2]
    return v


def normalize_surface_normals(normals_xyz):
    """
    Normalize explicit surface normals safely for torque integration.
    Used by: `starwinds_analysis/physics/torque.py`
    """
    n = np.array(normals_xyz)
    if n.shape[-1] != 3:
        raise ValueError("normals_xyz must have shape (..., 3)")
    nmag = np.sqrt(np.sum(n * n, axis=-1, keepdims=True))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.divide(n, nmag, out=np.full_like(n, np.nan), where=nmag > 0)
    return out


def radial_surface_normals(xyz):
    """
    Build radial normals from explicit Cartesian surface points.
    Used by: `starwinds_analysis/physics/torque.py`
    """
    xyz = np.array(xyz)
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
    Used by: `test/test_surface_torque_analysis.py`, `starwinds_analysis/physics/orbit_surface.py`,
      `starwinds_analysis/physics/torque.py`
    """
    # TODO(griblet): The local explicit-surface torque terms (`T1..T4`, `total`) are
    # physical quantities and should eventually be requestable via SmartDs/griblet in
    # SI units, with geometry inputs supplied explicitly.
    xyz = np.array(xyz_m)
    n = normalize_surface_normals(normals_xyz)
    area = np.array(area_m2)
    rho = np.array(rho_kg_m3)
    u = np.array(u_xyz_m_s)
    b = np.array(b_xyz_t)

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
        p = np.array(pressure_pa)
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
    Used by: `test/test_surface_torque_analysis.py`, `starwinds_analysis/physics/orbit_surface.py`,
      `starwinds_analysis/physics/torque.py`
    """
    area = np.array(terms["area [m^2]"])
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
        integral, cov = integrate_shell_scalar(np.array(terms[key]), area)
        out[key.replace("[N/m]", "[Nm]")] = np.array(integral)
        component_integrals.append(np.array(integral))
        coverages.append(np.array(cov))
    out["total [Nm]"] = np.sum(np.stack(component_integrals, axis=0), axis=0)
    if "total [N/m]" in terms:
        direct_total, cov_total = integrate_shell_scalar(np.array(terms["total [N/m]"]), area)
        out["total_direct [Nm]"] = np.array(direct_total)
        coverages.append(np.array(cov_total))
    if coverages:
        out["coverage [none]"] = np.min(np.stack(coverages, axis=0), axis=0)
    log.debug(
        "integrate_surface_torque_terms: components=%d, coverage_shape=%s",
        len(component_keys),
        np.shape(out.get("coverage [none]")),
    )
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
    coordinate_fields=("X [R]", "Y [R]", "Z [R]"),
    area_field: str = "dA [m^2]",
):
    """
    Convenience wrapper for explicit-surface torque terms on shell samples.
    Used by: `starwinds_analysis/physics/torque.py`
    """
    x_name, y_name, z_name = coordinate_fields
    xyz_r = np.stack(
        [
            np.array(shells(x_name)),
            np.array(shells(y_name)),
            np.array(shells(z_name)),
        ],
        axis=-1,
    )
    xyz_m = xyz_r * float(body_radius_m)
    normals = radial_surface_normals(xyz_m) if normals_xyz is None else normals_xyz
    return surface_torque_density_terms(
        xyz_m=xyz_m,
        normals_xyz=normals,
        area_m2=np.array(shells(area_field)),
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
    Used by: `test/test_surface_torque_analysis.py`
    """
    try:
        n_radii = len(radii)
    except TypeError:
        n_radii = None
    log.info(
        "surface_torque_vs_radius start: n_radii=%s, sampling=%s, method=%s, include_pressure=%s",
        n_radii,
        sampling,
        method,
        include_pressure_term,
    )
    smart_ds.add_batsrus_graph(body_radius_m=body_radius_m)
    body_radius_m = infer_body_radius_m(smart_ds, body_radius_m=body_radius_m)
    rho_name = "Rho [kg/m^3]"
    ux_name, uy_name, uz_name = "U_x [m/s]", "U_y [m/s]", "U_z [m/s]"
    bx_name, by_name, bz_name = "B_x [T]", "B_y [T]", "B_z [T]"

    fields = [rho_name, ux_name, uy_name, uz_name, bx_name, by_name, bz_name]
    p_name = None
    if include_pressure_term:
        p_name = "thermal_pressure [Pa]"
        fields.append(p_name)

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
    shells.add_batsrus_graph(body_radius_m=body_radius_m, merge=False)

    rho = np.array(shells(rho_name))
    u_xyz = np.array(shells("U_xyz [m/s]"))
    b_xyz = np.array(shells("B_xyz [T]"))
    p = None if p_name is None else np.array(shells(p_name))

    terms = surface_torque_terms_on_shell_samples(
        shells,
        rho_kg_m3=rho,
        u_xyz_m_s=u_xyz,
        b_xyz_t=b_xyz,
        pressure_pa=p,
        angvel_rad_s=angvel_rad_s,
        body_radius_m=body_radius_m,
        use_rotating_frame=True,
        coordinate_fields=coordinate_fields,
    )
    ints = integrate_surface_torque_terms(terms)
    r_field = np.array(shells("R [R]"))
    radii_profile = np.nanmean(r_field.reshape(r_field.shape[0], -1), axis=1)
    log.info(
        "surface_torque_vs_radius done: n_shells=%d, finite_total=%d",
        radii_profile.size,
        np.count_nonzero(np.isfinite(ints["total [Nm]"])),
    )

    return {
        "radius [R]": radii_profile,
        "height [R]": radii_profile - 1.0,
        "T1_magnetic [Nm]": np.array(ints["T1_magnetic [Nm]"]),
        "T2_pressure [Nm]": np.array(ints["T2_pressure [Nm]"]),
        "T3_corotation [Nm]": np.array(ints["T3_corotation [Nm]"]),
        "T4_dynamic [Nm]": np.array(ints["T4_dynamic [Nm]"]),
        "total [Nm]": np.array(ints["total [Nm]"]),
        "coverage [none]": np.array(ints["coverage [none]"]),
        "shell_samples": shells,
        "surface_terms": terms,
        "sampling": sampling,
    }
