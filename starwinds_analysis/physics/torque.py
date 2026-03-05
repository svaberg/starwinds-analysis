"""Torque-related formulas and temporary torque workflows.
"""

# It groups local torque-density formulas and explicit-surface integration helpers.


from __future__ import annotations

import logging

import numpy as np

from starwinds_analysis.analysis.shells import integrate_shell_scalar
from starwinds_analysis.constants import MU0

log = logging.getLogger(__name__)


def spherical_wind_torque_density_terms(
    *,
    rho,
    U_r,
    U_a,
    B_r,
    B_a,
    cylindrical_radius,
):
    """
    Spherical-shell wind torque-density terms about +z.
    Used by: no external call sites found
    """
    # TODO(griblet): These local spherical torque-density terms should be available
    # via SmartDs/griblet for SI fields, instead of being recomputed in shell/orbit
    # diagnostics.
    magnetic = -cylindrical_radius * B_a * B_r / MU0
    dynamic = cylindrical_radius * rho * U_a * U_r
    return magnetic, dynamic

def rotational_frame_velocity(U_xyz, xyz, angvel):
    """
    Convert inertial velocity `u` to rotating-frame velocity `V = u - Omega x r`
    Used by: `starwinds_analysis/physics/torque.py`
    """
    u = np.array(U_xyz)
    xyz = np.array(xyz)
    if u.shape != xyz.shape or u.shape[-1] != 3:
        log.error("rotational_frame_velocity failed: U_xyz shape=%s xyz shape=%s", u.shape, xyz.shape)
        raise ValueError("U_xyz and xyz must have the same shape (..., 3)")
    omega = float(angvel)
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
        log.error("normalize_surface_normals failed: shape=%s", n.shape)
        raise ValueError("normals_xyz must have shape (..., 3)")
    nmag = np.sqrt(np.sum(n * n, axis=-1, keepdims=True))
    if np.any(nmag == 0):
        log.error("normalize_surface_normals failed: zero-length normals")
        raise ValueError("normals_xyz contains zero-length vectors")
    return n / nmag


def radial_surface_normals(xyz):
    """
    Build radial normals from explicit Cartesian surface points.
    Used by: `starwinds_analysis/physics/torque.py`
    """
    xyz = np.array(xyz)
    if xyz.shape[-1] != 3:
        log.error("radial_surface_normals failed: shape=%s", xyz.shape)
        raise ValueError("xyz must have shape (..., 3)")
    return normalize_surface_normals(xyz)


def surface_torque_density_terms(
    *,
    xyz,
    normals_xyz,
    area,
    rho,
    U_xyz,
    B_xyz,
    pressure=None,
    angvel: float = 0.0,
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
    xyz = np.array(xyz)
    n = normalize_surface_normals(normals_xyz)
    area = np.array(area)
    rho = np.array(rho)
    u = np.array(U_xyz)
    b = np.array(B_xyz)

    if xyz.shape != u.shape or xyz.shape != b.shape or xyz.shape != n.shape:
        log.error(
            "surface_torque_density_terms failed: xyz=%s normals=%s U=%s B=%s",
            xyz.shape,
            n.shape,
            u.shape,
            b.shape,
        )
        raise ValueError("xyz, normals_xyz, U_xyz, and B_xyz must match shape (..., 3)")
    if xyz.shape[-1] != 3:
        log.error("surface_torque_density_terms failed: vector inputs last dim=%d", xyz.shape[-1])
        raise ValueError("vector inputs must have shape (..., 3)")

    scalar_shape = xyz.shape[:-1]
    if area.shape != scalar_shape:
        area = np.broadcast_to(area, scalar_shape)
    if rho.shape != scalar_shape:
        rho = np.broadcast_to(rho, scalar_shape)

    if pressure is None:
        p = np.zeros(scalar_shape, dtype=float)
    else:
        p = np.array(pressure)
        if p.shape != scalar_shape:
            p = np.broadcast_to(p, scalar_shape)

    omega = float(angvel)
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

    out = {
        "area [m^2]": area,
        "B_dot_n [T]": bdotn,
        "V_dot_n [m/s]": vdotn,
        "T1_magnetic [N/m]": t1,
        "T2_pressure [N/m]": t2,
        "T3_corotation [N/m]": t3,
        "T4_dynamic [N/m]": t4,
        "total [N/m]": total,
    }
    non_finite = int(np.count_nonzero(~np.isfinite(total)))
    if non_finite > 0:
        log.warning("surface_torque_density_terms total has %d/%d non-finite values", non_finite, total.size)
    log.info("surface_torque_density_terms done")
    return out


def integrate_surface_torque_terms(terms):
    """
    Integrate per-area torque-density terms from `surface_torque_density_terms(...)`.
    Used by: `test/test_surface_torque_analysis.py`, `starwinds_analysis/physics/orbit_surface.py`,
      `starwinds_analysis/physics/torque.py`
    """
    log.info("integrate_surface_torque_terms start")
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
    log.info("integrate_surface_torque_terms done")
    return out


def surface_torque_terms_on_shell_samples(
    shells,
    *,
    rho,
    U_xyz,
    B_xyz,
    pressure=None,
    angvel: float = 0.0,
    body_radius: float = 1.0,
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
    xyz = xyz_r * float(body_radius)
    normals = radial_surface_normals(xyz) if normals_xyz is None else normals_xyz
    out = surface_torque_density_terms(
        xyz=xyz,
        normals_xyz=normals,
        area=np.array(shells(area_field)),
        rho=rho,
        U_xyz=U_xyz,
        B_xyz=B_xyz,
        pressure=pressure,
        angvel=angvel,
        use_rotating_frame=use_rotating_frame,
    )
    log.info("surface_torque_terms_on_shell_samples done")
    return out
    log.info("surface_torque_density_terms start")
    log.info("surface_torque_terms_on_shell_samples start")
