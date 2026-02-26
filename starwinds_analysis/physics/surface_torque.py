"""THIS FILE contains local explicit-surface torque physics primitives.

It defines pointwise torque-density terms and local frame/normal transforms for
explicit surfaces. It does not sample shells or build radius profiles.
"""

from __future__ import annotations

import numpy as np

from starwinds_analysis.physics.torque import MU0


def rotational_frame_velocity(u_xyz_m_s, xyz_m, angvel_rad_s):
    """
    Convert inertial velocity `u` to rotating-frame velocity `V = u - Omega x r`
    for rotation about +z with scalar angular speed `angvel_rad_s`.
    """
    u = np.array(u_xyz_m_s, dtype=float)
    xyz = np.array(xyz_m, dtype=float)
    if u.shape != xyz.shape or u.shape[-1] != 3:
        raise ValueError("u_xyz_m_s and xyz_m must have the same shape (..., 3)")
    omega = float(angvel_rad_s)
    v = np.array(u, copy=True)
    v[..., 0] = u[..., 0] + omega * xyz[..., 1]
    v[..., 1] = u[..., 1] - omega * xyz[..., 0]
    v[..., 2] = u[..., 2]
    return v


def normalize_surface_normals(normals_xyz):
    n = np.array(normals_xyz, dtype=float)
    if n.shape[-1] != 3:
        raise ValueError("normals_xyz must have shape (..., 3)")
    nmag = np.sqrt(np.sum(n * n, axis=-1, keepdims=True))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.divide(n, nmag, out=np.full_like(n, np.nan), where=nmag > 0)
    return out


def radial_surface_normals(xyz):
    xyz = np.array(xyz, dtype=float)
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
    """
    # TODO(griblet): The local explicit-surface torque terms (`T1..T4`, `total`) are
    # physical quantities and should eventually be requestable via SmartDs/griblet in
    # SI units, with geometry inputs supplied explicitly.
    xyz = np.array(xyz_m, dtype=float)
    n = normalize_surface_normals(normals_xyz)
    area = np.array(area_m2, dtype=float)
    rho = np.array(rho_kg_m3, dtype=float)
    u = np.array(u_xyz_m_s, dtype=float)
    b = np.array(b_xyz_t, dtype=float)

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
        p = np.array(pressure_pa, dtype=float)
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


__all__ = [
    "MU0",
    "normalize_surface_normals",
    "radial_surface_normals",
    "rotational_frame_velocity",
    "surface_torque_density_terms",
]
