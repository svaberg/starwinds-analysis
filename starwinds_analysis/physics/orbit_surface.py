"""THIS FILE contains surface-of-revolution sampling and diagnostics.

It builds explicit surfaces from trajectory points and evaluates pressure/torque
components on them. It should reuse pressure/torque core functions rather
than redefining those quantities.
"""

# TODO(debt): This file combines geometry generation, resampling, pressure/torque
# quantity assembly, and summaries. It behaves like a workflow/pipeline but currently
# lives in `physics`.
# TODO(debt): This remains a workflow-heavy module in `physics`; keep moving shared
# orbit/surface sampling pieces into neutral primitives.

from __future__ import annotations

import logging
import math

import numpy as np

from starwinds_analysis.analysis.orbits import sample_curve
from starwinds_analysis.analysis.stats import summarize_samples
from starwinds_analysis.physics.pressure import magnetospheric_standoff_distance
from starwinds_analysis.physics.pressure import ram_pressure
from starwinds_analysis.physics.torque import integrate_surface_torque_terms
from starwinds_analysis.physics.torque import surface_torque_density_terms

log = logging.getLogger(__name__)

def surface_of_revolution_from_trajectory(points, *, n_longitudes: int = 199):
    """Surface of revolution around the z-axis from explicit trajectory points."""
    pts = np.array(points)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    n = pts.shape[0]
    if n < 2:
        raise ValueError("at least 2 trajectory points are required")
    n_longitudes = int(n_longitudes)
    if n_longitudes < 4:
        raise ValueError("n_longitudes must be >= 4")

    rxy = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    z = pts[:, 2]
    az = np.linspace(0.0, 2.0 * math.pi, n_longitudes, endpoint=False)
    c = np.cos(az)[None, :]
    s = np.sin(az)[None, :]

    x = rxy[:, None] * c
    y = rxy[:, None] * s
    z2 = z[:, None] * np.ones((1, n_longitudes), dtype=float)
    surface = np.stack([x, y, z2], axis=-1)
    return {
        "points": surface,
        "azimuth [rad]": az,
        "cyl_radius [R]": rxy[:, None] * np.ones((1, n_longitudes), dtype=float),
        "R [R]": np.sqrt(np.sum(surface * surface, axis=-1)),
    }

def surface_sample_weights(n_phase, n_longitudes, *, time_weight=None):
    """Build integration/summary weights for one sampled surface."""
    az_w = np.full(int(n_longitudes), 1.0 / float(n_longitudes), dtype=float)
    if time_weight is None:
        t_w = np.full(int(n_phase), 1.0 / float(n_phase), dtype=float)
    else:
        t_w = np.array(time_weight)
        if t_w.shape != (int(n_phase),):
            raise ValueError("time_weight shape mismatch")
        t_w = np.where(np.isfinite(t_w) & (t_w >= 0), t_w, 0.0)
        sw = float(np.sum(t_w))
        t_w = t_w / sw if sw > 0 else np.full(int(n_phase), 1.0 / float(n_phase), dtype=float)
    return np.outer(t_w, az_w)

def phase_quantile_rows(values_2d, q=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """Compute phase-binned quantiles for 2D sampled values."""
    arr = np.array(values_2d)
    if arr.ndim != 2:
        raise ValueError("values_2d must be 2D")
    q = np.array(q)
    out = np.full((arr.shape[0], q.size), np.nan, dtype=float)
    for i in range(arr.shape[0]):
        row = arr[i]
        m = np.isfinite(row)
        if np.any(m):
            out[i] = np.quantile(row[m], q)
    return q, out

def surface_point_normals_and_areas(surface_points):
    """Estimate point normals and point-associated areas on a structured surface."""
    pts = np.array(surface_points)
    if pts.ndim != 3 or pts.shape[-1] != 3:
        raise ValueError("surface_points must have shape (n_phase, n_lon, 3)")
    if pts.shape[0] < 3 or pts.shape[1] < 4:
        raise ValueError("surface grid is too small for centered-difference geometry")

    d_phase = 0.5 * (np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0))
    d_lon = 0.5 * (np.roll(pts, -1, axis=1) - np.roll(pts, 1, axis=1))
    cross = np.cross(d_phase, d_lon, axis=-1)
    area = np.sqrt(np.sum(cross * cross, axis=-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        normals = np.divide(
            cross,
            area[..., None],
            out=np.full_like(cross, np.nan),
            where=area[..., None] > 0,
        )
    return normals, area

def phase_line_integrals(values_2d, area_2d):
    """Integrate sampled surface density values over longitude for each phase."""
    v = np.array(values_2d)
    a = np.array(area_2d)
    if v.shape != a.shape:
        a = np.broadcast_to(a, v.shape)
    mask = np.isfinite(v) & np.isfinite(a)
    integ = np.sum(np.where(mask, v * a, 0.0), axis=1)
    covered = np.sum(np.where(mask, a, 0.0), axis=1)
    total = np.sum(np.where(np.isfinite(a), a, 0.0), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = np.divide(
            covered,
            total,
            out=np.full_like(integ, np.nan, dtype=float),
            where=total > 0,
        )
    return integ, cov

def sample_surface_revolution(
    smart_ds,
    *,
    fields,
    trajectory_points,
    phase=None,
    time=None,
    time_weight=None,
    velocity_xyz=None,
    trajectory_meta=None,
    coordinate_fields=("X [R]", "Y [R]", "Z [R]"),
    method: str = "nearest",
    fill_value: float = np.nan,
    zone: str = "surface",
    n_longitudes: int = 199,
):
    """Sample explicit fields on a surface of revolution generated from explicit trajectory points."""
    fields = tuple(fields)
    log.info(
        "sample_surface_revolution start: n_fields=%s, n_longitudes=%d, method=%s",
        len(fields),
        n_longitudes,
        method,
    )
    trajectory_points = np.array(trajectory_points)
    if trajectory_points.ndim != 2 or trajectory_points.shape[1] != 3:
        raise ValueError("trajectory_points must have shape (n_phase, 3)")
    n_phase = trajectory_points.shape[0]
    if n_phase < 2:
        raise ValueError("trajectory_points must contain at least 2 points")

    if phase is None:
        phase_arr = np.linspace(0.0, 1.0, n_phase, endpoint=False)
    else:
        phase_arr = np.array(phase)
        if phase_arr.shape != (n_phase,):
            raise ValueError("phase must have shape (n_phase,)")

    if time is None:
        time_arr = None
    else:
        time_arr = np.array(time)
        if time_arr.shape != (n_phase,):
            raise ValueError("time must have shape (n_phase,)")

    if time_weight is None:
        time_weight_arr = np.full(n_phase, 1.0 / float(n_phase), dtype=float)
    else:
        time_weight_arr = np.array(time_weight)
        if time_weight_arr.shape != (n_phase,):
            raise ValueError("time_weight must have shape (n_phase,)")

    if velocity_xyz is None:
        velocity = None
    else:
        velocity = np.array(velocity_xyz)
        if velocity.shape != (n_phase, 3):
            raise ValueError("velocity_xyz must have shape (n_phase, 3)")

    meta = dict(trajectory_meta or {})
    surf = surface_of_revolution_from_trajectory(trajectory_points, n_longitudes=n_longitudes)
    pts = surf["points"].reshape(-1, 3)

    sampled_curve = sample_curve(
        smart_ds,
        pts,
        fields=fields,
        coordinate_fields=coordinate_fields,
        method=method,
        fill_value=fill_value,
    )
    n_lon = surf["points"].shape[1]
    sampled = {
        "surface_points": surf["points"],
        "X [R]": surf["points"][..., 0],
        "Y [R]": surf["points"][..., 1],
        "Z [R]": surf["points"][..., 2],
        "R [R]": surf["R [R]"],
        "cyl_radius [R]": surf["cyl_radius [R]"],
        "azimuth [rad]": surf["azimuth [rad]"],
        "phase [turns]": phase_arr,
        "time_weight [none]": time_weight_arr,
        "trajectory_meta": meta,
        "zone": zone,
    }
    if time_arr is not None:
        sampled["t [s]"] = time_arr
    if velocity is not None:
        sampled["V_x [m/s]"] = np.repeat(velocity[:, 0][:, None], n_lon, axis=1)
        sampled["V_y [m/s]"] = np.repeat(velocity[:, 1][:, None], n_lon, axis=1)
        sampled["V_z [m/s]"] = np.repeat(velocity[:, 2][:, None], n_lon, axis=1)
        sampled["V_xyz [m/s]"] = np.repeat(velocity[:, None, :], n_lon, axis=1)
    for key in fields:
        arr = np.array(sampled_curve(key))
        if arr.shape == (n_phase * n_lon,):
            sampled[key] = arr.reshape(n_phase, n_lon)
    if all(name in sampled for name in ("U_x [m/s]", "U_y [m/s]", "U_z [m/s]")):
        sampled["U_xyz [m/s]"] = np.array(sampled_curve("U_xyz [m/s]")).reshape(n_phase, n_lon, 3)
    if all(name in sampled for name in ("B_x [T]", "B_y [T]", "B_z [T]")):
        sampled["B_xyz [T]"] = np.array(sampled_curve("B_xyz [T]")).reshape(n_phase, n_lon, 3)
    log.info(
        "sample_surface_revolution done: n_phase=%d, n_lon=%d",
        n_phase,
        n_lon,
    )
    return sampled


def pressure_components_on_surface(
    sampled,
    *,
    include_relative_ram: bool = True,
    standoff_b0: float = 0.7e-4,
    quantiles=(0.0, 0.25, 0.5, 0.75, 1.0),
):
    """Pressure-component analytics on an already sampled surface of revolution."""
    log.info(
        "pressure_components_on_surface start: include_relative=%s",
        include_relative_ram,
    )
    rho = np.array(sampled["Rho [kg/m^3]"])
    u_xyz = np.array(sampled["U_xyz [m/s]"])
    u = np.array(sampled["U [m/s]"])
    b = np.array(sampled["B [T]"])
    magnetic_pressure = np.array(sampled["magnetic_pressure [Pa]"])
    ram = np.array(sampled["ram_pressure [Pa]"])
    thermal_pressure = np.array(sampled["thermal_pressure [Pa]"])
    standoff = np.array(sampled["standoff_distance [m]"])

    object_velocity = None
    if include_relative_ram and "V_xyz [m/s]" in sampled:
        object_velocity = np.array(sampled["V_xyz [m/s]"])

    comps = {
        "U [m/s]": u,
        "B [T]": b,
        "magnetic_pressure [Pa]": magnetic_pressure,
        "ram_pressure [Pa]": ram,
        "thermal_pressure [Pa]": thermal_pressure,
        "standoff_distance [m]": standoff,
    }

    # TODO(griblet): Relative-speed/relative-ram and standoff quantities still use
    # local workflow logic because they depend on the trajectory velocity.
    if object_velocity is not None:
        U_minus_V = u_xyz - object_velocity
        U_minus_V_speed = np.sqrt(np.sum(U_minus_V * U_minus_V, axis=-1))
        comps["V [m/s]"] = np.sqrt(np.sum(object_velocity * object_velocity, axis=-1))
        comps["U_minus_V [m/s]"] = U_minus_V_speed
        comps["relative_ram_pressure [Pa]"] = ram_pressure(rho, U_minus_V_speed)
        comps["standoff_distance [m]"] = magnetospheric_standoff_distance(
            rho,
            U_minus_V_speed,
            b0=standoff_b0,
        )

    weights = surface_sample_weights(
        rho.shape[0], rho.shape[1], time_weight=sampled["time_weight [none]"]
    )
    summaries = {}
    phase_profiles = {}
    q = np.array(quantiles)
    for key, arr in comps.items():
        flat = np.array(arr).reshape(-1)
        summaries[key] = summarize_samples(flat, weights=weights.reshape(-1))
        qq, qarr = phase_quantile_rows(arr, q=q)
        phase_profiles[key] = {
            "phase [turns]": np.array(sampled["phase [turns]"]),
            "quantiles [none]": qq,
            "values": qarr,
        }

    out = {
        "sampled_surface": sampled,
        "rho [kg/m^3]": rho,
        **comps,
        "summary": summaries,
        "phase_quantiles": phase_profiles,
    }
    meta = sampled.get("trajectory_meta", {})
    if "semi_major_axis [R]" in meta:
        out["semi_major_axis [R]"] = float(meta["semi_major_axis [R]"])
        out["eccentricity [none]"] = float(meta.get("eccentricity [none]", np.nan))
    elif "radius [R]" in meta:
        out["radius [R]"] = float(meta["radius [R]"])
    log.info(
        "pressure_components_on_surface done: finite_ram=%d",
        np.count_nonzero(np.isfinite(out["ram_pressure [Pa]"])),
    )
    return out


def torque_components_on_surface(
    sampled,
    *,
    body_radius: float,
    include_pressure_term: bool = True,
    angvel: float = 0.0,
    quantiles=(0.0, 0.25, 0.5, 0.75, 1.0),
):
    """Explicit-surface torque diagnostics on an already sampled surface of revolution."""
    log.info(
        "torque_components_on_surface start: include_pressure=%s",
        include_pressure_term,
    )
    body_radius = float(body_radius)
    rho = np.array(sampled["Rho [kg/m^3]"])
    u_xyz = np.array(sampled["U_xyz [m/s]"])
    b_xyz = np.array(sampled["B_xyz [T]"])
    p = None
    if include_pressure_term and "thermal_pressure [Pa]" in sampled:
        p = np.array(sampled["thermal_pressure [Pa]"])

    points = np.array(sampled["surface_points"]) * body_radius
    normals, area = surface_point_normals_and_areas(points)

    terms = surface_torque_density_terms(
        xyz=points,
        normals_xyz=normals,
        area=area,
        rho=rho,
        U_xyz=u_xyz,
        B_xyz=b_xyz,
        pressure=p,
        angvel=angvel,
        use_rotating_frame=True,
    )
    totals = integrate_surface_torque_terms(terms)

    q = np.array(quantiles)
    phase_quantiles = {}
    phase_integrals = {}
    phase = np.array(sampled["phase [turns]"])
    for src_key, out_key in (
        ("T1_magnetic [N/m]", "T1_magnetic"),
        ("T2_pressure [N/m]", "T2_pressure"),
        ("T3_corotation [N/m]", "T3_corotation"),
        ("T4_dynamic [N/m]", "T4_dynamic"),
        ("total [N/m]", "total"),
    ):
        arr = np.array(terms[src_key])
        qq, qarr = phase_quantile_rows(arr, q=q)
        integ, cov = phase_line_integrals(arr, area)
        phase_quantiles[out_key] = {
            "phase [turns]": phase,
            "quantiles [none]": qq,
            "values [N/m]": qarr,
        }
        phase_integrals[out_key] = {
            "phase [turns]": phase,
            "integral [Nm]": integ,
            "coverage [none]": cov,
        }

    weights = area.reshape(-1)
    summary = {}
    for src_key, out_key in (
        ("T1_magnetic [N/m]", "T1_magnetic [N/m]"),
        ("T2_pressure [N/m]", "T2_pressure [N/m]"),
        ("T3_corotation [N/m]", "T3_corotation [N/m]"),
        ("T4_dynamic [N/m]", "T4_dynamic [N/m]"),
        ("total [N/m]", "total [N/m]"),
    ):
        summary[out_key] = summarize_samples(np.array(terms[src_key]).reshape(-1), weights=weights)

    out = {
        "sampled_surface": sampled,
        "surface_points [m]": points,
        "surface_normals [none]": normals,
        "surface_area [m^2]": area,
        "rho [kg/m^3]": rho,
        "phase_quantiles": phase_quantiles,
        "phase_integrals": phase_integrals,
        "summary": summary,
        "T1_magnetic [Nm]": np.array(totals["T1_magnetic [Nm]"]),
        "T2_pressure [Nm]": np.array(totals["T2_pressure [Nm]"]),
        "T3_corotation [Nm]": np.array(totals["T3_corotation [Nm]"]),
        "T4_dynamic [Nm]": np.array(totals["T4_dynamic [Nm]"]),
        "total [Nm]": np.array(totals["total [Nm]"]),
        "coverage [none]": np.array(totals["coverage [none]"]),
        "surface_terms": terms,
    }
    meta = sampled.get("trajectory_meta", {})
    if "semi_major_axis [R]" in meta:
        out["semi_major_axis [R]"] = float(meta["semi_major_axis [R]"])
        out["eccentricity [none]"] = float(meta.get("eccentricity [none]", np.nan))
    if "radius [R]" in meta:
        out["radius [R]"] = float(meta["radius [R]"])
    log.info(
        "torque_components_on_surface done: total=%s",
        float(out["total [Nm]"]),
    )
    return out
