"""THIS FILE contains orbit-surface (surface-of-revolution) sampling and diagnostics.

It builds explicit surfaces from orbital paths and evaluates pressure/torque components on them.
It should reuse pressure/torque core functions rather than redefining those quantities.
"""

# TODO(debt): This file combines geometry generation, resampling, pressure/torque
# quantity assembly, and summaries. It behaves like a workflow/pipeline but currently
# lives in `physics`.
# TODO(debt): This remains a workflow-heavy module in `physics`; keep moving shared
# orbit/surface sampling pieces into neutral primitives.

from __future__ import annotations

import math

import numpy as np

from starwinds_analysis.analysis.stats import summarize_samples
from starwinds_analysis.physics.orbits import orbital_period
from starwinds_analysis.sampling.orbits import (
    circular_orbit_points,
    elliptic_orbit_points,
    sample_points,
)
from starwinds_analysis.physics.pressure import (
    magnetospheric_standoff_distance,
    pressure_components,
)
from starwinds_analysis.physics.torque import (
    integrate_surface_torque_terms,
    surface_torque_density_terms,
)
from starwinds_analysis.analysis.shells import (
    infer_body_radius_m,
)
from starwinds_analysis.analysis.stats import weighted_quantile

def _pressure_field_name_and_scale(smart_ds):
    if smart_ds.has_field("P [Pa]"):
        return "P [Pa]", 1.0
    if smart_ds.has_field("P [dyne/cm^2]"):
        return "P [dyne/cm^2]", 0.1
    raise KeyError("Could not find pressure field in SI or cgs form")

def surface_of_revolution_from_path(points, *, n_longitudes: int = 199):
    """
    Surface of revolution around the z-axis from a sampled orbit path.

    The path is represented only by cylindrical radius and z-coordinate, matching the
    old `elliptic_orbit.surface_from_orbit(...)` behavior.
    """
    pts = np.array(points)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    n = pts.shape[0]
    if n < 2:
        raise ValueError("at least 2 path points are required")
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
        "cyl_radius [surface]": rxy[:, None] * np.ones((1, n_longitudes), dtype=float),
        "radius [surface]": np.sqrt(np.sum(surface * surface, axis=-1)),
    }

def _periodic_orbit_velocity(points_r, phase_turns, period_s, body_radius_m):
    points = np.array(points_r) * float(body_radius_m)
    phase = np.array(phase_turns)
    n = points.shape[0]
    if n < 3:
        return np.full_like(points, np.nan, dtype=float)
    t = phase * float(period_s)
    p_prev = np.roll(points, 1, axis=0)
    p_next = np.roll(points, -1, axis=0)
    t_prev = np.roll(t, 1)
    t_next = np.roll(t, -1)
    dt_prev = t - t_prev
    dt_next = t_next - t
    dt_prev[0] += float(period_s)
    dt_next[-1] += float(period_s)
    denom = dt_prev + dt_next
    return np.divide(
        p_next - p_prev,
        denom[:, None],
        out=np.full_like(points, np.nan, dtype=float),
        where=denom[:, None] != 0,
    )

def _make_surface_sample_weights(n_phase, n_longitudes, *, time_weight=None):
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

def _phase_quantiles(values_2d, q=(0.0, 0.25, 0.5, 0.75, 1.0)):
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

def surface_point_normals_and_areas(surface_points_xyz_m):
    """
    Estimate point normals and point-associated areas on a periodic structured surface.

    Uses centered differences in both directions with periodic wrapping. The resulting
    area weights are suitable for integrating sampled fields over the orbit surface.
    """
    pts = np.array(surface_points_xyz_m)
    if pts.ndim != 3 or pts.shape[-1] != 3:
        raise ValueError("surface_points_xyz_m must have shape (n_phase, n_lon, 3)")
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

def _phase_line_integrals(values_2d, area_2d):
    """
    Integrate sampled surface density values over longitude for each orbit phase.
    """
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

def sample_orbit_surface_revolution(
    smart_ds,
    *,
    fields,
    orbit,
    coordinate_fields=("X [R]", "Y [R]", "Z [R]"),
    method: str = "nearest",
    fill_value: float = np.nan,
    zone: str = "orbit-surface",
    n_longitudes: int = 199,
):
    """
    Sample explicit fields on a surface of revolution generated from an orbit path.

    `orbit` can be:
    - a float (circular radius in `R`)
    - a dict orbit spec with `semi_major_axis`, `eccentricity`, etc. (Kepler ellipse)
    """
    if isinstance(orbit, dict):
        spec = dict(orbit)
        kind = str(spec.pop("kind", "kepler")).lower()
        if kind not in {"kepler", "elliptic", "ellipse"}:
            raise ValueError(f"Unsupported orbit kind: {kind}")
        a = float(spec.pop("semi_major_axis", spec.pop("a", np.nan)))
        if not np.isfinite(a):
            raise ValueError("orbit spec requires 'semi_major_axis' (or 'a')")
        e = float(spec.pop("eccentricity", 0.0))
        plane = str(spec.pop("plane", "xy"))
        n_points = int(spec.pop("n_points", 360))
        angle0 = float(spec.pop("angle0", 0.0))
        phase0 = float(spec.pop("phase0", 0.0))
        sample = str(spec.pop("sample", "eccentric_anomaly"))
        center = spec.pop("center", (0.0, 0.0, 0.0))
        spec.pop("label", None)
        if spec:
            raise ValueError(f"Unknown orbit spec keys: {sorted(spec)}")
        orbit_info = elliptic_orbit_points(
            a,
            eccentricity=e,
            n_points=n_points,
            plane=plane,
            angle0=angle0,
            phase0=phase0,
            center=center,
            sample=sample,
            return_info=True,
        )
        path_points = orbit_info["points"]
        path_meta = {
            "kind": "elliptic",
            "plane": plane,
            "phase [turns]": orbit_info["phase [turns]"],
            "time_weight [none]": orbit_info["time_weight [none]"],
            "true_anomaly [rad]": orbit_info["true_anomaly [rad]"],
            "eccentricity [none]": float(e),
            "semi_major_axis [R]": float(a),
            "sample_parameter": sample,
        }
    else:
        r = float(orbit)
        path_points = circular_orbit_points(r, n_points=360, plane="xy")
        n = path_points.shape[0]
        path_meta = {
            "kind": "circular",
            "plane": "xy",
            "phase [turns]": np.arange(n, dtype=float) / n,
            "time_weight [none]": np.full(n, 1.0 / n, dtype=float),
            "radius [R]": float(r),
        }

    surf = surface_of_revolution_from_path(path_points, n_longitudes=n_longitudes)
    pts = surf["points"].reshape(-1, 3)

    sampled_flat = sample_points(
        smart_ds,
        pts,
        fields=fields,
        coordinate_fields=coordinate_fields,
        method=method,
        fill_value=fill_value,
    )
    n_phase, n_lon = surf["points"].shape[:2]
    sampled = {
        "surface_points": surf["points"],
        "X [surface]": surf["points"][..., 0],
        "Y [surface]": surf["points"][..., 1],
        "Z [surface]": surf["points"][..., 2],
        "R [surface]": surf["radius [surface]"],
        "C [surface]": surf["cyl_radius [surface]"],
        "azimuth [rad]": surf["azimuth [rad]"],
        "phase [turns]": np.array(path_meta["phase [turns]"]),
        "time_weight [none]": np.array(path_meta["time_weight [none]"]),
        "orbit_meta": path_meta,
        "zone": zone,
    }
    for key, val in sampled_flat.items():
        if key in {"X [sample]", "Y [sample]", "Z [sample]", "R [sample]"}:
            continue
        arr = np.array(val)
        if arr.shape == (n_phase * n_lon,):
            sampled[key] = arr.reshape(n_phase, n_lon)
    return sampled

def pressure_components_on_orbit_surface(
    smart_ds,
    orbit,
    *,
    body_radius_m: float | None = None,
    n_longitudes: int = 199,
    method: str = "nearest",
    star_mass_kg: float | None = None,
    include_relative_ram: bool = True,
    standoff_b0_t: float = 0.7e-4,
    quantiles=(0.0, 0.25, 0.5, 0.75, 1.0),
):
    """
    Pressure-component analytics on a surface of revolution around an orbit path.
    """
    body_radius_m = infer_body_radius_m(smart_ds, body_radius_m=body_radius_m)
    smart_ds.add_batsrus_graph(body_radius_m=body_radius_m)
    rho_name = "Rho [kg/m^3]"
    ux_name, uy_name, uz_name = "U_x [m/s]", "U_y [m/s]", "U_z [m/s]"
    bx_name, by_name, bz_name = "B_x [T]", "B_y [T]", "B_z [T]"
    p_name, p_scale = _pressure_field_name_and_scale(smart_ds)

    sampled = sample_orbit_surface_revolution(
        smart_ds,
        fields=(rho_name, ux_name, uy_name, uz_name, bx_name, by_name, bz_name, p_name),
        orbit=orbit,
        method=method,
        n_longitudes=n_longitudes,
    )

    rho = np.array(sampled[rho_name])
    u_xyz = np.stack(
        [sampled[ux_name], sampled[uy_name], sampled[uz_name]], axis=-1
    )
    b_xyz = np.stack(
        [sampled[bx_name], sampled[by_name], sampled[bz_name]], axis=-1
    )
    p_therm = p_scale * np.array(sampled[p_name])

    object_velocity = None
    orbit_meta = sampled["orbit_meta"]
    a_r = orbit_meta.get("semi_major_axis [R]")
    if (
        include_relative_ram
        and star_mass_kg is not None
        and a_r is not None
        and np.isfinite(a_r)
    ):
        path_points = np.column_stack(
            [sampled["X [surface]"][:, 0], sampled["Y [surface]"][:, 0], sampled["Z [surface]"][:, 0]]
        )
        phase = np.array(sampled["phase [turns]"])
        if phase.shape == (path_points.shape[0],):
            period_s = orbital_period(float(a_r) * body_radius_m, star_mass_kg)
            v_path = _periodic_orbit_velocity(path_points, phase, period_s, body_radius_m)
            object_velocity = np.repeat(v_path[:, None, :], sampled["X [surface]"].shape[1], axis=1)

    comps = pressure_components(
        rho.reshape(-1),
        u_xyz.reshape(-1, 3),
        b_xyz.reshape(-1, 3),
        thermal_pressure_pa=p_therm.reshape(-1),
        object_velocity_xyz_m_s=None if object_velocity is None else object_velocity.reshape(-1, 3),
    )
    comps = {k: np.array(v).reshape(rho.shape) for k, v in comps.items()}
    speed_for_standoff = comps.get("relative_speed [m/s]", comps["U [m/s]"])
    comps["standoff_distance [m]"] = magnetospheric_standoff_distance(
        rho,
        speed_for_standoff,
        b0_t=standoff_b0_t,
    )

    weights = _make_surface_sample_weights(
        rho.shape[0], rho.shape[1], time_weight=sampled["time_weight [none]"]
    )
    summaries = {}
    phase_profiles = {}
    q = np.array(quantiles)
    for key, arr in comps.items():
        flat = np.array(arr).reshape(-1)
        summaries[key] = summarize_samples(flat, weights=weights.reshape(-1))
        qq, qarr = _phase_quantiles(arr, q=q)
        phase_profiles[key] = {
            "phase [turns]": np.array(sampled["phase [turns]"]),
            "quantiles [none]": qq,
            "values": qarr,
        }

    out = {
        "orbit_surface": sampled,
        "rho [kg/m^3]": rho,
        **comps,
        "summary": summaries,
        "phase_quantiles": phase_profiles,
    }
    if "semi_major_axis [R]" in orbit_meta:
        out["semi_major_axis [R]"] = float(orbit_meta["semi_major_axis [R]"])
        out["eccentricity [none]"] = float(orbit_meta.get("eccentricity [none]", np.nan))
    elif "radius [R]" in orbit_meta:
        out["radius [R]"] = float(orbit_meta["radius [R]"])
    return out

def torque_components_on_orbit_surface(
    smart_ds,
    orbit,
    *,
    body_radius_m: float | None = None,
    n_longitudes: int = 199,
    method: str = "nearest",
    include_pressure_term: bool = True,
    angvel_rad_s: float = 0.0,
    quantiles=(0.0, 0.25, 0.5, 0.75, 1.0),
):
    """
    Explicit-surface torque diagnostics on an orbit surface of revolution (non-VTK).
    """
    body_radius_m = infer_body_radius_m(smart_ds, body_radius_m=body_radius_m)
    smart_ds.add_batsrus_graph(body_radius_m=body_radius_m)
    rho_name = "Rho [kg/m^3]"
    ux_name, uy_name, uz_name = "U_x [m/s]", "U_y [m/s]", "U_z [m/s]"
    bx_name, by_name, bz_name = "B_x [T]", "B_y [T]", "B_z [T]"

    fields = [rho_name, ux_name, uy_name, uz_name, bx_name, by_name, bz_name]
    p_name = p_scale = None
    if include_pressure_term:
        try:
            p_name, p_scale = _pressure_field_name_and_scale(smart_ds)
            fields.append(p_name)
        except Exception:
            p_name = p_scale = None

    sampled = sample_orbit_surface_revolution(
        smart_ds,
        fields=tuple(fields),
        orbit=orbit,
        method=method,
        n_longitudes=n_longitudes,
    )

    rho = np.array(sampled[rho_name])
    u_xyz = np.stack(
        [sampled[ux_name], sampled[uy_name], sampled[uz_name]], axis=-1
    )
    b_xyz = np.stack(
        [sampled[bx_name], sampled[by_name], sampled[bz_name]], axis=-1
    )
    p = None if p_name is None else p_scale * np.array(sampled[p_name])

    points_m = np.array(sampled["surface_points"]) * float(body_radius_m)
    normals, area = surface_point_normals_and_areas(points_m)

    terms = surface_torque_density_terms(
        xyz_m=points_m,
        normals_xyz=normals,
        area_m2=area,
        rho_kg_m3=rho,
        u_xyz_m_s=u_xyz,
        b_xyz_t=b_xyz,
        pressure_pa=p,
        angvel_rad_s=angvel_rad_s,
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
        qq, qarr = _phase_quantiles(arr, q=q)
        integ, cov = _phase_line_integrals(arr, area)
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
        "orbit_surface": sampled,
        "surface_points [m]": points_m,
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
    orbit_meta = sampled["orbit_meta"]
    if "semi_major_axis [R]" in orbit_meta:
        out["semi_major_axis [R]"] = float(orbit_meta["semi_major_axis [R]"])
        out["eccentricity [none]"] = float(orbit_meta.get("eccentricity [none]", np.nan))
    if "radius [R]" in orbit_meta:
        out["radius [R]"] = float(orbit_meta["radius [R]"])
    return out
