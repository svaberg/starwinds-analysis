"""THIS FILE contains explicit-surface torque analysis wrappers on sampled surfaces.

It integrates sampled torque-density terms and builds shell/radius workflows.
Local explicit-surface torque physics primitives live in `starwinds_analysis.physics.surface_torque`.
"""

from __future__ import annotations

import numpy as np

from starwinds_analysis.analysis.shells import (
    infer_body_radius_m,
    integrate_shell_scalar,
    resolve_batsrus_density_si,
    resolve_batsrus_vector_xyz_si,
    sample_spherical_shells_by_strategy,
    shell_profile_radius_height,
)

from starwinds_analysis.physics.surface_torque import (
    MU0,
    normalize_surface_normals,
    radial_surface_normals,
    rotational_frame_velocity,
    surface_torque_density_terms,
)


def integrate_surface_torque_terms(terms):
    """
    Integrate per-area torque-density terms from `surface_torque_density_terms(...)`.
    """
    area = np.array(terms["area [m^2]"], dtype=float)
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
        integral, cov = integrate_shell_scalar(np.array(terms[key], dtype=float), area)
        out[key.replace("[N/m]", "[Nm]")] = np.array(integral, dtype=float)
        component_integrals.append(np.array(integral, dtype=float))
        coverages.append(np.array(cov, dtype=float))
    # Define the total as the sum of integrated terms, so bookkeeping is exact even
    # when individual terms have different finite masks.
    out["total [Nm]"] = np.sum(np.stack(component_integrals, axis=0), axis=0)
    if "total [N/m]" in terms:
        _direct_total, cov_total = integrate_shell_scalar(np.array(terms["total [N/m]"], dtype=float), area)
        out["total_direct [Nm]"] = np.array(_direct_total, dtype=float)
        coverages.append(np.array(cov_total, dtype=float))
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
        "T1_magnetic [Nm]": np.array(ints["T1_magnetic [Nm]"], dtype=float),
        "T2_pressure [Nm]": np.array(ints["T2_pressure [Nm]"], dtype=float),
        "T3_corotation [Nm]": np.array(ints["T3_corotation [Nm]"], dtype=float),
        "T4_dynamic [Nm]": np.array(ints["T4_dynamic [Nm]"], dtype=float),
        "total [Nm]": np.array(ints["total [Nm]"], dtype=float),
        "coverage [none]": np.array(ints["coverage [none]"], dtype=float),
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
