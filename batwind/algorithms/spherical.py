"""Pure spherical coordinate and vector transforms.
"""

# It provides low-level Cartesian/spherical conversions only.
# It should stay independent of SmartDs, griblet recipe wiring, and plotting.


import logging

import numpy as np

log = logging.getLogger(__name__)


def cartesian_to_spherical_coordinates(x, y, z):
    """
    Convert Cartesian coordinates to spherical coordinates.
    Used by: `batwind/recipes/spherical.py`
    """
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    if x.shape != y.shape or x.shape != z.shape:
        log.error("cartesian_to_spherical_coordinates failed: x=%s y=%s z=%s", x.shape, y.shape, z.shape)
        raise ValueError("x, y, z must have matching shapes")

    r = np.sqrt(x * x + y * y + z * z)
    rho_xy = np.sqrt(x * x + y * y)

    polar = np.full_like(r, np.nan, dtype=float)
    azimuth = np.full_like(r, np.nan, dtype=float)

    with np.errstate(invalid="ignore", divide="ignore"):
        mask_r = r > 0
        cos_polar = np.empty_like(r, dtype=float)
        cos_polar.fill(np.nan)
        cos_polar[mask_r] = np.clip(z[mask_r] / r[mask_r], -1.0, 1.0)
        polar[mask_r] = np.arccos(cos_polar[mask_r])

        # Keep azimuth undefined on the axis instead of pretending it is zero.
        mask_azimuth = rho_xy > 0
        azimuth[mask_azimuth] = np.arctan2(y[mask_azimuth], x[mask_azimuth])

    undefined_polar = int(np.count_nonzero(~np.isfinite(polar)))
    undefined_azimuth = int(np.count_nonzero(~np.isfinite(azimuth)))
    if undefined_polar > 0 or undefined_azimuth > 0:
        log.warning(
            "cartesian_to_spherical_coordinates undefined polar=%d azimuth=%d",
            undefined_polar,
            undefined_azimuth,
        )
    return r, polar, azimuth


def spherical_to_cartesian_coordinates(r, polar, azimuth):
    """
    Convert spherical coordinates into Cartesian coordinates.
    Used by: `batwind/recipes/spherical.py`
    """
    r = np.array(r)
    polar = np.array(polar)
    azimuth = np.array(azimuth)
    if r.shape != polar.shape or r.shape != azimuth.shape:
        log.error("spherical_to_cartesian_coordinates failed: r=%s polar=%s azimuth=%s", r.shape, polar.shape, azimuth.shape)
        raise ValueError("r, polar, azimuth must have matching shapes")
    sin_polar = np.sin(polar)
    x = r * sin_polar * np.cos(azimuth)
    y = r * sin_polar * np.sin(azimuth)
    z = r * np.cos(polar)
    log.debug("spherical_to_cartesian_coordinates complete shape=%s", x.shape)
    return x, y, z


def polar_azimuth_to_latitude_longitude(polar, azimuth):
    """
    Convert polar/azimuth coordinates to latitude/longitude.
    Used by: `batwind/recipes/spherical.py`
    """
    polar = np.array(polar)
    azimuth = np.array(azimuth)
    if polar.shape != azimuth.shape:
        log.error("polar_azimuth_to_latitude_longitude failed: polar=%s azimuth=%s", polar.shape, azimuth.shape)
        raise ValueError("polar and azimuth must have matching shapes")
    latitude = (0.5 * np.pi) - polar
    longitude = np.array(azimuth)
    return latitude, longitude


def latitude_longitude_to_polar_azimuth(latitude, longitude):
    """
    Convert latitude/longitude to polar/azimuth coordinates.
    Used by: `batwind/recipes/spherical.py`
    """
    latitude = np.array(latitude)
    longitude = np.array(longitude)
    if latitude.shape != longitude.shape:
        log.error("latitude_longitude_to_polar_azimuth failed: latitude=%s longitude=%s", latitude.shape, longitude.shape)
        raise ValueError("latitude and longitude must have matching shapes")
    polar = (0.5 * np.pi) - latitude
    azimuth = np.array(longitude)
    return polar, azimuth


def cartesian_vector_to_spherical_components(vx, vy, vz, x, y, z):
    """
    Return ``(v_r, v_p, v_a)`` using `polar` and `azimuth` coordinates.
    Used by: `test/test_shell_analysis.py`, `batwind/recipes/spherical.py`
    """
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    vx = np.array(vx)
    vy = np.array(vy)
    vz = np.array(vz)
    if not (x.shape == y.shape == z.shape == vx.shape == vy.shape == vz.shape):
        log.error(
            "cartesian_vector_to_spherical_components failed: shapes x=%s y=%s z=%s vx=%s vy=%s vz=%s",
            x.shape,
            y.shape,
            z.shape,
            vx.shape,
            vy.shape,
            vz.shape,
        )
        raise ValueError("x, y, z, vx, vy, vz must have matching shapes")

    r = np.sqrt(x * x + y * y + z * z)
    rho_xy = np.sqrt(x * x + y * y)

    v_r = np.full_like(r, np.nan, dtype=float)
    v_p = np.full_like(r, np.nan, dtype=float)
    v_a = np.full_like(r, np.nan, dtype=float)

    with np.errstate(invalid="ignore", divide="ignore"):
        mask_r = r > 0
        v_r[mask_r] = (vx[mask_r] * x[mask_r] + vy[mask_r] * y[mask_r] + vz[mask_r] * z[mask_r]) / r[mask_r]

        # Keep polar/azimuth components undefined at the axis, where the basis is singular.
        mask_axis = mask_r & (rho_xy > 0)
        if np.any(mask_axis):
            rho = rho_xy[mask_axis]
            rr = r[mask_axis]
            xx = x[mask_axis]
            yy = y[mask_axis]
            zz = z[mask_axis]
            vxx = vx[mask_axis]
            vyy = vy[mask_axis]
            vzz = vz[mask_axis]

            v_p[mask_axis] = (zz * (xx * vxx + yy * vyy) - (xx * xx + yy * yy) * vzz) / (rr * rho)
            v_a[mask_axis] = (-yy * vxx + xx * vyy) / rho

    undefined_p = int(np.count_nonzero(~np.isfinite(v_p)))
    undefined_a = int(np.count_nonzero(~np.isfinite(v_a)))
    if undefined_p > 0 or undefined_a > 0:
        log.warning(
            "cartesian_vector_to_spherical_components undefined v_p=%d v_a=%d",
            undefined_p,
            undefined_a,
        )
    return v_r, v_p, v_a


def spherical_vector_to_cartesian_components(v_r, v_p, v_a, polar, azimuth):
    """
    Convert spherical vector components `(r, p, a)` into Cartesian components.
    Used by: `batwind/recipes/spherical.py`
    """
    v_r = np.array(v_r)
    v_p = np.array(v_p)
    v_a = np.array(v_a)
    polar = np.array(polar)
    azimuth = np.array(azimuth)
    if not (v_r.shape == v_p.shape == v_a.shape == polar.shape == azimuth.shape):
        log.error(
            "spherical_vector_to_cartesian_components failed: v_r=%s v_p=%s v_a=%s polar=%s azimuth=%s",
            v_r.shape,
            v_p.shape,
            v_a.shape,
            polar.shape,
            azimuth.shape,
        )
        raise ValueError("v_r, v_p, v_a, polar, azimuth must have matching shapes")
    sin_polar = np.sin(polar)
    cos_polar = np.cos(polar)
    sin_azimuth = np.sin(azimuth)
    cos_azimuth = np.cos(azimuth)
    vx = v_r * sin_polar * cos_azimuth + v_p * cos_polar * cos_azimuth - v_a * sin_azimuth
    vy = v_r * sin_polar * sin_azimuth + v_p * cos_polar * sin_azimuth + v_a * cos_azimuth
    vz = v_r * cos_polar - v_p * sin_polar
    log.debug("spherical_vector_to_cartesian_components complete shape=%s", vx.shape)
    return vx, vy, vz
