"""THIS FILE contains local magnetic-field component helpers.

It defines pointwise magnetic-field component transforms/combinations and simple unit
scales, without shell sampling or plotting orchestration.
"""

from __future__ import annotations

import numpy as np

from starwinds_analysis.recipes.spherical import spherical_vector_components


def magnetic_field_unit_scale(unit: str) -> tuple[float, str]:
    key = str(unit).strip()
    table = {
        "T": (1.0, "T"),
        "Tesla": (1.0, "T"),
        "G": (1e4, "G"),
        "Gauss": (1e4, "G"),
        "nT": (1e9, "nT"),
    }
    if key not in table:
        raise ValueError(f"Unsupported magnetic display unit '{unit}'")
    return table[key]


def magnetic_shell_components_from_cartesian(bx_t, by_t, bz_t, x, y, z):
    """
    Magnetic shell components from Cartesian magnetic field and coordinates.

    Returns a dict containing radial/colatitudinal/azimuthal components and common
    latitude-map combinations (`meridional`, `tangential`) in Tesla.
    """
    bx = np.array(bx_t, dtype=float)
    by = np.array(by_t, dtype=float)
    bz = np.array(bz_t, dtype=float)
    xx = np.array(x, dtype=float)
    yy = np.array(y, dtype=float)
    zz = np.array(z, dtype=float)
    b_r, b_theta, b_phi = spherical_vector_components(bx, by, bz, xx, yy, zz)
    b_meridional = -b_theta
    b_tangential = np.sqrt(b_phi * b_phi + b_meridional * b_meridional)
    return {
        "B_r [T]": np.array(b_r, dtype=float),
        "B_theta [T]": np.array(b_theta, dtype=float),
        "B_phi [T]": np.array(b_phi, dtype=float),
        "B_meridional [T]": np.array(b_meridional, dtype=float),
        "B_tangential [T]": np.array(b_tangential, dtype=float),
    }


__all__ = ["magnetic_field_unit_scale", "magnetic_shell_components_from_cartesian"]
