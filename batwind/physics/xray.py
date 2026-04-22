from __future__ import annotations

from pathlib import Path

import numpy as np
from batcamp import Octree
from batcamp import OctreeInterpolator

from batwind.smart_ds import SmartDs

DEFAULT_RESPONSE_FUNCTION_PATH = Path("/Users/dagfev/Documents/starwinds/g_lambda_T/TestResposne.dat")
BAND_COMPONENTS = {
    "hard": ("Hard_line", "Hard_cont"),
    "rosat": ("ROSAT_line", "ROSAT_cont"),
    "euv": ("EUV_line", "EUV_cont"),
}
# The legacy response table stores contribution-function values in
# 10^-26 erg cm^3 s^-1 sr^-1. Convert that once to SI:
# 10^-26 * (1e-7 J / erg) * (1e-6 m^3 / cm^3) = 1e-39 W m^3 sr^-1.
LEGACY_RESPONSE_SCALE_TO_SI = 1.0e-39
# After multiplying by ``Ne^2`` in ``cm^-6``, the legacy emissivity units become
# 10^-26 erg cm^-3 s^-1 sr^-1, which convert to SI as
# 10^-26 * (1e-7 W s / erg) * (1e6 cm^3 / m^3) = 1e-27 W m^-3 sr^-1.
LEGACY_EMISSIVITY_SCALE_TO_SI = 1.0e-27


def load_response_table(
    response_path: Path = DEFAULT_RESPONSE_FUNCTION_PATH,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Load the legacy response-function table.

    Returns:
    - ``log10_temperature`` on the table grid
    - one mapping of component name to table values in legacy units
    """
    with response_path.open("r", encoding="utf-8") as stream:
        next(stream)
        next(stream)
        shape = tuple(int(value) for value in next(stream).split())[::-1]
        names = next(stream).split()
        data = np.loadtxt(stream)
    if tuple(names[:2]) != ("l10T", "l10ne"):
        raise ValueError(f"Unexpected response-table columns in {response_path!s}: {names[:2]!r}")
    if shape[0] != 1:
        raise ValueError(f"Expected a single-density response table in {response_path!s}, got shape {shape!r}")
    log10_temperature = np.asarray(data[:, 0].reshape(shape)[0], dtype=float)
    components = {
        name: np.asarray(data[:, col_id].reshape(shape)[0], dtype=float)
        for col_id, name in enumerate(names[2:], start=2)
    }
    return log10_temperature, components


def band_response_values_si(
    response_components: dict[str, np.ndarray],
    band_name: str,
) -> np.ndarray:
    """
    Return one band-integrated contribution function on the table grid in SI.

    The returned units are ``W m^3 sr^-1``.
    """
    try:
        component_names = BAND_COMPONENTS[band_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported band_name {band_name!r}") from exc
    response_values_legacy = np.zeros_like(np.asarray(next(iter(response_components.values()))), dtype=float)
    for component_name in component_names:
        response_values_legacy = response_values_legacy + np.asarray(response_components[component_name], dtype=float)
    return LEGACY_RESPONSE_SCALE_TO_SI * response_values_legacy


def band_response_values_legacy(
    response_components: dict[str, np.ndarray],
    band_name: str,
) -> np.ndarray:
    """
    Return one band-integrated contribution function in the table's legacy units.
    """
    return band_response_values_si(response_components, band_name) / LEGACY_RESPONSE_SCALE_TO_SI


def interpolate_band_contribution_function_si(
    temperature_k: np.ndarray,
    response_log10_temperature: np.ndarray,
    band_response_values_si: np.ndarray,
) -> np.ndarray:
    """
    Interpolate one band-integrated contribution function onto one temperature field.

    Units:
    - input response values: ``W m^3 sr^-1``
    - output values: ``W m^3 sr^-1``
    """
    temperature_k = np.asarray(temperature_k, dtype=float)
    response_log10_temperature = np.asarray(response_log10_temperature, dtype=float)
    band_response_values_si = np.asarray(band_response_values_si, dtype=float)
    target_log10_temperature = np.log10(np.clip(temperature_k, 10 ** response_log10_temperature[0], None))
    return np.interp(
        target_log10_temperature,
        response_log10_temperature,
        band_response_values_si,
        left=band_response_values_si[0],
        right=band_response_values_si[-1],
    )


def band_emissivity_si(
    smart_ds: SmartDs,
    response_log10_temperature: np.ndarray,
    band_response_values_si: np.ndarray,
) -> np.ndarray:
    """
    Return one band emissivity field in SI units.

    Units:
    - contribution function ``G(T)``: ``W m^3 sr^-1``
    - electron density ``n_e``: ``m^-3``
    - emissivity ``epsilon = G(T) n_e^2``: ``W m^-3 sr^-1``
    """
    contribution_function_si = interpolate_band_contribution_function_si(
        np.asarray(smart_ds["te [K]"], dtype=float),
        response_log10_temperature,
        band_response_values_si,
    )
    electron_density_m3 = np.asarray(smart_ds["Ne [1/m^3]"], dtype=float)
    return contribution_function_si * electron_density_m3**2


def band_emissivity_from_response_table_si(
    smart_ds: SmartDs,
    band_name: str,
    *,
    response_path: Path = DEFAULT_RESPONSE_FUNCTION_PATH,
) -> np.ndarray:
    """
    Return one band emissivity field from the legacy response table in SI units.
    """
    response_log10_temperature, response_components = load_response_table(response_path)
    response_values_si = band_response_values_si(response_components, band_name)
    return band_emissivity_si(smart_ds, response_log10_temperature, response_values_si)


def band_emissivity_from_response_table_legacy(
    smart_ds: SmartDs,
    band_name: str,
    *,
    response_path: Path = DEFAULT_RESPONSE_FUNCTION_PATH,
) -> np.ndarray:
    """
    Return one band emissivity field in the legacy paper-table units.

    This keeps the original ``Ne [1/cm^3]`` and response-table convention used
    by the old volume quicklook path.
    """
    response_log10_temperature, response_components = load_response_table(response_path)
    response_values_legacy = band_response_values_legacy(response_components, band_name)
    temperature_k = np.asarray(smart_ds["te [K]"], dtype=float)
    electron_density_cm3 = np.asarray(smart_ds["Ne [1/cm^3]"], dtype=float)
    target_log10_temperature = np.log10(np.clip(temperature_k, 10 ** response_log10_temperature[0], None))
    band_response_legacy = np.interp(
        target_log10_temperature,
        response_log10_temperature,
        response_values_legacy,
        left=response_values_legacy[0],
        right=response_values_legacy[-1],
    )
    return np.asarray(electron_density_cm3**2 * band_response_legacy, dtype=float)


def unblocked_solid_angle(radial_distance_r: np.ndarray) -> np.ndarray:
    """
    Return the unblocked solid angle outside one opaque stellar sphere.

    Units:
    - input radius: ``R_*``
    - output solid angle: ``sr``
    """
    radial_distance_r = np.asarray(radial_distance_r, dtype=float)
    return 2.0 * np.pi * (1.0 + np.sqrt(np.clip(1.0 - radial_distance_r**-2, 0.0, None)))


def point_radius_r(smart_ds: SmartDs) -> np.ndarray:
    """
    Return point radii in stellar-radius units.

    Prefer the graph-backed ``R [R]`` field when available, otherwise compute
    the radius directly from the raw Cartesian point coordinates.
    """
    try:
        return np.asarray(smart_ds["R [R]"], dtype=float)
    except IndexError:
        x_r = np.asarray(smart_ds["X [R]"], dtype=float)
        y_r = np.asarray(smart_ds["Y [R]"], dtype=float)
        z_r = np.asarray(smart_ds["Z [R]"], dtype=float)
        return np.sqrt(x_r**2 + y_r**2 + z_r**2)


def point_unblocked_solid_angle_sr(smart_ds: SmartDs) -> np.ndarray:
    """
    Return the exterior unblocked solid angle in steradians at every dataset point.

    Prefer the graph-backed field when available so the same geometry primitive
    is reused consistently across the library.
    """
    try:
        return np.asarray(smart_ds["unblocked_solid_angle [sr]"], dtype=float)
    except IndexError:
        return unblocked_solid_angle(point_radius_r(smart_ds))


def band_luminosity_si(
    smart_ds: SmartDs,
    point_emissivity_w_m3_sr: np.ndarray,
    *,
    occultation: bool = True,
    tree: Octree | None = None,
) -> float:
    """
    Return one band luminosity in SI units.

    This implements the section-3.4 quantity
    ``L = \\int_V \\omega(r) epsilon dV``.

    Units:
    - emissivity ``epsilon``: ``W m^-3 sr^-1``
    - solid-angle factor ``omega``: ``sr``
    - volume ``dV``: ``m^3``
    - luminosity ``L``: ``W``

    Implementation note:
    - the self-occultation factor ``omega(r)`` is evaluated at every dataset
      point and folded into a point-valued luminosity-density field
    - the volume integral of that weighted point field is then delegated to
      ``batcamp`` via exact whole-cell trilinear integrals
    """
    point_emissivity_w_m3_sr = np.asarray(point_emissivity_w_m3_sr, dtype=float)
    if tree is None:
        tree = Octree.from_ds(smart_ds.raw)
    body_radius_m = float(smart_ds["RBODY [m]"])
    leaf_count = int(np.asarray(tree.corners).shape[0])
    leaf_ids = np.arange(leaf_count, dtype=int)
    if occultation:
        solid_angle_sr = point_unblocked_solid_angle_sr(smart_ds)
    else:
        solid_angle_sr = np.full_like(point_radius_r(smart_ds), 4.0 * np.pi)
    point_luminosity_density_w_m3 = point_emissivity_w_m3_sr * solid_angle_sr
    luminosity_integral_w = (
        np.asarray(OctreeInterpolator(tree, point_luminosity_density_w_m3).cell_integrals(leaf_ids), dtype=float)
        * body_radius_m**3
    )
    return float(np.sum(luminosity_integral_w))
