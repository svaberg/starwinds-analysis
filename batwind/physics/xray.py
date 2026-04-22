from __future__ import annotations

from pathlib import Path

import numpy as np
from batcamp import Octree

from batwind.algorithms.octree_integration import integrate_leaf_mean_field_with_cell_weight
from batwind.algorithms.octree_integration import compute_octree_leaf_centers_and_volumes
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


def unblocked_solid_angle(radial_distance_r: np.ndarray) -> np.ndarray:
    """
    Return the unblocked solid angle outside one opaque stellar sphere.

    Units:
    - input radius: ``R_*``
    - output solid angle: ``sr``
    """
    radial_distance_r = np.asarray(radial_distance_r, dtype=float)
    return 2.0 * np.pi * (1.0 + np.sqrt(np.clip(1.0 - radial_distance_r**-2, 0.0, None)))


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
    """
    point_emissivity_w_m3_sr = np.asarray(point_emissivity_w_m3_sr, dtype=float)
    if tree is None:
        tree = Octree.from_ds(smart_ds.raw)
    body_radius_m = float(smart_ds["RBODY [m]"])
    leaf_centers_r, _ = compute_octree_leaf_centers_and_volumes(tree, length_scale=body_radius_m)
    radial_distance_r = np.linalg.norm(leaf_centers_r, axis=1)
    if occultation:
        solid_angle_sr = unblocked_solid_angle(radial_distance_r)
    else:
        solid_angle_sr = np.full_like(radial_distance_r, 4.0 * np.pi)
    return integrate_leaf_mean_field_with_cell_weight(
        tree,
        point_emissivity_w_m3_sr,
        solid_angle_sr,
        length_scale=body_radius_m,
    )
