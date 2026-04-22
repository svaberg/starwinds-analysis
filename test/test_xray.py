import numpy as np

from batcamp import Octree
from batcamp import OctreeInterpolator
from batread import Dataset

from batwind.physics.xray import band_emissivity_si
from batwind.physics.xray import band_emissivity_from_response_table_si
from batwind.physics.xray import band_emissivity_from_response_table_legacy
from batwind.physics.xray import band_luminosity_si
from batwind.physics.xray import LEGACY_EMISSIVITY_SCALE_TO_SI
from batwind.physics.xray import unblocked_solid_angle
from batwind.recipes.batsrus import build_batsrus_graph
from batwind.smart_ds import SmartDs


def make_one_cell_dataset(origin_r: tuple[float, float, float], width_r: float, *, variables: list[str]) -> Dataset:
    x0, y0, z0 = origin_r
    x1 = x0 + float(width_r)
    y1 = y0 + float(width_r)
    z1 = z0 + float(width_r)
    corners_xyz = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=float,
    )
    points = np.zeros((8, len(variables)), dtype=float)
    points[:, :3] = corners_xyz
    corners = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=int)
    return Dataset(points, corners, aux={}, title="one-cell", variables=variables, zone="z0")


def test_band_emissivity_si_tracks_si_units():
    dataset = make_one_cell_dataset((1.2, 0.0, 0.0), 0.4, variables=["X [R]", "Y [R]", "Z [R]", "te [K]", "Rho [kg/m^3]"])
    dataset.points[:, 3] = 1.0e5
    dataset.points[:, 4] = 2.0 * 1.67262192595e-27
    sds = SmartDs(dataset)
    sds.merge_computation_graph(build_batsrus_graph(tuple(dataset.variables), body_radius_m=1.0))

    response_log10_temperature = np.array([4.0, 6.0], dtype=float)
    band_response_values_si = np.array([2.0, 2.0], dtype=float)
    emissivity = band_emissivity_si(sds, response_log10_temperature, band_response_values_si)

    np.testing.assert_allclose(emissivity, 8.0)


def test_band_luminosity_si_matches_off_star_single_cell_formula():
    dataset = make_one_cell_dataset((1.0, 1.0, 1.0), 1.0, variables=["X [R]", "Y [R]", "Z [R]", "R [R]"])
    dataset.points[:, 3] = np.sqrt(np.sum(dataset.points[:, :3] ** 2, axis=1))
    sds = SmartDs(dataset)
    sds.merge_computation_graph(build_batsrus_graph(tuple(dataset.variables), body_radius_m=2.0))
    point_emissivity = np.full(8, 3.0, dtype=float)

    expected_volume_m3 = (1.0 * 2.0) ** 3
    tree = Octree.from_ds(dataset)

    luminosity_unocculted = band_luminosity_si(sds, point_emissivity, occultation=False)
    luminosity_occulted = band_luminosity_si(sds, point_emissivity, occultation=True)

    np.testing.assert_allclose(luminosity_unocculted, 3.0 * 4.0 * np.pi * expected_volume_m3)
    point_radius_r = np.sqrt(
        np.asarray(sds["X [R]"], dtype=float) ** 2
        + np.asarray(sds["Y [R]"], dtype=float) ** 2
        + np.asarray(sds["Z [R]"], dtype=float) ** 2
    )
    point_luminosity_density = point_emissivity * unblocked_solid_angle(point_radius_r)
    expected_occulted = float(
        np.asarray(OctreeInterpolator(tree, point_luminosity_density).cell_integrals(np.array([0], dtype=int)), dtype=float)[0]
        * float(sds["RBODY [m]"]) ** 3
    )
    np.testing.assert_allclose(luminosity_occulted, expected_occulted)


def test_legacy_and_si_response_table_emissivities_match_by_unit_conversion():
    dataset = make_one_cell_dataset((1.2, 0.0, 0.0), 0.4, variables=["X [R]", "Y [R]", "Z [R]", "te [K]", "Rho [kg/m^3]"])
    dataset.points[:, 3] = 1.0e5
    dataset.points[:, 4] = 2.0 * 1.67262192595e-27
    sds = SmartDs(dataset)
    sds.merge_computation_graph(build_batsrus_graph(tuple(dataset.variables), body_radius_m=1.0))

    legacy = band_emissivity_from_response_table_legacy(sds, "rosat")
    si = band_emissivity_from_response_table_si(sds, "rosat")

    np.testing.assert_allclose(LEGACY_EMISSIVITY_SCALE_TO_SI * legacy, si)
