import numpy as np

from batread.dataset import Dataset

from batwind.physics.alfven_radius import alfven_radius_map
from batwind.physics.alfven_radius import projected_solid_angle_weights
from batwind.physics.alfven_radius import summarize_alfven_radius
from batwind.smart_ds import SmartDs


def make_shell_demo(*, no_cross_at_00: bool = False):
    radii_r = np.array([2.0, 4.0, 6.0], dtype=float)
    polar = np.array([np.pi / 4.0, np.pi / 2.0], dtype=float)
    azimuth = np.array([-np.pi / 2.0, 0.0, np.pi / 2.0], dtype=float)

    r = np.broadcast_to(radii_r[:, None, None], (3, 2, 3)).copy()
    p = np.broadcast_to(polar[None, :, None], (3, 2, 3)).copy()
    a = np.broadcast_to(azimuth[None, None, :], (3, 2, 3)).copy()
    m_a = r / 5.0
    if no_cross_at_00:
        m_a[:, 0, 0] = 0.6

    d_omega = np.array(
        [
            [1.0, 2.0, 1.0],
            [2.0, 1.0, 2.0],
        ],
        dtype=float,
    )
    r_m = 2.0 * r
    d_a_m2 = np.square(r_m) * d_omega[None, :, :]

    points = np.stack((r, r_m, p, a, m_a, d_a_m2), axis=-1)
    dataset = Dataset(
        points,
        np.empty((0, 0), dtype=int),
        aux={},
        title="shell-demo",
        variables=[
            "R [R]",
            "R [m]",
            "polar [rad]",
            "azimuth [rad]",
            "M_A [none]",
            "dA [m^2]",
        ],
        zone="shell-demo-zone",
    )
    return SmartDs(dataset), d_omega


def test_alfven_radius_map_first_outward_crossing():
    shell_ds, _ = make_shell_demo()
    radius_map = alfven_radius_map(shell_ds)
    np.testing.assert_allclose(radius_map, 5.0, rtol=0.0, atol=1e-12)


def test_alfven_radius_map_nan_when_no_outward_crossing():
    shell_ds, _ = make_shell_demo(no_cross_at_00=True)
    radius_map = alfven_radius_map(shell_ds)
    assert np.isnan(radius_map[0, 0])
    assert np.isfinite(radius_map[1, 1])


def test_alfven_radius_map_equal_level_is_not_crossing():
    shell_ds, _ = make_shell_demo()
    m_a = np.array(shell_ds("M_A [none]"), dtype=float).copy()
    m_a[:, 0, 1] = np.array([0.6, 1.0, 1.2], dtype=float)
    m_a_idx = shell_ds.raw.variables.index("M_A [none]")
    shell_ds.raw.points[..., m_a_idx] = m_a

    radius_map = alfven_radius_map(shell_ds)
    assert np.isnan(radius_map[0, 1])


def test_alfven_radius_summary_matches_expected_weighted_values():
    shell_ds, d_omega = make_shell_demo()
    radius_map = alfven_radius_map(shell_ds)
    weights = projected_solid_angle_weights(shell_ds)
    polar_map = np.array(shell_ds("polar [rad]"))[0]

    min_r, max_r, avg_r, avg_cyl_r, coverage = summarize_alfven_radius(
        radius_map,
        polar_map,
        weights=weights,
    )

    expected_avg = 5.0
    expected_avg_cyl = 5.0 * (
        np.sum(np.sin(polar_map) * d_omega) / np.sum(d_omega)
    )

    assert np.isclose(min_r, 5.0)
    assert np.isclose(max_r, 5.0)
    assert np.isclose(avg_r, expected_avg)
    assert np.isclose(avg_cyl_r, expected_avg_cyl)
    assert np.isclose(coverage, 1.0)
