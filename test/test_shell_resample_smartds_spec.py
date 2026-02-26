import importlib.util
from pathlib import Path

import numpy as np
import pytest

from starwinds_analysis.analysis.shells import sample_spherical_shells
from starwinds_analysis.smart_ds import SmartDs


EXAMPLE_PLT = Path("sample_data/3d__var_1_n00060000.plt")


@pytest.mark.skipif(
    importlib.util.find_spec("scipy") is None,
    reason="scipy is required for spherical shell resampling",
)
@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_sample_spherical_shells_returns_structured_smartds_with_free_and_bound_coordinates():
    """
    Spec test for the shell-resampling workflow.

    Desired behavior:
    - resampling returns a NEW SmartDs (not a per-function container)
    - the new dataset is structured (arrays are not flattened point lists)
    - free coordinates (`R`, `theta`, `phi`) are populated
    - bound coordinates (`X`, `Y`, `Z`) are populated
    - requested source fields are sampled into the new dataset
    """
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    source_field = next(
        name for name in sds.variables if name not in {"X [R]", "Y [R]", "Z [R]"}
    )

    radii = [2.0, 4.0]
    n_polar = 8
    n_azimuth = 16

    shell_ds = sample_spherical_shells(
        sds,
        radii,
        fields=(source_field,),
        n_polar=n_polar,
        n_azimuth=n_azimuth,
        method="nearest",
    )

    assert isinstance(shell_ds, SmartDs)
    assert shell_ds is not sds

    expected_shape = (len(radii), n_polar, n_azimuth)

    x = np.array(shell_ds.variable("X [R]"))
    y = np.array(shell_ds.variable("Y [R]"))
    z = np.array(shell_ds.variable("Z [R]"))
    r = np.array(shell_ds.variable("R [R]"))
    theta = np.array(shell_ds.variable("theta [rad]"))
    phi = np.array(shell_ds.variable("phi [rad]"))
    sampled = np.array(shell_ds.variable(source_field))

    assert x.shape == expected_shape
    assert y.shape == expected_shape
    assert z.shape == expected_shape
    assert r.shape == expected_shape
    assert theta.shape == expected_shape
    assert phi.shape == expected_shape
    assert sampled.shape == expected_shape

    r_direct = np.sqrt(x * x + y * y + z * z)
    np.testing.assert_allclose(r, r_direct, rtol=0, atol=1e-10)

    theta_direct = np.arccos(np.clip(z / r_direct, -1.0, 1.0))
    np.testing.assert_allclose(theta, theta_direct, rtol=0, atol=1e-10)

    phi_direct = np.arctan2(y, x)
    dphi = (phi - phi_direct + np.pi) % (2.0 * np.pi) - np.pi
    np.testing.assert_allclose(dphi, 0.0, rtol=0, atol=1e-10)

    unique_r = np.unique(np.round(r.reshape(len(radii), -1).mean(axis=1), decimals=12))
    np.testing.assert_allclose(unique_r, radii, rtol=0, atol=1e-10)

    # If fields are not specified, the shell SmartDs should contain all parent raw fields
    # (plus free spherical coordinates added by the shell sampler).
    shell_ds_all = sample_spherical_shells(
        sds,
        radii,
        n_polar=4,
        n_azimuth=8,
        method="nearest",
    )
    for name in sds.variables:
        assert name in shell_ds_all.variables
    for name in ("R [R]", "theta [rad]", "phi [rad]"):
        assert name in shell_ds_all.variables
