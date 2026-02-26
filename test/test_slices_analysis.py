from pathlib import Path

import numpy as np
import pytest

from starwinds_analysis.analysis.slices import (
    infer_range,
    resample_structured_xz_slice,
    structured_quad_corners,
)
from starwinds_analysis.smart_ds import SmartDs


EXAMPLE_PLT = Path("sample_data/3d__var_1_n00060000.plt")


def test_structured_quad_corners_shape_and_values():
    c = structured_quad_corners(nx=3, nz=2)
    np.testing.assert_array_equal(c, np.array([[0, 1, 4, 3], [1, 2, 5, 4]]))


def test_infer_range_symmetric_padding():
    lo, hi = infer_range([-1, 2], symmetric=True, padding_frac=0.1)
    assert lo < -2 and hi > 2
    assert np.isclose(abs(lo), abs(hi))


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_resample_structured_xz_slice_on_example():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    out = resample_structured_xz_slice(
        sds,
        nx=16,
        nz=12,
        fields=("Rho [g/cm^3]", "B_x [Gauss]"),
        method="nearest",
        symmetric_ranges=True,
    )

    assert out.raw.corners.shape[1] == 4
    assert out.raw.corners.shape[0] == (16 - 1) * (12 - 1)
    np.testing.assert_allclose(out.variable("Y [R]"), 0.0)
    assert out.variable("Rho [g/cm^3]").shape == (16 * 12,)

