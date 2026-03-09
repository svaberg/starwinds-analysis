from __future__ import annotations

import os
from time import perf_counter

import numpy as np
import pytest
from starwinds_readplt.dataset import Dataset

from starwinds_analysis.data.samples import data_file
from starwinds_analysis.octree import OctreeInterpolator

pytestmark = [
    pytest.mark.design_lockin,
    pytest.mark.skipif(
        os.getenv("STARWINDS_RUN_PERF_TESTS", "0") != "1",
        reason="Set STARWINDS_RUN_PERF_TESTS=1 to run performance checks.",
    ),
]


def test_octree_build_and_query_runtime_on_mixed_level_3d_file() -> None:
    """Performance guardrail for octree interpolator build/query on representative data."""
    input_file = data_file("difflevels-3d__var_1_n00000000.dat")
    assert input_file.exists(), f"Missing sample file: {input_file}"

    ds = Dataset.from_file(str(input_file))
    rng = np.random.default_rng(0)

    t0 = perf_counter()
    interp = OctreeInterpolator(ds, "Rho [g/cm^3]", query_space="xyz")
    build_s = perf_counter() - t0

    xyz = np.column_stack(
        [
            np.asarray(ds.variable("X [R]"), dtype=float),
            np.asarray(ds.variable("Y [R]"), dtype=float),
            np.asarray(ds.variable("Z [R]"), dtype=float),
        ]
    )
    n_query = min(4000, xyz.shape[0])
    q = xyz[rng.choice(xyz.shape[0], size=n_query, replace=False)]

    t0 = perf_counter()
    vals = interp(q)
    query_s = perf_counter() - t0

    # Guardrails are intentionally loose to avoid flaky failures across machines,
    # while still catching major regressions.
    assert build_s < 60.0, f"octree build took {build_s:.3f}s (expected < 60s)"
    assert query_s < 10.0, f"octree query took {query_s:.3f}s for {n_query} points (expected < 10s)"
    assert np.isfinite(vals).any(), "expected at least some finite interpolation values"
