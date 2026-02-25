import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from starwinds_analysis.analysis.native_slice import (
    native_slice_geometry,
    plot_alfven_mach_slice,
    plot_native_slice_tripcolor,
)


class DummySliceDs:
    def __init__(self):
        x1 = np.array([-1.0, 0.0, 1.0])
        y1 = np.array([-1.0, 0.0, 1.0])
        xx, yy = np.meshgrid(x1, y1, indexing="xy")
        zz = np.zeros_like(xx)

        self._vars = {
            "X [R]": xx.ravel(),
            "Y [R]": yy.ravel(),
            "Z [R]": zz.ravel(),
            "Rho [g/cm^3]": (1.0 + xx * xx + yy * yy).ravel(),
            "M_A [none]": (10.0 ** xx).ravel(),
        }
        corners = []
        nx = xx.shape[1]
        ny = xx.shape[0]
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                p0 = iy * nx + ix
                p1 = p0 + 1
                p2 = p0 + nx + 1
                p3 = p0 + nx
                corners.append([p0, p1, p2, p3])
        self.corners = np.asarray(corners, dtype=int)
        self.variables = tuple(self._vars)

    def variable(self, name):
        return self._vars[name]

    def has_field(self, name):
        return name in self._vars

    def add_batsrus_graph(self, **kwargs):  # pragma: no cover - not used here
        return self


def test_native_slice_geometry_autodetects_xy():
    ds = DummySliceDs()
    geom = native_slice_geometry(ds)
    assert geom.x_field == "X [R]"
    assert geom.y_field == "Y [R]"
    assert geom.triangulation.triangles.shape[0] == 2 * ds.corners.shape[0]


def test_plot_native_slice_tripcolor_smoke():
    ds = DummySliceDs()
    fig, ax, extra = plot_native_slice_tripcolor(ds, "Rho [g/cm^3]")
    try:
        assert ax.get_xlabel() == "X [R]"
        assert ax.get_ylabel() == "Y [R]"
        assert extra["scale"] in {"linear", "positive_log"}
    finally:
        plt.close(fig)


def test_plot_alfven_mach_slice_smoke():
    ds = DummySliceDs()
    fig, ax, extra = plot_alfven_mach_slice(ds, ensure_batsrus_graph=False, vmin=1e-2, vmax=1e2)
    try:
        assert ax.get_title() == "Alfvén Mach number"
        assert "contour_drawn" in extra
        assert extra["contour_drawn"] is True
    finally:
        plt.close(fig)
