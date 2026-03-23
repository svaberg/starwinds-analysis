from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import numpy as np

from batread import Dataset
from batwind.data.field_names import DEFAULT_XYZ_NAMES
from batwind.smart_ds import SmartDs

DEFAULT_RADIAL_MAP_A = 0.0021
DEFAULT_RADIAL_MAP_B = 0.03
DEFAULT_RADIAL_MAP_P = 1.95


def powrational_from_logr(logr: np.ndarray | float, a: float, b: float, p: float) -> np.ndarray | float:
    if b <= 0.0:
        raise ValueError("radial-map-b must be > 0")
    if p <= 0.0:
        raise ValueError("radial-map-p must be > 0")
    arr = np.asarray(logr, dtype=float)
    return arr - a / np.power(arr + b, p) + a / (b**p)


def xyz_to_logr_theta_phi(
    x: np.ndarray | float,
    y: np.ndarray | float,
    z: np.ndarray | float,
    r2_eps: float = 1.0e-30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    zz = np.asarray(z, dtype=float)
    r2 = xx * xx + yy * yy + zz * zz
    safe_r2 = np.maximum(r2, float(r2_eps))
    r = np.sqrt(safe_r2)
    logr = np.log(np.maximum(r, 1.0))
    theta = np.arccos(np.clip(zz / r, -1.0, 1.0))
    phi = np.arctan2(yy, xx)
    return logr, theta, phi


try:
    from scipy.interpolate import LinearNDInterpolator
except Exception as _scipy_linear_exc:
    LinearNDInterpolator = None
    SCIPY_LINEAR_IMPORT_ERROR = _scipy_linear_exc
else:
    SCIPY_LINEAR_IMPORT_ERROR = None

try:
    from scipy.interpolate import NearestNDInterpolator
except Exception as _scipy_nearest_exc:
    NearestNDInterpolator = None
    SCIPY_NEAREST_IMPORT_ERROR = _scipy_nearest_exc
else:
    SCIPY_NEAREST_IMPORT_ERROR = None


def hexes_to_tetrahedra(corners: np.ndarray) -> np.ndarray:
    """
    Split each hexahedron (8 corners) into 6 tetrahedra using a fixed diagonal.
    """
    if corners.ndim != 2 or corners.shape[1] != 8:
        raise ValueError(f"Expected corners with shape (n, 8), got {corners.shape}")
    pattern = np.array(
        [
            [0, 1, 2, 6],
            [0, 2, 3, 6],
            [0, 3, 7, 6],
            [0, 7, 4, 6],
            [0, 4, 5, 6],
            [0, 5, 1, 6],
        ],
        dtype=np.int64,
    )
    return corners[:, pattern].reshape(-1, 4)


class _PrecomputedTetCore:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        tets: np.ndarray,
        values: np.ndarray,
        tol: float = 1e-12,
    ) -> None:
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.z = np.asarray(z, dtype=float)
        self.tets = np.asarray(tets, dtype=np.int64)
        self.values = np.asarray(values, dtype=float)
        self.tol = float(tol)

        if self.x.shape != self.y.shape or self.x.shape != self.z.shape or self.x.shape != self.values.shape:
            raise ValueError("x, y, z, and values must have identical shapes")
        if self.tets.ndim != 2 or self.tets.shape[1] != 4:
            raise ValueError("tets must have shape (ntet, 4)")

        i0 = self.tets[:, 0]
        i1 = self.tets[:, 1]
        i2 = self.tets[:, 2]
        i3 = self.tets[:, 3]

        self.x0 = self.x[i0]
        self.y0 = self.y[i0]
        self.z0 = self.z[i0]
        self.x1 = self.x[i1]
        self.y1 = self.y[i1]
        self.z1 = self.z[i1]
        self.x2 = self.x[i2]
        self.y2 = self.y[i2]
        self.z2 = self.z[i2]
        self.x3 = self.x[i3]
        self.y3 = self.y[i3]
        self.z3 = self.z[i3]

        v0 = self.values[i0]
        v1 = self.values[i1]
        v2 = self.values[i2]
        v3 = self.values[i3]

        e1 = np.stack([self.x1 - self.x0, self.y1 - self.y0, self.z1 - self.z0], axis=1)
        e2 = np.stack([self.x2 - self.x0, self.y2 - self.y0, self.z2 - self.z0], axis=1)
        e3 = np.stack([self.x3 - self.x0, self.y3 - self.y0, self.z3 - self.z0], axis=1)

        m = np.stack([e1, e2, e3], axis=2)  # (ntet, 3, 3), columns are edge vectors
        det = np.einsum("ij,ij->i", e1, np.cross(e2, e3))
        self.valid = np.abs(det) > self.tol

        invm = np.zeros_like(m)
        if np.any(self.valid):
            invm[self.valid] = np.linalg.inv(m[self.valid])

        # w1,w2,w3 = A*[x,y,z] + c
        self.a1 = invm[:, 0, 0]
        self.b1 = invm[:, 0, 1]
        self.c1 = invm[:, 0, 2]
        self.a2 = invm[:, 1, 0]
        self.b2 = invm[:, 1, 1]
        self.c2 = invm[:, 1, 2]
        self.a3 = invm[:, 2, 0]
        self.b3 = invm[:, 2, 1]
        self.c3 = invm[:, 2, 2]

        self.d1 = -(self.a1 * self.x0 + self.b1 * self.y0 + self.c1 * self.z0)
        self.d2 = -(self.a2 * self.x0 + self.b2 * self.y0 + self.c2 * self.z0)
        self.d3 = -(self.a3 * self.x0 + self.b3 * self.y0 + self.c3 * self.z0)

        # value = av*x + bv*y + cv*z + dv
        dv1 = v1 - v0
        dv2 = v2 - v0
        dv3 = v3 - v0
        self.av = dv1 * self.a1 + dv2 * self.a2 + dv3 * self.a3
        self.bv = dv1 * self.b1 + dv2 * self.b2 + dv3 * self.b3
        self.cv = dv1 * self.c1 + dv2 * self.c2 + dv3 * self.c3
        self.dv = v0 + dv1 * self.d1 + dv2 * self.d2 + dv3 * self.d3

        self.tet_xmin = np.minimum(np.minimum(self.x0, self.x1), np.minimum(self.x2, self.x3))
        self.tet_xmax = np.maximum(np.maximum(self.x0, self.x1), np.maximum(self.x2, self.x3))
        self.tet_ymin = np.minimum(np.minimum(self.y0, self.y1), np.minimum(self.y2, self.y3))
        self.tet_ymax = np.maximum(np.maximum(self.y0, self.y1), np.maximum(self.y2, self.y3))
        self.tet_zmin = np.minimum(np.minimum(self.z0, self.z1), np.minimum(self.z2, self.z3))
        self.tet_zmax = np.maximum(np.maximum(self.z0, self.z1), np.maximum(self.z2, self.z3))

    def _eval_candidates(self, px: float, py: float, pz: float, candidates: np.ndarray) -> float:
        for t in candidates:
            if not self.valid[t]:
                continue
            if px < self.tet_xmin[t] - self.tol or px > self.tet_xmax[t] + self.tol:
                continue
            if py < self.tet_ymin[t] - self.tol or py > self.tet_ymax[t] + self.tol:
                continue
            if pz < self.tet_zmin[t] - self.tol or pz > self.tet_zmax[t] + self.tol:
                continue

            w1 = self.a1[t] * px + self.b1[t] * py + self.c1[t] * pz + self.d1[t]
            w2 = self.a2[t] * px + self.b2[t] * py + self.c2[t] * pz + self.d2[t]
            w3 = self.a3[t] * px + self.b3[t] * py + self.c3[t] * pz + self.d3[t]
            w0 = 1.0 - w1 - w2 - w3

            if (
                w0 >= -self.tol
                and w1 >= -self.tol
                and w2 >= -self.tol
                and w3 >= -self.tol
                and w0 <= 1.0 + self.tol
                and w1 <= 1.0 + self.tol
                and w2 <= 1.0 + self.tol
                and w3 <= 1.0 + self.tol
            ):
                return self.av[t] * px + self.bv[t] * py + self.cv[t] * pz + self.dv[t]
        return np.nan


class NaiveLinearInterpolator3D(_PrecomputedTetCore):
    """
    Naive 3D interpolator: scan all tetrahedra for each query point.
    """

    def __call__(self, xq: np.ndarray, yq: np.ndarray, zq: np.ndarray) -> np.ndarray:
        xq = np.asarray(xq, dtype=float)
        yq = np.asarray(yq, dtype=float)
        zq = np.asarray(zq, dtype=float)
        if xq.shape != yq.shape or xq.shape != zq.shape:
            raise ValueError("xq, yq, zq must have identical shapes")

        out = np.full(xq.size, np.nan, dtype=float)
        all_tets = np.arange(self.tets.shape[0], dtype=np.int64)
        fx = xq.ravel()
        fy = yq.ravel()
        fz = zq.ravel()

        for i, (px, py, pz) in enumerate(zip(fx, fy, fz)):
            out[i] = self._eval_candidates(float(px), float(py), float(pz), all_tets)
        return out.reshape(xq.shape)


class SciPyLinearNDInterpolator3D:
    """
    Wrapper around scipy.interpolate.LinearNDInterpolator for 3D points.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        tets: np.ndarray,
        values: np.ndarray,
        tol: float = 1e-12,
    ) -> None:
        if LinearNDInterpolator is None:
            raise RuntimeError(f"SciPy LinearNDInterpolator unavailable: {SCIPY_LINEAR_IMPORT_ERROR}")
        _ = tets, tol
        points = np.column_stack(
            [np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(z, dtype=float)]
        )
        self._interp = LinearNDInterpolator(points, np.asarray(values, dtype=float), fill_value=np.nan)

    def __call__(self, xq: np.ndarray, yq: np.ndarray, zq: np.ndarray) -> np.ndarray:
        xq = np.asarray(xq, dtype=float)
        yq = np.asarray(yq, dtype=float)
        zq = np.asarray(zq, dtype=float)
        if xq.shape != yq.shape or xq.shape != zq.shape:
            raise ValueError("xq, yq, zq must have identical shapes")

        pts = np.column_stack([xq.ravel(), yq.ravel(), zq.ravel()])
        out = self._interp(pts)
        return np.asarray(out, dtype=float).reshape(xq.shape)


class SciPyLinearNDInterpolator3DLogRThetaPhi:
    """
    SciPy LinearNDInterpolator in raw (log-r, theta, phi) coordinates.

    No seam or pole handling is applied; this is a direct coordinate transform
    benchmark only.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        tets: np.ndarray,
        values: np.ndarray,
        tol: float = 1e-12,
    ) -> None:
        if LinearNDInterpolator is None:
            raise RuntimeError(f"SciPy LinearNDInterpolator unavailable: {SCIPY_LINEAR_IMPORT_ERROR}")
        _ = tets, tol
        logr, theta, phi = xyz_to_logr_theta_phi(x, y, z)
        points = np.column_stack([logr, theta, phi])
        self._interp = LinearNDInterpolator(points, np.asarray(values, dtype=float), fill_value=np.nan)

    def __call__(self, xq: np.ndarray, yq: np.ndarray, zq: np.ndarray) -> np.ndarray:
        xq = np.asarray(xq, dtype=float)
        yq = np.asarray(yq, dtype=float)
        zq = np.asarray(zq, dtype=float)
        if xq.shape != yq.shape or xq.shape != zq.shape:
            raise ValueError("xq, yq, zq must have identical shapes")

        logrq, thetaq, phiq = xyz_to_logr_theta_phi(xq.ravel(), yq.ravel(), zq.ravel())
        pts = np.column_stack([logrq, thetaq, phiq])
        out = self._interp(pts)
        return np.asarray(out, dtype=float).reshape(xq.shape)


class SciPyNearestNDInterpolator3D:
    """
    Wrapper around scipy.interpolate.NearestNDInterpolator for 3D points.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        tets: np.ndarray,
        values: np.ndarray,
        tol: float = 1e-12,
    ) -> None:
        if NearestNDInterpolator is None:
            raise RuntimeError(f"SciPy NearestNDInterpolator unavailable: {SCIPY_NEAREST_IMPORT_ERROR}")
        _ = tets, tol
        points = np.column_stack(
            [np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(z, dtype=float)]
        )
        self._interp = NearestNDInterpolator(points, np.asarray(values, dtype=float))

    def __call__(self, xq: np.ndarray, yq: np.ndarray, zq: np.ndarray) -> np.ndarray:
        xq = np.asarray(xq, dtype=float)
        yq = np.asarray(yq, dtype=float)
        zq = np.asarray(zq, dtype=float)
        if xq.shape != yq.shape or xq.shape != zq.shape:
            raise ValueError("xq, yq, zq must have identical shapes")

        pts = np.column_stack([xq.ravel(), yq.ravel(), zq.ravel()])
        out = self._interp(pts)
        return np.asarray(out, dtype=float).reshape(xq.shape)


class AABBBinnedPrecomputedLinearInterpolator3D(_PrecomputedTetCore):
    """
    3D interpolator with AABB binning and precomputed tetra coefficients.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        tets: np.ndarray,
        values: np.ndarray,
        tol: float = 1e-12,
        bins: int | tuple[int, int, int] = 20,
    ) -> None:
        super().__init__(x, y, z, tets, values, tol=tol)

        if isinstance(bins, tuple):
            self.nx, self.ny, self.nz = int(bins[0]), int(bins[1]), int(bins[2])
        else:
            self.nx = int(bins)
            self.ny = int(bins)
            self.nz = int(bins)
        if self.nx <= 0 or self.ny <= 0 or self.nz <= 0:
            raise ValueError("bins must be positive")

        self.domain_xmin = float(np.min(self.x))
        self.domain_xmax = float(np.max(self.x))
        self.domain_ymin = float(np.min(self.y))
        self.domain_ymax = float(np.max(self.y))
        self.domain_zmin = float(np.min(self.z))
        self.domain_zmax = float(np.max(self.z))

        self.hx = (self.domain_xmax - self.domain_xmin) / self.nx
        self.hy = (self.domain_ymax - self.domain_ymin) / self.ny
        self.hz = (self.domain_zmax - self.domain_zmin) / self.nz
        if self.hx <= 0 or self.hy <= 0 or self.hz <= 0:
            raise ValueError("Degenerate coordinate bounds for 3D binning")

        cell_lists = [[] for _ in range(self.nx * self.ny * self.nz)]
        for t in range(self.tets.shape[0]):
            ix0 = int(np.floor((self.tet_xmin[t] - self.domain_xmin) / self.hx))
            ix1 = int(np.floor((self.tet_xmax[t] - self.domain_xmin) / self.hx))
            iy0 = int(np.floor((self.tet_ymin[t] - self.domain_ymin) / self.hy))
            iy1 = int(np.floor((self.tet_ymax[t] - self.domain_ymin) / self.hy))
            iz0 = int(np.floor((self.tet_zmin[t] - self.domain_zmin) / self.hz))
            iz1 = int(np.floor((self.tet_zmax[t] - self.domain_zmin) / self.hz))

            ix0 = max(0, min(self.nx - 1, ix0))
            ix1 = max(0, min(self.nx - 1, ix1))
            iy0 = max(0, min(self.ny - 1, iy0))
            iy1 = max(0, min(self.ny - 1, iy1))
            iz0 = max(0, min(self.nz - 1, iz0))
            iz1 = max(0, min(self.nz - 1, iz1))

            for iz in range(iz0, iz1 + 1):
                zbase = iz * self.nx * self.ny
                for iy in range(iy0, iy1 + 1):
                    ybase = zbase + iy * self.nx
                    for ix in range(ix0, ix1 + 1):
                        cell_lists[ybase + ix].append(t)
        self.cells = [np.asarray(v, dtype=np.int64) for v in cell_lists]

    def _cell_index(self, px: float, py: float, pz: float) -> int | None:
        if (
            px < self.domain_xmin
            or px > self.domain_xmax
            or py < self.domain_ymin
            or py > self.domain_ymax
            or pz < self.domain_zmin
            or pz > self.domain_zmax
        ):
            return None
        ix = int(np.floor((px - self.domain_xmin) / self.hx))
        iy = int(np.floor((py - self.domain_ymin) / self.hy))
        iz = int(np.floor((pz - self.domain_zmin) / self.hz))
        ix = max(0, min(self.nx - 1, ix))
        iy = max(0, min(self.ny - 1, iy))
        iz = max(0, min(self.nz - 1, iz))
        return iz * self.nx * self.ny + iy * self.nx + ix

    def __call__(self, xq: np.ndarray, yq: np.ndarray, zq: np.ndarray) -> np.ndarray:
        xq = np.asarray(xq, dtype=float)
        yq = np.asarray(yq, dtype=float)
        zq = np.asarray(zq, dtype=float)
        if xq.shape != yq.shape or xq.shape != zq.shape:
            raise ValueError("xq, yq, zq must have identical shapes")

        out = np.full(xq.size, np.nan, dtype=float)
        fx = xq.ravel()
        fy = yq.ravel()
        fz = zq.ravel()

        for i, (px, py, pz) in enumerate(zip(fx, fy, fz)):
            cidx = self._cell_index(float(px), float(py), float(pz))
            if cidx is None:
                continue
            out[i] = self._eval_candidates(float(px), float(py), float(pz), self.cells[cidx])
        return out.reshape(xq.shape)


class SphericalAABBBinnedPrecomputedLinearInterpolator3D(_PrecomputedTetCore):
    """
    3D interpolator with naive spherical (r, theta, phi) AABB binning.

    Important: this intentionally does NO seam handling for phi at [-pi, pi].
    It is intended for speed tests only.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        tets: np.ndarray,
        values: np.ndarray,
        tol: float = 1e-12,
        bins: int | tuple[int, int, int] = 20,
        radial_map_a: float = DEFAULT_RADIAL_MAP_A,
        radial_map_b: float = DEFAULT_RADIAL_MAP_B,
        radial_map_p: float = DEFAULT_RADIAL_MAP_P,
    ) -> None:
        super().__init__(x, y, z, tets, values, tol=tol)

        if isinstance(bins, tuple):
            self.nr, self.nt, self.np = int(bins[0]), int(bins[1]), int(bins[2])
        else:
            self.nr = int(bins)
            self.nt = int(bins)
            self.np = int(bins)
        if self.nr <= 0 or self.nt <= 0 or self.np <= 0:
            raise ValueError("bins must be positive")
        self.radial_map_a = float(radial_map_a)
        self.radial_map_b = float(radial_map_b)
        self.radial_map_p = float(radial_map_p)
        if self.radial_map_b <= 0.0:
            raise ValueError("radial_map_b must be > 0")
        if self.radial_map_p <= 0.0:
            raise ValueError("radial_map_p must be > 0")

        self.r2_eps = max(float(self.tol), 1.0e-15) ** 2

        def _spherical(xx: np.ndarray, yy: np.ndarray, zz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            r2 = xx * xx + yy * yy + zz * zz
            safe_r2 = np.maximum(r2, self.r2_eps)
            r = np.sqrt(safe_r2)
            logr = np.log(np.maximum(r, 1.0))
            fr = powrational_from_logr(logr, self.radial_map_a, self.radial_map_b, self.radial_map_p)
            inv_r = 1.0 / np.sqrt(safe_r2)
            theta = np.arccos(np.clip(zz * inv_r, -1.0, 1.0))
            phi = np.arctan2(yy, xx)
            return fr, theta, phi

        fr0, t0, p0 = _spherical(self.x0, self.y0, self.z0)
        fr1, t1, p1 = _spherical(self.x1, self.y1, self.z1)
        fr2, t2, p2 = _spherical(self.x2, self.y2, self.z2)
        fr3, t3, p3 = _spherical(self.x3, self.y3, self.z3)

        self.tet_frmin = np.minimum(np.minimum(fr0, fr1), np.minimum(fr2, fr3))
        self.tet_frmax = np.maximum(np.maximum(fr0, fr1), np.maximum(fr2, fr3))
        self.tet_tmin = np.minimum(np.minimum(t0, t1), np.minimum(t2, t3))
        self.tet_tmax = np.maximum(np.maximum(t0, t1), np.maximum(t2, t3))
        self.tet_pmin = np.minimum(np.minimum(p0, p1), np.minimum(p2, p3))
        self.tet_pmax = np.maximum(np.maximum(p0, p1), np.maximum(p2, p3))

        frall, tall, pall = _spherical(self.x, self.y, self.z)
        self.frmin = float(np.min(frall))
        self.frmax = float(np.max(frall))
        self.tmin = float(np.min(tall))
        self.tmax = float(np.max(tall))
        self.pmin = float(np.min(pall))
        self.pmax = float(np.max(pall))

        self.hfr = (self.frmax - self.frmin) / self.nr
        self.ht = (self.tmax - self.tmin) / self.nt
        self.hp = (self.pmax - self.pmin) / self.np
        if self.hfr <= 0 or self.ht <= 0 or self.hp <= 0:
            raise ValueError("Degenerate spherical bounds for binning")

        cell_lists = [[] for _ in range(self.nr * self.nt * self.np)]
        for t in range(self.tets.shape[0]):
            ir0 = int(np.floor((self.tet_frmin[t] - self.frmin) / self.hfr))
            ir1 = int(np.floor((self.tet_frmax[t] - self.frmin) / self.hfr))
            it0 = int(np.floor((self.tet_tmin[t] - self.tmin) / self.ht))
            it1 = int(np.floor((self.tet_tmax[t] - self.tmin) / self.ht))
            ip0 = int(np.floor((self.tet_pmin[t] - self.pmin) / self.hp))
            ip1 = int(np.floor((self.tet_pmax[t] - self.pmin) / self.hp))

            ir0 = max(0, min(self.nr - 1, ir0))
            ir1 = max(0, min(self.nr - 1, ir1))
            it0 = max(0, min(self.nt - 1, it0))
            it1 = max(0, min(self.nt - 1, it1))
            ip0 = max(0, min(self.np - 1, ip0))
            ip1 = max(0, min(self.np - 1, ip1))

            for ip in range(ip0, ip1 + 1):
                pbase = ip * self.nr * self.nt
                for it in range(it0, it1 + 1):
                    tbase = pbase + it * self.nr
                    for ir in range(ir0, ir1 + 1):
                        cell_lists[tbase + ir].append(t)
        self.cells = [np.asarray(v, dtype=np.int64) for v in cell_lists]

    def _cell_index(self, px: float, py: float, pz: float) -> int | None:
        r2 = float(px * px + py * py + pz * pz)
        safe_r2 = max(r2, self.r2_eps)
        rr = float(np.sqrt(safe_r2))
        logr = float(np.log(max(rr, 1.0)))
        fr = float(powrational_from_logr(logr, self.radial_map_a, self.radial_map_b, self.radial_map_p))
        tt = float(np.arccos(np.clip(pz / np.sqrt(safe_r2), -1.0, 1.0)))
        pp = float(np.arctan2(py, px))

        if fr < self.frmin or fr > self.frmax or tt < self.tmin or tt > self.tmax or pp < self.pmin or pp > self.pmax:
            return None

        ir = int(np.floor((fr - self.frmin) / self.hfr))
        it = int(np.floor((tt - self.tmin) / self.ht))
        ip = int(np.floor((pp - self.pmin) / self.hp))
        ir = max(0, min(self.nr - 1, ir))
        it = max(0, min(self.nt - 1, it))
        ip = max(0, min(self.np - 1, ip))
        return ip * self.nr * self.nt + it * self.nr + ir

    def __call__(self, xq: np.ndarray, yq: np.ndarray, zq: np.ndarray) -> np.ndarray:
        xq = np.asarray(xq, dtype=float)
        yq = np.asarray(yq, dtype=float)
        zq = np.asarray(zq, dtype=float)
        if xq.shape != yq.shape or xq.shape != zq.shape:
            raise ValueError("xq, yq, zq must have identical shapes")

        out = np.full(xq.size, np.nan, dtype=float)
        fx = xq.ravel()
        fy = yq.ravel()
        fz = zq.ravel()

        for i, (px, py, pz) in enumerate(zip(fx, fy, fz)):
            cidx = self._cell_index(float(px), float(py), float(pz))
            if cidx is None:
                continue
            out[i] = self._eval_candidates(float(px), float(py), float(pz), self.cells[cidx])
        return out.reshape(xq.shape)


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    default_file = repo_root / "sample_data" / "3d__var_4_n00000000.dat"

    parser = ArgumentParser(description="Read a plain BATSRUS 3D .dat dataset.")
    parser.add_argument(
        "data_file",
        nargs="?",
        default=str(default_file),
        help="Path to BATSRUS 3D file (default: sample_data/3d__var_4_n00000000.dat)",
    )
    parser.add_argument(
        "--grid-n",
        type=int,
        default=6,
        help="Number of samples per axis for benchmark grid (default: 6)",
    )
    parser.add_argument(
        "--bin-n",
        type=int,
        default=20,
        help="Number of AABB bins per axis for 3D binned interpolator (default: 20)",
    )
    parser.add_argument(
        "--include-heavy",
        action="store_true",
        help="Include heavy 3D models (LinearND, naive, and custom tetra AABB).",
    )
    parser.add_argument(
        "--radial-map-a",
        type=float,
        default=DEFAULT_RADIAL_MAP_A,
        help="Powrational radial-map parameter a (default: 0.0021).",
    )
    parser.add_argument(
        "--radial-map-b",
        type=float,
        default=DEFAULT_RADIAL_MAP_B,
        help="Powrational radial-map parameter b (default: 0.03).",
    )
    parser.add_argument(
        "--radial-map-p",
        type=float,
        default=DEFAULT_RADIAL_MAP_P,
        help="Powrational radial-map parameter p (default: 1.95).",
    )
    return parser.parse_args()


def bin_equality_metrics_3d(
    cart_interp: AABBBinnedPrecomputedLinearInterpolator3D,
    sph_interp: SphericalAABBBinnedPrecomputedLinearInterpolator3D,
    xq: np.ndarray,
    yq: np.ndarray,
    zq: np.ndarray,
) -> dict[str, float]:
    """
    Compare Cartesian-AABB and spherical-AABB candidate bin behavior.
    """
    fx = np.asarray(xq, dtype=float).ravel()
    fy = np.asarray(yq, dtype=float).ravel()
    fz = np.asarray(zq, dtype=float).ravel()

    n = fx.size
    cart_counts = np.zeros(n, dtype=np.int64)
    sph_counts = np.zeros(n, dtype=np.int64)
    same_count = 0
    both_valid = 0
    same_set = 0

    for i, (px, py, pz) in enumerate(zip(fx, fy, fz)):
        cidx = cart_interp._cell_index(float(px), float(py), float(pz))
        sidx = sph_interp._cell_index(float(px), float(py), float(pz))

        carr = cart_interp.cells[cidx] if cidx is not None else np.empty(0, dtype=np.int64)
        sarr = sph_interp.cells[sidx] if sidx is not None else np.empty(0, dtype=np.int64)

        cart_counts[i] = carr.size
        sph_counts[i] = sarr.size

        if carr.size == sarr.size:
            same_count += 1
        if cidx is not None and sidx is not None:
            both_valid += 1
            if carr.shape == sarr.shape and np.array_equal(carr, sarr):
                same_set += 1

    def _qs(arr: np.ndarray) -> dict[str, float]:
        q = np.quantile(arr, [0.05, 0.25, 0.50, 0.75, 0.95])
        return {
            "q05": float(q[0]),
            "q25": float(q[1]),
            "q50": float(q[2]),
            "q75": float(q[3]),
            "q95": float(q[4]),
        }

    cart_bin_occ = np.asarray([len(c) for c in cart_interp.cells], dtype=float)
    sph_bin_occ = np.asarray([len(c) for c in sph_interp.cells], dtype=float)

    mean_cart = float(np.mean(cart_counts))
    mean_sph = float(np.mean(sph_counts))
    ratio = float(mean_sph / mean_cart) if mean_cart > 0 else float("nan")

    out = {
        "n_points": float(n),
        "mean_cart": mean_cart,
        "mean_sph": mean_sph,
        "ratio_sph_over_cart": ratio,
        "equal_count_frac": float(same_count / n),
        "both_valid_frac": float(both_valid / n),
        "same_set_frac_given_both": float(same_set / both_valid) if both_valid > 0 else float("nan"),
    }
    for k, v in _qs(cart_counts).items():
        out[f"query_cart_{k}"] = v
    for k, v in _qs(sph_counts).items():
        out[f"query_sph_{k}"] = v
    for k, v in _qs(cart_bin_occ).items():
        out[f"occ_cart_{k}"] = v
    for k, v in _qs(sph_bin_occ).items():
        out[f"occ_sph_{k}"] = v
    return out


def run_smartds_octree_demo(
    data_file: Path,
    var_name: str,
    probe_x: np.ndarray,
    probe_y: np.ndarray,
    probe_z: np.ndarray,
    probe_true: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
) -> None:
    print("")
    print("=== SmartDs octree demo ===")

    sds = SmartDs.from_file(str(data_file), batsrus=False, spherical=False)

    probe_points = np.column_stack([probe_x, probe_y, probe_z])
    t0 = perf_counter()
    probe_ds = sds.resample(
        probe_points,
        coordinate_fields=DEFAULT_XYZ_NAMES,
        fields=[var_name],
        method="octree",
    )
    t_probe = perf_counter() - t0
    probe_pred = np.asarray(probe_ds[var_name], dtype=float)
    probe_max_abs_err = float(np.nanmax(np.abs(probe_pred - probe_true)))

    grid_points = np.stack([gx, gy, gz], axis=-1)
    t0 = perf_counter()
    grid_ds = sds.resample(
        grid_points,
        coordinate_fields=DEFAULT_XYZ_NAMES,
        fields=[var_name],
        method="octree",
    )
    t_grid = perf_counter() - t0
    grid_values = np.asarray(grid_ds[var_name], dtype=float)
    finite = int(np.isfinite(grid_values).sum())

    print(f"probe time [s]: {t_probe:.4f}")
    print(f"probe max |error| on mesh points: {probe_max_abs_err:.3e}")
    print(f"grid time [s]: {t_grid:.4f}")
    print(f"grid finite values: {finite}/{grid_values.size}")
    print(f"output coordinate fields: {DEFAULT_XYZ_NAMES}")
    print(f"output variables: {grid_ds.raw.variables}")


def main() -> None:
    args = parse_args()
    data_file = Path(args.data_file).expanduser().resolve()
    if not data_file.exists():
        raise FileNotFoundError(f"Missing data file: {data_file}")

    ds = Dataset.from_file(str(data_file))
    x = np.asarray(ds["X [R]"], dtype=float)
    y = np.asarray(ds["Y [R]"], dtype=float)
    z = np.asarray(ds["Z [R]"], dtype=float)
    tets = hexes_to_tetrahedra(ds.corners)
    r2 = x * x + y * y + z * z
    r2_eps = 1.0e-30
    safe_r2 = np.maximum(r2, r2_eps)
    r = np.sqrt(safe_r2)
    logr = np.log(np.maximum(r, 1.0))
    fr = powrational_from_logr(logr, args.radial_map_a, args.radial_map_b, args.radial_map_p)
    inv_r = 1.0 / np.sqrt(safe_r2)
    theta = np.arccos(np.clip(z * inv_r, -1.0, 1.0))
    phi = np.arctan2(y, x)

    var_name = "Rho [g/cm^3]" if "Rho [g/cm^3]" in ds.variables else str(ds.variables[0])
    values = np.asarray(ds[var_name], dtype=float)

    print(f"data file: {data_file}")
    print(f"title: {ds.title}")
    print(f"zone: {ds.zone}")
    print(f"points shape: {ds.points.shape}")
    print(f"X range [R]: [{float(np.min(x)):.6g}, {float(np.max(x)):.6g}]")
    print(f"Y range [R]: [{float(np.min(y)):.6g}, {float(np.max(y)):.6g}]")
    print(f"Z range [R]: [{float(np.min(z)):.6g}, {float(np.max(z)):.6g}]")
    print(f"R range [R]: [{float(np.min(r)):.6g}, {float(np.max(r)):.6g}]")
    print(f"log(R) range: [{float(np.min(logr)):.6g}, {float(np.max(logr)):.6g}]")
    print(f"mapped-r range [arb]: [{float(np.min(fr)):.6g}, {float(np.max(fr)):.6g}]")
    print(f"Theta range [rad]: [{float(np.min(theta)):.6g}, {float(np.max(theta)):.6g}]")
    print(f"Phi range [rad]: [{float(np.min(phi)):.6g}, {float(np.max(phi)):.6g}]")
    print(f"corners shape: {ds.corners.shape}")
    print(f"tetrahedra shape: {tets.shape}")
    print(f"number of variables: {len(ds.variables)}")
    print(f"interpolation variable: {var_name}")
    print(f"benchmark grid resolution: {args.grid_n} x {args.grid_n} x {args.grid_n}")
    print(f"AABB bins per axis: {args.bin_n}")
    print(
        f"spherical bins per axis: {args.bin_n}  "
        f"(mapped-r enabled, a={args.radial_map_a:g}, b={args.radial_map_b:g}, p={args.radial_map_p:g}; "
        "naive phi seam handling: OFF)"
    )
    print(f"include heavy benchmarks: {args.include_heavy}")
    if LinearNDInterpolator is None:
        print(f"SciPy LinearNDInterpolator: unavailable ({type(SCIPY_LINEAR_IMPORT_ERROR).__name__})")
    else:
        print("SciPy LinearNDInterpolator: available")
    if NearestNDInterpolator is None:
        print(f"SciPy NearestNDInterpolator: unavailable ({type(SCIPY_NEAREST_IMPORT_ERROR).__name__})")
    else:
        print("SciPy NearestNDInterpolator: available")

    probe_idx = np.linspace(0, x.size - 1, 12, dtype=int)
    probe_x = x[probe_idx]
    probe_y = y[probe_idx]
    probe_z = z[probe_idx]
    probe_true = values[probe_idx]

    gx, gy, gz = np.meshgrid(
        np.linspace(float(x.min()), float(x.max()), args.grid_n),
        np.linspace(float(y.min()), float(y.max()), args.grid_n),
        np.linspace(float(z.min()), float(z.max()), args.grid_n),
        indexing="ij",
    )

    interpolators = []
    if NearestNDInterpolator is not None:
        interpolators.append(("scipy-nearestnd-3d", SciPyNearestNDInterpolator3D))
    if args.include_heavy and LinearNDInterpolator is not None:
        interpolators.append(("scipy-linearnd-3d", SciPyLinearNDInterpolator3D))
        interpolators.append(
            ("scipy-linearnd-3d-logr-theta-phi", SciPyLinearNDInterpolator3DLogRThetaPhi)
        )
    if args.include_heavy:
        interpolators += [
            ("naive-linear-3d", NaiveLinearInterpolator3D),
            (
                "aabb-binned-precomputed-3d",
                lambda px, py, pz, pt, pv: AABBBinnedPrecomputedLinearInterpolator3D(
                    px, py, pz, pt, pv, bins=args.bin_n
                ),
            ),
            (
                "spherical-aabb-binned-precomputed-3d",
                lambda px, py, pz, pt, pv: SphericalAABBBinnedPrecomputedLinearInterpolator3D(
                    px,
                    py,
                    pz,
                    pt,
                    pv,
                    bins=args.bin_n,
                    radial_map_a=args.radial_map_a,
                    radial_map_b=args.radial_map_b,
                    radial_map_p=args.radial_map_p,
                ),
            ),
        ]

    results = []
    built_interps = {}
    for label, interp_cls in interpolators:
        print("")
        print(f"Starting benchmark: {label}", flush=True)
        t0 = perf_counter()
        interp = interp_cls(x, y, z, tets, values)
        t_build = perf_counter() - t0
        if hasattr(interp, "valid") and hasattr(interp, "tets"):
            valid_tets = int(np.count_nonzero(interp.valid))
            total_tets = int(interp.tets.shape[0])
            valid_txt = f"{valid_tets}/{total_tets}"
        else:
            valid_txt = "-"

        probe_pred = interp(probe_x, probe_y, probe_z)
        max_abs_err = np.nanmax(np.abs(probe_pred - probe_true))

        t0 = perf_counter()
        gv = interp(gx, gy, gz)
        t_grid = perf_counter() - t0
        finite = int(np.isfinite(gv).sum())

        print(f"[{label}]")
        print(f"build time [s]: {t_build:.4f}")
        print(f"valid tetrahedra: {valid_txt}")
        print(f"grid time [s]: {t_grid:.4f}")
        print(f"grid finite values: {finite}/{gv.size}")
        print(f"probe max |error| on mesh points: {max_abs_err:.3e}")
        built_interps[label] = interp

        results.append(
            {
                "label": label,
                "build_s": float(t_build),
                "grid_s": float(t_grid),
                "valid_txt": valid_txt,
                "finite": int(finite),
                "grid_total": int(gv.size),
                "probe_err": float(max_abs_err),
            }
        )

    print("")
    print("=== Scoreboard (3D, sorted by total time = build + grid) ===")
    print(
        f"{'name':34s} {'build[s]':>10s} {'grid[s]':>10s} {'total[s]':>10s} {'valid_tets':>16s} {'finite':>17s} {'probe_err':>12s}"
    )
    for row in sorted(results, key=lambda r: r["build_s"] + r["grid_s"]):
        total_s = row["build_s"] + row["grid_s"]
        finite_pct = 100.0 * row["finite"] / row["grid_total"]
        finite_txt = f"{row['finite']}/{row['grid_total']} ({finite_pct:5.1f}%)"
        print(
            f"{row['label']:34s} "
            f"{row['build_s']:10.4f} "
            f"{row['grid_s']:10.4f} "
            f"{total_s:10.4f} "
            f"{row['valid_txt']:>16s} "
            f"{finite_txt:>17s} "
            f"{row['probe_err']:12.3e}"
        )

    cart_key = "aabb-binned-precomputed-3d"
    sph_key = "spherical-aabb-binned-precomputed-3d"
    if cart_key in built_interps and sph_key in built_interps:
        m = bin_equality_metrics_3d(built_interps[cart_key], built_interps[sph_key], gx, gy, gz)
        print("")
        print("=== Bin Equality (Cartesian vs Spherical AABB) ===")
        print(f"points compared: {int(m['n_points'])}")
        print(
            f"mean candidate count: cart={m['mean_cart']:.2f}, "
            f"spherical={m['mean_sph']:.2f}, ratio={m['ratio_sph_over_cart']:.3f}"
        )
        print(f"equal candidate-count fraction: {100.0 * m['equal_count_frac']:.1f}%")
        print(f"both-bin-valid fraction: {100.0 * m['both_valid_frac']:.1f}%")
        print(
            "exact candidate-set equality (given both bins valid): "
            f"{100.0 * m['same_set_frac_given_both']:.1f}%"
        )
        print(
            "query candidate-count quantiles "
            "(q05,q25,q50,q75,q95): "
            f"cart=({m['query_cart_q05']:.1f},{m['query_cart_q25']:.1f},{m['query_cart_q50']:.1f},{m['query_cart_q75']:.1f},{m['query_cart_q95']:.1f}) "
            f"spherical=({m['query_sph_q05']:.1f},{m['query_sph_q25']:.1f},{m['query_sph_q50']:.1f},{m['query_sph_q75']:.1f},{m['query_sph_q95']:.1f})"
        )
        print(
            "bin-occupancy quantiles "
            "(q05,q25,q50,q75,q95): "
            f"cart=({m['occ_cart_q05']:.1f},{m['occ_cart_q25']:.1f},{m['occ_cart_q50']:.1f},{m['occ_cart_q75']:.1f},{m['occ_cart_q95']:.1f}) "
            f"spherical=({m['occ_sph_q05']:.1f},{m['occ_sph_q25']:.1f},{m['occ_sph_q50']:.1f},{m['occ_sph_q75']:.1f},{m['occ_sph_q95']:.1f})"
        )

    run_smartds_octree_demo(
        data_file,
        var_name,
        probe_x,
        probe_y,
        probe_z,
        probe_true,
        gx,
        gy,
        gz,
    )


if __name__ == "__main__":
    main()
