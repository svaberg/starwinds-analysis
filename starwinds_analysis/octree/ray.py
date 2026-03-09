#!/usr/bin/env python3
"""Ray traversal and ray-based interpolation helpers for octrees."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from .base import Octree

if TYPE_CHECKING:
    from .interpolator import OctreeInterpolator

logger = logging.getLogger(__name__)

HEX_TETS_INDEX = np.array(
    (
        (0, 1, 2, 6),
        (0, 2, 3, 6),
        (0, 3, 7, 6),
        (0, 7, 4, 6),
        (0, 4, 5, 6),
        (0, 5, 1, 6),
    ),
    dtype=np.int64,
)
TET_FACES_INDEX = np.array(
    (
        (1, 2, 3),
        (0, 3, 2),
        (0, 1, 3),
        (0, 2, 1),
    ),
    dtype=np.int64,
)


def _normalize_direction(direction_xyz: np.ndarray) -> np.ndarray:
    """Normalize one Cartesian ray direction."""
    d = np.asarray(direction_xyz, dtype=float).reshape(3)
    dn = float(np.linalg.norm(d))
    if not math.isfinite(dn) or dn == 0.0:
        raise ValueError("direction_xyz must be finite and non-zero.")
    return d / dn


def _as_xyz(point_xyz: np.ndarray) -> np.ndarray:
    """Coerce one Cartesian point to shape `(3,)` float array."""
    return np.asarray(point_xyz, dtype=float).reshape(3)


@dataclass(frozen=True)
class RaySegment:
    """Ray interval `[t_enter, t_exit]` that remains inside one leaf cell."""

    cell_id: int
    t_enter: float
    t_exit: float


@dataclass(frozen=True)
class RayLinearPiece:
    """Linear function f(t) = slope*t + intercept over [t_start, t_end]."""

    t_start: float
    t_end: float
    cell_id: int
    tet_id: int
    slope: np.ndarray
    intercept: np.ndarray


class OctreeRayTracer:
    """Ray tracer operating on an already-built and bound `Octree`."""

    def __init__(self, tree: Octree) -> None:
        """Store the tree used for cell-segment tracing."""
        self.tree = tree

    def trace(
        self,
        origin_xyz: np.ndarray,
        direction_xyz: np.ndarray,
        t_start: float,
        t_end: float,
        *,
        max_steps: int = 100000,
        bisect_iters: int = 48,
        boundary_tol: float = 1e-9,
    ) -> list[RaySegment]:
        """Trace a Cartesian ray into contiguous per-cell segments.

        Algorithm:
        - Start from the containing cell at `t_start`.
        - Expand a trial step size until leaving the current cell.
        - Bisect the exit point for a tight `t_exit`.
        - Step slightly forward and re-resolve near the previous cell.
        Consumes:
        - Ray origin/direction, `t` bounds, and tracing controls.
        Returns:
        - Ordered list of `RaySegment` intervals.
        """
        o = _as_xyz(origin_xyz)
        d = _normalize_direction(direction_xyz)
        return self.trace_prepared(
            o,
            d,
            float(t_start),
            float(t_end),
            max_steps=max_steps,
            bisect_iters=bisect_iters,
            boundary_tol=boundary_tol,
        )

    def trace_prepared(
        self,
        origin_xyz: np.ndarray,
        direction_xyz_unit: np.ndarray,
        t_start: float,
        t_end: float,
        *,
        max_steps: int = 100000,
        bisect_iters: int = 48,
        boundary_tol: float = 1e-9,
    ) -> list[RaySegment]:
        """Trace ray segments for pre-normalized inputs."""
        o = origin_xyz
        d = direction_xyz_unit

        t0 = float(t_start)
        t1 = float(t_end)
        if t1 <= t0:
            return []

        abs_eps = max(float(boundary_tol) * (1.0 + abs(t1 - t0)), 1e-12)
        t = t0
        p = o + t * d
        hit = self.tree.lookup_point(p, space="xyz")
        if hit is None:
            logger.warning("Ray start point is outside interpolation domain.")
            return []

        segments: list[RaySegment] = []
        lookup = self.tree.lookup
        for _ in range(max_steps):
            if t >= (t1 - abs_eps):
                break
            cid = int(hit.cell_id)

            if not self.tree.contains_cell(cid, p, space="xyz", tol=1e-8):
                near_hit = self.tree.lookup_local(p, cid)
                if near_hit is None:
                    break
                hit = near_hit
                cid = int(hit.cell_id)

            dt = float(lookup.cell_step_hint(cid))

            t_lo = t
            t_hi = min(t1, t + dt)
            p_hi = o + t_hi * d
            while t_hi < t1 and self.tree.contains_cell(cid, p_hi, space="xyz", tol=1e-8):
                dt *= 2.0
                t_hi = min(t1, t + dt)
                p_hi = o + t_hi * d

            if self.tree.contains_cell(cid, p_hi, space="xyz", tol=1e-8):
                segments.append(RaySegment(cell_id=cid, t_enter=t, t_exit=t1))
                break

            lo = t_lo
            hi = t_hi
            for _ in range(bisect_iters):
                mid = 0.5 * (lo + hi)
                p_mid = o + mid * d
                if self.tree.contains_cell(cid, p_mid, space="xyz", tol=1e-8):
                    lo = mid
                else:
                    hi = mid
                if (hi - lo) <= abs_eps:
                    break
            t_exit = max(t, lo)
            segments.append(RaySegment(cell_id=cid, t_enter=t, t_exit=t_exit))

            t_next = min(t1, hi + abs_eps)
            if t_next <= t + abs_eps * 0.25:
                t_next = min(t1, t + abs_eps)
            p_next = o + t_next * d
            next_hit = self.tree.lookup_local(p_next, cid)
            if next_hit is None:
                break
            if int(next_hit.cell_id) == cid and t_next < t1:
                t_next = min(t1, t_next + 4.0 * abs_eps)
                p_next = o + t_next * d
                next_hit = self.tree.lookup_local(p_next, cid)
                if next_hit is None:
                    break

            t = t_next
            p = p_next
            hit = next_hit

        return segments


class OctreeRayInterpolator:
    """Ray sampling and piecewise-linear extraction on `OctreeInterpolator`."""

    def __init__(self, interpolator: "OctreeInterpolator") -> None:
        """Store the interpolator and its bound tree for ray operations."""
        self.interpolator = interpolator
        self.tree = interpolator.tree

    @staticmethod
    def point_in_tet(point_xyz: np.ndarray, tet_xyz: np.ndarray, *, tol: float = 1e-10) -> bool:
        """Test whether a point is inside/on one tetrahedron."""
        a = tet_xyz[0]
        mat = np.column_stack((tet_xyz[1] - a, tet_xyz[2] - a, tet_xyz[3] - a))
        try:
            uvw = np.linalg.solve(mat, point_xyz - a)
        except np.linalg.LinAlgError:
            return False
        b0 = 1.0 - float(np.sum(uvw))
        bary = np.array([b0, float(uvw[0]), float(uvw[1]), float(uvw[2])], dtype=float)
        return bool(np.all(bary >= -tol) and np.all(bary <= 1.0 + tol))

    def find_tet_in_hex(self, cell_xyz: np.ndarray, point_xyz: np.ndarray, *, tol: float = 1e-10) -> int | None:
        """Find which tet in the local 6-tet split contains a point."""
        p = np.array(point_xyz, dtype=float)
        for tet_idx, tet in enumerate(HEX_TETS_INDEX):
            tet_xyz = cell_xyz[tet]
            if self.point_in_tet(p, tet_xyz, tol=tol):
                return int(tet_idx)
        return None

    @staticmethod
    def ray_triangle_hit_t(
        ray_origin: np.ndarray,
        ray_dir: np.ndarray,
        tri_xyz: np.ndarray,
        *,
        t_min: float,
        t_max: float,
        tol: float,
    ) -> float | None:
        """Intersect one triangle with a ray and return hit `t` when valid."""
        p0 = tri_xyz[0]
        p1 = tri_xyz[1]
        p2 = tri_xyz[2]
        e1 = p1 - p0
        e2 = p2 - p0
        h = np.cross(ray_dir, e2)
        a = float(np.dot(e1, h))
        if abs(a) <= tol:
            return None
        f = 1.0 / a
        s = ray_origin - p0
        u = f * float(np.dot(s, h))
        if u < -tol or u > 1.0 + tol:
            return None
        q = np.cross(s, e1)
        v = f * float(np.dot(ray_dir, q))
        if v < -tol or (u + v) > 1.0 + tol:
            return None
        t = f * float(np.dot(e2, q))
        if t <= t_min + tol or t > t_max + tol:
            return None
        return float(t)

    @staticmethod
    def tet_ray_linear_coefficients(
        ray_origin: np.ndarray,
        ray_dir: np.ndarray,
        tet_xyz: np.ndarray,
        tet_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute linear-in-`t` coefficients on one linear tetrahedron."""
        mat = np.column_stack((tet_xyz, np.ones(4, dtype=float)))
        rhs = tet_values.reshape(4, -1)
        coef = np.linalg.solve(mat, rhs)
        slope = coef[0] * ray_dir[0] + coef[1] * ray_dir[1] + coef[2] * ray_dir[2]
        intercept = (
            coef[0] * ray_origin[0]
            + coef[1] * ray_origin[1]
            + coef[2] * ray_origin[2]
            + coef[3]
        )
        trailing = tet_values.shape[1:]
        return slope.reshape(trailing), intercept.reshape(trailing)

    def sample(
        self,
        origin_xyz: np.ndarray,
        direction_xyz: np.ndarray,
        t_start: float,
        t_end: float,
        n_samples: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[RaySegment]]:
        """Sample interpolated values at uniform `t` points on one ray.

        This method is coord-system agnostic: it evaluates values through the
        interpolator's generic callable interface with `query_space="xyz"`.
        """
        if int(n_samples) <= 0:
            raise ValueError("n_samples must be positive.")

        o = _as_xyz(origin_xyz)
        d = _normalize_direction(direction_xyz)
        t_values = np.linspace(float(t_start), float(t_end), int(n_samples))

        tracer = OctreeRayTracer(self.tree)
        segments = tracer.trace_prepared(o, d, float(t_start), float(t_end))
        query_xyz = o[None, :] + t_values[:, None] * d[None, :]
        values, cell_ids = self.interpolator(
            query_xyz,
            query_space="xyz",
            return_cell_ids=True,
        )
        return t_values, values, np.array(cell_ids, dtype=np.int64), segments

    def linear_pieces_for_cell_segment(
        self,
        ray_origin: np.ndarray,
        ray_dir: np.ndarray,
        cell_id: int,
        t_enter: float,
        t_exit: float,
        *,
        tol: float = 1e-10,
        max_steps: int = 128,
    ) -> list[RayLinearPiece]:
        """Split one ray/cell interval into piecewise-linear tet intervals."""
        cid = int(cell_id)
        if t_exit <= t_enter:
            return []

        interp = self.interpolator
        corner_ids = interp._corners[cid]
        cell_xyz = interp.lookup._points[corner_ids]
        cell_vals = interp._point_values[corner_ids]
        eps = max(1e-12, 1e-9 * (1.0 + abs(t_exit - t_enter)))

        t = float(t_enter)
        t_probe = min(float(t_exit), t + eps)
        p_probe = ray_origin + t_probe * ray_dir
        tet_idx = self.find_tet_in_hex(cell_xyz, p_probe, tol=1e-8)
        if tet_idx is None:
            p_mid = ray_origin + (0.5 * (t_enter + t_exit)) * ray_dir
            tet_idx = self.find_tet_in_hex(cell_xyz, p_mid, tol=1e-7)
        if tet_idx is None:
            return []

        pieces: list[RayLinearPiece] = []
        for _ in range(max_steps):
            if t >= t_exit - eps:
                break
            tet = HEX_TETS_INDEX[tet_idx]
            tet_xyz = cell_xyz[tet]
            tet_vals = cell_vals[tet]

            t_next = float(t_exit)
            for face in TET_FACES_INDEX:
                tri = tet_xyz[face]
                t_hit = self.ray_triangle_hit_t(
                    ray_origin,
                    ray_dir,
                    tri,
                    t_min=t + eps,
                    t_max=t_exit,
                    tol=tol,
                )
                if t_hit is not None and t_hit < t_next:
                    t_next = float(t_hit)

            if t_next <= t + eps * 0.25:
                t_next = min(float(t_exit), t + eps)

            slope, intercept = self.tet_ray_linear_coefficients(
                ray_origin,
                ray_dir,
                tet_xyz,
                tet_vals,
            )
            pieces.append(
                RayLinearPiece(
                    t_start=float(t),
                    t_end=float(t_next),
                    cell_id=cid,
                    tet_id=int(tet_idx),
                    slope=slope,
                    intercept=intercept,
                )
            )

            if t_next >= t_exit - eps:
                break

            t = float(t_next)
            next_tet: int | None = None
            for mult in (1.0, 4.0, 16.0, 64.0):
                t_probe = min(float(t_exit), t + mult * eps)
                p_probe = ray_origin + t_probe * ray_dir
                probe = self.find_tet_in_hex(cell_xyz, p_probe, tol=1e-7)
                if probe is not None:
                    next_tet = int(probe)
                    break
            if next_tet is None:
                break
            tet_idx = next_tet

        return pieces

    def linear_pieces(
        self,
        origin_xyz: np.ndarray,
        direction_xyz: np.ndarray,
        t_start: float,
        t_end: float,
    ) -> list[RayLinearPiece]:
        """Return stitched piecewise-linear `f(t)=m*t+b` intervals on a ray."""
        o = _as_xyz(origin_xyz)
        d = _normalize_direction(direction_xyz)

        tracer = OctreeRayTracer(self.tree)
        segments = tracer.trace_prepared(o, d, float(t_start), float(t_end))
        pieces: list[RayLinearPiece] = []
        for seg in segments:
            pieces.extend(
                self.linear_pieces_for_cell_segment(
                    o,
                    d,
                    int(seg.cell_id),
                    float(seg.t_enter),
                    float(seg.t_exit),
                )
            )
        if not pieces:
            return pieces

        span = abs(float(t_end) - float(t_start))
        stitch_tol = max(1e-10, 1e-8 * (1.0 + span))
        out: list[RayLinearPiece] = [pieces[0]]
        for piece in pieces[1:]:
            prev = out[-1]
            a = float(piece.t_start)
            b = float(piece.t_end)
            if abs(a - prev.t_end) <= stitch_tol:
                a = float(prev.t_end)
            if b <= a:
                continue
            out.append(
                RayLinearPiece(
                    t_start=a,
                    t_end=b,
                    cell_id=int(piece.cell_id),
                    tet_id=int(piece.tet_id),
                    slope=piece.slope,
                    intercept=piece.intercept,
                )
            )

        first = out[0]
        if abs(first.t_start - float(t_start)) <= stitch_tol:
            out[0] = RayLinearPiece(
                t_start=float(t_start),
                t_end=float(first.t_end),
                cell_id=int(first.cell_id),
                tet_id=int(first.tet_id),
                slope=first.slope,
                intercept=first.intercept,
            )
        last = out[-1]
        if abs(last.t_end - float(t_end)) <= stitch_tol:
            out[-1] = RayLinearPiece(
                t_start=float(last.t_start),
                t_end=float(t_end),
                cell_id=int(last.cell_id),
                tet_id=int(last.tet_id),
                slope=last.slope,
                intercept=last.intercept,
            )
        return out
