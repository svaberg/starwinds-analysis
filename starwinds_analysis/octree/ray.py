#!/usr/bin/env python3
"""Ray traversal and ray-based interpolation helpers for octrees."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import TYPE_CHECKING

from numba import njit
from numba import prange
import numpy as np

from .base import Octree
from .cartesian import CartesianLookupKernelState
from .cartesian import lookup_xyz_cell_id_kernel
from .spherical import LookupKernelState
from .spherical import lookup_rpa_cell_id_kernel

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

_TRACE_CONTAIN_TOL = 1e-8


@njit(cache=True)
def _contains_xyz_from_state(
    cid: int,
    x: float,
    y: float,
    z: float,
    lookup_state: CartesianLookupKernelState,
    tol: float = _TRACE_CONTAIN_TOL,
) -> bool:
    """Check Cartesian point containment using packed lookup state arrays."""
    if x < (lookup_state.cell_x_min[cid] - tol) or x > (lookup_state.cell_x_max[cid] + tol):
        return False
    if y < (lookup_state.cell_y_min[cid] - tol) or y > (lookup_state.cell_y_max[cid] + tol):
        return False
    if z < (lookup_state.cell_z_min[cid] - tol) or z > (lookup_state.cell_z_max[cid] + tol):
        return False
    return True


@njit(cache=True)
def _xyz_to_rpa_components(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert one Cartesian point to spherical `(r, polar, azimuth)`."""
    r = math.sqrt(x * x + y * y + z * z)
    if r == 0.0:
        polar = 0.0
    else:
        zr = z / r
        if zr < -1.0:
            zr = -1.0
        elif zr > 1.0:
            zr = 1.0
        polar = math.acos(zr)
    azimuth = math.atan2(y, x) % (2.0 * math.pi)
    return r, polar, azimuth


@njit(cache=True)
def _contains_rpa_from_components(
    cid: int,
    r: float,
    polar: float,
    azimuth: float,
    lookup_state: LookupKernelState,
    tol: float = _TRACE_CONTAIN_TOL,
) -> bool:
    """Check spherical point containment using packed lookup state arrays."""
    if r < (lookup_state.cell_r_min[cid] - tol) or r > (lookup_state.cell_r_max[cid] + tol):
        return False
    if polar < (lookup_state.cell_theta_min[cid] - tol) or polar > (lookup_state.cell_theta_max[cid] + tol):
        return False
    width = lookup_state.cell_phi_width[cid]
    dphi = (azimuth - lookup_state.cell_phi_start[cid]) % (2.0 * math.pi)
    if width >= ((2.0 * math.pi) - tol):
        return True
    return dphi <= (width + tol)


@njit(cache=True)
def _trace_segments_xyz_kernel(
    origin_xyz: np.ndarray,
    direction_xyz_unit: np.ndarray,
    t_start: float,
    t_end: float,
    max_steps: int,
    bisect_iters: int,
    boundary_tol: float,
    lookup_state: CartesianLookupKernelState,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Trace Cartesian ray segments with exact per-cell boundary walk.

    This kernel computes the next boundary crossing analytically from the
    current point and direction, avoiding expand-and-bisect stepping.
    """
    cell_ids = np.empty(max_steps, dtype=np.int64)
    enters = np.empty(max_steps, dtype=np.float64)
    exits = np.empty(max_steps, dtype=np.float64)

    if t_end <= t_start or max_steps <= 0:
        return 0, cell_ids, enters, exits

    abs_eps = max(boundary_tol * (1.0 + abs(t_end - t_start)), 1e-12)
    t = t_start
    x = origin_xyz[0] + t * direction_xyz_unit[0]
    y = origin_xyz[1] + t * direction_xyz_unit[1]
    z = origin_xyz[2] + t * direction_xyz_unit[2]
    cid = lookup_xyz_cell_id_kernel(x, y, z, lookup_state, -1)
    if cid < 0:
        return 0, cell_ids, enters, exits

    d0 = direction_xyz_unit[0]
    d1 = direction_xyz_unit[1]
    d2 = direction_xyz_unit[2]
    dir_eps = 1e-15
    n_seg = 0
    for _ in range(max_steps):
        if t >= (t_end - abs_eps):
            break

        if not _contains_xyz_from_state(cid, x, y, z, lookup_state):
            cid_near = lookup_xyz_cell_id_kernel(x, y, z, lookup_state, cid)
            if cid_near < 0:
                break
            cid = cid_near

        tx = np.inf
        ty = np.inf
        tz = np.inf
        if abs(d0) > dir_eps:
            if d0 > 0.0:
                tx = (lookup_state.cell_x_max[cid] - x) / d0
            else:
                tx = (lookup_state.cell_x_min[cid] - x) / d0
            if tx <= abs_eps:
                tx = np.inf
        if abs(d1) > dir_eps:
            if d1 > 0.0:
                ty = (lookup_state.cell_y_max[cid] - y) / d1
            else:
                ty = (lookup_state.cell_y_min[cid] - y) / d1
            if ty <= abs_eps:
                ty = np.inf
        if abs(d2) > dir_eps:
            if d2 > 0.0:
                tz = (lookup_state.cell_z_max[cid] - z) / d2
            else:
                tz = (lookup_state.cell_z_min[cid] - z) / d2
            if tz <= abs_eps:
                tz = np.inf

        dt_exit = tx
        if ty < dt_exit:
            dt_exit = ty
        if tz < dt_exit:
            dt_exit = tz
        if not math.isfinite(dt_exit):
            dt_exit = t_end - t
        if dt_exit <= abs_eps:
            dt_exit = abs_eps
        t_exit = t + dt_exit
        if t_exit > t_end:
            t_exit = t_end
        if t_exit < t:
            t_exit = t
        cell_ids[n_seg] = cid
        enters[n_seg] = t
        exits[n_seg] = t_exit
        n_seg += 1
        if n_seg >= max_steps:
            break
        if t_exit >= (t_end - abs_eps):
            break

        t_next = t_exit + abs_eps
        if t_next > t_end:
            t_next = t_end
        if t_next <= t + abs_eps * 0.25:
            t_next = t + abs_eps
            if t_next > t_end:
                t_next = t_end
        x_next = origin_xyz[0] + t_next * d0
        y_next = origin_xyz[1] + t_next * d1
        z_next = origin_xyz[2] + t_next * d2
        cid_next = lookup_xyz_cell_id_kernel(x_next, y_next, z_next, lookup_state, cid)
        if cid_next < 0:
            break
        if cid_next == cid and t_next < t_end:
            t_next = t_next + 4.0 * abs_eps
            if t_next > t_end:
                t_next = t_end
            x_next = origin_xyz[0] + t_next * d0
            y_next = origin_xyz[1] + t_next * d1
            z_next = origin_xyz[2] + t_next * d2
            cid_next = lookup_xyz_cell_id_kernel(x_next, y_next, z_next, lookup_state, cid)
            if cid_next < 0:
                break

        t = t_next
        x = x_next
        y = y_next
        z = z_next
        cid = cid_next

    return n_seg, cell_ids, enters, exits


@njit(cache=True)
def _trace_segments_rpa_kernel(
    origin_xyz: np.ndarray,
    direction_xyz_unit: np.ndarray,
    t_start: float,
    t_end: float,
    max_steps: int,
    bisect_iters: int,
    boundary_tol: float,
    lookup_state: LookupKernelState,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Trace Cartesian ray segments on spherical trees using compiled kernels."""
    cell_ids = np.empty(max_steps, dtype=np.int64)
    enters = np.empty(max_steps, dtype=np.float64)
    exits = np.empty(max_steps, dtype=np.float64)

    if t_end <= t_start or max_steps <= 0:
        return 0, cell_ids, enters, exits

    abs_eps = max(boundary_tol * (1.0 + abs(t_end - t_start)), 1e-12)
    t = t_start
    x = origin_xyz[0] + t * direction_xyz_unit[0]
    y = origin_xyz[1] + t * direction_xyz_unit[1]
    z = origin_xyz[2] + t * direction_xyz_unit[2]
    r, polar, azimuth = _xyz_to_rpa_components(x, y, z)
    cid = lookup_rpa_cell_id_kernel(r, polar, azimuth, lookup_state, -1)
    if cid < 0:
        return 0, cell_ids, enters, exits

    n_seg = 0
    for _ in range(max_steps):
        if t >= (t_end - abs_eps):
            break

        if not _contains_rpa_from_components(cid, r, polar, azimuth, lookup_state):
            cid_near = lookup_rpa_cell_id_kernel(r, polar, azimuth, lookup_state, cid)
            if cid_near < 0:
                break
            cid = cid_near

        r_span = lookup_state.cell_r_max[cid] - lookup_state.cell_r_min[cid]
        theta_span = lookup_state.cell_theta_max[cid] - lookup_state.cell_theta_min[cid]
        phi_width = lookup_state.cell_phi_width[cid]
        phi_span = phi_width
        two_pi = 2.0 * math.pi
        if phi_span > two_pi:
            phi_span = two_pi
        length_scale = lookup_state.cell_r_max[cid]
        if length_scale < 1.0:
            length_scale = 1.0
        dt = r_span
        t_theta = length_scale * theta_span
        if t_theta > dt:
            dt = t_theta
        t_phi = length_scale * phi_span
        if t_phi > dt:
            dt = t_phi
        if dt < 1e-6:
            dt = 1e-6

        t_hi = t + dt
        if t_hi > t_end:
            t_hi = t_end
        x_hi = origin_xyz[0] + t_hi * direction_xyz_unit[0]
        y_hi = origin_xyz[1] + t_hi * direction_xyz_unit[1]
        z_hi = origin_xyz[2] + t_hi * direction_xyz_unit[2]
        r_hi, polar_hi, azimuth_hi = _xyz_to_rpa_components(x_hi, y_hi, z_hi)
        while t_hi < t_end and _contains_rpa_from_components(cid, r_hi, polar_hi, azimuth_hi, lookup_state):
            dt *= 2.0
            t_hi = t + dt
            if t_hi > t_end:
                t_hi = t_end
            x_hi = origin_xyz[0] + t_hi * direction_xyz_unit[0]
            y_hi = origin_xyz[1] + t_hi * direction_xyz_unit[1]
            z_hi = origin_xyz[2] + t_hi * direction_xyz_unit[2]
            r_hi, polar_hi, azimuth_hi = _xyz_to_rpa_components(x_hi, y_hi, z_hi)

        if _contains_rpa_from_components(cid, r_hi, polar_hi, azimuth_hi, lookup_state):
            cell_ids[n_seg] = cid
            enters[n_seg] = t
            exits[n_seg] = t_end
            n_seg += 1
            break

        lo = t
        hi = t_hi
        for _ in range(bisect_iters):
            mid = 0.5 * (lo + hi)
            x_mid = origin_xyz[0] + mid * direction_xyz_unit[0]
            y_mid = origin_xyz[1] + mid * direction_xyz_unit[1]
            z_mid = origin_xyz[2] + mid * direction_xyz_unit[2]
            r_mid, polar_mid, azimuth_mid = _xyz_to_rpa_components(x_mid, y_mid, z_mid)
            if _contains_rpa_from_components(cid, r_mid, polar_mid, azimuth_mid, lookup_state):
                lo = mid
            else:
                hi = mid
            if (hi - lo) <= abs_eps:
                break

        t_exit = lo
        if t_exit < t:
            t_exit = t
        cell_ids[n_seg] = cid
        enters[n_seg] = t
        exits[n_seg] = t_exit
        n_seg += 1
        if n_seg >= max_steps:
            break

        t_next = hi + abs_eps
        if t_next > t_end:
            t_next = t_end
        if t_next <= t + abs_eps * 0.25:
            t_next = t + abs_eps
            if t_next > t_end:
                t_next = t_end
        x_next = origin_xyz[0] + t_next * direction_xyz_unit[0]
        y_next = origin_xyz[1] + t_next * direction_xyz_unit[1]
        z_next = origin_xyz[2] + t_next * direction_xyz_unit[2]
        r_next, polar_next, azimuth_next = _xyz_to_rpa_components(x_next, y_next, z_next)
        cid_next = lookup_rpa_cell_id_kernel(r_next, polar_next, azimuth_next, lookup_state, cid)
        if cid_next < 0:
            break
        if cid_next == cid and t_next < t_end:
            t_next = t_next + 4.0 * abs_eps
            if t_next > t_end:
                t_next = t_end
            x_next = origin_xyz[0] + t_next * direction_xyz_unit[0]
            y_next = origin_xyz[1] + t_next * direction_xyz_unit[1]
            z_next = origin_xyz[2] + t_next * direction_xyz_unit[2]
            r_next, polar_next, azimuth_next = _xyz_to_rpa_components(x_next, y_next, z_next)
            cid_next = lookup_rpa_cell_id_kernel(r_next, polar_next, azimuth_next, lookup_state, cid)
            if cid_next < 0:
                break

        t = t_next
        x = x_next
        y = y_next
        z = z_next
        r = r_next
        polar = polar_next
        azimuth = azimuth_next
        cid = cid_next

    return n_seg, cell_ids, enters, exits


@njit(cache=True)
def _trilinear_scalar_from_cell_xyz_state(
    cell_id: int,
    x: float,
    y: float,
    z: float,
    interp_state,
) -> float:
    """Evaluate scalar trilinear interpolation (component 0) in one Cartesian cell."""
    cid = int(cell_id)
    u = (x - interp_state.cell_x0[cid]) / interp_state.cell_xden[cid]
    if u < 0.0:
        u = 0.0
    elif u > 1.0:
        u = 1.0
    v = (y - interp_state.cell_y0[cid]) / interp_state.cell_yden[cid]
    if v < 0.0:
        v = 0.0
    elif v > 1.0:
        v = 1.0
    w = (z - interp_state.cell_z0[cid]) / interp_state.cell_zden[cid]
    if w < 0.0:
        w = 0.0
    elif w > 1.0:
        w = 1.0

    w0 = (1.0 - u) * (1.0 - v) * (1.0 - w)
    w1 = u * (1.0 - v) * (1.0 - w)
    w2 = (1.0 - u) * v * (1.0 - w)
    w3 = u * v * (1.0 - w)
    w4 = (1.0 - u) * (1.0 - v) * w
    w5 = u * (1.0 - v) * w
    w6 = (1.0 - u) * v * w
    w7 = u * v * w

    local = interp_state.corners[cid]
    map_row = interp_state.bin_to_corner[cid]
    c0 = int(local[int(map_row[0])])
    c1 = int(local[int(map_row[1])])
    c2 = int(local[int(map_row[2])])
    c3 = int(local[int(map_row[3])])
    c4 = int(local[int(map_row[4])])
    c5 = int(local[int(map_row[5])])
    c6 = int(local[int(map_row[6])])
    c7 = int(local[int(map_row[7])])
    return (
        w0 * interp_state.point_values_2d[c0, 0]
        + w1 * interp_state.point_values_2d[c1, 0]
        + w2 * interp_state.point_values_2d[c2, 0]
        + w3 * interp_state.point_values_2d[c3, 0]
        + w4 * interp_state.point_values_2d[c4, 0]
        + w5 * interp_state.point_values_2d[c5, 0]
        + w6 * interp_state.point_values_2d[c6, 0]
        + w7 * interp_state.point_values_2d[c7, 0]
    )


@njit(cache=True, parallel=True)
def _integrate_axis_aligned_xyz_scalar_kernel(
    origins_xyz: np.ndarray,
    direction_xyz_unit: np.ndarray,
    t_start: float,
    t_end: float,
    axis: int,
    scale: float,
    max_steps: int,
    boundary_tol: float,
    lookup_state: CartesianLookupKernelState,
    interp_state,
) -> np.ndarray:
    """Integrate scalar field along many axis-aligned Cartesian rays."""
    n_rays = int(origins_xyz.shape[0])
    out = np.full(n_rays, np.nan, dtype=np.float64)
    if t_end <= t_start or n_rays == 0:
        return out

    d0 = direction_xyz_unit[0]
    d1 = direction_xyz_unit[1]
    d2 = direction_xyz_unit[2]
    abs_eps = max(boundary_tol * (1.0 + abs(t_end - t_start)), 1e-12)

    for i in prange(n_rays):
        ox = origins_xyz[i, 0]
        oy = origins_xyz[i, 1]
        oz = origins_xyz[i, 2]
        t = t_start
        x = ox + t * d0
        y = oy + t * d1
        z = oz + t * d2
        cid = lookup_xyz_cell_id_kernel(x, y, z, lookup_state, -1)
        if cid < 0:
            continue

        col = 0.0
        for _ in range(max_steps):
            if t >= (t_end - abs_eps):
                break

            if not _contains_xyz_from_state(cid, x, y, z, lookup_state):
                cid = lookup_xyz_cell_id_kernel(x, y, z, lookup_state, cid)
                if cid < 0:
                    break

            if axis == 0:
                if d0 > 0.0:
                    t_exit = (lookup_state.cell_x_max[cid] - ox) / d0
                else:
                    t_exit = (lookup_state.cell_x_min[cid] - ox) / d0
            elif axis == 1:
                if d1 > 0.0:
                    t_exit = (lookup_state.cell_y_max[cid] - oy) / d1
                else:
                    t_exit = (lookup_state.cell_y_min[cid] - oy) / d1
            else:
                if d2 > 0.0:
                    t_exit = (lookup_state.cell_z_max[cid] - oz) / d2
                else:
                    t_exit = (lookup_state.cell_z_min[cid] - oz) / d2

            if not math.isfinite(t_exit):
                break
            if t_exit > t_end:
                t_exit = t_end
            if t_exit <= t + abs_eps * 0.25:
                t_exit = t + abs_eps
                if t_exit > t_end:
                    t_exit = t_end

            x1 = ox + t_exit * d0
            y1 = oy + t_exit * d1
            z1 = oz + t_exit * d2
            v0 = _trilinear_scalar_from_cell_xyz_state(cid, x, y, z, interp_state)
            v1 = _trilinear_scalar_from_cell_xyz_state(cid, x1, y1, z1, interp_state)
            col += 0.5 * (v0 + v1) * (t_exit - t)

            if t_exit >= (t_end - abs_eps):
                t = t_exit
                x = x1
                y = y1
                z = z1
                break

            t_next = t_exit + abs_eps
            if t_next > t_end:
                t_next = t_end
            x_next = ox + t_next * d0
            y_next = oy + t_next * d1
            z_next = oz + t_next * d2
            cid_next = lookup_xyz_cell_id_kernel(x_next, y_next, z_next, lookup_state, cid)
            if cid_next < 0:
                break

            t = t_next
            x = x_next
            y = y_next
            z = z_next
            cid = cid_next

        out[i] = scale * col

    return out


@njit(cache=True, parallel=True)
def _integrate_axis_aligned_xyz_scalar_midpoint_kernel(
    origins_xyz: np.ndarray,
    direction_xyz_unit: np.ndarray,
    t_start: float,
    t_end: float,
    axis: int,
    scale: float,
    max_steps: int,
    boundary_tol: float,
    lookup_state: CartesianLookupKernelState,
    interp_state,
) -> np.ndarray:
    """Integrate scalar field along many axis-aligned rays with midpoint rule."""
    n_rays = int(origins_xyz.shape[0])
    out = np.full(n_rays, np.nan, dtype=np.float64)
    if t_end <= t_start or n_rays == 0:
        return out

    d0 = direction_xyz_unit[0]
    d1 = direction_xyz_unit[1]
    d2 = direction_xyz_unit[2]
    abs_eps = max(boundary_tol * (1.0 + abs(t_end - t_start)), 1e-12)

    for i in prange(n_rays):
        ox = origins_xyz[i, 0]
        oy = origins_xyz[i, 1]
        oz = origins_xyz[i, 2]
        t = t_start
        x = ox + t * d0
        y = oy + t * d1
        z = oz + t * d2
        cid = lookup_xyz_cell_id_kernel(x, y, z, lookup_state, -1)
        if cid < 0:
            continue

        col = 0.0
        for _ in range(max_steps):
            if t >= (t_end - abs_eps):
                break

            if not _contains_xyz_from_state(cid, x, y, z, lookup_state):
                cid = lookup_xyz_cell_id_kernel(x, y, z, lookup_state, cid)
                if cid < 0:
                    break

            if axis == 0:
                if d0 > 0.0:
                    t_exit = (lookup_state.cell_x_max[cid] - ox) / d0
                else:
                    t_exit = (lookup_state.cell_x_min[cid] - ox) / d0
            elif axis == 1:
                if d1 > 0.0:
                    t_exit = (lookup_state.cell_y_max[cid] - oy) / d1
                else:
                    t_exit = (lookup_state.cell_y_min[cid] - oy) / d1
            else:
                if d2 > 0.0:
                    t_exit = (lookup_state.cell_z_max[cid] - oz) / d2
                else:
                    t_exit = (lookup_state.cell_z_min[cid] - oz) / d2

            if not math.isfinite(t_exit):
                break
            if t_exit > t_end:
                t_exit = t_end
            if t_exit <= t + abs_eps * 0.25:
                t_exit = t + abs_eps
                if t_exit > t_end:
                    t_exit = t_end

            tm = 0.5 * (t + t_exit)
            xm = ox + tm * d0
            ym = oy + tm * d1
            zm = oz + tm * d2
            vm = _trilinear_scalar_from_cell_xyz_state(cid, xm, ym, zm, interp_state)
            col += vm * (t_exit - t)

            if t_exit >= (t_end - abs_eps):
                break

            t_next = t_exit + abs_eps
            if t_next > t_end:
                t_next = t_end
            x_next = ox + t_next * d0
            y_next = oy + t_next * d1
            z_next = oz + t_next * d2
            cid_next = lookup_xyz_cell_id_kernel(x_next, y_next, z_next, lookup_state, cid)
            if cid_next < 0:
                break

            t = t_next
            x = x_next
            y = y_next
            z = z_next
            cid = cid_next

        out[i] = scale * col

    return out


@njit(cache=True, parallel=True)
def _integrate_xyz_scalar_exact_kernel(
    origins_xyz: np.ndarray,
    direction_xyz_unit: np.ndarray,
    t_start: float,
    t_end: float,
    scale: float,
    max_steps: int,
    boundary_tol: float,
    lookup_state: CartesianLookupKernelState,
    interp_state,
) -> np.ndarray:
    """Integrate scalar field along many Cartesian rays (arbitrary direction, exact per-cell linear integral)."""
    n_rays = int(origins_xyz.shape[0])
    out = np.full(n_rays, np.nan, dtype=np.float64)
    if t_end <= t_start or n_rays == 0:
        return out

    d0 = direction_xyz_unit[0]
    d1 = direction_xyz_unit[1]
    d2 = direction_xyz_unit[2]
    abs_eps = max(boundary_tol * (1.0 + abs(t_end - t_start)), 1e-12)
    dir_eps = 1e-15

    for i in prange(n_rays):
        ox = origins_xyz[i, 0]
        oy = origins_xyz[i, 1]
        oz = origins_xyz[i, 2]
        t = t_start
        x = ox + t * d0
        y = oy + t * d1
        z = oz + t * d2
        cid = lookup_xyz_cell_id_kernel(x, y, z, lookup_state, -1)
        if cid < 0:
            continue

        col = 0.0
        for _ in range(max_steps):
            if t >= (t_end - abs_eps):
                break

            if not _contains_xyz_from_state(cid, x, y, z, lookup_state):
                cid = lookup_xyz_cell_id_kernel(x, y, z, lookup_state, cid)
                if cid < 0:
                    break

            tx = np.inf
            ty = np.inf
            tz = np.inf
            if abs(d0) > dir_eps:
                if d0 > 0.0:
                    tx = (lookup_state.cell_x_max[cid] - x) / d0
                else:
                    tx = (lookup_state.cell_x_min[cid] - x) / d0
                if tx <= abs_eps:
                    tx = np.inf
            if abs(d1) > dir_eps:
                if d1 > 0.0:
                    ty = (lookup_state.cell_y_max[cid] - y) / d1
                else:
                    ty = (lookup_state.cell_y_min[cid] - y) / d1
                if ty <= abs_eps:
                    ty = np.inf
            if abs(d2) > dir_eps:
                if d2 > 0.0:
                    tz = (lookup_state.cell_z_max[cid] - z) / d2
                else:
                    tz = (lookup_state.cell_z_min[cid] - z) / d2
                if tz <= abs_eps:
                    tz = np.inf

            dt_exit = tx
            if ty < dt_exit:
                dt_exit = ty
            if tz < dt_exit:
                dt_exit = tz
            if not math.isfinite(dt_exit):
                dt_exit = t_end - t
            if dt_exit <= abs_eps:
                dt_exit = abs_eps

            t_exit = t + dt_exit
            if t_exit > t_end:
                t_exit = t_end
            if t_exit < t:
                t_exit = t

            x1 = ox + t_exit * d0
            y1 = oy + t_exit * d1
            z1 = oz + t_exit * d2
            v0 = _trilinear_scalar_from_cell_xyz_state(cid, x, y, z, interp_state)
            v1 = _trilinear_scalar_from_cell_xyz_state(cid, x1, y1, z1, interp_state)
            col += 0.5 * (v0 + v1) * (t_exit - t)

            if t_exit >= (t_end - abs_eps):
                break

            t_next = t_exit + abs_eps
            if t_next > t_end:
                t_next = t_end
            if t_next <= t + abs_eps * 0.25:
                t_next = t + abs_eps
                if t_next > t_end:
                    t_next = t_end
            x_next = ox + t_next * d0
            y_next = oy + t_next * d1
            z_next = oz + t_next * d2
            cid_next = lookup_xyz_cell_id_kernel(x_next, y_next, z_next, lookup_state, cid)
            if cid_next < 0:
                break
            if cid_next == cid and t_next < t_end:
                t_next = t_next + 4.0 * abs_eps
                if t_next > t_end:
                    t_next = t_end
                x_next = ox + t_next * d0
                y_next = oy + t_next * d1
                z_next = oz + t_next * d2
                cid_next = lookup_xyz_cell_id_kernel(x_next, y_next, z_next, lookup_state, cid)
                if cid_next < 0:
                    break

            t = t_next
            x = x_next
            y = y_next
            z = z_next
            cid = cid_next

        out[i] = scale * col

    return out


@njit(cache=True, parallel=True)
def _integrate_xyz_scalar_midpoint_kernel(
    origins_xyz: np.ndarray,
    direction_xyz_unit: np.ndarray,
    t_start: float,
    t_end: float,
    scale: float,
    max_steps: int,
    boundary_tol: float,
    lookup_state: CartesianLookupKernelState,
    interp_state,
) -> np.ndarray:
    """Integrate scalar field along many Cartesian rays (arbitrary direction, midpoint rule)."""
    n_rays = int(origins_xyz.shape[0])
    out = np.full(n_rays, np.nan, dtype=np.float64)
    if t_end <= t_start or n_rays == 0:
        return out

    d0 = direction_xyz_unit[0]
    d1 = direction_xyz_unit[1]
    d2 = direction_xyz_unit[2]
    abs_eps = max(boundary_tol * (1.0 + abs(t_end - t_start)), 1e-12)
    dir_eps = 1e-15

    for i in prange(n_rays):
        ox = origins_xyz[i, 0]
        oy = origins_xyz[i, 1]
        oz = origins_xyz[i, 2]
        t = t_start
        x = ox + t * d0
        y = oy + t * d1
        z = oz + t * d2
        cid = lookup_xyz_cell_id_kernel(x, y, z, lookup_state, -1)
        if cid < 0:
            continue

        col = 0.0
        for _ in range(max_steps):
            if t >= (t_end - abs_eps):
                break

            if not _contains_xyz_from_state(cid, x, y, z, lookup_state):
                cid = lookup_xyz_cell_id_kernel(x, y, z, lookup_state, cid)
                if cid < 0:
                    break

            tx = np.inf
            ty = np.inf
            tz = np.inf
            if abs(d0) > dir_eps:
                if d0 > 0.0:
                    tx = (lookup_state.cell_x_max[cid] - x) / d0
                else:
                    tx = (lookup_state.cell_x_min[cid] - x) / d0
                if tx <= abs_eps:
                    tx = np.inf
            if abs(d1) > dir_eps:
                if d1 > 0.0:
                    ty = (lookup_state.cell_y_max[cid] - y) / d1
                else:
                    ty = (lookup_state.cell_y_min[cid] - y) / d1
                if ty <= abs_eps:
                    ty = np.inf
            if abs(d2) > dir_eps:
                if d2 > 0.0:
                    tz = (lookup_state.cell_z_max[cid] - z) / d2
                else:
                    tz = (lookup_state.cell_z_min[cid] - z) / d2
                if tz <= abs_eps:
                    tz = np.inf

            dt_exit = tx
            if ty < dt_exit:
                dt_exit = ty
            if tz < dt_exit:
                dt_exit = tz
            if not math.isfinite(dt_exit):
                dt_exit = t_end - t
            if dt_exit <= abs_eps:
                dt_exit = abs_eps

            t_exit = t + dt_exit
            if t_exit > t_end:
                t_exit = t_end
            if t_exit < t:
                t_exit = t

            tm = 0.5 * (t + t_exit)
            xm = ox + tm * d0
            ym = oy + tm * d1
            zm = oz + tm * d2
            vm = _trilinear_scalar_from_cell_xyz_state(cid, xm, ym, zm, interp_state)
            col += vm * (t_exit - t)

            if t_exit >= (t_end - abs_eps):
                break

            t_next = t_exit + abs_eps
            if t_next > t_end:
                t_next = t_end
            if t_next <= t + abs_eps * 0.25:
                t_next = t + abs_eps
                if t_next > t_end:
                    t_next = t_end
            x_next = ox + t_next * d0
            y_next = oy + t_next * d1
            z_next = oz + t_next * d2
            cid_next = lookup_xyz_cell_id_kernel(x_next, y_next, z_next, lookup_state, cid)
            if cid_next < 0:
                break
            if cid_next == cid and t_next < t_end:
                t_next = t_next + 4.0 * abs_eps
                if t_next > t_end:
                    t_next = t_end
                x_next = ox + t_next * d0
                y_next = oy + t_next * d1
                z_next = oz + t_next * d2
                cid_next = lookup_xyz_cell_id_kernel(x_next, y_next, z_next, lookup_state, cid)
                if cid_next < 0:
                    break

            t = t_next
            x = x_next
            y = y_next
            z = z_next
            cid = cid_next

        out[i] = scale * col

    return out


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
        """Trace ray segments for pre-normalized inputs.

        Uses compiled traversal kernels when lookup state supports them;
        falls back to the Python path otherwise.
        """
        o = origin_xyz
        d = direction_xyz_unit
        t0 = float(t_start)
        t1 = float(t_end)
        if t1 <= t0:
            return []

        lookup = self.tree.lookup
        coord_system = str(self.tree.coord_system)
        if coord_system == "xyz":
            try:
                n_seg, cids, enters, exits = _trace_segments_xyz_kernel(
                    o,
                    d,
                    t0,
                    t1,
                    int(max_steps),
                    int(bisect_iters),
                    float(boundary_tol),
                    lookup._lookup_state,
                )
                return [
                    RaySegment(
                        cell_id=int(cids[i]),
                        t_enter=float(enters[i]),
                        t_exit=float(exits[i]),
                    )
                    for i in range(int(n_seg))
                ]
            except Exception:
                logger.debug("Falling back to Python ray tracer path for xyz tree.", exc_info=True)
        elif coord_system == "rpa":
            try:
                n_seg, cids, enters, exits = _trace_segments_rpa_kernel(
                    o,
                    d,
                    t0,
                    t1,
                    int(max_steps),
                    int(bisect_iters),
                    float(boundary_tol),
                    lookup._lookup_state,
                )
                if n_seg == 0:
                    p0 = o + t0 * d
                    if self.tree.lookup_point(p0, space="xyz") is None:
                        logger.warning("Ray start point is outside interpolation domain.")
                return [
                    RaySegment(
                        cell_id=int(cids[i]),
                        t_enter=float(enters[i]),
                        t_exit=float(exits[i]),
                    )
                    for i in range(int(n_seg))
                ]
            except Exception:
                logger.debug("Falling back to Python ray tracer path for rpa tree.", exc_info=True)

        return self._trace_prepared_python(
            o,
            d,
            t0,
            t1,
            max_steps=max_steps,
            bisect_iters=bisect_iters,
            boundary_tol=boundary_tol,
        )

    def _trace_prepared_python(
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
        """Python fallback implementation of ray tracing."""
        o = origin_xyz
        d = direction_xyz_unit
        abs_eps = max(float(boundary_tol) * (1.0 + abs(t_end - t_start)), 1e-12)
        t = float(t_start)
        p = o + t * d
        hit = self.tree.lookup_point(p, space="xyz")
        if hit is None:
            logger.warning("Ray start point is outside interpolation domain.")
            return []
        segments: list[RaySegment] = []
        lookup = self.tree.lookup
        for _ in range(max_steps):
            if t >= (t_end - abs_eps):
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
            t_hi = min(t_end, t + dt)
            p_hi = o + t_hi * d
            while t_hi < t_end and self.tree.contains_cell(cid, p_hi, space="xyz", tol=1e-8):
                dt *= 2.0
                t_hi = min(t_end, t + dt)
                p_hi = o + t_hi * d

            if self.tree.contains_cell(cid, p_hi, space="xyz", tol=1e-8):
                segments.append(RaySegment(cell_id=cid, t_enter=t, t_exit=t_end))
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

            t_next = min(t_end, hi + abs_eps)
            if t_next <= t + abs_eps * 0.25:
                t_next = min(t_end, t + abs_eps)
            p_next = o + t_next * d
            next_hit = self.tree.lookup_local(p_next, cid)
            if next_hit is None:
                break
            if int(next_hit.cell_id) == cid and t_next < t_end:
                t_next = min(t_end, t_next + 4.0 * abs_eps)
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
        self.ray_tracer = OctreeRayTracer(self.tree)

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

        segments = self.ray_tracer.trace_prepared(o, d, float(t_start), float(t_end))
        query_xyz = o[None, :] + t_values[:, None] * d[None, :]
        values, cell_ids = self.interpolator(
            query_xyz,
            query_space="xyz",
            return_cell_ids=True,
        )
        return t_values, values, np.array(cell_ids, dtype=np.int64), segments

    def integrate_field_along_rays(
        self,
        origins_xyz: np.ndarray,
        direction_xyz: np.ndarray,
        t_start: float,
        t_end: float,
        *,
        chunk_size: int = 2048,
        scale: float = 1.0,
    ) -> np.ndarray:
        """Integrate interpolated field(s) along many rays.

        This computes `scale * integral f(t) dt` for each ray using exact
        per-cell linearity on octree leaf segments:
        - trace cell segments along each ray;
        - evaluate field at segment endpoints in batch;
        - use trapezoidal exactness on linear segments.
        """
        origins = np.asarray(origins_xyz, dtype=float)
        if origins.ndim == 1:
            if origins.shape[0] != 3:
                raise ValueError("origins_xyz must have shape (n_rays, 3) or (3,).")
            origins = origins.reshape(1, 3)
        if origins.ndim != 2 or origins.shape[1] != 3:
            raise ValueError("origins_xyz must have shape (n_rays, 3) or (3,).")
        if int(chunk_size) <= 0:
            raise ValueError("chunk_size must be positive.")

        d = _normalize_direction(direction_xyz)
        t0 = float(t_start)
        t1 = float(t_end)
        if t1 <= t0:
            raise ValueError("t_end must be greater than t_start.")

        n_rays = int(origins.shape[0])

        # Fastest path: Cartesian scalar field on axis-aligned rays.
        abs_d = np.abs(d)
        axis = int(np.argmax(abs_d))
        axis_tol = 1e-12
        is_axis_aligned = bool(
            abs_d[axis] >= (1.0 - axis_tol)
            and abs_d[(axis + 1) % 3] <= axis_tol
            and abs_d[(axis + 2) % 3] <= axis_tol
        )
        if (
            is_axis_aligned
            and str(self.tree.coord_system) == "xyz"
            and int(getattr(self.interpolator, "_n_value_components", 0)) == 1
            and hasattr(self.interpolator, "_interp_state_xyz")
            and hasattr(self.tree.lookup, "_lookup_state")
        ):
            return _integrate_axis_aligned_xyz_scalar_kernel(
                origins,
                d,
                t0,
                t1,
                int(axis),
                float(scale),
                200000,
                1e-9,
                self.tree.lookup._lookup_state,
                self.interpolator._interp_state_xyz,
            )

        if (
            str(self.tree.coord_system) == "xyz"
            and int(getattr(self.interpolator, "_n_value_components", 0)) == 1
            and hasattr(self.interpolator, "_interp_state_xyz")
            and hasattr(self.tree.lookup, "_lookup_state")
        ):
            return _integrate_xyz_scalar_exact_kernel(
                origins,
                d,
                t0,
                t1,
                float(scale),
                200000,
                1e-9,
                self.tree.lookup._lookup_state,
                self.interpolator._interp_state_xyz,
            )

        use_xyz_kernel = bool(str(self.tree.coord_system) == "xyz" and hasattr(self.tree.lookup, "_lookup_state"))
        trace_max_steps = 16384
        trace_boundary_tol = 1e-9
        out_2d: np.ndarray | None = None

        for i0 in range(0, n_rays, int(chunk_size)):
            i1 = min(n_rays, i0 + int(chunk_size))
            n_chunk = i1 - i0

            endpoint_offsets = np.zeros(n_chunk + 1, dtype=np.int64)
            seg_dt_list: list[np.ndarray] = []
            query_points: list[tuple[float, float, float]] = []
            k = 0
            if use_xyz_kernel:
                lookup_state = self.tree.lookup._lookup_state
                for j in range(n_chunk):
                    endpoint_offsets[j] = int(k)
                    o = origins[i0 + j]
                    n_seg, cids, enters, exits = _trace_segments_xyz_kernel(
                        o,
                        d,
                        t0,
                        t1,
                        int(trace_max_steps),
                        0,
                        float(trace_boundary_tol),
                        lookup_state,
                    )
                    dts: list[float] = []
                    for si in range(int(n_seg)):
                        if int(cids[si]) < 0:
                            continue
                        ta = float(enters[si])
                        tb = float(exits[si])
                        dt = tb - ta
                        if dt <= 0.0:
                            continue
                        pa = o + ta * d
                        pb = o + tb * d
                        query_points.append((float(pa[0]), float(pa[1]), float(pa[2])))
                        query_points.append((float(pb[0]), float(pb[1]), float(pb[2])))
                        dts.append(float(dt))
                        k += 2
                    seg_dt_list.append(np.asarray(dts, dtype=float))
                endpoint_offsets[n_chunk] = int(k)
            else:
                segment_lists: list[list[RaySegment]] = []
                endpoint_count = 0
                for j in range(n_chunk):
                    segs = self.ray_tracer.trace_prepared(origins[i0 + j], d, t0, t1)
                    segment_lists.append(segs)
                    endpoint_count += 2 * len(segs)
                if endpoint_count == 0:
                    if out_2d is None:
                        continue
                    continue
                query_xyz = np.empty((endpoint_count, 3), dtype=float)
                kk = 0
                for j, segs in enumerate(segment_lists):
                    endpoint_offsets[j] = int(kk)
                    o = origins[i0 + j]
                    dts: list[float] = []
                    for seg in segs:
                        ta = float(seg.t_enter)
                        tb = float(seg.t_exit)
                        dt = tb - ta
                        if dt <= 0.0:
                            continue
                        query_xyz[kk] = o + ta * d
                        kk += 1
                        query_xyz[kk] = o + tb * d
                        kk += 1
                        dts.append(float(dt))
                    seg_dt_list.append(np.asarray(dts, dtype=float))
                endpoint_offsets[n_chunk] = int(kk)
                query_xyz = query_xyz[:kk]
            if endpoint_offsets[n_chunk] <= 0:
                if out_2d is None:
                    continue
                continue
            if use_xyz_kernel:
                query_xyz = np.asarray(query_points, dtype=float)

            values, cell_ids = self.interpolator(
                query_xyz,
                query_space="xyz",
                return_cell_ids=True,
            )
            value_arr = np.asarray(values, dtype=float)
            if value_arr.ndim == 1:
                value_arr_2d = value_arr.reshape(-1, 1)
                scalar_fields = True
            else:
                value_arr_2d = value_arr
                scalar_fields = False
            cids = np.asarray(cell_ids, dtype=np.int64).reshape(-1)

            if out_2d is None:
                out_2d = np.full((n_rays, value_arr_2d.shape[1]), np.nan, dtype=float)

            for j in range(n_chunk):
                s = int(endpoint_offsets[j])
                e = int(endpoint_offsets[j + 1])
                if e <= s:
                    continue
                if np.any(cids[s:e] < 0):
                    continue
                v = value_arr_2d[s:e]
                col = np.zeros(v.shape[1], dtype=float)
                dts = seg_dt_list[j]
                for si, dt in enumerate(dts):
                    col += 0.5 * (v[2 * si] + v[2 * si + 1]) * float(dt)
                out_2d[i0 + j] = float(scale) * col

        if out_2d is None:
            # All rays missed the domain. Preserve scalar/vector behavior.
            return np.full((n_rays,), np.nan, dtype=float)
        if out_2d.shape[1] == 1:
            return out_2d[:, 0]
        return out_2d

    def adaptive_midpoint_rule(
        self,
        origins_xyz: np.ndarray,
        direction_xyz: np.ndarray,
        t_start: float,
        t_end: float,
        *,
        chunk_size: int = 2048,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build flattened adaptive midpoint quadrature data for many rays.

        For each crossed leaf segment `[t_enter, t_exit]`, this emits:
        - midpoint query point `x(t_mid)`,
        - weight `dt = t_exit - t_enter`,
        and returns `ray_offsets` so ray `i` uses samples in
        `[ray_offsets[i], ray_offsets[i+1])`.
        """
        origins = np.asarray(origins_xyz, dtype=float)
        if origins.ndim == 1:
            if origins.shape[0] != 3:
                raise ValueError("origins_xyz must have shape (n_rays, 3) or (3,).")
            origins = origins.reshape(1, 3)
        if origins.ndim != 2 or origins.shape[1] != 3:
            raise ValueError("origins_xyz must have shape (n_rays, 3) or (3,).")
        if int(chunk_size) <= 0:
            raise ValueError("chunk_size must be positive.")

        d = _normalize_direction(direction_xyz)
        t0 = float(t_start)
        t1 = float(t_end)
        if t1 <= t0:
            raise ValueError("t_end must be greater than t_start.")

        n_rays = int(origins.shape[0])
        ray_offsets = np.zeros(n_rays + 1, dtype=np.int64)
        midpoints_list: list[tuple[float, float, float]] = []
        weights_list: list[float] = []
        global_count = 0
        use_xyz_kernel = bool(str(self.tree.coord_system) == "xyz" and hasattr(self.tree.lookup, "_lookup_state"))
        trace_max_steps = 16384
        trace_boundary_tol = 1e-9

        for i0 in range(0, n_rays, int(chunk_size)):
            i1 = min(n_rays, i0 + int(chunk_size))
            n_chunk = i1 - i0
            if use_xyz_kernel:
                lookup_state = self.tree.lookup._lookup_state
                for j in range(n_chunk):
                    ray_offsets[i0 + j] = int(global_count)
                    o = origins[i0 + j]
                    n_seg, cids, enters, exits = _trace_segments_xyz_kernel(
                        o,
                        d,
                        t0,
                        t1,
                        int(trace_max_steps),
                        0,
                        float(trace_boundary_tol),
                        lookup_state,
                    )
                    for si in range(int(n_seg)):
                        if int(cids[si]) < 0:
                            continue
                        ta = float(enters[si])
                        tb = float(exits[si])
                        dt = tb - ta
                        if dt <= 0.0:
                            continue
                        tm = 0.5 * (ta + tb)
                        pm = o + tm * d
                        midpoints_list.append((float(pm[0]), float(pm[1]), float(pm[2])))
                        weights_list.append(float(dt))
                        global_count += 1
            else:
                for j in range(n_chunk):
                    ray_offsets[i0 + j] = int(global_count)
                    o = origins[i0 + j]
                    segs = self.ray_tracer.trace_prepared(o, d, t0, t1)
                    for seg in segs:
                        ta = float(seg.t_enter)
                        tb = float(seg.t_exit)
                        dt = tb - ta
                        if dt <= 0.0:
                            continue
                        tm = 0.5 * (ta + tb)
                        pm = o + tm * d
                        midpoints_list.append((float(pm[0]), float(pm[1]), float(pm[2])))
                        weights_list.append(float(dt))
                        global_count += 1

        ray_offsets[n_rays] = int(global_count)
        if global_count == 0:
            return (
                np.empty((0, 3), dtype=float),
                np.empty((0,), dtype=float),
                ray_offsets,
            )

        midpoints_xyz = np.asarray(midpoints_list, dtype=float)
        weights = np.asarray(weights_list, dtype=float)
        return midpoints_xyz, weights, ray_offsets

    def integrate_midpoint_rule(
        self,
        midpoints_xyz: np.ndarray,
        weights: np.ndarray,
        ray_offsets: np.ndarray,
        *,
        scale: float = 1.0,
    ) -> np.ndarray:
        """Integrate the interpolator field using flattened midpoint quadrature."""
        mids = np.asarray(midpoints_xyz, dtype=float)
        w = np.asarray(weights, dtype=float).reshape(-1)
        offsets = np.asarray(ray_offsets, dtype=np.int64).reshape(-1)
        if mids.ndim != 2 or mids.shape[1] != 3:
            raise ValueError("midpoints_xyz must have shape (n_samples, 3).")
        if mids.shape[0] != w.shape[0]:
            raise ValueError("weights length must equal number of midpoint samples.")
        if offsets.size < 1:
            raise ValueError("ray_offsets must have length >= 1.")
        if offsets[0] != 0:
            raise ValueError("ray_offsets must start at 0.")
        if offsets[-1] != mids.shape[0]:
            raise ValueError("ray_offsets must end at n_samples.")
        if np.any(np.diff(offsets) < 0):
            raise ValueError("ray_offsets must be non-decreasing.")

        n_rays = int(offsets.size - 1)
        if mids.shape[0] == 0:
            return np.full((n_rays,), np.nan, dtype=float)

        values, cell_ids = self.interpolator(
            mids,
            query_space="xyz",
            return_cell_ids=True,
        )
        value_arr = np.asarray(values, dtype=float)
        if value_arr.ndim == 1:
            value_arr_2d = value_arr.reshape(-1, 1)
        else:
            value_arr_2d = value_arr
        cids = np.asarray(cell_ids, dtype=np.int64).reshape(-1)

        out_2d = np.full((n_rays, value_arr_2d.shape[1]), np.nan, dtype=float)
        for i in range(n_rays):
            s = int(offsets[i])
            e = int(offsets[i + 1])
            if e <= s:
                continue
            if np.any(cids[s:e] < 0):
                continue
            seg_vals = value_arr_2d[s:e]
            seg_w = w[s:e].reshape(-1, 1)
            out_2d[i] = float(scale) * np.sum(seg_vals * seg_w, axis=0)

        if out_2d.shape[1] == 1:
            return out_2d[:, 0]
        return out_2d

    def integrate_field_along_rays_midpoint(
        self,
        origins_xyz: np.ndarray,
        direction_xyz: np.ndarray,
        t_start: float,
        t_end: float,
        *,
        chunk_size: int = 2048,
        scale: float = 1.0,
    ) -> np.ndarray:
        """One-shot adaptive midpoint integration on many rays."""
        origins = np.asarray(origins_xyz, dtype=float)
        if origins.ndim == 1:
            if origins.shape[0] != 3:
                raise ValueError("origins_xyz must have shape (n_rays, 3) or (3,).")
            origins = origins.reshape(1, 3)
        if origins.ndim != 2 or origins.shape[1] != 3:
            raise ValueError("origins_xyz must have shape (n_rays, 3) or (3,).")

        d = _normalize_direction(direction_xyz)
        t0 = float(t_start)
        t1 = float(t_end)
        if t1 <= t0:
            raise ValueError("t_end must be greater than t_start.")

        # Specialized midpoint path: Cartesian scalar field, axis-aligned rays.
        abs_d = np.abs(d)
        axis = int(np.argmax(abs_d))
        axis_tol = 1e-12
        is_axis_aligned = bool(
            abs_d[axis] >= (1.0 - axis_tol)
            and abs_d[(axis + 1) % 3] <= axis_tol
            and abs_d[(axis + 2) % 3] <= axis_tol
        )
        if (
            is_axis_aligned
            and str(self.tree.coord_system) == "xyz"
            and int(getattr(self.interpolator, "_n_value_components", 0)) == 1
            and hasattr(self.interpolator, "_interp_state_xyz")
            and hasattr(self.tree.lookup, "_lookup_state")
        ):
            return _integrate_axis_aligned_xyz_scalar_midpoint_kernel(
                origins,
                d,
                t0,
                t1,
                int(axis),
                float(scale),
                200000,
                1e-9,
                self.tree.lookup._lookup_state,
                self.interpolator._interp_state_xyz,
            )

        if (
            str(self.tree.coord_system) == "xyz"
            and int(getattr(self.interpolator, "_n_value_components", 0)) == 1
            and hasattr(self.interpolator, "_interp_state_xyz")
            and hasattr(self.tree.lookup, "_lookup_state")
        ):
            return _integrate_xyz_scalar_midpoint_kernel(
                origins,
                d,
                t0,
                t1,
                float(scale),
                200000,
                1e-9,
                self.tree.lookup._lookup_state,
                self.interpolator._interp_state_xyz,
            )

        mids, weights, offsets = self.adaptive_midpoint_rule(
            origins,
            d,
            t0,
            t1,
            chunk_size=chunk_size,
        )
        return self.integrate_midpoint_rule(mids, weights, offsets, scale=scale)

    def linear_pieces_axis_aligned(
        self,
        origin_xyz: np.ndarray,
        direction_xyz_unit: np.ndarray,
        segments: list[RaySegment],
    ) -> list[RayLinearPiece] | None:
        """Build cellwise linear pieces quickly for axis-aligned Cartesian rays.

        On Cartesian trees with trilinear interpolation, an axis-aligned ray
        is linear within each cell segment, so endpoint values define exact
        `f(t)=m*t+b` pieces without tet subdivision.
        """
        if str(self.tree.coord_system) != "xyz":
            return None
        if not segments:
            return []

        abs_d = np.abs(direction_xyz_unit)
        axis = int(np.argmax(abs_d))
        axis_tol = 1e-12
        if abs_d[axis] < (1.0 - axis_tol):
            return None
        if abs_d[(axis + 1) % 3] > axis_tol or abs_d[(axis + 2) % 3] > axis_tol:
            return None

        n_seg = len(segments)
        t_bounds = np.empty(2 * n_seg, dtype=float)
        for i, seg in enumerate(segments):
            t_bounds[2 * i] = float(seg.t_enter)
            t_bounds[2 * i + 1] = float(seg.t_exit)
        query_xyz = origin_xyz[None, :] + t_bounds[:, None] * direction_xyz_unit[None, :]

        values, cell_ids = self.interpolator(
            query_xyz,
            query_space="xyz",
            return_cell_ids=True,
        )
        cid_arr = np.asarray(cell_ids, dtype=np.int64).reshape(-1)
        if np.any(cid_arr < 0):
            return None
        value_arr = np.asarray(values, dtype=float)
        if value_arr.ndim == 1:
            value_arr = value_arr.reshape(-1, 1)
            scalar_field = True
        else:
            scalar_field = False

        out: list[RayLinearPiece] = []
        for i, seg in enumerate(segments):
            t0 = float(seg.t_enter)
            t1 = float(seg.t_exit)
            dt = t1 - t0
            if dt <= 0.0:
                continue
            v0 = value_arr[2 * i]
            v1 = value_arr[2 * i + 1]
            slope = (v1 - v0) / dt
            intercept = v0 - slope * t0
            if scalar_field:
                slope_out: np.ndarray | float = float(slope[0])
                intercept_out: np.ndarray | float = float(intercept[0])
            else:
                slope_out = slope.copy()
                intercept_out = intercept.copy()
            out.append(
                RayLinearPiece(
                    t_start=t0,
                    t_end=t1,
                    cell_id=int(seg.cell_id),
                    tet_id=-1,
                    slope=slope_out,
                    intercept=intercept_out,
                )
            )
        return out

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

        segments = self.ray_tracer.trace_prepared(o, d, float(t_start), float(t_end))
        pieces = self.linear_pieces_axis_aligned(o, d, segments)
        if pieces is None:
            pieces = []
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
