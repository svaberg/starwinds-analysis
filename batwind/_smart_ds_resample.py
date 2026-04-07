from __future__ import annotations

from copy import deepcopy
import logging
from time import perf_counter

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from scipy.spatial import cKDTree

from batcamp import Octree
from batcamp import OctreeInterpolator
from batread import Dataset
RESAMPLE_METHODS = ("nearest", "linear", "octree")

log = logging.getLogger(__name__)


def _get_spatial_cache(smart_ds, coordinate_fields):
    spatial_cache = smart_ds._resample_spatial_cache.get(coordinate_fields)
    if spatial_cache is None:
        log.debug("_get_spatial_cache creating cache for coordinate_fields=%s", coordinate_fields)
        spatial_cache = {
            "nearest_tree": None,
            "linear_triangulation": None,
            "octree_tree": None,
        }
        smart_ds._resample_spatial_cache[coordinate_fields] = spatial_cache
    else:
        log.debug("_get_spatial_cache reusing cache for coordinate_fields=%s", coordinate_fields)
    return spatial_cache


def _interpolate_field(
    smart_ds,
    source_coords,
    coord_mask,
    coordinate_fields,
    spatial_cache,
    name,
    values,
    flat_sample_points,
    *,
    method: str,
    fill_value: float,
    nearest_indices,
):
    if method == "nearest":
        valid = coord_mask & np.isfinite(values)
        if not np.any(valid):
            log.debug("_interpolate_field nearest skipped field=%s no finite values", name)
            return None, nearest_indices
        if np.array_equal(valid, coord_mask):
            nearest_tree = spatial_cache["nearest_tree"]
            if nearest_tree is None:
                log.debug("_interpolate_field nearest building shared KD-tree field=%s", name)
                nearest_tree = cKDTree(source_coords[coord_mask])
                spatial_cache["nearest_tree"] = nearest_tree
            if nearest_indices is None:
                log.debug("_interpolate_field nearest querying shared KD-tree field=%s", name)
                nearest_indices = nearest_tree.query(flat_sample_points)[1]
            return values[coord_mask][nearest_indices], nearest_indices

        log.debug("_interpolate_field nearest building field-local KD-tree field=%s", name)
        nearest_tree = cKDTree(source_coords[valid])
        nearest_indices = nearest_tree.query(flat_sample_points)[1]
        return values[valid][nearest_indices], nearest_indices

    if method == "linear":
        valid = coord_mask & np.isfinite(values)
        if not np.any(valid):
            log.debug("_interpolate_field linear skipped field=%s no finite values", name)
            return None, nearest_indices
        if np.array_equal(valid, coord_mask):
            linear_triangulation = spatial_cache["linear_triangulation"]
            if linear_triangulation is None:
                log.debug("_interpolate_field linear building shared triangulation field=%s", name)
                linear_triangulation = Delaunay(source_coords[coord_mask])
                spatial_cache["linear_triangulation"] = linear_triangulation
            interpolator = LinearNDInterpolator(
                linear_triangulation,
                values[coord_mask],
                fill_value=fill_value,
            )
        else:
            log.debug("_interpolate_field linear using field-local triangulation field=%s", name)
            interpolator = LinearNDInterpolator(
                source_coords[valid],
                values[valid],
                fill_value=fill_value,
            )
        out = np.asarray(interpolator(flat_sample_points), dtype=float)
        if out.ndim == 0:
            out = out[np.newaxis]
        return out, nearest_indices

    raise ValueError(f"method must be one of {RESAMPLE_METHODS!r}")


def _interpolate_fields(
    smart_ds,
    source_coords,
    coord_mask,
    coordinate_fields,
    output_variables,
    flat_sample_points,
    *,
    method: str,
    fill_value: float,
    spatial_cache,
):
    value_names = [name for name in output_variables if name not in coordinate_fields]
    if not value_names:
        log.debug("_interpolate_fields no value fields beyond coordinates")
        return value_names, np.empty((flat_sample_points.shape[0], 0), dtype=float)

    stage_start = perf_counter()
    log.debug(
        "_interpolate_fields method=%s coordinate_fields=%s value_fields=%d",
        method,
        coordinate_fields,
        len(value_names),
    )
    if method == "octree":
        octree_tree = spatial_cache["octree_tree"]
        if octree_tree is None:
            log.debug("_interpolate_fields octree building shared tree")
            octree_tree = Octree.from_ds(smart_ds.raw)
            spatial_cache["octree_tree"] = octree_tree

        if coordinate_fields == ("X [R]", "Y [R]", "Z [R]"):
            octree_points = flat_sample_points
            query_coord = "xyz"
        elif coordinate_fields == ("R [R]", "polar [rad]", "azimuth [rad]"):
            if octree_tree.tree_coord == "rpa":
                octree_points = flat_sample_points
                query_coord = "rpa"
            else:
                radius = flat_sample_points[:, 0]
                polar = flat_sample_points[:, 1]
                azimuth = flat_sample_points[:, 2]
                sin_polar = np.sin(polar)
                octree_points = np.column_stack(
                    (
                        radius * sin_polar * np.cos(azimuth),
                        radius * sin_polar * np.sin(azimuth),
                        radius * np.cos(polar),
                    )
                )
                query_coord = "xyz"
        else:
            raise ValueError(
                "method='octree' requires coordinate_fields to be "
                "('X [R]', 'Y [R]', 'Z [R]') or "
                "('R [R]', 'polar [rad]', 'azimuth [rad]')."
            )

        value_columns = []
        for name in value_names:
            values = np.asarray(smart_ds[name]).ravel()
            if values.shape[0] != source_coords.shape[0]:
                raise ValueError(
                    f"Field '{name}' has length {values.shape[0]} but coordinates have "
                    f"length {source_coords.shape[0]}"
                )
            value_columns.append(values)
        point_values = np.column_stack(value_columns)
        interpolator = OctreeInterpolator(
            octree_tree,
            point_values,
            fill_value=fill_value,
        )
        out_values = np.asarray(
            interpolator(octree_points, query_coord=query_coord),
            dtype=float,
        )
        if out_values.ndim == 1:
            out_values = out_values[:, np.newaxis]
        log.debug(
            "_interpolate_fields complete method=%s value_fields=%d in %.2f s.",
            method,
            len(value_names),
            perf_counter() - stage_start,
        )
        return value_names, out_values

    out_values = np.full((flat_sample_points.shape[0], len(value_names)), np.nan, dtype=float)
    nearest_indices = None
    for i, name in enumerate(value_names):
        values = np.asarray(smart_ds[name]).ravel()
        if values.shape[0] != source_coords.shape[0]:
            raise ValueError(
                f"Field '{name}' has length {values.shape[0]} but coordinates have "
                f"length {source_coords.shape[0]}"
            )
        out, nearest_indices = _interpolate_field(
            smart_ds,
            source_coords,
            coord_mask,
            coordinate_fields,
            spatial_cache,
            name,
            values,
            flat_sample_points,
            method=method,
            fill_value=fill_value,
            nearest_indices=nearest_indices,
        )
        if out is None:
            continue
        out_values[:, i] = out

    log.debug(
        "_interpolate_fields complete method=%s value_fields=%d in %.2f s.",
        method,
        len(value_names),
        perf_counter() - stage_start,
    )
    return value_names, out_values


def _build_resampled_dataset(
    smart_ds,
    out_points,
    sample_shape,
    output_variables,
    *,
    corners,
    copy_aux: bool,
    title: str | None,
    zone: str | None,
):
    if corners is None:
        corners_arr = np.empty((0, 0), dtype=int)
    else:
        corners_arr = np.asarray(corners)

    if copy_aux:
        aux = deepcopy(smart_ds._dataset.aux)
    else:
        aux = smart_ds._dataset.aux

    if title is None:
        title = smart_ds._dataset.title
    if zone is None:
        zone = f"{smart_ds._dataset.zone} (resampled)"

    log.debug(
        "_build_resampled_dataset zone=%s sample_shape=%s variables=%d",
        zone,
        sample_shape,
        len(output_variables),
    )
    return Dataset(
        out_points.reshape(*sample_shape, len(output_variables)),
        corners_arr,
        aux,
        title,
        output_variables,
        zone,
    )


def resample_smart_ds(
    smart_ds,
    sample_points,
    *,
    coordinate_fields,
    fields,
    method: str,
    fill_value: float,
    corners,
    copy_aux: bool,
    title: str | None,
    zone: str | None,
):
    """
    Resample scalar fields onto new point locations and return a new wrapped dataset.
    """
    # Normalize target points to a flat (n_points, ndim) form for interpolation,
    # then reshape back to the requested grid shape at the end.
    log.info("resample_smart_ds...")
    stage_start = perf_counter()
    sample_points = np.asarray(sample_points, dtype=float)
    if sample_points.ndim == 1:
        sample_points = sample_points[np.newaxis, :]
    if sample_points.ndim < 2:
        raise ValueError("sample_points must have shape (..., ndim)")
    sample_shape = sample_points.shape[:-1]
    flat_sample_points = sample_points.reshape(-1, sample_points.shape[-1])
    ndim = sample_points.shape[-1]
    if coordinate_fields is None:
        raise ValueError("coordinate_fields must be provided")
    coordinate_fields = tuple(coordinate_fields)
    if len(coordinate_fields) != ndim:
        raise ValueError(
            f"Expected {ndim} coordinate fields, got {len(coordinate_fields)}: "
            f"{coordinate_fields}"
        )

    for coord_name in coordinate_fields:
        if coord_name not in smart_ds:
            raise IndexError(f"Coordinate field '{coord_name}' not available")

    # Method intent:
    # - `octree` is the intended 3D resampler.
    # - `linear` is the intended 2D / structured-grid resampler.
    # - `nearest` is mainly for showing the grid resolution.

    if fields is None:
        output_variables = list(smart_ds._dataset.variables)
    else:
        # Always keep the coordinate columns in the output dataset, then append the
        # requested fields without duplication.
        output_variables = list(coordinate_fields)
        for name in fields:
            if name not in output_variables:
                output_variables.append(name)
    log.debug(
        "resample_smart_ds method=%s sample_shape=%s coordinate_fields=%s output_variables=%d",
        method,
        sample_shape,
        coordinate_fields,
        len(output_variables),
    )

    # Reuse coordinate-dependent spatial structures across resample calls with the
    # same coordinate field choice.
    coord_start = perf_counter()
    source_coords = np.column_stack(
        [np.asarray(smart_ds[name]).ravel() for name in coordinate_fields]
    )
    coord_mask = np.isfinite(source_coords).all(axis=1)
    if not np.any(coord_mask):
        raise ValueError("No finite source coordinates available for resampling")
    log.debug(
        "resample_smart_ds source_coords_shape=%s finite_source_points=%d/%d assembled in %.2f s.",
        source_coords.shape,
        int(np.count_nonzero(coord_mask)),
        coord_mask.size,
        perf_counter() - coord_start,
    )
    spatial_cache = _get_spatial_cache(smart_ds, coordinate_fields)

    out_points = np.full((flat_sample_points.shape[0], len(output_variables)), np.nan, dtype=float)
    out_index = {name: i for i, name in enumerate(output_variables)}

    for dim, coord_name in enumerate(coordinate_fields):
        if coord_name in out_index:
            out_points[:, out_index[coord_name]] = flat_sample_points[:, dim]

    value_names, out_values = _interpolate_fields(
        smart_ds,
        source_coords,
        coord_mask,
        coordinate_fields,
        output_variables,
        flat_sample_points,
        method=method,
        fill_value=fill_value,
        spatial_cache=spatial_cache,
    )
    for i, name in enumerate(value_names):
        out_points[:, out_index[name]] = out_values[:, i]

    # Build the resampled raw Dataset and wrap it back into the same SmartDs type,
    # carrying forward the derived-field graph.
    new_dataset = _build_resampled_dataset(
        smart_ds,
        out_points,
        sample_shape,
        output_variables,
        corners=corners,
        copy_aux=copy_aux,
        title=title,
        zone=zone,
    )

    out = type(smart_ds)(
        new_dataset,
        cache_enabled=smart_ds._cache_enabled,
        computation_graph=smart_ds.recipe_graph,
    )
    log.debug("resample_smart_ds complete in %.2f s.", perf_counter() - stage_start)
    return out


__all__ = ["resample_smart_ds"]
