from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from scipy.spatial import cKDTree

from batread.dataset import Dataset

RESAMPLE_METHODS = ("nearest", "linear", "octree")


def _get_spatial_cache(smart_ds, coordinate_fields):
    spatial_cache = smart_ds._resample_spatial_cache.get(coordinate_fields)
    if spatial_cache is None:
        source_coords = np.column_stack(
            [np.asarray(smart_ds[name]).ravel() for name in coordinate_fields]
        )
        coord_mask = np.isfinite(source_coords).all(axis=1)
        if not np.any(coord_mask):
            raise ValueError("No finite source coordinates available for resampling")
        spatial_cache = {
            "source_coords": source_coords,
            "nearest_tree": None,
            "linear_triangulation": None,
        }
        smart_ds._resample_spatial_cache[coordinate_fields] = spatial_cache
    return spatial_cache


def _interpolate_field(
    source_coords,
    coord_mask,
    spatial_cache,
    values,
    flat_sample_points,
    *,
    method: str,
    fill_value: float,
    nearest_indices,
):
    valid = coord_mask & np.isfinite(values)
    if not np.any(valid):
        return None, nearest_indices

    if method == "nearest":
        if np.array_equal(valid, coord_mask):
            nearest_tree = spatial_cache["nearest_tree"]
            if nearest_tree is None:
                nearest_tree = cKDTree(source_coords[coord_mask])
                spatial_cache["nearest_tree"] = nearest_tree
            if nearest_indices is None:
                nearest_indices = nearest_tree.query(flat_sample_points)[1]
            return values[coord_mask][nearest_indices], nearest_indices

        nearest_tree = cKDTree(source_coords[valid])
        nearest_indices = nearest_tree.query(flat_sample_points)[1]
        return values[valid][nearest_indices], nearest_indices

    if method == "linear":
        if np.array_equal(valid, coord_mask):
            linear_triangulation = spatial_cache["linear_triangulation"]
            if linear_triangulation is None:
                linear_triangulation = Delaunay(source_coords[coord_mask])
                spatial_cache["linear_triangulation"] = linear_triangulation
            interpolator = LinearNDInterpolator(
                linear_triangulation,
                values[coord_mask],
                fill_value=fill_value,
            )
        else:
            interpolator = LinearNDInterpolator(
                source_coords[valid],
                values[valid],
                fill_value=fill_value,
            )
        out = np.asarray(interpolator(flat_sample_points), dtype=float)
        if out.ndim == 0:
            out = out[np.newaxis]
        return out, nearest_indices

    if method == "octree":
        raise NotImplementedError("method='octree' is not implemented")

    raise ValueError(f"method must be one of {RESAMPLE_METHODS!r}")


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

    if fields is None:
        output_variables = list(smart_ds._dataset.variables)
    else:
        # Always keep the coordinate columns in the output dataset, then append the
        # requested fields without duplication.
        output_variables = list(coordinate_fields)
        for name in fields:
            if name not in output_variables:
                output_variables.append(name)

    # Reuse coordinate-dependent spatial structures across resample calls with the
    # same coordinate field choice.
    spatial_cache = _get_spatial_cache(smart_ds, coordinate_fields)
    source_coords = spatial_cache["source_coords"]
    coord_mask = np.isfinite(source_coords).all(axis=1)
    if not np.any(coord_mask):
        raise ValueError("No finite source coordinates available for resampling")

    out_points = np.full((flat_sample_points.shape[0], len(output_variables)), np.nan, dtype=float)
    out_index = {name: i for i, name in enumerate(output_variables)}

    for dim, coord_name in enumerate(coordinate_fields):
        if coord_name in out_index:
            out_points[:, out_index[coord_name]] = flat_sample_points[:, dim]

    nearest_indices = None

    # Interpolate each non-coordinate field onto the target points. 
    # Cached spatial structures are used for efficiency.
    for name in output_variables:
        if name in coordinate_fields:
            continue

        values = np.asarray(smart_ds[name]).ravel()
        if values.shape[0] != source_coords.shape[0]:
            raise ValueError(
                f"Field '{name}' has length {values.shape[0]} but coordinates have "
                f"length {source_coords.shape[0]}"
            )
        out, nearest_indices = _interpolate_field(
            source_coords,
            coord_mask,
            spatial_cache,
            values,
            flat_sample_points,
            method=method,
            fill_value=fill_value,
            nearest_indices=nearest_indices,
        )
        if out is None:
            continue
        out_points[:, out_index[name]] = out

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

    return type(smart_ds)(
        new_dataset,
        cache_enabled=smart_ds._cache_enabled,
        computation_graph=smart_ds._computation_graph,
    )


__all__ = ["resample_smart_ds"]
