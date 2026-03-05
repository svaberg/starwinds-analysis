"""SmartDs resampling internals.
"""

# It owns interpolation/resample implementation details and construction of new wrapped datasets.
# It should not contain domain-specific shell/orbit/slice analysis logic.


from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from scipy.spatial import cKDTree

from starwinds_readplt.dataset import Dataset

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
    Used by: `starwinds_analysis/smart_ds.py`
    """
    sample_points = np.array(sample_points)
    if sample_points.ndim == 1:
        sample_points = sample_points[np.newaxis, :]
    if sample_points.ndim < 2:
        raise ValueError("sample_points must have shape (..., ndim)")
    if not np.isfinite(sample_points).all():
        raise ValueError("sample_points must be finite")

    grid_shape = tuple(sample_points.shape[:-1])
    flat_sample_points = sample_points.reshape(-1, sample_points.shape[-1])

    ndim = flat_sample_points.shape[1]
    if coordinate_fields is None:
        coordinate_fields = smart_ds._infer_coordinate_fields(ndim)
    coordinate_fields = tuple(coordinate_fields)
    if len(coordinate_fields) != ndim:
        raise ValueError(
            f"Expected {ndim} coordinate fields, got {len(coordinate_fields)}: "
            f"{coordinate_fields}"
        )

    for coord_name in coordinate_fields:
        if not smart_ds.has_field(coord_name):
            raise IndexError(f"Coordinate field '{coord_name}' not available")

    if fields is None:
        output_variables = list(smart_ds._dataset.variables)
    else:
        output_variables = list(coordinate_fields)
        for name in fields:
            if name not in output_variables:
                output_variables.append(name)

    spatial_cache = smart_ds._resample_spatial_cache.get(coordinate_fields)
    if spatial_cache is None:
        source_coords = np.column_stack(
            [np.array(smart_ds.variable(name)).ravel() for name in coordinate_fields]
        )
        if not np.isfinite(source_coords).all():
            raise ValueError("source coordinates contain non-finite values")
        spatial_cache = {
            "source_coords": source_coords,
            "nearest_tree": None,
            "linear_triangulation": None,
        }
        smart_ds._resample_spatial_cache[coordinate_fields] = spatial_cache

    out_points = np.full((flat_sample_points.shape[0], len(output_variables)), np.nan, dtype=float)
    out_index = {name: i for i, name in enumerate(output_variables)}

    for dim, coord_name in enumerate(coordinate_fields):
        if coord_name in out_index:
            out_points[:, out_index[coord_name]] = flat_sample_points[:, dim]

    nearest_indices = None

    for name in output_variables:
        if name in coordinate_fields:
            continue

        values = np.array(smart_ds.variable(name)).ravel()
        if values.shape[0] != spatial_cache["source_coords"].shape[0]:
            raise ValueError(
                f"Field '{name}' has length {values.shape[0]} but coordinates have "
                f"length {spatial_cache['source_coords'].shape[0]}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"source field '{name}' contains non-finite values")

        if method == "nearest":
            nearest_tree = spatial_cache["nearest_tree"]
            if nearest_tree is None:
                nearest_tree = cKDTree(spatial_cache["source_coords"])
                spatial_cache["nearest_tree"] = nearest_tree
            if nearest_indices is None:
                nearest_indices = nearest_tree.query(flat_sample_points)[1]
            out = values[nearest_indices]
        elif method == "linear":
            linear_triangulation = spatial_cache["linear_triangulation"]
            if linear_triangulation is None:
                linear_triangulation = Delaunay(spatial_cache["source_coords"])
                spatial_cache["linear_triangulation"] = linear_triangulation
            interpolator = LinearNDInterpolator(
                linear_triangulation,
                values,
                fill_value=fill_value,
            )
            out = interpolator(flat_sample_points)
        else:
            raise ValueError("method must be 'nearest' or 'linear'")

        out = np.array(out)
        if out.ndim == 0:
            out = out[np.newaxis]
        out_points[:, out_index[name]] = out

    out_points = out_points.reshape(*grid_shape, len(output_variables))

    if corners is None:
        corners_arr = np.empty((0, 0), dtype=int)
    else:
        corners_arr = np.array(corners)

    if copy_aux:
        aux = deepcopy(smart_ds._dataset.aux)
    else:
        aux = smart_ds._dataset.aux

    if title is None:
        title = smart_ds._dataset.title
    if zone is None:
        zone = f"{smart_ds._dataset.zone} (resampled)"

    new_dataset = Dataset(
        out_points,
        corners_arr,
        aux,
        title,
        output_variables,
        zone,
    )

    return type(smart_ds)(
        new_dataset,
        cache_enabled=smart_ds._cache_enabled,
        computation_graph=smart_ds._computation_graph,
        include_aux_in_loader=smart_ds._include_aux_in_loader,
    )
