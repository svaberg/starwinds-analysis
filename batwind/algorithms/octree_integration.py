from __future__ import annotations

import numpy as np

from batcamp.octree import Octree


def compute_octree_leaf_centers_and_volumes(tree: Octree, length_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Return leaf-cell centers and leaf-cell volumes in one explicit length unit.
    """
    leaf_count = int(np.asarray(tree.corners).shape[0])
    bounds = np.asarray(tree.cell_bounds, dtype=float)[:leaf_count]
    starts = bounds[:, :, 0]
    widths = bounds[:, :, 1]
    centers = starts + 0.5 * widths
    volumes = np.prod(widths, axis=1) * float(length_scale) ** 3
    return centers, volumes


def compute_octree_leaf_geometry(tree: Octree, body_radius_cm: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Return leaf-cell center radii in `R_*` and leaf-cell volumes in `cm^3`.

    The returned radius is the Euclidean radius of the leaf-cell center.
    Downstream radius-limited integrals therefore use a leaf-center inclusion
    rule: a cell is treated as inside a cutoff sphere when its center radius is
    below that cutoff. This is octree-aware, but not an exact partial-cell
    clipping against a sphere.
    """
    centers_r, cell_volume_cm3 = compute_octree_leaf_centers_and_volumes(tree, length_scale=body_radius_cm)
    radial_distance_r = np.linalg.norm(centers_r, axis=1)
    return radial_distance_r, cell_volume_cm3


def leaf_point_mean(tree: Octree, point_values: np.ndarray) -> np.ndarray:
    """
    Average one point-valued scalar field onto octree leaf cells.
    """
    return np.mean(np.asarray(point_values, dtype=float)[np.asarray(tree.corners, dtype=int)], axis=1)


def integrate_leaf_scalar(tree: Octree, point_values: np.ndarray, *, length_scale: float = 1.0) -> float:
    """
    Integrate one point-valued scalar field over octree leaf-cell volume.

    The scalar is averaged to each leaf from its corner point values and then
    multiplied by the full leaf volume.
    """
    _, leaf_volumes = compute_octree_leaf_centers_and_volumes(tree, length_scale=length_scale)
    leaf_values = leaf_point_mean(tree, point_values)
    return float(np.sum(leaf_values * leaf_volumes))


def integrate_leaf_mean_field_with_cell_weight(
    tree: Octree,
    point_values: np.ndarray,
    cell_weight: np.ndarray,
    *,
    length_scale: float = 1.0,
) -> float:
    """
    Integrate one point-valued field times one leaf-defined weight over octree volume.

    The point-valued field is first averaged to each leaf from its corner values.
    The returned scalar is then

        sum_leaf(leaf_mean_value * cell_weight * leaf_volume).

    This is the direct octree-aware shape needed for quantities such as
    ``\\int omega(r) epsilon dV``, where ``epsilon`` is point-valued on the
    octree corners, ``omega`` is naturally defined once per leaf, and ``dV`` is
    the leaf volume in one explicit length unit.
    """
    _, leaf_volumes = compute_octree_leaf_centers_and_volumes(tree, length_scale=length_scale)
    leaf_values = leaf_point_mean(tree, point_values)
    return float(np.sum(leaf_values * np.asarray(cell_weight, dtype=float) * leaf_volumes))


def cumulative_radius(radial_distance_r: np.ndarray, cell_emission: np.ndarray, fraction: float) -> float:
    """
    Return the radius containing the requested cumulative emission fraction.

    This uses one representative radius per leaf cell. The cumulative profile is
    therefore built by sorting leaf cells by center radius and summing their full
    cell emission in that order.
    """
    order = np.argsort(radial_distance_r)
    sorted_radius = np.asarray(radial_distance_r, dtype=float)[order]
    sorted_emission = np.asarray(cell_emission, dtype=float)[order]
    cumulative = np.cumsum(sorted_emission)
    total = float(cumulative[-1])
    if total <= 0.0:
        return float("nan")
    cumulative_fraction = cumulative / total
    return float(np.interp(float(fraction), cumulative_fraction, sorted_radius))


def radial_emission_profile(
    radial_distance_r: np.ndarray,
    cell_emission: np.ndarray,
    *,
    n_bins: int = 192,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin integrated cell emission radially.

    Each leaf cell contributes its full emission to the radial bin selected by
    its center radius. Cells intersecting a bin edge are not fractionally split.
    """
    r_min = max(1.0, float(np.nanmin(radial_distance_r[radial_distance_r > 0.0])))
    r_max = float(np.nanmax(radial_distance_r))
    radial_edges = np.geomspace(r_min, r_max, n_bins + 1)
    shell_emission = np.histogram(radial_distance_r, bins=radial_edges, weights=cell_emission)[0]
    radial_centers = np.sqrt(radial_edges[:-1] * radial_edges[1:])
    cumulative = np.cumsum(shell_emission)
    cumulative_fraction = cumulative / cumulative[-1] if cumulative.size and cumulative[-1] > 0.0 else cumulative
    return radial_centers, shell_emission, cumulative_fraction
