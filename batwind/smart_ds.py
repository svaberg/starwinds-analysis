from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence

import griblet
import numpy as np

from batread.dataset import Dataset
from batwind.data.field_names import DEFAULT_XYZ_NAMES
from batwind.param_in import stellar_aux_from_nearby_param_in
from griblet.dependency_solver import UnresolvableFieldError
from griblet.evaluate_tree import evaluate_tree
from batwind._smart_ds_resample import resample_smart_ds
from batwind.recipes.batsrus import build_batsrus_graph
from batwind.recipes.spherical import build_spherical_graph


class SmartDs:
    """
    Lightweight wrapper around ``batread.Dataset`` with graph-backed derived fields.
    """

    DEFAULT_COORD_FIELDS = DEFAULT_XYZ_NAMES

    def __init__(
        self,
        dataset: Dataset,
        *,
        cache_enabled: bool = True,
        computation_graph: griblet.ComputationGraph | None = None,
    ) -> None:
        self._dataset = dataset
        self._cache_enabled = bool(cache_enabled)
        self._cache: dict[str, np.ndarray] = {}
        self._resample_spatial_cache: dict[tuple[str, ...], dict[str, object]] = {}
        self._computation_graph = griblet.ComputationGraph()
        self.clear_computation_graph()
        if computation_graph is not None:
            self.merge_computation_graph(computation_graph)

    @classmethod
    def from_file(
        cls,
        file: str,
        *,
        batsrus: bool = True,
        spherical: bool = True,
        body_radius_m: float | None = None,
        **kwargs,
    ) -> "SmartDs":
        raw = Dataset.from_file(str(file))
        stellar_aux = stellar_aux_from_nearby_param_in(file)
        if stellar_aux:
            raw = Dataset(
                raw.points,
                raw.corners,
                dict(raw.aux) | dict(stellar_aux),
                raw.title,
                raw.variables,
                raw.zone,
            )
        if body_radius_m is None:
            radius_from_param = raw.aux.get("Star_radius_m")
            if radius_from_param is not None:
                body_radius_m = float(radius_from_param)
        sds = cls(raw, **kwargs)
        if batsrus:
            sds.computation_graph.merge(
                build_batsrus_graph(sds.raw.variables, gamma=sds.raw.aux.get("GAMMA"), body_radius_m=body_radius_m)
            )
        if spherical:
            sds.computation_graph.merge(build_spherical_graph(tuple(sds)))
        return sds

    @property
    def raw(self) -> Dataset:
        return self._dataset

    def __repr__(self) -> str:
        return (
            f"SmartDs(title={self._dataset.title!r}, zone={self._dataset.zone!r}, "
            f"points={np.shape(self._dataset.points)}, variables={len(self._dataset.variables)})"
        )

    def __str__(self) -> str:
        return "\n".join(
            (
                "SmartDs",
                f"  Title: {self._dataset.title}",
                f"  Zone : {self._dataset.zone}",
                f"  Points: {np.shape(self._dataset.points)}",
                f"  Variables: {len(self._dataset.variables)}",
            )
        )

    @property
    def computation_graph(self):
        return self._computation_graph

    def __iter__(self):
        names = list(self._dataset.variables)
        for name in self._computation_graph.list_fields():
            if name not in names:
                names.append(name)
        return iter(names)

    def __getitem__(self, index_or_name):
        if not isinstance(index_or_name, str):
            return self._dataset[index_or_name]

        name = index_or_name
        if self._cache_enabled and name in self._cache:
            return self._cache[name]

        if name in self._dataset.variables:
            value = self._dataset[name]
            if self._cache_enabled:
                self._cache[name] = value
            return value

        solver = griblet.DependencySolver(self._computation_graph)
        try:
            cost, tree = solver.resolve_field(name)
        except UnresolvableFieldError as e:
            raise IndexError(
                f"Field '{name}' not available. Raw fields: {self._dataset.variables}."
            ) from e
        if not np.isfinite(cost):
            raise IndexError(
                f"Field '{name}' not available. Raw fields: {self._dataset.variables}."
            )
        value = evaluate_tree(tree, self._computation_graph)
        if self._cache_enabled:
            self._cache[name] = value
        return value

    def clear_computation_graph(self):
        self._computation_graph = griblet.ComputationGraph()
        for raw_name in self._dataset.variables:
            self._computation_graph.add_recipe(
                field=raw_name,
                func=lambda raw_name=raw_name: self._dataset[raw_name],
                deps=[],
                cost=0.0,
                metadata={"description": "Dataset raw field", "loader": True},
            )
        for key, value in self._dataset.aux.items():
            self._computation_graph.add_recipe(
                field=key,
                func=lambda value=value: value,
                deps=[],
                cost=0.0,
                metadata={"description": "Dataset aux", "loader": True},
            )
        return self

    def merge_computation_graph(self, graph):
        # Loader recipes close over this SmartDs' dataset, so we must not blindly
        # merge them forward into another SmartDs that may wrap different raw data.
        for field, recipes in graph.recipes.items():
            for recipe in recipes:
                metadata = dict(recipe.get("metadata", {}) or {})
                if metadata.get("loader"):
                    continue
                self._computation_graph.add_recipe(
                    field=field,
                    func=recipe["func"],
                    deps=recipe["deps"],
                    cost=recipe["cost"],
                    metadata=metadata,
                )
        return self

    def clear_cache(self, *names: str) -> None:
        if not names:
            self._cache.clear()
            self._resample_spatial_cache.clear()
            return
        for name in names:
            self._cache.pop(name, None)

    def source_fields(self, fields: Sequence[str]) -> tuple[str, ...]:
        base_fields: list[str] = []
        solver = griblet.DependencySolver(self._computation_graph)

        for field in tuple(dict.fromkeys(fields)):
            if field in self._dataset.variables:
                base_fields.append(field)
                continue
            try:
                _cost, tree = solver.resolve_field(field)
            except UnresolvableFieldError:
                if field not in base_fields:
                    base_fields.append(field)
                continue
            stack = [tree]
            leaves: list[str] = []
            while stack:
                node = stack.pop()
                if node.deps:
                    stack.extend(reversed(node.deps))
                    continue
                if node.field in self._dataset.variables and node.field not in leaves:
                    leaves.append(node.field)
            for name in leaves or [field]:
                if name not in base_fields:
                    base_fields.append(name)

        return tuple(base_fields)

    def resample(
        self,
        sample_points,
        *,
        coordinate_fields: Sequence[str] | None = None,
        fields: Sequence[str] | None = None,
        # Method guidance:
        # - `octree` is the intended default for 3D resampling.
        # - `linear` is the intended default for 2D resampling and structured datasets.
        # - `nearest` is mainly for exposing the underlying grid resolution.
        method: str = "nearest",
        fill_value: float = np.nan,
        corners=None,
        copy_aux: bool = True,
        title: str | None = None,
        zone: str | None = None,
    ) -> "SmartDs":
        if coordinate_fields is None:
            preferred = DEFAULT_XYZ_NAMES
            ndim = np.asarray(sample_points, dtype=float).shape[-1]
            coordinate_fields = tuple(name for name in preferred if name in self._dataset.variables)
            if len(coordinate_fields) < ndim:
                raise ValueError(
                    "Could not infer coordinate fields. Pass coordinate_fields explicitly."
                )
            coordinate_fields = coordinate_fields[:ndim]
        return resample_smart_ds(
            self,
            sample_points,
            coordinate_fields=coordinate_fields,
            fields=fields,
            method=method,
            fill_value=fill_value,
            corners=corners,
            copy_aux=copy_aux,
            title=title,
            zone=zone,
        )

    def append_fields(
        self,
        extra_fields: Mapping[str, np.ndarray],
        *,
        zone_suffix: str = "derived fields",
    ) -> "SmartDs":
        if not extra_fields:
            return self

        base_points = np.asarray(self.raw.points)
        if base_points.ndim < 2:
            raise ValueError("Expected raw points to have shape (..., nvars)")
        base_shape = base_points.shape[:-1]

        arrays = []
        names = []
        for name, values in extra_fields.items():
            arr = np.asarray(values)
            if arr.shape != base_shape:
                raise ValueError(
                    f"Extra field '{name}' shape {arr.shape} does not match dataset grid shape {base_shape}"
                )
            arrays.append(arr[..., None])
            names.append(name)

        new_points = np.concatenate([base_points, *arrays], axis=-1)
        new_dataset = Dataset(
            new_points,
            self.raw.corners,
            self.raw.aux,
            self.raw.title,
            list(self.raw.variables) + names,
            f"{self.raw.zone} ({zone_suffix})",
        )
        return type(self)(
            new_dataset,
            cache_enabled=self._cache_enabled,
            computation_graph=self._computation_graph,
        )


__all__ = ["SmartDs"]
