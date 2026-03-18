from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence

import griblet
import numpy as np

from batread.dataset import Dataset
from griblet.dependency_solver import UnresolvableFieldError
from batwind._smart_ds_resample import resample_smart_ds


class SmartDs:
    """
    Lightweight wrapper around ``batread.Dataset`` with graph-backed derived fields.
    """

    DEFAULT_COORD_FIELDS = ("X [R]", "Y [R]", "Z [R]")

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
        self._computation_graph = griblet.ComputationGraph()
        self._next_recipe_id = 0
        self.clear_computation_graph()
        if computation_graph is not None:
            self.merge_computation_graph(computation_graph)

    @classmethod
    def from_file(cls, file: str, **kwargs) -> "SmartDs":
        return cls(Dataset.from_file(str(file)), **kwargs)

    @property
    def raw(self) -> Dataset:
        return self._dataset

    def __repr__(self) -> str:
        return (
            f"SmartDs(title={self.title!r}, zone={self.zone!r}, "
            f"points={np.shape(self.points)}, variables={len(self.variables)})"
        )

    def __str__(self) -> str:
        return "\n".join(
            (
                "SmartDs",
                f"  Title: {self.title}",
                f"  Zone : {self.zone}",
                f"  Points: {np.shape(self.points)}",
                f"  Variables: {len(self.variables)}",
            )
        )

    @property
    def aux(self):
        return self._dataset.aux

    @property
    def title(self):
        return self._dataset.title

    @property
    def zone(self):
        return self._dataset.zone

    @property
    def points(self):
        return self._dataset.points

    @property
    def corners(self):
        return self._dataset.corners

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(self._dataset.variables)

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
        value = self._evaluate_resolved_tree(tree)
        if self._cache_enabled:
            self._cache[name] = value
        return value

    def clear_computation_graph(self):
        self._computation_graph = griblet.ComputationGraph()
        self._next_recipe_id = 0
        for raw_name in self._dataset.variables:
            self._computation_graph.add_recipe(
                field=raw_name,
                func=lambda raw_name=raw_name: self._dataset[raw_name],
                deps=[],
                cost=0.0,
                metadata={
                    "description": "Dataset raw field",
                    "loader": True,
                    "recipe_id": self._next_recipe_id,
                },
            )
            self._next_recipe_id += 1
        for key, value in self._dataset.aux.items():
            self._computation_graph.add_recipe(
                field=key,
                func=lambda value=value: value,
                deps=[],
                cost=0.0,
                metadata={
                    "description": "Dataset aux",
                    "loader": True,
                    "recipe_id": self._next_recipe_id,
                },
            )
            self._next_recipe_id += 1
        return self

    def merge_computation_graph(self, graph):
        for field, recipes in graph.recipes.items():
            for recipe in recipes:
                metadata = dict(recipe.get("metadata", {}) or {})
                if metadata.get("loader"):
                    continue
                metadata["recipe_id"] = self._next_recipe_id
                self._next_recipe_id += 1
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
            return
        for name in names:
            self._cache.pop(name, None)

    def base_fields_for_resample(self, fields: Sequence[str]) -> tuple[str, ...]:
        base_fields: list[str] = []
        solver = griblet.DependencySolver(self._computation_graph)

        def add(name: str) -> None:
            if name not in base_fields:
                base_fields.append(name)

        def raw_leaves(node) -> list[str]:
            deps = list(getattr(node, "deps", []) or [])
            if not deps:
                field = getattr(node, "field", None)
                if isinstance(field, str) and field in self._dataset.variables:
                    return [field]
                return []
            leaves: list[str] = []
            for dep in deps:
                for leaf in raw_leaves(dep):
                    if leaf not in leaves:
                        leaves.append(leaf)
            return leaves

        for field in tuple(dict.fromkeys(fields)):
            if field in self._dataset.variables:
                add(field)
                continue
            try:
                _cost, tree = solver.resolve_field(field)
            except (IndexError, KeyError, RuntimeError, ValueError, UnresolvableFieldError):
                add(field)
                continue
            leaves = raw_leaves(tree)
            if leaves:
                for leaf in leaves:
                    add(leaf)
            else:
                add(field)

        return tuple(base_fields)
    def resample(
        self,
        sample_points,
        *,
        coordinate_fields: Sequence[str] | None = None,
        fields: Sequence[str] | None = None,
        method: str = "nearest",
        fill_value: float = np.nan,
        corners=None,
        copy_aux: bool = True,
        title: str | None = None,
        zone: str | None = None,
    ) -> "SmartDs":
        if coordinate_fields is None:
            preferred = ["X [R]", "Y [R]", "Z [R]"]
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

    def _evaluate_resolved_tree(self, node):
        values = [self._evaluate_resolved_tree(dep) for dep in node.deps]
        recipe_id = node.recipe_metadata["recipe_id"]
        recipe = next(
            recipe
            for recipe in self._computation_graph.recipes[node.field]
            if recipe["metadata"]["recipe_id"] == recipe_id
        )
        return recipe["func"](*values)


__all__ = ["SmartDs"]
