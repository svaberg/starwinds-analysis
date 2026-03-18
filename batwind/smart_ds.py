from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence

import griblet
import numpy as np

from batread.dataset import Dataset
from griblet.dependency_solver import UnresolvableFieldError
from batwind._smart_ds_resample import resample_smart_ds
from batwind.recipes.batsrus import build_griblet_batsrus_graph
from batwind.recipes.spherical import build_griblet_auto_vector_spherical_components_graph
from batwind.recipes.spherical import build_griblet_spherical_geometry_graph


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
        include_aux_in_loader: bool = True,
    ) -> None:
        self._dataset = dataset
        self._cache_enabled = bool(cache_enabled)
        self._cache: dict[str, np.ndarray] = {}
        self._computation_graph = griblet.ComputationGraph()
        if computation_graph is not None:
            self.merge_computation_graph(computation_graph)
        self._include_aux_in_loader = bool(include_aux_in_loader)

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

    def keys(self) -> tuple[str, ...]:
        names = list(self._dataset.variables)
        for name in self._computation_graph.list_fields():
            if name not in names:
                names.append(name)
        return tuple(names)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self.has_field(name)

    def __getitem__(self, index_or_name):
        if not isinstance(index_or_name, str):
            if isinstance(index_or_name, int) and hasattr(self._dataset, "variable"):
                return self._dataset.variable(self._dataset.variables[index_or_name])
            return self._dataset[index_or_name]

        name = index_or_name
        if self._cache_enabled and name in self._cache:
            return self._cache[name]

        raw_name = self._resolve_raw_name(name)
        if raw_name is not None:
            if hasattr(self._dataset, "variable"):
                value = self._dataset.variable(raw_name)
            else:
                value = self._dataset[raw_name]
            if self._cache_enabled:
                self._cache[name] = value
            return value

        value = self._compute_via_graph(name)
        if self._cache_enabled:
            self._cache[name] = value
        return value

    def has_raw_field(self, name: str) -> bool:
        return self._resolve_raw_name(name) is not None

    def has_field(self, name: str) -> bool:
        if self.has_raw_field(name):
            return True
        try:
            cost, _tree = self.resolve(name)
        except (IndexError, KeyError, RuntimeError, ValueError, UnresolvableFieldError):
            return False
        return np.isfinite(cost)

    def get(self, name: str, default=None):
        try:
            return self[name]
        except (IndexError, KeyError):
            return default

    def clear_computation_graph(self):
        self._computation_graph = griblet.ComputationGraph()
        return self

    def merge_computation_graph(self, graph):
        if not isinstance(graph, griblet.ComputationGraph):
            raise TypeError("graph must be a griblet.ComputationGraph")
        self._computation_graph.merge(graph)
        return self

    def add_spherical_graph(
        self,
        *,
        coord_fields: Sequence[str] = DEFAULT_COORD_FIELDS,
        vectors: Sequence[str] = ("B", "U"),
        components: Sequence[str] = ("r", "p", "a"),
        merge: bool = True,
    ):
        graph = build_griblet_spherical_geometry_graph(coord_fields=coord_fields)
        graph.merge(
            build_griblet_auto_vector_spherical_components_graph(
                self.keys(),
                coord_fields=coord_fields,
                prefixes=tuple(vectors),
                components=tuple(components),
            )
        )
        if not merge:
            self.clear_computation_graph()
        self.merge_computation_graph(graph)
        return self

    def add_batsrus_graph(
        self,
        *,
        body_radius_m: float | None = None,
        include_unit_normalization: bool = True,
        include_derived: bool = True,
        merge: bool = True,
    ):
        graph = build_griblet_batsrus_graph(
            self.variables,
            aux=self.aux,
            body_radius_m=body_radius_m,
            include_unit_normalization=include_unit_normalization,
            include_derived=include_derived,
        )
        if not merge:
            self.clear_computation_graph()
        self.merge_computation_graph(graph)
        return self

    def prepare(self, *, body_radius: float | None = None) -> "SmartDs":
        self.add_batsrus_graph(body_radius_m=body_radius)
        self.add_spherical_graph()
        return self

    def clear_cache(self, *names: str) -> None:
        if not names:
            self._cache.clear()
            return
        for name in names:
            self._cache.pop(name, None)

    def resolve(self, name: str):
        graph = self._build_runtime_graph()
        solver = griblet.DependencySolver(graph)
        return solver.resolve_field(name)

    def explain(self, name: str, *, return_tree: bool = False):
        cost, tree = self.resolve(name)
        if return_tree:
            return cost, tree

        lines: list[str] = []

        def walk(node, depth=0):
            meta = getattr(node, "recipe_metadata", {}) or {}
            desc = meta.get("description", "")
            planned = getattr(node, "cost", None)
            parts = [node.field]
            if planned is not None:
                parts.append(f"(cost={planned})")
            if desc:
                parts.append(f"- {desc}")
            lines.append("  " * depth + " ".join(parts))
            for dep in getattr(node, "deps", []):
                walk(dep, depth + 1)

        walk(tree)
        header = f"{name} total_cost={cost}"
        return "\n".join([header, *lines])

    def base_fields_for_resample(self, fields: Sequence[str]) -> tuple[str, ...]:
        base_fields: list[str] = []

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
            if self.has_raw_field(field):
                add(field)
                continue
            try:
                _cost, tree = self.resolve(field)
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

    def _compute_via_graph(self, name: str):
        graph = self._build_runtime_graph()
        solver = griblet.DependencySolver(graph)
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
        return self._evaluate_resolved_tree(tree, graph)

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
            include_aux_in_loader=self._include_aux_in_loader,
        )

    def _resolve_raw_name(self, name: str) -> str | None:
        if name in self._dataset.variables:
            return name
        return None

    def _infer_coordinate_fields(self, ndim: int) -> tuple[str, ...]:
        preferred = ["X [R]", "Y [R]", "Z [R]"]
        available = [name for name in preferred if name in self._dataset.variables]
        if len(available) >= ndim:
            return tuple(available[:ndim])

        raise ValueError(
            "Could not infer coordinate fields. Pass coordinate_fields explicitly."
        )

    def _build_runtime_graph(self):
        runtime_graph = griblet.ComputationGraph()
        runtime_graph.merge(self._build_loader_graph())
        runtime_graph.merge(self._computation_graph)
        return runtime_graph

    def _build_loader_graph(self):
        graph = griblet.ComputationGraph()

        for raw_name in self._dataset.variables:
            graph.add_recipe(
                field=raw_name,
                func=lambda raw_name=raw_name: (
                    self._dataset.variable(raw_name)
                    if hasattr(self._dataset, "variable")
                    else self._dataset[raw_name]
                ),
                deps=[],
                cost=0.0,
                metadata={"description": "Dataset raw field"},
            )

        if self._include_aux_in_loader:
            for key, value in self._dataset.aux.items():
                graph.add_recipe(
                    field=key,
                    func=lambda value=value: value,
                    deps=[],
                    cost=0.0,
                    metadata={"description": "Dataset aux"},
                )
        return graph

    def _evaluate_resolved_tree(self, node, graph):
        if getattr(node, "used_primary", False):
            for recipe in graph.recipes[node.field]:
                if len(recipe["deps"]) == 0:
                    return recipe["func"]()
            raise RuntimeError(f"No zero-dependency recipe for {node.field}")

        values = [self._evaluate_resolved_tree(dep, graph) for dep in node.deps]
        dep_fields = tuple(dep.field for dep in node.deps)
        for recipe in graph.recipes[node.field]:
            if tuple(recipe["deps"]) == dep_fields:
                return recipe["func"](*values)
        raise RuntimeError(f"No matching recipe for {node.field} with deps={dep_fields}")


__all__ = ["SmartDs"]
