from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import importlib

import numpy as np

from starwinds_readplt.dataset import Dataset


FieldFunction = Callable[["SmartDs"], np.ndarray]


class SmartDs:
    """
    Lightweight wrapper around ``starwinds_readplt.Dataset``.

    Initial goals:
    - Provide a stable place for on-demand derived fields (lazy + cached).
    - Support resampling into a new wrapped dataset without involving VTK/PyVista.

    The current implementation is intentionally simple: if a requested field exists
    in the underlying dataset it is returned directly, and optional registered field
    functions can be used for lazy derived quantities.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        field_functions: Mapping[str, FieldFunction] | None = None,
        aliases: Mapping[str, str | Sequence[str]] | None = None,
        cache_enabled: bool = True,
        computation_graph=None,
        include_aux_in_loader: bool = True,
    ) -> None:
        self._dataset = dataset
        self._field_functions: dict[str, FieldFunction] = dict(field_functions or {})
        self._aliases: dict[str, tuple[str, ...]] = {}
        self._cache_enabled = bool(cache_enabled)
        self._cache: dict[str, np.ndarray] = {}
        self._computation_graph = computation_graph
        self._include_aux_in_loader = bool(include_aux_in_loader)

        for name, candidates in (aliases or {}).items():
            self.set_alias(name, candidates)

    @classmethod
    def from_file(cls, file: str, **kwargs) -> "SmartDs":
        return cls(Dataset.from_file(file), **kwargs)

    @property
    def raw(self) -> Dataset:
        return self._dataset

    @property
    def dataset(self) -> Dataset:
        # Alias for readability at call sites.
        return self._dataset

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
    def field_functions(self) -> Mapping[str, FieldFunction]:
        return self._field_functions

    @property
    def computation_graph(self):
        return self._computation_graph

    def keys(self) -> tuple[str, ...]:
        """Known field names (raw + registered computed fields)."""
        names = list(self._dataset.variables)
        for name in self._field_functions:
            if name not in names:
                names.append(name)
        if self._computation_graph is not None and hasattr(self._computation_graph, "fields"):
            for name in self._computation_graph.fields():
                if name not in names:
                    names.append(name)
        return tuple(names)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self.has_field(name)

    def __call__(self, index_or_name):
        return self.variable(index_or_name)

    def __getitem__(self, index_or_name):
        return self.variable(index_or_name)

    def has_raw_field(self, name: str) -> bool:
        return self._resolve_raw_name(name) is not None

    def has_field(self, name: str) -> bool:
        if self.has_raw_field(name) or name in self._field_functions:
            return True
        if self._computation_graph is None:
            return False
        try:
            cost, _tree = self.resolve(name)
            return np.isfinite(cost)
        except Exception:
            return False

    def get(self, name: str, default=None):
        try:
            return self.variable(name)
        except (IndexError, KeyError):
            return default

    def set_alias(self, name: str, candidates: str | Sequence[str]) -> None:
        if isinstance(candidates, str):
            candidates = (candidates,)
        self._aliases[name] = tuple(candidates)

    def register_field(
        self,
        name: str,
        func: FieldFunction,
        *,
        overwrite: bool = False,
        aliases: str | Sequence[str] | None = None,
    ) -> None:
        if (not overwrite) and (name in self._field_functions):
            raise KeyError(f"Field function for '{name}' is already registered")
        self._field_functions[name] = func
        if aliases is not None:
            self.set_alias(name, aliases)
        self._cache.pop(name, None)

    def set_computation_graph(self, graph, *, merge: bool = False):
        """
        Attach a griblet computation graph used as a fallback for unresolved fields.

        Parameters
        ----------
        graph
            A ``griblet.ComputationGraph`` instance (or compatible object).
        merge
            If True and a graph is already present, merge the new graph into the
            existing graph using ``existing.merge(graph)``.
        """
        if graph is None:
            self._computation_graph = None
            return self

        if merge and self._computation_graph is not None:
            self._computation_graph.merge(graph)
        else:
            self._computation_graph = graph
        return self

    def add_spherical_fields(
        self,
        *,
        coord_fields: Sequence[str] = ("X [R]", "Y [R]", "Z [R]"),
        vectors: Sequence[str] = ("B", "U"),
        components: Sequence[str] = ("r", "theta", "phi"),
    ):
        """
        Register on-demand spherical geometry and vector-component fields.

        This uses local field functions (no griblet required), but the same recipe
        functions are available in ``starwinds_analysis.recipes.spherical`` for
        future griblet integration.
        """
        from starwinds_analysis.recipes.spherical import (
            auto_register_vector_spherical_components,
            register_spherical_geometry_fields,
        )

        register_spherical_geometry_fields(self, coord_fields=coord_fields)
        auto_register_vector_spherical_components(
            self,
            coord_fields=coord_fields,
            prefixes=tuple(vectors),
            components=tuple(components),
        )
        return self

    def clear_cache(self, *names: str) -> None:
        if not names:
            self._cache.clear()
            return
        for name in names:
            self._cache.pop(name, None)

    def variable(self, index_or_name):
        # Preserve Dataset behavior for integer indexing.
        if not isinstance(index_or_name, str):
            return self._dataset.variable(index_or_name)

        name = index_or_name
        if self._cache_enabled and name in self._cache:
            return self._cache[name]

        raw_name = self._resolve_raw_name(name)
        if raw_name is not None:
            value = self._dataset.variable(raw_name)
            if self._cache_enabled:
                self._cache[name] = value
            return value

        func = self._field_functions.get(name)
        if func is not None:
            value = np.asarray(func(self))
        else:
            value = self._compute_via_graph(name)
        if self._cache_enabled:
            self._cache[name] = value
        return value

    def resolve(self, name: str):
        """
        Resolve a field through the attached griblet computation graph.

        Returns ``(cost, tree)`` from ``griblet.DependencySolver.resolve_field``.
        """
        graph = self._build_runtime_graph()
        griblet = self._import_griblet()
        solver = griblet.DependencySolver(graph)
        return solver.resolve_field(name)

    def explain(self, name: str, *, return_tree: bool = False):
        """
        Return a human-readable computation path (if griblet is configured).
        """
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

    def _compute_via_graph(self, name: str):
        if self._computation_graph is None:
            raise IndexError(
                f"Field '{name}' not available. Raw fields: {self._dataset.variables}. "
                f"Computed fields: {list(self._field_functions)}."
            )

        graph = self._build_runtime_graph()
        griblet = self._import_griblet()
        solver = griblet.DependencySolver(graph)
        _cost, tree = solver.resolve_field(name)
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
        """
        Resample scalar fields onto new point locations and return a new wrapped dataset.

        Parameters
        ----------
        sample_points
            Array-like of shape ``(n_points, ndim)`` containing target coordinates.
        coordinate_fields
            Coordinate field names to use from the source dataset (e.g. ``("X [R]",
            "Y [R]", "Z [R]")``). If omitted, common BATSRUS coordinate names are
            inferred from the source dataset.
        fields
            Fields to include in the output dataset (coordinates are always included).
            Defaults to all raw fields.
        method
            ``"nearest"`` or ``"linear"``.
        fill_value
            Fill value for points outside the convex hull when ``method="linear"``.
        corners
            Optional cell connectivity for the resampled dataset. Defaults to an empty
            connectivity array, which is suitable for point-wise analysis.
        """
        sample_points = np.asarray(sample_points, dtype=float)
        if sample_points.ndim == 1:
            sample_points = sample_points[np.newaxis, :]
        if sample_points.ndim != 2:
            raise ValueError("sample_points must have shape (n_points, ndim)")

        ndim = sample_points.shape[1]
        if coordinate_fields is None:
            coordinate_fields = self._infer_coordinate_fields(ndim)
        coordinate_fields = tuple(coordinate_fields)
        if len(coordinate_fields) != ndim:
            raise ValueError(
                f"Expected {ndim} coordinate fields, got {len(coordinate_fields)}: "
                f"{coordinate_fields}"
            )

        for coord_name in coordinate_fields:
            if not self.has_field(coord_name):
                raise IndexError(f"Coordinate field '{coord_name}' not available")

        if fields is None:
            output_variables = list(self._dataset.variables)
        else:
            output_variables = list(coordinate_fields)
            for name in fields:
                if name not in output_variables:
                    output_variables.append(name)

        # Source coordinates used for interpolation.
        source_coords = np.column_stack(
            [np.asarray(self.variable(name)).ravel() for name in coordinate_fields]
        )
        coord_mask = np.isfinite(source_coords).all(axis=1)
        if not np.any(coord_mask):
            raise ValueError("No finite source coordinates available for resampling")

        # Build output points table matching Dataset(points, ..., variables=...).
        out_points = np.full((sample_points.shape[0], len(output_variables)), np.nan, dtype=float)
        out_index = {name: i for i, name in enumerate(output_variables)}

        for dim, coord_name in enumerate(coordinate_fields):
            if coord_name in out_index:
                out_points[:, out_index[coord_name]] = sample_points[:, dim]

        for name in output_variables:
            if name in coordinate_fields:
                continue

            values = np.asarray(self.variable(name)).ravel()
            if values.shape[0] != source_coords.shape[0]:
                raise ValueError(
                    f"Field '{name}' has length {values.shape[0]} but coordinates have "
                    f"length {source_coords.shape[0]}"
                )
            valid = coord_mask & np.isfinite(values)
            if not np.any(valid):
                continue

            out_points[:, out_index[name]] = self._interpolate(
                source_coords[valid],
                values[valid],
                sample_points,
                method=method,
                fill_value=fill_value,
            )

        if corners is None:
            corners_arr = np.empty((0, 0), dtype=int)
        else:
            corners_arr = np.asarray(corners)

        if copy_aux:
            aux = deepcopy(self._dataset.aux)
        else:
            aux = self._dataset.aux

        if title is None:
            title = self._dataset.title
        if zone is None:
            zone = f"{self._dataset.zone} (resampled)"

        new_dataset = Dataset(
            out_points,
            corners_arr,
            aux,
            title,
            output_variables,
            zone,
        )

        return type(self)(
            new_dataset,
            field_functions=self._field_functions,
            aliases=self._aliases,
            cache_enabled=self._cache_enabled,
            computation_graph=self._computation_graph,
            include_aux_in_loader=self._include_aux_in_loader,
        )

    def _build_runtime_graph(self):
        if self._computation_graph is None:
            raise RuntimeError("No computation graph attached")
        griblet = self._import_griblet()
        runtime_graph = griblet.ComputationGraph()
        runtime_graph.merge(self._build_loader_graph())
        runtime_graph.merge(self._computation_graph)
        return runtime_graph

    def _build_loader_graph(self):
        """
        Build a zero-dependency graph exposing raw dataset variables (+ selected aux).
        """
        griblet = self._import_griblet()
        graph = griblet.ComputationGraph()

        for raw_name in self._dataset.variables:
            graph.add_recipe(
                field=raw_name,
                func=lambda raw_name=raw_name: self._dataset.variable(raw_name),
                deps=[],
                cost=0.0,
                metadata={"description": "Dataset raw field"},
            )

        for alias_name, candidates in self._aliases.items():
            if alias_name in self._dataset.variables:
                continue
            raw_name = next((c for c in candidates if c in self._dataset.variables), None)
            if raw_name is None:
                continue
            graph.add_recipe(
                field=alias_name,
                func=lambda raw_name=raw_name: self._dataset.variable(raw_name),
                deps=[],
                cost=0.0,
                metadata={"description": f"Alias for {raw_name}"},
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

    def _resolve_raw_name(self, name: str) -> str | None:
        if name in self._dataset.variables:
            return name

        for candidate in self._aliases.get(name, ()):
            if candidate in self._dataset.variables:
                return candidate
        return None

    def _infer_coordinate_fields(self, ndim: int) -> tuple[str, ...]:
        preferred = [
            "X [R]",
            "Y [R]",
            "Z [R]",
        ]
        available = [name for name in preferred if name in self._dataset.variables]
        if len(available) >= ndim:
            return tuple(available[:ndim])

        raise ValueError(
            "Could not infer coordinate fields. Pass coordinate_fields explicitly."
        )

    @staticmethod
    def _import_griblet():
        try:
            return importlib.import_module("griblet")
        except ImportError as e:
            raise ImportError(
                "griblet is required for computation-graph resolution. Install griblet "
                "or use local register_field(...) functions."
            ) from e

    @staticmethod
    def _evaluate_resolved_tree(node, graph):
        """
        Evaluate a griblet-resolved computation tree.

        Uses tolerant dependency matching (tuple/list) to stay compatible with the
        current griblet recipe storage/evaluator behavior.
        """
        if getattr(node, "used_primary", False):
            for recipe in graph.recipes[node.field]:
                if len(recipe["deps"]) == 0:
                    return recipe["func"]()
            raise RuntimeError(f"No zero-dependency recipe for {node.field}")

        values = [SmartDs._evaluate_resolved_tree(dep, graph) for dep in node.deps]
        dep_fields = tuple(dep.field for dep in node.deps)
        for recipe in graph.recipes[node.field]:
            if tuple(recipe["deps"]) == dep_fields:
                return recipe["func"](*values)
        raise RuntimeError(f"No matching recipe for {node.field} with deps={dep_fields}")

    @staticmethod
    def _interpolate(source_points, values, sample_points, *, method: str, fill_value: float):
        try:
            from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
        except ImportError as e:
            raise ImportError(
                "Resampling requires scipy (scipy.interpolate). Install scipy to use "
                "SmartDs.resample()."
            ) from e

        if method == "nearest":
            interpolator = NearestNDInterpolator(source_points, values)
            out = interpolator(sample_points)
        elif method == "linear":
            interpolator = LinearNDInterpolator(source_points, values, fill_value=fill_value)
            out = interpolator(sample_points)
        else:
            raise ValueError("method must be 'nearest' or 'linear'")

        out = np.asarray(out, dtype=float)
        if out.ndim == 0:
            out = out[np.newaxis]
        return out


__all__ = ["SmartDs"]
