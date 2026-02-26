"""THIS FILE contains the public SmartDs dataset wrapper API.

It is the facade for raw field access, lazy computed fields, graph integration, and resampling delegation.
It should not contain domain-specific physics formulas or plotting code.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from os import PathLike

import numpy as np

from starwinds_readplt.dataset import Dataset
from starwinds_analysis._smart_ds_graph import (
    compute_via_graph as _compute_via_graph,
    explain_field as _explain_field,
    graph_field_names as _graph_field_names,
    resolve_field as _resolve_field,
)
from starwinds_analysis._smart_ds_resample import resample_smart_ds

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

    Centering note (current limitation):
    - SmartDs does not yet track field centering metadata (point-centered vs cell-centered).
    - In current resampling/sample workflows, values are effectively treated as samples
      at the provided coordinates (often used as cell-centered values in notebooks/analysis).
    - TODO geometry metrics: points/cells should be able to report finite geometric
      measures (e.g. `length [..]`, `area [..^2]`, `volume [..^3]`) for regular grids.
    """

    # Wrap a raw Dataset and initialize aliases, local field functions, and graph hooks.
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

        self._auto_register_builtin_fields()

    # Debug-style summary string for interactive use.
    def __repr__(self) -> str:
        return (
            f"SmartDs(title={self.title!r}, zone={self.zone!r}, "
            f"points={len(self.points)}, variables={len(self.variables)})"
        )

    # Human-readable dataset summary for notebooks/examples (`print(sds)`).
    def __str__(self) -> str:
        return "\n".join(
            (
                "SmartDs",
                f"  Title: {self.title}",
                f"  Zone : {self.zone}",
                f"  Points: {len(self.points)}",
                f"  Variables: {len(self.variables)}",
            )
        )

    @classmethod
    # Construct a SmartDs directly from a `.plt` file path.
    def from_file(cls, file: str | PathLike[str], **kwargs) -> "SmartDs":
        return cls(Dataset.from_file(str(file)), **kwargs)

    # Register built-in spherical geometry/vector fields when XYZ coordinates are present.
    def _auto_register_builtin_fields(self) -> None:
        """
        Register lightweight built-in derived fields that should be available by default.

        For now this auto-registers spherical geometry/vector-component fields when
        standard Cartesian BATSRUS-style coordinates are present.
        """
        coord_fields = ("X [R]", "Y [R]", "Z [R]")
        if not all(name in self._dataset.variables for name in coord_fields):
            return

        from starwinds_analysis.recipes.spherical import (
            auto_register_vector_spherical_components,
            register_spherical_geometry_fields,
        )

        register_spherical_geometry_fields(self, coord_fields=coord_fields)
        auto_register_vector_spherical_components(self, coord_fields=coord_fields)

    @property
    # Direct access to the underlying `starwinds_readplt.Dataset`.
    def raw(self) -> Dataset:
        return self._dataset

    @property
    # Readability alias for `raw`.
    def dataset(self) -> Dataset:
        # Alias for readability at call sites.
        return self._dataset

    @property
    # AUX metadata passthrough from the raw dataset.
    def aux(self):
        return self._dataset.aux

    @property
    # Dataset title passthrough.
    def title(self):
        return self._dataset.title

    @property
    # Zone name passthrough.
    def zone(self):
        return self._dataset.zone

    @property
    # Raw point array passthrough.
    def points(self):
        return self._dataset.points

    @property
    # Raw cell connectivity passthrough.
    def corners(self):
        return self._dataset.corners

    @property
    # Raw variable names as an immutable tuple.
    def variables(self) -> tuple[str, ...]:
        return tuple(self._dataset.variables)

    @property
    # Registered local field functions (lazy computed fields, no griblet).
    def field_functions(self) -> Mapping[str, FieldFunction]:
        return self._field_functions

    @property
    # Attached griblet graph (or None).
    def computation_graph(self):
        return self._computation_graph

    # Known field names from raw variables, local fields, and the attached graph.
    def keys(self) -> tuple[str, ...]:
        """Known field names (raw + registered computed fields)."""
        names = list(self._dataset.variables)
        for name in self._field_functions:
            if name not in names:
                names.append(name)
        for name in _graph_field_names(self):
            if name not in names:
                names.append(name)
        return tuple(names)

    # Support `name in sds` by checking field availability.
    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self.has_field(name)

    # `sds(name)` is shorthand for `sds.variable(name)`.
    def __call__(self, index_or_name):
        return self.variable(index_or_name)

    # `sds[name]` is raw-dataset access only (no aliases/local fields/griblet).
    def __getitem__(self, index_or_name):
        # `[]` is a raw-dataset passthrough (base fields only). Use `()` / `.variable()`
        # for SmartDs field resolution (aliases, lazy fields, griblet recipes).
        return self._dataset.variable(index_or_name)

    # Check only raw dataset fields (including alias fallback).
    def has_raw_field(self, name: str) -> bool:
        return self._resolve_raw_name(name) is not None

    # Check whether a field is available from raw data, local fields, or griblet.
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

    # Dict-like getter wrapper around `variable(...)` with a default.
    def get(self, name: str, default=None):
        try:
            return self.variable(name)
        except (IndexError, KeyError):
            return default

    # Map a canonical field name to one or more raw field candidates.
    def set_alias(self, name: str, candidates: str | Sequence[str]) -> None:
        if isinstance(candidates, str):
            candidates = (candidates,)
        self._aliases[name] = tuple(candidates)

    # Register a lazy local field function on this SmartDs instance.
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

    # Attach or merge a griblet computation graph used for unresolved fields.
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

    # Add local (non-griblet) spherical geometry/vector component fields.
    def add_spherical_fields(
        self,
        *,
        coord_fields: Sequence[str] = ("X [R]", "Y [R]", "Z [R]"),
        vectors: Sequence[str] | None = None,
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
            prefixes=None if vectors is None else tuple(vectors),
            components=tuple(components),
        )
        return self

    # Add griblet spherical geometry/vector recipes to the attached graph.
    def add_spherical_graph(
        self,
        *,
        coord_fields: Sequence[str] = ("X [R]", "Y [R]", "Z [R]"),
        vectors: Sequence[str] | None = None,
        components: Sequence[str] = ("r", "theta", "phi"),
        merge: bool = True,
    ):
        """
        Attach griblet recipes for spherical geometry/vector components.

        Unlike ``add_spherical_fields()``, this registers recipes in the attached
        computation graph and resolves them via ``griblet`` on demand.
        """
        from starwinds_analysis.recipes.spherical import (
            build_griblet_auto_vector_spherical_components_graph,
            build_griblet_spherical_geometry_graph,
        )

        graph = build_griblet_spherical_geometry_graph(coord_fields=coord_fields)
        graph.merge(
            build_griblet_auto_vector_spherical_components_graph(
                self.variables,
                coord_fields=coord_fields,
                prefixes=None if vectors is None else tuple(vectors),
                components=tuple(components),
            )
        )
        self.set_computation_graph(graph, merge=merge)
        return self

    # Add the BATSRUS SI-normalization and derived-field recipe graph.
    def add_batsrus_graph(
        self,
        *,
        body_radius_m: float | None = None,
        include_unit_normalization: bool = True,
        include_derived: bool = True,
        merge: bool = True,
    ):
        """
        Attach a BATSRUS-oriented griblet graph (SI normalization + derived fields).
        """
        from starwinds_analysis.recipes.batsrus import build_griblet_batsrus_graph

        graph = build_griblet_batsrus_graph(
            self.variables,
            aux=self.aux,
            body_radius_m=body_radius_m,
            include_unit_normalization=include_unit_normalization,
            include_derived=include_derived,
        )
        self.set_computation_graph(graph, merge=merge)
        return self

    # Clear cached field values (all or selected names).
    def clear_cache(self, *names: str) -> None:
        if not names:
            self._cache.clear()
            return
        for name in names:
            self._cache.pop(name, None)

    # Main field accessor: raw passthrough, then local fields, then griblet graph.
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
            value = np.array(func(self))
        else:
            value = self._compute_via_graph(name)
        if self._cache_enabled:
            self._cache[name] = value
        return value

    # Graph-path resolver (`cost, tree`) used by `has_field` and `explain`.
    def resolve(self, name: str):
        # TODO smartds-resolve:
        # This currently means "resolve computation path via griblet". The user-facing
        # field/unit resolution API should likely live on SmartDs too (returning data
        # + parsed unit string from bracketed field names), which may require renaming
        # this graph-planning method to avoid semantic collision.
        return _resolve_field(self, name)

    # Human-readable explanation of the chosen griblet path for a field.
    def explain(self, name: str, *, return_tree: bool = False):
        return _explain_field(self, name, return_tree=return_tree)

    # Evaluate a field via the attached griblet graph.
    def _compute_via_graph(self, name: str):
        return _compute_via_graph(self, name)

    # Generic resampling entry point returning a new SmartDs (flat or structured targets).
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

        Notes
        -----
        SmartDs currently does not track centering metadata. Resampling therefore
        treats source values as samples at the source coordinates and returns values
        at the requested target coordinates without explicit point/cell centering semantics.
        """
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

    # Resolve a requested name to an existing raw dataset field (via aliases).
    def _resolve_raw_name(self, name: str) -> str | None:
        if name in self._dataset.variables:
            return name

        for candidate in self._aliases.get(name, ()):
            if candidate in self._dataset.variables:
                return candidate
        return None

    # Infer coordinate fields from common BATSRUS-style XYZ names.
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
