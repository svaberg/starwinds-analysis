"""THIS FILE contains the public SmartDs dataset wrapper API.

It is the facade for raw field access, lazy computed fields, graph integration, and resampling delegation.
It should not contain domain-specific physics formulas or plotting code.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from os import PathLike

import numpy as np

from starwinds_readplt.dataset import Dataset
from starwinds_analysis._smart_ds_graph import compute_via_graph as _compute_via_graph
from starwinds_analysis._smart_ds_graph import explain_field as _explain_field
from starwinds_analysis._smart_ds_graph import graph_field_names as _graph_field_names
from starwinds_analysis._smart_ds_graph import resolve_field as _resolve_field
from starwinds_analysis._smart_ds_resample import resample_smart_ds

FieldFunction = Callable[["SmartDs"], np.ndarray]


def prepare_smartds(smart_ds: "SmartDs", *, body_radius_m: float) -> None:
    """
    Attach the standard SI and spherical field graphs used by common workflows.
    Used by: `starwinds_analysis/pipelines/slice.py`, `starwinds_analysis/pipelines/volume.py`, `starwinds_analysis/pipelines/shell.py`
    """
    smart_ds.add_batsrus_graph(body_radius_m=body_radius_m)
    try:
        smart_ds.add_spherical_graph(vectors=("B", "U"))
    except Exception:
        smart_ds.add_spherical_fields(vectors=("B", "U"))

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
        """
        Wrap a raw Dataset and initialize aliases, local field functions, and graph hooks.
        Used by: `SmartDs` users and internal methods
        """
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

    def __repr__(self) -> str:
        """
        Debug-style summary string for interactive use.
        Used by: `SmartDs` users and internal methods
        """
        return (
            f"SmartDs(title={self.title!r}, zone={self.zone!r}, "
            f"points={len(self.points)}, variables={len(self.variables)})"
        )

    def __str__(self) -> str:
        """
        Human-readable dataset summary for notebooks/examples (`print(sds)`).
        Used by: `SmartDs` users and internal methods
        """
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
    def from_file(cls, file: str | PathLike[str], **kwargs) -> "SmartDs":
        """
        Construct a SmartDs directly from a `.plt` file path.
        Used by: `SmartDs` users and internal methods
        """
        return cls(Dataset.from_file(str(file)), **kwargs)

    def _auto_register_builtin_fields(self) -> None:
        """
        Register built-in spherical geometry/vector fields when XYZ coordinates are present.
        Used by: `SmartDs` users and internal methods
        Register lightweight built-in derived fields that should be available by default.
        For now this auto-registers spherical geometry/vector-component fields when
        standard Cartesian BATSRUS-style coordinates are present.
        """
        coord_fields = ("X [R]", "Y [R]", "Z [R]")
        if not all(name in self._dataset.variables for name in coord_fields):
            return

        from starwinds_analysis.recipes.spherical import _vector_triplets
        from starwinds_analysis.recipes.spherical import register_spherical_geometry_fields
        from starwinds_analysis.recipes.spherical import register_vector_spherical_components

        register_spherical_geometry_fields(self, coord_fields=coord_fields)
        for prefix, unit in _vector_triplets(self.variables):
            register_vector_spherical_components(self, prefix=prefix, unit=unit, coord_fields=coord_fields)

    @property
    def raw(self) -> Dataset:
        """
        Direct access to the underlying `starwinds_readplt.Dataset`.
        Used by: `SmartDs` users and internal methods
        """
        return self._dataset

    @property
    def dataset(self) -> Dataset:
        # Alias for readability at call sites.
        """
        Readability alias for `raw`.
        Used by: `SmartDs` users and internal methods
        """
        return self._dataset

    @property
    def aux(self):
        """
        AUX metadata passthrough from the raw dataset.
        Used by: `SmartDs` users and internal methods
        """
        return self._dataset.aux

    @property
    def title(self):
        """
        Dataset title passthrough.
        Used by: `SmartDs` users and internal methods
        """
        return self._dataset.title

    @property
    def zone(self):
        """
        Zone name passthrough.
        Used by: `SmartDs` users and internal methods
        """
        return self._dataset.zone

    @property
    def points(self):
        """
        Raw point array passthrough.
        Used by: `SmartDs` users and internal methods
        """
        return self._dataset.points

    @property
    def corners(self):
        """
        Raw cell connectivity passthrough.
        Used by: `SmartDs` users and internal methods
        """
        return self._dataset.corners

    @property
    def variables(self) -> tuple[str, ...]:
        """
        Raw variable names as an immutable tuple.
        Used by: `SmartDs` users and internal methods
        """
        return tuple(self._dataset.variables)

    @property
    def field_functions(self) -> Mapping[str, FieldFunction]:
        """
        Registered local field functions (lazy computed fields, no griblet).
        Used by: `SmartDs` users and internal methods
        """
        return self._field_functions

    @property
    def computation_graph(self):
        """
        Attached griblet graph (or None).
        Used by: `SmartDs` users and internal methods
        """
        return self._computation_graph

    def keys(self) -> tuple[str, ...]:
        """
        Known field names from raw variables, local fields, and the attached graph.
        Used by: `SmartDs` users and internal methods
        Known field names (raw + registered computed fields).
        """
        names = list(self._dataset.variables)
        for name in self._field_functions:
            if name not in names:
                names.append(name)
        for name in _graph_field_names(self):
            if name not in names:
                names.append(name)
        return tuple(names)

    def __contains__(self, name: object) -> bool:
        """
        Support `name in sds` by checking field availability.
        Used by: `SmartDs` users and internal methods
        """
        return isinstance(name, str) and self.has_field(name)

    def __call__(self, index_or_name):
        """
        `sds(name)` is shorthand for `sds.variable(name)`.
        Used by: `SmartDs` users and internal methods
        """
        return self.variable(index_or_name)

    def __getitem__(self, index_or_name):
        # `[]` is a raw-dataset passthrough (base fields only). Use `()` / `.variable()`
        # for SmartDs field resolution (aliases, lazy fields, griblet recipes).
        """
        `sds[name]` is raw-dataset access only (no aliases/local fields/griblet).
        Used by: `SmartDs` users and internal methods
        """
        return self._dataset.variable(index_or_name)

    def has_raw_field(self, name: str) -> bool:
        """
        Check only raw dataset fields (including alias fallback).
        Used by: `SmartDs` users and internal methods
        """
        return self._resolve_raw_name(name) is not None

    def has_field(self, name: str) -> bool:
        """
        Check whether a field is available from raw data, local fields, or griblet.
        Used by: `SmartDs` users and internal methods
        """
        if self.has_raw_field(name) or name in self._field_functions:
            return True
        if self._computation_graph is None:
            return False
        return name in _graph_field_names(self)

    def get(self, name: str, default=None):
        """
        Dict-like getter wrapper around `variable(...)` with a default.
        Used by: `SmartDs` users and internal methods
        """
        try:
            return self.variable(name)
        except (IndexError, KeyError):
            return default

    def set_alias(self, name: str, candidates: str | Sequence[str]) -> None:
        """
        Map a canonical field name to one or more raw field candidates.
        Used by: `SmartDs` users and internal methods
        """
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
        """
        Register a lazy local field function on this SmartDs instance.
        Used by: `SmartDs` users and internal methods
        """
        if (not overwrite) and (name in self._field_functions):
            raise KeyError(f"Field function for '{name}' is already registered")
        self._field_functions[name] = func
        if aliases is not None:
            self.set_alias(name, aliases)
        self._cache.pop(name, None)

    def set_computation_graph(self, graph, *, merge: bool = False):
        """
        Attach or merge a griblet computation graph used for unresolved fields.
        Used by: `SmartDs` users and internal methods
        Attach a griblet computation graph used as a fallback for unresolved fields.
        Parameters
        ----------
        graph
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
        vectors: Sequence[str] | None = None,
    ):
        """
        Add local (non-griblet) spherical geometry/vector component fields.
        Used by: `SmartDs` users and internal methods
        Register on-demand spherical geometry and vector-component fields.
        This uses local field functions (no griblet required), but the same recipe
        functions are available in ``starwinds_analysis.recipes.spherical`` for
        future griblet integration.
        """
        from starwinds_analysis.recipes.spherical import _vector_triplets
        from starwinds_analysis.recipes.spherical import register_spherical_geometry_fields
        from starwinds_analysis.recipes.spherical import register_vector_spherical_components

        register_spherical_geometry_fields(self, coord_fields=coord_fields)
        for prefix, unit in _vector_triplets(self.variables, prefixes=vectors):
            register_vector_spherical_components(self, prefix=prefix, unit=unit, coord_fields=coord_fields)
        return self

    def add_spherical_graph(
        self,
        *,
        coord_fields: Sequence[str] = ("X [R]", "Y [R]", "Z [R]"),
        vectors: Sequence[str] | None = None,
        merge: bool = True,
    ):
        """
        Add griblet spherical geometry/vector recipes to the attached graph.
        Used by: `SmartDs` users and internal methods
        Attach griblet recipes for spherical geometry/vector components.
        Unlike ``add_spherical_fields()``, this registers recipes in the attached
        computation graph and resolves them via ``griblet`` on demand.
        """
        from starwinds_analysis.recipes.spherical import _vector_triplets
        from starwinds_analysis.recipes.spherical import build_griblet_spherical_geometry_graph
        from starwinds_analysis.recipes.spherical import build_griblet_vector_spherical_components_graph

        graph = build_griblet_spherical_geometry_graph(coord_fields=coord_fields)
        for prefix, unit in _vector_triplets(self.variables, prefixes=vectors):
            graph.merge(
                build_griblet_vector_spherical_components_graph(
                    prefix=prefix,
                    unit=unit,
                    coord_fields=coord_fields,
                )
            )
        self.set_computation_graph(graph, merge=merge)
        return self

    def add_batsrus_graph(
        self,
        *,
        body_radius_m: float | None = None,
        include_unit_normalization: bool = True,
        include_derived: bool = True,
        merge: bool = True,
    ):
        """
        Add the BATSRUS SI-normalization and derived-field recipe graph.
        Used by: `SmartDs` users and internal methods
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

    def clear_cache(self, *names: str) -> None:
        """
        Clear cached field values (all or selected names).
        Used by: `SmartDs` users and internal methods
        """
        if not names:
            self._cache.clear()
            return
        for name in names:
            self._cache.pop(name, None)

    def variable(self, index_or_name):
        # Preserve Dataset behavior for integer indexing.
        """
        Main field accessor: raw passthrough, then local fields, then griblet graph.
        Used by: `SmartDs` users and internal methods
        """
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

    def resolve(self, name: str):
        # TODO smartds-resolve:
        # This currently means "resolve computation path via griblet". The user-facing
        # field/unit resolution API should likely live on SmartDs too (returning data
        # + parsed unit string from bracketed field names), which may require renaming
        # this graph-planning method to avoid semantic collision.
        """
        Graph-path resolver (`cost, tree`) used by `has_field` and `explain`.
        Used by: `SmartDs` users and internal methods
        """
        return _resolve_field(self, name)

    def explain(self, name: str, *, return_tree: bool = False):
        """
        Human-readable explanation of the chosen griblet path for a field.
        Used by: `SmartDs` users and internal methods
        """
        return _explain_field(self, name, return_tree=return_tree)

    def _compute_via_graph(self, name: str):
        """
        Evaluate a field via the attached griblet graph.
        Used by: `SmartDs` users and internal methods
        """
        return _compute_via_graph(self, name)

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
        Generic resampling entry point returning a new SmartDs (flat or structured targets).
        Used by: `SmartDs` users and internal methods
        Resample scalar fields onto new point locations and return a new wrapped dataset.
        Parameters
        ----------
        sample_points
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

    def append_fields(
        self,
        extra_fields: Mapping[str, np.ndarray],
        *,
        zone_suffix: str = "derived fields",
    ) -> "SmartDs":
        """
        Return a new SmartDs with extra point-shaped fields appended to the raw dataset.
        Used by: `starwinds_analysis/analysis/shells.py`
        """
        if not extra_fields:
            return self

        base_points = np.array(self.raw.points)
        if base_points.ndim < 2:
            raise ValueError("Expected raw points to have shape (..., nvars)")
        base_shape = base_points.shape[:-1]

        arrays = []
        names = []
        for name, values in extra_fields.items():
            arr = np.array(values)
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
            field_functions=self._field_functions,
            aliases=self._aliases,
            cache_enabled=self._cache_enabled,
            computation_graph=self._computation_graph,
            include_aux_in_loader=self._include_aux_in_loader,
        )

    def _resolve_raw_name(self, name: str) -> str | None:
        """
        Resolve a requested name to an existing raw dataset field (via aliases).
        Used by: `SmartDs` users and internal methods
        """
        if name in self._dataset.variables:
            return name

        for candidate in self._aliases.get(name, ()):
            if candidate in self._dataset.variables:
                return candidate
        return None

    def _infer_coordinate_fields(self, ndim: int) -> tuple[str, ...]:
        """
        Infer coordinate fields from common BATSRUS-style XYZ names.
        Used by: `SmartDs` users and internal methods
        """
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
