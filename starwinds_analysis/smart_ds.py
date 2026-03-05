"""Public SmartDs dataset wrapper API.
"""

# It is the facade for raw field access, graph integration, and resampling delegation.
# It should not contain domain-specific physics formulas or plotting code.


from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from os import PathLike
from pathlib import Path

import numpy as np
from griblet.dependency_solver import UnresolvableFieldError

from starwinds_readplt.dataset import Dataset
from starwinds_analysis._smart_ds_graph import compute_via_graph as _compute_via_graph
from starwinds_analysis._smart_ds_graph import explain_field as _explain_field
from starwinds_analysis._smart_ds_graph import graph_field_names as _graph_field_names
from starwinds_analysis._smart_ds_resample import resample_smart_ds
from starwinds_analysis.param_in import stellar_aux_from_nearby_param_in


class SmartDs:
    """
    Lightweight wrapper around ``starwinds_readplt.Dataset``.

    Initial goals:
    - Provide a stable place for on-demand derived fields (lazy + cached).
    - Support resampling into a new wrapped dataset without involving VTK/PyVista.

    The current implementation is intentionally simple: if a requested field exists
    in the underlying dataset it is returned directly; otherwise the attached graph
    is used to derive it.

    Data model note (current limitation):
    - SmartDs reads the dataset as unstructured point samples.
    - SmartDs does not track centering metadata (point-centered vs cell-centered).
    - If cell-centered quantities are needed, they must be constructed explicitly from
      the available point data.
    - TODO geometry metrics: points/cells should be able to report finite geometric
      measures (e.g. `length [..]`, `area [..^2]`, `volume [..^3]`) for regular grids.
    """

    DEFAULT_COORD_FIELDS = ("X [R]", "Y [R]", "Z [R]")

    def __init__(
        self,
        dataset: Dataset,
        *,
        cache_enabled: bool = True,
        computation_graph=None,
        include_aux_in_loader: bool = True,
    ) -> None:
        """
        Wrap a raw Dataset and initialize graph hooks.
        Used by: `SmartDs` users and internal methods
        """
        self._dataset = dataset
        self._cache_enabled = bool(cache_enabled)
        self._cache: dict[str, np.ndarray] = {}
        self._computation_graph = computation_graph
        self._include_aux_in_loader = bool(include_aux_in_loader)

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
        dataset = Dataset.from_file(str(file))
        for key, value in stellar_aux_from_nearby_param_in(Path(file)).items():
            dataset.aux.setdefault(key, value)
        return cls(dataset, **kwargs)

    @property
    def raw(self) -> Dataset:
        """
        Direct access to the underlying `starwinds_readplt.Dataset`.
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
    def computation_graph(self):
        """
        Attached griblet graph (or None).
        Used by: `SmartDs` users and internal methods
        """
        return self._computation_graph

    def keys(self) -> tuple[str, ...]:
        """
        Known field names from raw variables and attached graph fields.
        Used by: `SmartDs` users and internal methods
        """
        names = list(self._dataset.variables)
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

    def __call__(self, name: str):
        """
        `sds(name)` is shorthand for `sds.variable(name)`.
        Used by: `SmartDs` users and internal methods
        """
        return self.variable(name)

    def has_raw_field(self, name: str) -> bool:
        """
        Check only raw dataset fields.
        Used by: `SmartDs` users and internal methods
        """
        return name in self._dataset.variables

    def has_field(self, name: str) -> bool:
        """
        Check whether a field is available from raw data or the attached graph.
        Used by: `SmartDs` users and internal methods
        """
        if self.has_raw_field(name):
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

    def set_computation_graph(self, graph, *, merge: bool = False):
        """
        Attach or merge a griblet computation graph used for unresolved fields.
        Used by: `SmartDs` users and internal methods
        """
        if graph is None:
            self._computation_graph = None
            return self

        if merge and self._computation_graph is not None:
            self._computation_graph.merge(graph)
        else:
            self._computation_graph = graph
        return self

    def add_spherical_graph(
        self,
        *,
        coord_fields: Sequence[str] | None = None,
        vectors: Sequence[str] | None = None,
        merge: bool = True,
    ):
        """
        Add griblet spherical geometry/vector recipes to the attached graph.
        Used by: `SmartDs` users and internal methods
        """
        from starwinds_analysis.recipes.spherical import _vector_triplets
        from starwinds_analysis.recipes.spherical import build_griblet_spherical_geometry_graph
        from starwinds_analysis.recipes.spherical import build_griblet_vector_spherical_components_graph

        if coord_fields is None:
            coord_fields = self.DEFAULT_COORD_FIELDS
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
        body_radius: float | None = None,
        include_unit_normalization: bool = True,
        include_derived: bool = True,
        merge: bool = True,
    ):
        """
        Add the BATSRUS SI-normalization and derived-field recipe graph.
        Used by: `SmartDs` users and internal methods
        """
        from starwinds_analysis.recipes.batsrus import build_griblet_batsrus_graph

        graph = build_griblet_batsrus_graph(
            self.variables,
            aux=self.aux,
            body_radius=body_radius,
            include_unit_normalization=include_unit_normalization,
            include_derived=include_derived,
        )
        self.set_computation_graph(graph, merge=merge)
        return self

    def prepare(self, *, body_radius: float | None = None) -> "SmartDs":
        """
        Attach the standard SI and spherical graphs used by common workflows.
        If `body_radius` is omitted, the BATSRUS graph must be able to infer it
        from available metadata (for example nearby `PARAM.in` stellar parameters).
        Used by: `starwinds_analysis/pipelines/slice.py`, `starwinds_analysis/pipelines/volume.py`, `starwinds_analysis/pipelines/shell.py`
        """
        self.add_batsrus_graph(body_radius=body_radius)
        self.add_spherical_graph()
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

    def variable(self, name: str):
        """
        Main name-based field accessor. Raw loader fields win when present; otherwise
        SmartDs produces the field from the attached graph.
        Used by: `SmartDs` users and internal methods
        """
        if not isinstance(name, str):
            raise TypeError("SmartDs fields must be requested by name")
        if self._cache_enabled and name in self._cache:
            return self._cache[name]

        if name in self._dataset.variables:
            value = self._dataset.variable(name)
        else:
            try:
                value = _compute_via_graph(self, name)
            except UnresolvableFieldError as exc:
                raise IndexError(str(exc)) from exc

        if self._cache_enabled:
            self._cache[name] = value
        return value

    def explain(self, name: str, *, return_tree: bool = False):
        """
        Human-readable explanation of the chosen griblet path for a field.
        Used by: `SmartDs` users and internal methods
        """
        return _explain_field(self, name, return_tree=return_tree)

    def base_fields_for_resample(self, fields: Sequence[str]) -> tuple[str, ...]:
        """
        Resolve requested fields to raw field dependencies for interpolation.
        Used by: `starwinds_analysis/analysis/trajectories.py`, `starwinds_analysis/analysis/shells.py`,
          `starwinds_analysis/analysis/slices.py`
        """
        base_fields: list[str] = []

        def add(name: str) -> None:
            """Append a field name once while preserving insertion order."""
            if name not in base_fields:
                base_fields.append(name)

        def raw_leaves(node) -> list[str]:
            """Collect raw dataset leaf fields from one resolved dependency tree node."""
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
                _cost, tree = self.explain(field, return_tree=True)
            except (IndexError, KeyError, RuntimeError):
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
        """
        Generic resampling entry point returning a new SmartDs (flat or structured targets).

        Expected workflow:
        1. Create target points.
        2. Resample this SmartDs onto those points.
        3. Append any extra context fields to the returned SmartDs if needed.

        Used by: `SmartDs` users and internal methods
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
            cache_enabled=self._cache_enabled,
            computation_graph=self._computation_graph,
            include_aux_in_loader=self._include_aux_in_loader,
        )

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
