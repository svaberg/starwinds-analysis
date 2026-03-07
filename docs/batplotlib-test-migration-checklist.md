# Batplotlib Test Migration Checklist

Last reviewed: 2026-03-07 (`dev`)

## Purpose

Track migration coverage from old
`/Users/dagfev/Documents/starwinds/batplotlib/tests`
into
`/Users/dagfev/Documents/starwinds/starwinds-analysis/test`.

This is a coverage bookkeeping document, not a requirement for strict 1:1 test ports.

## Snapshot (Current)

- Old `batplotlib` test files: `23`
- New repo test files: `23`
- Exact filename overlap: `1` (`test_volumetric.py`)

Notes:

- New tests are mostly split by domain (`shell`, `orbit`, `surface`, `pipeline`, `param`, `smart_ds`).
- Several legacy paths were replaced by new architecture (SmartDs + recipes + pipelines).

Verification baseline (migration-critical subset):

- Last verified: 2026-03-07
- Environment: `starwinds-analysis` conda env
- Command:

```bash
conda run -n starwinds-analysis python -m pytest -q \
  test/test_shell_analysis.py \
  test/test_surface_torque_analysis.py \
  test/test_orbit_analysis.py \
  test/test_orbit_pressure.py \
  test/test_orbit_surface_analysis.py
```

- Result: `39 passed, 2 warnings in 0.67s`
- Warning source: `starwinds_analysis/physics/orbit_surface.py` runtime warnings in the tested path.

## Status Legend

- `Migrated`: behavior is covered in the new suite (possibly split across files)
- `Partial`: core formulas/workflows are covered, but not all legacy edges
- `Deferred`: intentionally postponed
- `Out of Scope`: not part of current migration goals

## Mapping (Old -> New)

| Old batplotlib test | New equivalent(s) | Status | Notes |
| --- | --- | --- | --- |
| `test_elliptic_orbit.py` | `test/test_orbit_analysis.py`, `test/test_orbit_pressure.py`, `test/test_orbit_surface_analysis.py`, `test/test_planetary_orbits.py` | Migrated | Orbit geometry/sampling and derived diagnostics split across focused modules |
| `test_integral_physical.py` | `test/test_shell_analysis.py`, `test/test_surface_torque_analysis.py` | Partial | Core shell/surface integrals are covered |
| `test_torque.py` | `test/test_shell_analysis.py`, `test/test_surface_torque_analysis.py`, `test/test_orbit_surface_analysis.py` | Partial | Shell and explicit-surface torque paths are covered |
| `test_histograms.py` | (none dedicated) | Deferred | Plot primitives exist; dedicated migration tests still pending |
| `test_fibonacci_sphere.py` | `test/test_shell_analysis.py`, `test/test_surface_torque_analysis.py` | Partial | Covered indirectly through shell and torque workflows |
| `test_polar_azimuthal_plot.py` | `test/test_shell_analysis.py` | Partial | Covered indirectly through shell-grid workflows |
| `test_geometry.py` | `test/test_shell_analysis.py`, `test/test_orbit_surface_analysis.py` | Partial | Geometry checks exist in shell/surface tests |
| `test_quantiles.py` | `test/test_shell_analysis.py` | Partial | Weighted quantile/summaries covered in shell analytics |
| `test_vector_fields.py` | `test/test_smart_ds.py`, `test/test_shell_analysis.py`, `test/test_surface_torque_analysis.py` | Partial | Vector/spherical usage covered in downstream diagnostics |
| `test_zone_coordinate_transforms.py` | `test/test_smart_ds.py`, `test/test_shell_analysis.py` | Partial | Coordinate/derived behavior covered functionally |
| `test_load_file.py` | `test/test_smart_ds.py`, `test/test_read_plt.py` | Partial | New loader path tested; legacy reader expectations differ |
| `test_integral.py` | `test/test_shell_analysis.py`, `test/test_surface_torque_analysis.py` | Partial | Formula-level coverage present |
| `test_numpy_save.py` | `test/test_sw_pipe.py`, `test/test_sw_pipe_results.py` | Migrated | Recorder-backed JSON persistence + inspector tooling tested |
| `test_units.py` | (none dedicated) | Deferred | Unit-framework choices remain intentionally constrained to SI-first graph access |
| `test_confidence_bands.py` | (none dedicated) | Deferred | No dedicated confidence-band migration tests yet |
| `test_color_maps.py` | (none) | Out of Scope | Legacy color-map specifics are not migration drivers |
| `test_cartopy.py` | (none) | Out of Scope | Cartopy workflow not in current scope |
| `test_chiantipy_spectrum.py` | (none) | Out of Scope | Chianti/spectrum functionality not in scope |
| `test_swmf_log_parser.py` | (none) | Out of Scope | SWMF log parser not part of current migration |
| `test_field_rejection.py` | (none) | Out of Scope | Legacy field-filtering path not part of current design |
| `test_of_pytest.py` | (none) | Out of Scope | Framework/demo utility |
| `test_of_test_context.py` | (none) | Out of Scope | Framework/context helper |
| `test_volumetric.py` | `test/test_volumetric.py` | Deferred | 3D visualisation path remains secondary |

## Suggested Next Migration Targets

1. Add dedicated algorithm tests for `fibonacci_sphere` and `PolarAzimuthalGrid`.
2. Add dedicated transform tests for core spherical/vector conversion primitives.
3. Keep legacy-compat tests only when they validate current architecture goals.
