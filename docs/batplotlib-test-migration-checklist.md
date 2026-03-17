# Batplotlib Test Migration Checklist

## Purpose

Track migration status from old `/Users/dagfev/Documents/starwinds/batplotlib/tests`
into this repo's test suite (`/Users/dagfev/Documents/starwinds/batwind/test`).

This is file-level bookkeeping, not a claim that tests must be migrated 1:1.
Many old tests were tied to older plotting stacks or test-framework-specific behavior and
are intentionally out of scope for the current NumPy/SciPy-first quicklook migration.

## Snapshot (Current)

- Old `batplotlib` test files: `23`
- New repo test files: `16`
- Exact filename overlap: `1` (`test_volumetric.py`)
- Current full suite in `batwind` env: `74 passed, 14 skipped`

Notes:

- Several legacy tests in this repo were previously skipped when the old `reader` API was
  unavailable.
- The new repo has many renamed/split tests that cover old quicklook behavior in a more
  modular way (shells, orbits, quicklook2d, orbit-surface, surface-torque).

## Status Legend

- `Migrated`: Functionality is covered in the new suite (possibly split across files).
- `Partial`: Core behavior or formulas are covered, but not all old workflows/cases.
- `Deferred`: Intentionally postponed (often legacy-plotting-adjacent or non-quicklook).
- `Out of Scope`: Not part of the current quicklook/core analysis migration.

## Mapping (Old -> New)

| Old batplotlib test | Quicklook relevance | New equivalent(s) in this repo | Status | Notes |
| --- | --- | --- | --- | --- |
| `test_elliptic_orbit.py` | High | `test/test_orbit_analysis.py`, `test/test_orbit_pressure.py`, `test/test_orbit_surface_analysis.py`, `test/test_planetary_orbits.py`, `test/test_quicklook2d.py` | Migrated | Split into orbit geometry/sampling, local estimates, pressure, orbit-surface diagnostics, and runner coverage. |
| `test_integral_physical.py` | High | `test/test_shell_analysis.py`, `test/test_surface_torque_analysis.py` | Partial | Physical shell integrals (mass/open flux/torque comparisons) covered; Tecplot integral machinery not ported. |
| `test_torque.py` | High | `test/test_shell_analysis.py`, `test/test_surface_torque_analysis.py`, `test/test_orbit_surface_analysis.py` | Partial | Spherical-shell torque + explicit-surface torque core are covered; automatic surface extraction workflows are deferred. |
| `test_histograms.py` | High | `test/test_quicklook2d.py` | Partial | Radius scatter/binned/CDF + modernized `hist2d` mode covered; not a byte-for-byte Tecplot-style histogram port. |
| `test_fibonacci_sphere.py` | Medium | `test/test_shell_analysis.py`, `test/test_surface_torque_analysis.py` | Partial | Fibonacci sampling is exercised via shell area/exactness and torque integration tests; no standalone algorithm-only test yet. |
| `test_polar_azimuthal_plot.py` | Medium | `test/test_shell_analysis.py` | Partial | Polar/azimuthal grid behavior covered indirectly via shell sampling and axisymmetric flux diagnostics. |
| `test_geometry.py` | Medium | `test/test_shell_analysis.py`, `test/test_orbit_surface_analysis.py` | Partial | Shell/orbit-surface geometry checks exist, but no direct port of old geometry utility tests. |
| `test_quantiles.py` | Medium | `test/test_shell_analysis.py`, `test/test_quicklook2d.py` | Partial | Weighted quantiles/summaries and phase quantile outputs covered in new analytics. |
| `test_vector_fields.py` | Medium | `test/test_smart_ds.py`, `test/test_shell_analysis.py`, `test/test_surface_torque_analysis.py` | Partial | Spherical vector usage covered in downstream analytics; no dedicated vector-transform test module yet. |
| `test_zone_coordinate_transforms.py` | Medium | `test/test_smart_ds.py`, `test/test_shell_analysis.py` | Partial | Spherical-coordinate/derived-component behavior covered functionally, not as direct transform unit tests. |
| `test_load_file.py` | Medium | `test/test_smart_ds.py`, `test/test_read_plt.py` (legacy) | Partial | New `SmartDs`/reader path is tested, but legacy `reader` import compatibility path is currently skipped. |
| `test_integral.py` | Medium | `test/test_shell_analysis.py`, `test/test_surface_torque_analysis.py` | Partial | New shell/surface integration formulas tested; old Tecplot integration API tests are not ported. |
| `test_numpy_save.py` | Medium | `test/test_quicklook2d.py` | Partial | New JSON/NPZ bundle export is tested; old exact save/load patterns not ported 1:1. |
| `test_units.py` | Medium | (none; unit tests deferred) | Deferred | Unit-framework decisions intentionally deferred while pursuing SI gatekeeping at wrapper boundary. |
| `test_confidence_bands.py` | Low | `test/test_quicklook2d.py` (indirect plotting summaries) | Deferred | Plot utility coverage exists, but confidence-band-specific test logic not ported. |
| `test_color_maps.py` | Low | (none) | Out of Scope | Tecplot color-map tooling. |
| `test_cartopy.py` | Low | (none) | Out of Scope | Cartopy mapping workflow not part of current quicklook/core migration. |
| `test_chiantipy_spectrum.py` | Low | (none) | Out of Scope | Chianti/spectrum-specific functionality not in current scope. |
| `test_swmf_log_parser.py` | Low | (none) | Out of Scope | SWMF log parsing not part of current quicklook migration. |
| `test_field_rejection.py` | Low | (none) | Out of Scope | Legacy field-filtering behavior not currently part of migrated quicklook path. |
| `test_of_pytest.py` | Low | (none) | Out of Scope | Test-framework/demo utility. |
| `test_of_test_context.py` | Low | (none) | Out of Scope | Test-context/helper behavior for old repo. |
| `test_volumetric.py` | Low (for current plan) | `test/test_volumetric.py` (legacy, currently skipped on missing `reader`) | Deferred | 3D volumetric workflows are not a priority for the current quicklook migration. |

## New Test Modules With No Direct Old Filename Match

These are mostly the modern replacements for old quicklook monolith behavior:

- `test/test_smart_ds.py`
- `test/test_shell_analysis.py`
- `test/test_slices_analysis.py`
- `test/test_quicklook2d.py`
- `test/test_orbit_analysis.py`
- `test/test_orbit_pressure.py`
- `test/test_orbit_surface_analysis.py`
- `test/test_surface_torque_analysis.py`
- `test/test_planetary_orbits.py`

## Suggested Next Migration Targets (If We Keep Going)

1. Add standalone algorithm tests for `fibonacci_sphere(...)` and `PolarAzimuthalGrid` (clean port of old sampling tests).
2. Add dedicated vector/spherical transform tests (instead of only downstream analytic coverage).
3. Decide whether to migrate legacy `reader` tests or permanently retire them in favor of `SmartDs` tests.
4. Revisit `test_units.py` only after the SI/unit-framework direction is finalized.
