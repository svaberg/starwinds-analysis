# Shim-Sized Function Audit
Short-function scan for `starwinds_analysis/` to find functions that look like shims/wrappers (low-signal code).
This is a heuristic audit, not an automatic delete list. The point is to spot places where a function call replaces fewer than ~3-4 real lines without protecting tricky logic.
## Heuristic
- `stmt_count <= 2`, or `stmt_count <= 4` and mostly returns/delegates (`callish`)
- Excludes nothing automatically; exceptions are listed explicitly below
- Current counts: `209` total functions/methods, `93` with `<=4` statements, `45` shim-like by heuristic
## Strong Shim-Smell Candidates (manual review first)
- `starwinds_analysis/data/samples.py:11` `sample_data_dir` (1 stmts): One-line path wrapper; user explicitly chose to keep this for future complexity.
## SmartDs Passthrough/Delegator Cluster (likely intentional API surface, but review size creep)
- `starwinds_analysis/smart_ds.py:72` `SmartDs.__repr__` (1 stmts)
- `starwinds_analysis/smart_ds.py:82` `SmartDs.__str__` (1 stmts)
- `starwinds_analysis/smart_ds.py:98` `SmartDs.from_file` (1 stmts)
- `starwinds_analysis/smart_ds.py:126` `SmartDs.raw` (1 stmts)
- `starwinds_analysis/smart_ds.py:134` `SmartDs.dataset` (1 stmts)
- `starwinds_analysis/smart_ds.py:143` `SmartDs.aux` (1 stmts)
- `starwinds_analysis/smart_ds.py:151` `SmartDs.title` (1 stmts)
- `starwinds_analysis/smart_ds.py:159` `SmartDs.zone` (1 stmts)
- `starwinds_analysis/smart_ds.py:167` `SmartDs.points` (1 stmts)
- `starwinds_analysis/smart_ds.py:175` `SmartDs.corners` (1 stmts)
- `starwinds_analysis/smart_ds.py:183` `SmartDs.variables` (1 stmts)
- `starwinds_analysis/smart_ds.py:191` `SmartDs.field_functions` (1 stmts)
- `starwinds_analysis/smart_ds.py:199` `SmartDs.computation_graph` (1 stmts)
- `starwinds_analysis/smart_ds.py:221` `SmartDs.__contains__` (1 stmts)
- `starwinds_analysis/smart_ds.py:228` `SmartDs.__call__` (1 stmts)
- `starwinds_analysis/smart_ds.py:235` `SmartDs.__getitem__` (1 stmts)
- `starwinds_analysis/smart_ds.py:244` `SmartDs.has_raw_field` (1 stmts)
- `starwinds_analysis/smart_ds.py:266` `SmartDs.get` (1 stmts)
- `starwinds_analysis/smart_ds.py:276` `SmartDs.set_alias` (2 stmts)
- `starwinds_analysis/smart_ds.py:409` `SmartDs.clear_cache` (2 stmts)
- `starwinds_analysis/smart_ds.py:449` `SmartDs.resolve` (1 stmts)
- `starwinds_analysis/smart_ds.py:461` `SmartDs.explain` (1 stmts)
- `starwinds_analysis/smart_ds.py:468` `SmartDs._compute_via_graph` (1 stmts)
- `starwinds_analysis/smart_ds.py:475` `SmartDs.resample` (1 stmts)
## Likely Acceptable Compact Functions (exceptions / easy-to-get-wrong logic)
- `starwinds_analysis/_smart_ds_graph.py:138` `import_griblet` (1 stmts)
- `starwinds_analysis/algorithms/sphere_sampling.py:71` `PolarAzimuthalGrid.polar_edges` (1 stmts)
- `starwinds_analysis/algorithms/sphere_sampling.py:79` `PolarAzimuthalGrid.azimuthal_edges` (1 stmts)
- `starwinds_analysis/algorithms/sphere_sampling.py:116` `PolarAzimuthalGrid.cell_area` (2 stmts)
- `starwinds_analysis/algorithms/sphere_sampling.py:137` `PolarAzimuthalGrid.corners_cartesian` (1 stmts)
- `starwinds_analysis/algorithms/sphere_sampling.py:144` `PolarAzimuthalGrid.centres_cartesian` (1 stmts)
- `starwinds_analysis/recipes/spherical.py:311` `build_griblet_spherical_geometry_graph._r` (2 stmts)
- `starwinds_analysis/recipes/spherical.py:319` `build_griblet_spherical_geometry_graph._theta` (2 stmts)
- `starwinds_analysis/recipes/spherical.py:327` `build_griblet_spherical_geometry_graph._phi` (2 stmts)
- `starwinds_analysis/recipes/spherical.py:392` `build_griblet_vector_spherical_components_graph._all` (1 stmts)
## Recently Removed Shim Candidates
- `starwinds_analysis/_smart_ds_resample.py:interpolate_nd` (inlined into `resample_smart_ds(...)`)
- `starwinds_analysis/pipelines/quicklook2d.py:_resolve_first_field` (inlined at the only call site)
- `starwinds_analysis/pipelines/quicklook2d.py:_orbit_result_title` (inlined at orbit quicklook title call sites)
- `starwinds_analysis/pipelines/quicklook2d.py:_open_wind_magnetisation_from_diagnostics` (inlined at the two quicklook call sites)
- `starwinds_analysis/recipes/batsrus.py:_parse_float` (inlined at the two call sites)
- `starwinds_analysis/analysis/shells.py:shell_profile_radius_height` (inlined at shell profile return sites)
## Full Heuristic List (shim-like)
| File | Line | Function | Stmts | Span | Notes |
|---|---:|---|---:|---:|---|
| `starwinds_analysis/_smart_ds_graph.py` | 138 | `import_griblet` | 1 | 12 | likely-acceptable |
| `starwinds_analysis/algorithms/sphere_sampling.py` | 71 | `PolarAzimuthalGrid.polar_edges` | 1 | 6 | likely-acceptable |
| `starwinds_analysis/algorithms/sphere_sampling.py` | 79 | `PolarAzimuthalGrid.azimuthal_edges` | 1 | 6 | likely-acceptable |
| `starwinds_analysis/algorithms/sphere_sampling.py` | 116 | `PolarAzimuthalGrid.cell_area` | 2 | 7 | likely-acceptable |
| `starwinds_analysis/algorithms/sphere_sampling.py` | 137 | `PolarAzimuthalGrid.corners_cartesian` | 1 | 6 | likely-acceptable, delegates |
| `starwinds_analysis/algorithms/sphere_sampling.py` | 144 | `PolarAzimuthalGrid.centres_cartesian` | 1 | 10 | likely-acceptable, delegates |
| `starwinds_analysis/analysis/shells.py` | 15 | `_resample_shell_points` | 2 | 27 |  |
| `starwinds_analysis/data/samples.py` | 11 | `sample_data_dir` | 1 | 6 | strong-smell |
| `starwinds_analysis/physics/local_estimates.py` | 14 | `local_mass_loss_estimates` | 1 | 8 |  |
| `starwinds_analysis/physics/pressure.py` | 13 | `magnetic_pressure` | 1 | 6 |  |
| `starwinds_analysis/physics/pressure.py` | 20 | `ram_pressure` | 1 | 7 |  |
| `starwinds_analysis/pipelines/quicklook2d.py` | 131 | `_load_slice_styles` | 2 | 24 |  |
| `starwinds_analysis/pipelines/quicklook2d.py` | 675 | `_plot_phase_quantile_band._pick` | 2 | 7 |  |
| `starwinds_analysis/recipes/spherical.py` | 311 | `build_griblet_spherical_geometry_graph._r` | 2 | 7 | likely-acceptable |
| `starwinds_analysis/recipes/spherical.py` | 319 | `build_griblet_spherical_geometry_graph._theta` | 2 | 7 | likely-acceptable |
| `starwinds_analysis/recipes/spherical.py` | 327 | `build_griblet_spherical_geometry_graph._phi` | 2 | 7 | likely-acceptable |
| `starwinds_analysis/recipes/spherical.py` | 392 | `build_griblet_vector_spherical_components_graph._all` | 1 | 6 | likely-acceptable, delegates |
| `starwinds_analysis/smart_ds.py` | 72 | `SmartDs.__repr__` | 1 | 9 | SmartDs API |
| `starwinds_analysis/smart_ds.py` | 82 | `SmartDs.__str__` | 1 | 14 | SmartDs API, delegates |
| `starwinds_analysis/smart_ds.py` | 98 | `SmartDs.from_file` | 1 | 6 | SmartDs API, delegates |
| `starwinds_analysis/smart_ds.py` | 126 | `SmartDs.raw` | 1 | 6 | SmartDs API, delegates |
| `starwinds_analysis/smart_ds.py` | 134 | `SmartDs.dataset` | 1 | 7 | SmartDs API, delegates |
| `starwinds_analysis/smart_ds.py` | 143 | `SmartDs.aux` | 1 | 6 | SmartDs API, delegates |
| `starwinds_analysis/smart_ds.py` | 151 | `SmartDs.title` | 1 | 6 | SmartDs API, delegates |
| `starwinds_analysis/smart_ds.py` | 159 | `SmartDs.zone` | 1 | 6 | SmartDs API, delegates |
| `starwinds_analysis/smart_ds.py` | 167 | `SmartDs.points` | 1 | 6 | SmartDs API, delegates |
| `starwinds_analysis/smart_ds.py` | 175 | `SmartDs.corners` | 1 | 6 | SmartDs API, delegates |
| `starwinds_analysis/smart_ds.py` | 183 | `SmartDs.variables` | 1 | 6 | SmartDs API, delegates |
| `starwinds_analysis/smart_ds.py` | 191 | `SmartDs.field_functions` | 1 | 6 | SmartDs API, delegates |
| `starwinds_analysis/smart_ds.py` | 199 | `SmartDs.computation_graph` | 1 | 6 | SmartDs API, delegates |
| `starwinds_analysis/smart_ds.py` | 221 | `SmartDs.__contains__` | 1 | 6 | SmartDs API |
| `starwinds_analysis/smart_ds.py` | 228 | `SmartDs.__call__` | 1 | 6 | SmartDs API, delegates |
| `starwinds_analysis/smart_ds.py` | 235 | `SmartDs.__getitem__` | 1 | 8 | SmartDs API, delegates |
| `starwinds_analysis/smart_ds.py` | 244 | `SmartDs.has_raw_field` | 1 | 6 | SmartDs API |
| `starwinds_analysis/smart_ds.py` | 266 | `SmartDs.get` | 1 | 9 | SmartDs API |
| `starwinds_analysis/smart_ds.py` | 276 | `SmartDs.set_alias` | 2 | 8 | SmartDs API |
| `starwinds_analysis/smart_ds.py` | 409 | `SmartDs.clear_cache` | 2 | 10 | SmartDs API |
| `starwinds_analysis/smart_ds.py` | 449 | `SmartDs.resolve` | 1 | 11 | SmartDs API, delegates |
| `starwinds_analysis/smart_ds.py` | 461 | `SmartDs.explain` | 1 | 6 | SmartDs API, delegates |
| `starwinds_analysis/smart_ds.py` | 468 | `SmartDs._compute_via_graph` | 1 | 6 | SmartDs API, delegates |
| `starwinds_analysis/smart_ds.py` | 475 | `SmartDs.resample` | 1 | 33 | SmartDs API, delegates |
| `starwinds_analysis/utils.py` | 63 | `extract_index` | 2 | 8 |  |
