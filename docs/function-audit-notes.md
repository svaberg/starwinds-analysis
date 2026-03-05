# Function Audit Notes (What They Do / Where Used)

Short notes for library functions/classes in `starwinds_analysis`.
Usage notes are direct caller locations (grep-based, then manually tightened for key APIs).

Status note:
- This is a historical audit snapshot.
- Entries mentioning removed `quicklook2d.py`, removed `test/test_quicklook2d.py`, or removed `theta`/`phi` field aliases are stale and should be treated as audit debt until this file is rerun against the current codebase.
- Additional stale markers in this snapshot:
  - Orbit-surface APIs were renamed to trajectory-oriented names (`sample_surface_revolution`, `pressure_components_on_surface`, `torque_components_on_surface`).
  - `periodic_curve_velocity(...)` was removed; use `trajectory_velocity(...)` with explicit `time`.

## `starwinds_analysis/_smart_ds_graph.py`
- `graph_field_names`: List available field names from the runtime griblet graph. Used in no external call sites found.
- `resolve_field`: Resolve a field through the attached griblet computation graph. Used in `starwinds_analysis/_smart_ds_graph.py`.
- `explain_field`: Build a human-readable explanation of the chosen graph path. Used in no external call sites found.
- `compute_via_graph`: Compute a field by resolving + evaluating a griblet graph path. Used in no external call sites found.
- `build_runtime_graph`: Merge loader graph and user graph into a runtime graph for one SmartDs instance. Used in `starwinds_analysis/_smart_ds_graph.py`.
- `build_loader_graph`: Build a zero-dependency graph exposing raw dataset variables (+ selected aux). Used in `starwinds_analysis/_smart_ds_graph.py`.
- `import_griblet`: Import griblet lazily and return required runtime pieces. Used in `starwinds_analysis/_smart_ds_graph.py`.
- `evaluate_resolved_tree`: Evaluate a griblet-resolved computation tree. Used in `starwinds_analysis/_smart_ds_graph.py`.

## `starwinds_analysis/_smart_ds_resample.py`
- `resample_smart_ds`: Resample scalar fields onto new point locations and return a new wrapped dataset. Used in `starwinds_analysis/smart_ds.py`.
- `interpolate_nd`: Interpolate one field onto target points using SciPy nearest/linear ND interpolators. Used in `starwinds_analysis/_smart_ds_resample.py`.

## `starwinds_analysis/algorithms/sphere_sampling.py`
- `fibonacci_sphere`: Generate approximately uniformly distributed points on the unit sphere Used in `test/test_surface_torque_analysis.py`, `starwinds_analysis/analysis/shells.py`.
- `PolarAzimuthalGrid` (class): Spherical surface grid defined by polar (colatitude) and azimuthal edges. Used by shell map notebooks/tests and shell sampling geometry helpers for polar/azimuthal grids.
  - `PolarAzimuthalGrid.polar_edges`: Returns the polar-angle edge grid (theta edges). Used by shell-map notebooks for edge geometry and plotting extents.
  - `PolarAzimuthalGrid.azimuthal_edges`: Returns the azimuth edge grid (phi edges). Used with `polar_edges` for shell map corners.
  - `PolarAzimuthalGrid.polar_centres`: Returns theta cell centres from polar edges. Used when centre sampling is desired on a polar/azimuth grid.
  - `PolarAzimuthalGrid.azimuthal_centres`: Returns phi cell centres from azimuth edges. Used with `polar_centres` for centre grids.
  - `PolarAzimuthalGrid.cell_solid_angle`: Returns per-cell solid angle on the unit sphere. Used to build shell areas and integration weights.
  - `PolarAzimuthalGrid.cell_area`: Returns per-cell area for a supplied radius. Used by shell workflows that want explicit finite cell measures.
  - `PolarAzimuthalGrid.corners_cartesian`: Maps edge-angle grid to Cartesian corner points at a radius. Used by the magnetic ZDI notebook and shell corner resampling.
  - `PolarAzimuthalGrid.centres_cartesian`: Maps centre-angle grid to Cartesian centre points at a radius. Used in shell workflows/notebooks when centre sampling is chosen.

## `starwinds_analysis/analysis/shell_summary.py`
- `boxcar_shell_weights`: Boxcar weights over shell radii in units of body radii. Used in `test/test_shell_analysis.py`, `starwinds_analysis/analysis/shell_summary.py`.
- `summarize_shell_series`: Weighted summary (mean/std/quantiles) for a 1D shell profile series. Used in `test/test_shell_analysis.py`, `starwinds_analysis/analysis/shell_summary.py`.
- `summarize_shell_diagnostics_band`: Summarize all 1D shell-profile series in a diagnostics bundle over a shell-radius band. Used in `test/test_shell_analysis.py`, `starwinds_analysis/pipelines/quicklook2d.py`.

## `starwinds_analysis/analysis/shells.py`
- `_resample_shell_points`: Resample requested fields onto explicit shell points and return a shell SmartDs. Used in `starwinds_analysis/analysis/shells.py`.
- `unit_from_brackets`: Extract the unit substring from a bracketed field name like `X [R]`. Used in `starwinds_analysis/analysis/shells.py`.
- `_append_fields_to_smart_ds`: Attach derived arrays (free coords/areas/etc.) to a resampled shell SmartDs. Used in `starwinds_analysis/analysis/shells.py`.
- `infer_body_radius_m`: Infer the body radius in meters from args/aux so shell/orbit lengths can be converted to SI. Used in `starwinds_analysis/physics/orbit_local.py`, `starwinds_analysis/physics/orbit_surface.py`, `starwinds_analysis/physics/fluxes.py`, `starwinds_analysis/physics/mass_loss.py`, `starwinds_analysis/physics/torque.py` (+1 more).
- `infer_cartesian_axis_radii`: Infer available shell radii from points lying on a Cartesian axis. Used in `test/test_shell_analysis.py`.
- `sample_spherical_shells`: Resample fields onto spherical shell cell centers. Used in `test/test_shell_magnetic_analysis.py`, `test/test_shell_analysis.py`, `test/test_shell_resample_smartds_spec.py`, `examples/smartds_quicklook_profiles.ipynb`, `examples/smartds_shell_mass_flux.ipynb` (+1 more).
- `sample_spherical_shells_fibonacci`: Resample fields onto equal-area Fibonacci sphere points on each shell. Used in `test/test_shell_analysis.py`, `examples/smartds_shell_mass_flux.ipynb`, `starwinds_analysis/analysis/shells.py`.
- `integrate_shell_scalar`: Integrate scalar values over shell surfaces with NaN-safe area weighting. Used in `test/test_shell_magnetic_analysis.py`, `test/test_shell_analysis.py`, `examples/smartds_inner_boundary_magnetic_zdi.ipynb`, `examples/smartds_quicklook_profiles.ipynb`, `examples/smartds_shell_mass_flux.ipynb` (+3 more).
- `shell_profile_radius_height`: Build standard radius/height profile arrays from a shell SmartDs. Used in `starwinds_analysis/physics/fluxes.py`, `starwinds_analysis/physics/mass_loss.py`, `starwinds_analysis/physics/torque.py`.

## `starwinds_analysis/analysis/slices.py`
- `structured_quad_corners`: Quad connectivity for a row-major `(nz, nx)` point grid. Used in `test/test_slices_analysis.py`, `starwinds_analysis/analysis/slices.py`.
- `infer_range`: Infer a plotting/resampling range from data with optional symmetry/padding. Used in `test/test_slices_analysis.py`, `starwinds_analysis/analysis/slices.py`.
- `resample_structured_xz_slice`: Resample a 3D dataset onto a structured XZ plane and return a new `SmartDs`. Used in `test/test_slices_analysis.py`, `starwinds_analysis/pipelines/quicklook2d.py`.

## `starwinds_analysis/analysis/stats.py`
- `weighted_mean_std`: Weighted mean and standard deviation over finite values. Used in `test/test_shell_analysis.py`, `starwinds_analysis/analysis/shell_summary.py`, `starwinds_analysis/analysis/stats.py`.
- `weighted_quantile`: Weighted quantiles for 1D data. Used in `test/test_shell_analysis.py`, `starwinds_analysis/analysis/shell_summary.py`, `starwinds_analysis/analysis/stats.py`.
- `summarize_samples`: Weighted quantiles + mean/std summary for 1D samples. Used in `test/test_shell_analysis.py`, `starwinds_analysis/physics/orbit_local.py`, `starwinds_analysis/physics/orbit_surface.py`, `starwinds_analysis/physics/orbit_pressure.py`.

## `starwinds_analysis/data/samples.py`
- `sample_data_dir`: Return the repository's `sample_data` directory. Used in `test/test_sample_data_helpers.py`, `starwinds_analysis/data/samples.py`.
- `get_sample`: Return an absolute `Path` to a file in `sample_data`. Used in `test/test_shell_magnetic_analysis.py`, `test/test_sample_data_helpers.py`, `examples/smartds_radial_histograms.ipynb`, `examples/smartds_inner_boundary_magnetic_zdi.ipynb`, `examples/smartds_quicklook_profiles.ipynb` (+2 more).

## `starwinds_analysis/physics/fluxes.py`
- `open_magnetic_flux_vs_radius`: Signed/unsigned magnetic flux on spherical shells. Used in `test/test_shell_analysis.py`, `starwinds_analysis/pipelines/quicklook2d.py`, `starwinds_analysis/physics/fluxes.py`.
- `axisymmetric_open_flux_vs_radius`: Axisymmetric open magnetic flux and fraction using shell-sampled B_r. Used in `test/test_shell_analysis.py`, `starwinds_analysis/pipelines/quicklook2d.py`.
- `energy_flux_vs_radius`: Radial energy flux profile using `E * U_r`. Used in `test/test_shell_analysis.py`, `starwinds_analysis/pipelines/quicklook2d.py`.

## `starwinds_analysis/physics/local_estimates.py`
- `local_mass_loss_estimates`: Pointwise local mass-loss estimates using `4*pi*r^2*rho*u_r`. Used in `test/test_shell_analysis.py`, `starwinds_analysis/physics/orbit_local.py`.
- `local_torque_estimates`: Pointwise local torque estimates using the spherical-shell scaling from old quicklook. Used in `test/test_shell_analysis.py`, `starwinds_analysis/physics/orbit_local.py`.

## `starwinds_analysis/physics/magnetic.py`
- `magnetic_field_unit_scale`: Return scale factor + label for plotting magnetic field in `T` or `G`. Used in `test/test_shell_magnetic_analysis.py`.

## `starwinds_analysis/physics/mass_loss.py`
- `mass_loss_vs_radius`: Wind mass-loss profile on spherical shells. Used in `test/test_shell_analysis.py`, `examples/smartds_shell_mass_flux.ipynb`, `starwinds_analysis/pipelines/quicklook2d.py`, `starwinds_analysis/physics/orbit_local.py`.

## `starwinds_analysis/physics/orbit_local.py`
- `_interp_profile`: 1D interpolate a shell profile onto orbit sample radii (with NaNs outside range). Used in `starwinds_analysis/physics/orbit_local.py`.
- `_local_mass_loss_from_orbit_sample`: Compute local mass-loss estimates on one sampled orbit and compare to shell profile values. Used in `starwinds_analysis/physics/orbit_local.py`.
- `_local_torque_from_orbit_sample`: Compute local torque estimates on one sampled orbit and compare to shell torque profile values. Used in `starwinds_analysis/physics/orbit_local.py`.
- `local_mass_loss_on_circular_orbit`: Sample a circular orbit and compute local-vs-shell mass-loss comparisons. Used in `test/test_orbit_analysis.py`, `starwinds_analysis/pipelines/quicklook2d.py`.
- `local_torque_on_circular_orbit`: Sample a circular orbit and compute local-vs-shell torque comparisons. Used in `test/test_orbit_analysis.py`, `starwinds_analysis/pipelines/quicklook2d.py`.
- `local_mass_loss_on_elliptic_orbit`: Sample an elliptic orbit and compute local-vs-shell mass-loss comparisons. Used in `test/test_orbit_analysis.py`, `starwinds_analysis/pipelines/quicklook2d.py`.
- `local_torque_on_elliptic_orbit`: Sample an elliptic orbit and compute local-vs-shell torque comparisons. Used in `test/test_orbit_analysis.py`, `starwinds_analysis/pipelines/quicklook2d.py`.

## `starwinds_analysis/physics/orbit_pressure.py`
- Removed from the current tree; trajectory pressure diagnostics now live in `starwinds_analysis/physics/curve.py` and `starwinds_analysis/physics/orbit_surface.py`.

## `starwinds_analysis/physics/orbit_surface.py`
- `surface_of_revolution_from_trajectory`: Surface of revolution around the z-axis from explicit trajectory points. Used in `test/test_orbit_surface_analysis.py`, `starwinds_analysis/physics/orbit_surface.py`.
- `surface_sample_weights`: Build integration/summary weights for orbit-surface sampled data. Used in `starwinds_analysis/physics/orbit_surface.py`.
- `phase_quantile_rows`: Compute phase-binned quantiles for 2D orbit-surface sampled values. Used in `starwinds_analysis/physics/orbit_surface.py`.
- `surface_point_normals_and_areas`: Estimate point normals and point-associated areas on a periodic structured surface. Used in `test/test_orbit_surface_analysis.py`, `starwinds_analysis/physics/orbit_surface.py`.
- `phase_line_integrals`: Integrate sampled surface density values over longitude for each orbit phase. Used in `starwinds_analysis/physics/orbit_surface.py`.
- `sample_surface_revolution`: Sample explicit fields on a surface of revolution generated from trajectory points. Used in `test/test_orbit_surface_analysis.py`, `starwinds_analysis/physics/orbit_surface.py`.
- `pressure_components_on_surface`: Pressure-component analytics on a sampled surface of revolution. Used in `test/test_orbit_surface_analysis.py`, `examples/orbit_surface_revolution.ipynb`.
- `torque_components_on_surface`: Explicit-surface torque diagnostics on a sampled surface of revolution. Used in `test/test_orbit_surface_analysis.py`, `examples/orbit_surface_revolution.ipynb`.

## `starwinds_analysis/physics/orbits.py`
- `orbital_period`: Keplerian orbital period for a test particle around a point mass. Used in `test/test_orbit_analysis.py`, `starwinds_analysis/physics/orbit_surface.py`, `starwinds_analysis/physics/planetary_orbits.py`, `starwinds_analysis/physics/orbit_pressure.py`.
- `orbital_velocity`: Vis-viva orbital speed. Used in `test/test_orbit_analysis.py`.

## `starwinds_analysis/physics/planetary_orbits.py`
- `PlanetOrbitElements` (class): Typed container for orbital constants in the built-in table. Used in `test/test_planetary_orbits.py`, `starwinds_analysis/physics/planetary_orbits.py`, `starwinds_analysis/pipelines/quicklook2d.py`.
- `SOLAR_SYSTEM_PLANETS` (constant): Built-in orbital constant table used directly by quicklook and tests.

## `starwinds_analysis/physics/pressure.py`
- `magnetic_pressure`: Magnetic pressure `B^2 / (2 mu0)` in Pa. Used in `test/test_orbit_pressure.py`, `starwinds_analysis/physics/pressure.py`.
- `ram_pressure`: Ram pressure `rho * u^2` in Pa. Used in `test/test_orbit_pressure.py`, `starwinds_analysis/physics/pressure.py`, `starwinds_analysis/physics/orbit_pressure.py`.
- `pressure_components`: Compute thermal/magnetic/ram pressure components from local samples. Used in `test/test_orbit_pressure.py`, `starwinds_analysis/physics/orbit_surface.py`.
- `magnetospheric_standoff_distance`: Vidotto-style stand-off distance proxy from pressure balance. Used in `test/test_orbit_pressure.py`, `starwinds_analysis/physics/orbit_surface.py`, `starwinds_analysis/physics/orbit_pressure.py`.

## `starwinds_analysis/physics/torque.py`
- `spherical_wind_torque_density_terms`: Spherical-shell wind torque-density terms about +z. Used in no external call sites found.
- `torque_vs_radius`: Spherical-shell wind torque profile (magnetic + dynamic + total). Used in `test/test_surface_torque_analysis.py`, `test/test_shell_analysis.py`, `starwinds_analysis/pipelines/quicklook2d.py`, `starwinds_analysis/physics/orbit_local.py`.
- `rotational_frame_velocity`: Convert inertial velocity `u` to rotating-frame velocity `V = u - Omega x r` Used in `starwinds_analysis/physics/torque.py`.
- `normalize_surface_normals`: Normalize explicit surface normals safely for torque integration. Used in `starwinds_analysis/physics/torque.py`.
- `radial_surface_normals`: Build radial normals from explicit Cartesian surface points. Used in `starwinds_analysis/physics/torque.py`.
- `surface_torque_density_terms`: Mestel/Vidotto-like z-angular-momentum flux terms on an explicit surface. Used in `test/test_surface_torque_analysis.py`, `starwinds_analysis/physics/orbit_surface.py`, `starwinds_analysis/physics/torque.py`.
- `integrate_surface_torque_terms`: Integrate per-area torque-density terms from `surface_torque_density_terms(...)`. Used in `test/test_surface_torque_analysis.py`, `starwinds_analysis/physics/orbit_surface.py`, `starwinds_analysis/physics/torque.py`.
- `surface_torque_terms_on_shell_samples`: Convenience wrapper for explicit-surface torque terms on shell samples. Used in `starwinds_analysis/physics/torque.py`.
- `surface_torque_vs_radius`: Explicit-surface torque profile on spherical shells using general T1..T4 terms. Used in `test/test_surface_torque_analysis.py`.

## `starwinds_analysis/physics/wind_scaling.py`
- `surface_escape_speed`: Surface escape speed `sqrt(2GM/R)`. Used in `test/test_shell_analysis.py`, `starwinds_analysis/physics/wind_scaling.py`.
- `open_wind_magnetisation`: Reville-style open wind magnetisation used in the old quicklook (`Upsilon_open`). Used in `test/test_shell_analysis.py`, `starwinds_analysis/pipelines/quicklook2d.py`.

## `starwinds_analysis/pipelines/quicklook2d.py`
- `SlicePreset` (class): SlicePreset class. Used inside quicklook2d slice preset registries for field/plot configuration.
- `_open_wind_magnetisation_from_diagnostics`: Local quicklook adapter from shell-profile dicts to the `Upsilon_open` formula. Used in `starwinds_analysis/pipelines/quicklook2d.py`.
- `_load_slice_styles`: Lazy-import slice plotting styles/helpers only when slice quicklooks are used. Used in `starwinds_analysis/pipelines/quicklook2d.py`.
- `_has_field`: Check if a SmartDs has a field without raising in quicklook field selection. Used in `starwinds_analysis/pipelines/quicklook2d.py`.
- `_resolve_first_field`: Pick the first available field from a candidate list. Used in `starwinds_analysis/pipelines/quicklook2d.py`.
- `_normalize_overlays`: Normalize overlay specs into a list for quicklook plotting. Used in `starwinds_analysis/pipelines/quicklook2d.py`.
- `plot_slice_quicklook`: Thin 2D quicklook wrapper over existing slice plotting helpers. Used in `test/test_quicklook2d.py`, `starwinds_analysis/pipelines/quicklook2d.py`.
- `plot_radius_quicklook`: Radius/scatter/cumulative/hist2d quicklook wrapper over `visualisation.histograms`. Used in `test/test_quicklook2d.py`, `starwinds_analysis/pipelines/quicklook2d.py`.
- `compute_shell_diagnostics`: Compute a bundle of shell diagnostics used by 2D quicklook summaries. Used in `starwinds_analysis/pipelines/quicklook2d.py`.
- `plot_shell_diagnostics`: Plot a compact shell-diagnostics summary figure. Used in `starwinds_analysis/pipelines/quicklook2d.py`.
- `quicklook_shell_figure`: Convenience shell-diagnostics figure builder used by quicklook and tests. Used in `test/test_quicklook2d.py`, `starwinds_analysis/pipelines/quicklook2d.py`.
- `plot_orbit_mass_loss_comparison`: Plot local-vs-shell mass-loss comparison for one orbit result bundle. Used in `starwinds_analysis/pipelines/quicklook2d.py`.
- `plot_orbit_torque_comparison`: Plot local-vs-shell torque comparison for one orbit result bundle. Used in `starwinds_analysis/pipelines/quicklook2d.py`.
- `_orbit_phase`: Extract orbit phase array from result bundles with a safe empty fallback. Used in `starwinds_analysis/pipelines/quicklook2d.py`.
- `_orbit_result_title`: Build a compact title string for orbit quicklook result figures. Used in `starwinds_analysis/pipelines/quicklook2d.py`.
- `plot_orbit_pressure_components`: Plot orbit pressure-component series and derived standoff proxy. Used in `starwinds_analysis/pipelines/quicklook2d.py`.
- `orbit_pressure_figure`: Orbit pressure quicklook (thermal/magnetic/ram and stand-off proxy). Used in `test/test_quicklook2d.py`.
- `_plot_phase_quantile_band`: Plot filled phase-quantile bands for orbit-surface diagnostics. Used in `starwinds_analysis/pipelines/quicklook2d.py`.
- `orbit_surface_pressure_figure`: Surface-of-revolution orbit pressure quicklook (pure NumPy/SciPy resampling). Used in `test/test_quicklook2d.py`, `starwinds_analysis/pipelines/quicklook2d.py`.
- `orbit_surface_torque_figure`: Surface-of-revolution torque quicklook (`T1..T4` + total), independent of dedicated 3D visualisation tooling. Used in `test/test_quicklook2d.py`, `starwinds_analysis/pipelines/quicklook2d.py`.
- `orbit_local_comparison_figure`: Compute and plot local-vs-shell comparisons for mass loss and torque on one orbit. Used in `test/test_quicklook2d.py`, `starwinds_analysis/pipelines/quicklook2d.py`.
- `summarize_shell_diagnostics`: JSON-friendly summary (stats only) of shell diagnostics. Used in `starwinds_analysis/pipelines/quicklook2d.py`.
- `flatten_shell_diagnostics_arrays`: Flatten shell diagnostic arrays for `np.savez`. Used in `starwinds_analysis/pipelines/quicklook2d.py`.
- `summarize_orbit_results`: JSON-friendly summary of orbit local-vs-shell comparison results. Used in `starwinds_analysis/pipelines/quicklook2d.py`.
- `flatten_orbit_results_arrays`: Flatten selected orbit result arrays for `np.savez`. Used in `starwinds_analysis/pipelines/quicklook2d.py`.
- `save_shell_diagnostics_json`: Save shell diagnostics summary JSON to disk. Used in `starwinds_analysis/pipelines/quicklook2d.py`.
- `save_shell_diagnostics_npz`: Save shell diagnostics arrays to NPZ. Used in `starwinds_analysis/pipelines/quicklook2d.py`.
- `save_orbit_results_json`: Save orbit-result summaries to JSON. Used in `starwinds_analysis/pipelines/quicklook2d.py`.
- `save_orbit_results_npz`: Save orbit-result arrays to NPZ. Used in `starwinds_analysis/pipelines/quicklook2d.py`.
- `save_quicklook2d_bundle`: Save figures and shell summaries (JSON/NPZ) as a small quicklook bundle. Used in `test/test_quicklook2d.py`, `starwinds_analysis/pipelines/quicklook2d.py`.
- `_slug_key`: Make a filesystem-safe-ish slug for summary/array keys. Used in `starwinds_analysis/pipelines/quicklook2d.py`.
- `_array_summary`: Return small summary stats for an array (shape/finite/min/max etc.). Used in `starwinds_analysis/pipelines/quicklook2d.py`.
- `_summarize_result_object`: Recursively summarize nested result dicts into JSON-friendly metadata. Used in `starwinds_analysis/pipelines/quicklook2d.py`.
- `_flatten_result_arrays`: Collect arrays from nested result objects into a flat dict for NPZ export. Used in `starwinds_analysis/pipelines/quicklook2d.py`.
- `prepare_smartds_for_quicklook`: Best-effort setup of common BATSRUS + spherical derived fields. Used in `starwinds_analysis/pipelines/quicklook2d.py`.
- `run_quicklook2d`: End-to-end non-3D quicklook runner (figures + shell diagnostics + optional save). Used in `test/test_quicklook2d.py`.

## `starwinds_analysis/recipes/batsrus.py`
- `build_griblet_batsrus_graph`: Build a griblet graph for BATSRUS-style fields. Used in `starwinds_analysis/smart_ds.py`.
- `build_griblet_unit_normalization_graph`: Add raw->SI unit conversion recipes (BATSRUS naming conventions). Used in `starwinds_analysis/recipes/batsrus.py`.
- `build_griblet_common_derived_graph`: Add common BATSRUS derived SI quantities (pressures, Mach numbers, fluxes, torque densities). Used in `starwinds_analysis/recipes/batsrus.py`.
- `build_griblet_vector_magnitude_graph`: Add vector-magnitude recipes (e.g. `|U|`, `|B|`) for available Cartesian triplets. Used in `starwinds_analysis/recipes/batsrus.py`.
- `_parse_var_name`: Parse BATSRUS variable names. Used in `starwinds_analysis/recipes/batsrus.py`.
- `_parse_float`: Parse a float from aux/meta strings with safe fallback to `None`. Used in `starwinds_analysis/recipes/batsrus.py`.
- `_safe_gamma`: Return a physically valid adiabatic index fallback when metadata is missing/bad. Used in `starwinds_analysis/recipes/batsrus.py`.
- `_resolve_body_radius_m`: Resolve body radius in meters from explicit arg or BATSRUS aux metadata. Used in `starwinds_analysis/recipes/batsrus.py`.

## `starwinds_analysis/recipes/spherical.py`
- `cartesian_to_spherical_angles`: Convert Cartesian coordinates to spherical coordinates. Used in `starwinds_analysis/recipes/spherical.py`.
- `radial_component`: Project a Cartesian vector onto the radial direction defined by `(x,y,z)`. Used in no external call sites found.
- `spherical_vector_components`: Return ``(v_r, v_theta, v_phi)`` using physics convention ``theta=colatitude``. Used in `test/test_shell_analysis.py`, `starwinds_analysis/recipes/spherical.py`.
- `register_spherical_geometry_fields`: Register local on-demand spherical coordinate fields on a SmartDs wrapper. Used in `starwinds_analysis/smart_ds.py`.
- `register_vector_spherical_components`: Register local on-demand spherical vector components for one Cartesian vector triplet. Used in `starwinds_analysis/recipes/spherical.py`.
- `_vector_triplets`: Find vector component triplets named like ``prefix_x [unit]``. Used in `starwinds_analysis/recipes/spherical.py`, `starwinds_analysis/smart_ds.py`, and `starwinds_analysis/recipes/batsrus.py`.
- `build_griblet_spherical_geometry_graph`: Build a griblet graph for spherical geometry fields. Used in `test/test_smart_ds.py`, `starwinds_analysis/smart_ds.py`, `starwinds_analysis/recipes/batsrus.py`.
- `build_griblet_vector_spherical_components_graph`: Build griblet recipes for ``prefix_{r,theta,phi}`` from Cartesian components. Used in `starwinds_analysis/recipes/spherical.py`.
- `build_griblet_vector_spherical_components_graph`: Build spherical-component recipes for one detected vector triplet. Used in `starwinds_analysis/recipes/spherical.py`, `starwinds_analysis/smart_ds.py`, and `starwinds_analysis/recipes/batsrus.py`.
- `_infer_radius_name_from_coord`: Infer the matching radius field name/unit from coordinate field names. Used in `starwinds_analysis/recipes/spherical.py`.

## `starwinds_analysis/analysis/trajectories.py`
- `trajectory_velocity`: Velocity from explicit trajectory points and strictly increasing time. Used in `test/test_orbit_analysis.py`, `test/test_orbit_pressure.py`, `test/test_orbit_surface_analysis.py`.
- `circular_orbit_points`: Cartesian points on a circular XY orbit (same coordinate unit as `radius`). Used in `test/test_orbit_analysis.py`, `test/test_orbit_pressure.py`, `test/test_orbit_surface_analysis.py`, `starwinds_analysis/physics/orbit_surface.py`.
- `sample_curve`: Resample requested fields onto explicit Cartesian points and return a curve `SmartDs`. Used in `test/test_orbit_analysis.py`, `starwinds_analysis/analysis/trajectories.py`, `starwinds_analysis/physics/orbit_surface.py`.
- `sample_trajectory`: Resample fields onto trajectory points and append `t` and optional `V_xyz` context fields. Used in `test/test_orbit_pressure.py`.

## `starwinds_analysis/smart_ds.py`
- `SmartDs` (class): Lightweight wrapper around ``starwinds_readplt.Dataset``. Used across notebooks/tests and most physics/sampling workflows as the main dataset wrapper.
  - `SmartDs.from_file`: Constructs a wrapper from a BATSRUS output file path. Used by tests and all example notebooks.
  - `SmartDs.__str__`: Prints a compact summary (title/zone/points/variables). Used in example notebooks via `print(sds)`.
  - `SmartDs.__call__`: Shorthand for `variable(...)` (full SmartDs/griblet path). Used heavily in notebooks and shell/orbit workflows.
  - `SmartDs.__getitem__`: Raw-only field access passthrough to the underlying dataset. Used when callers explicitly want base/raw fields only.
  - `SmartDs.has_field`: Checks raw/alias/computed availability of a field name. Used by recipes/quicklook field selection and diagnostics setup.
  - `SmartDs.add_spherical_fields`: Registers local spherical derived fields without griblet graphs. Used in some notebooks/resampled shell workflows when local spherical fields are enough.
  - `SmartDs.add_spherical_graph`: Attaches spherical griblet recipes (geometry + vector components). Used when graph-based resolution/explainability is desired.
  - `SmartDs.add_batsrus_graph`: Attaches BATSRUS SI/unit/derived recipes into the computation graph. Used by most SI analysis workflows and notebooks before requesting derived SI fields.
  - `SmartDs.variable`: Primary field request API (raw -> alias -> local fields -> griblet). Called across the codebase by analysis/physics modules and notebooks.
  - `SmartDs.resolve`: Returns a griblet resolution/explanation object for a target field. Used mainly for graph introspection/debugging, not ordinary analysis math.
  - `SmartDs.explain`: Human-readable graph path explanation for a requested field. Used in tests and debugging of recipe paths.
  - `SmartDs.resample`: Interpolates onto flat or structured target points and returns a new SmartDs. Used by shell/slice/orbit sampling paths and example notebooks.

## `starwinds_analysis/utils.py`
- `auto_coords`: Detect the two varying coordinates in a nominal 2D slice dataset. Used in `examples/smartds_2d_xy_points.ipynb`, `examples/planet.py`, `starwinds_analysis/utils.py`, `examples/earth-xuv-neutrals/earth-xuv-neutrals.py`.
- `triangles`: Build a Matplotlib triangulation from 2D quad-cell connectivity. Used in `examples/smartds_2d_xy_points.ipynb`, `examples/planet.py`, `starwinds_analysis/pipelines/quicklook2d.py`, `examples/earth-xuv-neutrals/earth-xuv-neutrals.py`.
- `extract_index`: Extract the step number from a filename of the form '..._n00060000.dat'. Used in `examples/planet.py`, `examples/earth-xuv-neutrals/earth-xuv-neutrals.py`.
- `sort_key`: Sort by the number in the filename, with trailing zeros prioritized. Used in no external call sites found.

## `starwinds_analysis/visualisation/histograms.py`
- `plot_cumulative_hists`: Cumulative histogram (CDF line) for each field on the provided axes. Used in `examples/smartds_radial_histograms.ipynb`, `examples/planet.py`, `starwinds_analysis/pipelines/quicklook2d.py`, `examples/earth-xuv-neutrals/earth-xuv-neutrals.py`.
- `plot_vs_radius`: Scatter raw samples vs radius for one or more fields. Used in `examples/smartds_radial_histograms.ipynb`, `examples/smartds_quicklook_profiles.ipynb`, `examples/planet.py`, `starwinds_analysis/pipelines/quicklook2d.py`, `examples/earth-xuv-neutrals/earth-xuv-neutrals.py`.
- `plot_binned_vs_radius`: Plot binned radial summaries (mean/median/sum) for one or more fields. Used in `examples/smartds_radial_histograms.ipynb`, `examples/smartds_quicklook_profiles.ipynb`, `examples/planet.py`, `starwinds_analysis/pipelines/quicklook2d.py`, `examples/earth-xuv-neutrals/earth-xuv-neutrals.py`.
- `plot_radial_hist2d`: 2D histogram (radius vs field value) as a compact replacement for old "monster" plots. Used in `examples/smartds_radial_histograms.ipynb`, `starwinds_analysis/pipelines/quicklook2d.py`.

## `starwinds_analysis/visualisation/profile_plots.py`
- `shell_profile_height`: Return `height [R]` from a shell-profile dict (fallback from radius). Used in `test/test_profile_plotting.py`, `starwinds_analysis/pipelines/quicklook2d.py`, `starwinds_analysis/visualisation/profile_plots.py`.
- `plot_shell_height_series`: Generic shell-profile line plot primitive (height on x, chosen quantity on y). Used in `test/test_profile_plotting.py`, `starwinds_analysis/pipelines/quicklook2d.py`.

## `starwinds_analysis/vtk_utils.py`
- `read`: Read a `.plt` file and optionally convert to base SI after 3D-grid conversion. Used in `test/test_isosurface.py`, `test/test_read_plt.py`, `test/test_volumetric.py`, `test/test_integrals.py`.
- `convert`: Convert a `starwinds_readplt.Dataset` into an unstructured 3D grid. Used in `starwinds_analysis/vtk_utils.py`.
- `convert_to_base_si`: Rename/scale common BATSRUS variables in a 3D visualisation grid into base SI units. Used in `starwinds_analysis/vtk_utils.py`.
